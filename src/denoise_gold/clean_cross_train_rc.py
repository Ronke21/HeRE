"""
Cross-trained RC (Relation Classification) for silver data cleaning.

For each (model, K) combination:
  1. Split silver data into K folds (stratified by predicate).
  2. For fold i: fine-tune `model` on fold i's data only.
     Task: multi-class predicate classification from (text, subject, object).
     Silver rows are treated as positive (distant-supervision assumption).
  3. Apply model_i to all folds j ≠ i → argmax predicted predicate for each row.
  4. Each silver row collects K-1 votes (one per model not trained on its fold):
       vote_correct = 1 if predicted_predicate == original_predicate, else 0
     Majority vote → cleaned relation_present label (threshold configurable).
  5. Apply all K fold-models to gold data → K votes per gold row → majority → evaluate.

Cleaning rationale:
  Each fold model sees 1/K of the silver data (noisy). When it correctly predicts
  the predicate of a held-out row, that row is more likely to genuinely express the
  relation. Rows that multiple independent (differently-noisy) models consistently
  mis-classify are likely false positives in the silver data.

Models (base HuggingFace encoders, NOT the fine-tuned NLI checkpoints):
  xlmroberta   — FacebookAI/xlm-roberta-large
  mmbert       — jhu-clsp/mmBERT-base
  neodictabert — dicta-il/neodictabert
  alephbert    — onlplab/alephbert-base
  me5large     — intfloat/multilingual-e5-large

K values: 3, 5, 7, 10

Output layout:
  outputs/cross_train_rc/
    summary.txt                    ← combined table across all (model, K)
    {model_tag}/
      summary.txt                  ← per-model table across K values
      k{K}/
        fold_{i}/
          train.log
          gold_preds.csv           ← 500 gold rows with fold-i prediction
        silver_cleaned.csv         ← silver rows with vote_count + cleaned_label
        gold_classified.csv        ← 500 gold rows with K votes + final pred + metrics
        summary.txt

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m scripts_clean_data.clean_cross_train_rc
  CUDA_VISIBLE_DEVICES=0 python -m scripts_clean_data.clean_cross_train_rc --models xlmroberta mmbert
  CUDA_VISIBLE_DEVICES=0 python -m scripts_clean_data.clean_cross_train_rc --k-values 3 5 --max-silver 200000
  CUDA_VISIBLE_DEVICES=0 python -m scripts_clean_data.clean_cross_train_rc --debug 5000 --k-values 3

Conda env: hreb_relation_extraction
"""

import os
import csv
import gc
import json
import time
import logging
import argparse
import random
import contextlib
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from tqdm import tqdm

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except Exception:
    _BNB_AVAILABLE = False

try:
    import flash_attn  # noqa: F401
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths / macros
# ---------------------------------------------------------------------------

SILVER_FILE = "data/prepared_silver.parquet"
GOLD_FILE   = "data/prepared_gold_500.csv"   # preprocessed text + relation_present
OUTPUT_BASE = "outputs/cross_train_rc"

MODEL_LIST = [
    ("FacebookAI/xlm-roberta-large",    "xlmroberta"),
    ("jhu-clsp/mmBERT-base",            "mmbert"),
    ("dicta-il/neodictabert",           "neodictabert"),
    # ("onlplab/alephbert-base",          "alephbert"),   # CUDA embedding assertion — disabled
    ("intfloat/multilingual-e5-large",  "me5large"),
    ("facebook/xlm-roberta-xl",         "xlmrobertaxl"),
    ("facebook/xlm-roberta-xxl",        "xlmrobertaxxl"),
]

K_VALUES         = [3, 5, 7, 10]
TRAIN_STEPS_LIST  = [10000]      # default: step budgets to sweep
TRAIN_EPOCHS_LIST = []           # if non-empty, sweep epoch counts instead of steps
EVAL_EVERY_N_STEPS = 500         # evaluate on gold during training every N steps
TRAIN_BATCH_SIZE  = 256
EVAL_BATCH_SIZE   = 1024
LEARNING_RATE    = 2e-5
WEIGHT_DECAY     = 0.01
MAX_SEQ_LENGTH   = 256
WARMUP_RATIO     = 0.1
MIN_PRED_FREQ    = 20     # silver predicates below this are excluded from vocab
VOTE_THRESHOLD   = 0.5   # fraction of correct votes to label cleaned_label=1
RANDOM_SEED      = 42
LOG_EVERY_N      = 200


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RCDataset(Dataset):
    """
    Pair classification for predicate prediction.
      First segment : "{subject} | {object}"
      Second segment: passage text
      Label         : predicate index
    Rows with predicates outside the vocab are silently dropped.
    """

    def __init__(self, rows: list[dict], pred2idx: dict, tokenizer, max_length: int):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.rows   = [r for r in rows if r["predicate"] in pred2idx]
        self.labels = [pred2idx[r["predicate"]] for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row   = self.rows[idx]
        enc   = self.tokenizer(
            f"{row['subject']} | {row['object']}",
            row["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Pre-tokenized dataset (fast path when silver tokens are cached)
# ---------------------------------------------------------------------------

class PreTokenizedDataset(Dataset):
    """Dataset backed by pre-computed token tensors (see build_silver_tokens)."""
    def __init__(self, tokens: dict):
        self.input_ids      = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]
        self.token_type_ids = tokens.get("token_type_ids")
        self.labels         = tokens["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {
            "input_ids":      self.input_ids[i],
            "attention_mask": self.attention_mask[i],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[i]
        item["label"] = self.labels[i]
        return item


def get_amp_ctx(amp_dtype: Optional[torch.dtype]):
    """Return an autocast context manager, or nullcontext if amp disabled."""
    if amp_dtype is None or not torch.cuda.is_available():
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=amp_dtype)


def load_tokenizer(model_id: str, log: logging.Logger):
    log.info(f"  loading tokenizer: {model_id}")
    return transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def build_silver_tokens(
    rows:       list[dict],
    pred2idx:   dict,
    tokenizer,
    max_length: int,
    model_tag:  str,
    cache_dir:  str,
    log:        logging.Logger,
    batch_size: int = 512,
) -> dict:
    """
    Pre-tokenize all silver rows whose predicate is in pred2idx.
    Cached to cache_dir/silver_tokens_{model_tag}_l{max_length}.pt.

    Returns dict:
      input_ids, attention_mask, [token_type_ids] : LongTensor (N, L)
      labels       : LongTensor (N,)   — predicate index
      orig_indices : np.int64 (N,)     — index back into silver_rows
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"silver_tokens_{model_tag}_l{max_length}.pt")

    if os.path.exists(cache_path):
        log.info(f"  [tokens] loading cached → {cache_path}")
        return torch.load(cache_path, weights_only=False)

    valid_idx  = [i for i, r in enumerate(rows) if r["predicate"] in pred2idx]
    valid_rows = [rows[i]                        for i in valid_idx]
    labels     = [pred2idx[rows[i]["predicate"]] for i in valid_idx]
    dropped    = len(rows) - len(valid_idx)

    log.info(
        f"  [tokens] tokenizing {len(valid_rows):,} / {len(rows):,} silver rows  "
        f"(dropped {dropped:,} unknown predicates) ..."
    )
    t0 = time.time()

    ids_list, attn_list, tti_list, has_tti = [], [], [], None
    for start in tqdm(range(0, len(valid_rows), batch_size), desc="  pre-tokenize"):
        batch = valid_rows[start : start + batch_size]
        enc = tokenizer(
            [f"{r['subject']} | {r['object']}" for r in batch],
            [r["text"] for r in batch],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        ids_list.append(enc["input_ids"])
        attn_list.append(enc["attention_mask"])
        if has_tti is None:
            has_tti = "token_type_ids" in enc
        if has_tti:
            tti_list.append(enc["token_type_ids"])

    result = {
        "input_ids":      torch.cat(ids_list,  dim=0),
        "attention_mask": torch.cat(attn_list, dim=0),
        "labels":         torch.tensor(labels, dtype=torch.long),
        "orig_indices":   np.array(valid_idx,  dtype=np.int64),
    }
    if has_tti:
        result["token_type_ids"] = torch.cat(tti_list, dim=0)

    torch.save(result, cache_path)
    log.info(f"  [tokens] done  elapsed={_fmt(time.time()-t0)}  cached → {cache_path}")
    return result


def subset_tokens(tokens: dict, mask: np.ndarray) -> dict:
    """Return a subset of a tokens dict selected by a boolean mask over N rows."""
    where = np.where(mask)[0]
    idx   = torch.from_numpy(where).long()
    result = {
        "input_ids":      tokens["input_ids"][idx],
        "attention_mask": tokens["attention_mask"][idx],
        "labels":         tokens["labels"][idx],
        "orig_indices":   tokens["orig_indices"][where],
    }
    if "token_type_ids" in tokens:
        result["token_type_ids"] = tokens["token_type_ids"][idx]
    return result


def predict_from_tokens(
    tokens:          dict,
    idx2pred:        dict,
    model,
    device:          torch.device,
    log:             logging.Logger,
    eval_batch_size: int,
    amp_dtype:       Optional[torch.dtype],
    desc:            str = "infer",
) -> list[str]:
    """Inference from pre-tokenized tensors. Returns predicted predicate strings."""
    n       = len(tokens["labels"])
    has_tti = "token_type_ids" in tokens
    preds   = []

    for start in tqdm(range(0, n, eval_batch_size),
                      desc=f"    {desc}", file=TqdmToLogger(log), leave=False):
        end    = min(start + eval_batch_size, n)
        inputs = {
            "input_ids":      tokens["input_ids"][start:end].to(device),
            "attention_mask": tokens["attention_mask"][start:end].to(device),
        }
        if has_tti:
            inputs["token_type_ids"] = tokens["token_type_ids"][start:end].to(device)
        with torch.no_grad(), get_amp_ctx(amp_dtype):
            out    = model(**inputs)
            logits = out.logits if hasattr(out, "logits") else out[0]
        preds.extend(idx2pred.get(i, "UNKNOWN") for i in logits.argmax(dim=-1).cpu().tolist())

    return preds


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class TqdmToLogger:
    def __init__(self, logger): self._logger = logger
    def write(self, msg):
        if msg := msg.strip(): self._logger.info(msg)
    def flush(self): pass


def setup_logger(log_path: str, name: str, mode: str = "w") -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False   # prevent duplicate output from parent loggers
    return logger


def _fmt(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, s   = divmod(rem, 60)
    if h:  return f"{h}h {m}m {s}s"
    if m:  return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str, max_rows: Optional[int], log: logging.Logger) -> list[dict]:
    log.info(f"  loading: {path}")
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        if max_rows:
            df = df.iloc[:max_rows]
        rows = df.astype(str).to_dict(orient="records")
    else:
        rows = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if max_rows and i >= max_rows:
                    break
                rows.append(row)
    log.info(f"  loaded {len(rows):,} rows")
    return rows


def build_pred_vocab(
    silver_rows: list[dict],
    gold_rows:   list[dict],
    min_freq:    int,
    log:         logging.Logger,
) -> tuple[dict, dict]:
    """
    Returns (pred2idx, idx2pred).
    All gold predicates are guaranteed to be in the vocab (even if rare in silver)
    so inference on gold always produces a meaningful score for the correct class.
    """
    freq: dict[str, int] = defaultdict(int)
    for r in silver_rows:
        freq[r["predicate"]] += 1

    vocab = {p: i for i, p in enumerate(
        sorted(p for p, c in freq.items() if c >= min_freq)
    )}
    n_from_silver = len(vocab)

    extra = 0
    for r in gold_rows:
        if r["predicate"] not in vocab:
            vocab[r["predicate"]] = len(vocab)
            extra += 1

    log.info(
        f"  vocab: {n_from_silver} silver predicates (min_freq={min_freq}) "
        f"+ {extra} gold-only = {len(vocab)} total  "
        f"(dropped {sum(1 for c in freq.values() if c < min_freq)} rare)"
    )
    idx2pred = {v: k for k, v in vocab.items()}
    return vocab, idx2pred


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_id: str, num_labels: int, log: logging.Logger, tokenizer=None,
               load_dtype: Optional[torch.dtype] = None, device_map: Optional[str] = None,
               flash_attn: bool = False):
    log.info(f"  loading: {model_id}  (num_labels={num_labels})")
    t0 = time.time()
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
    config = transformers.AutoConfig.from_pretrained(
        model_id, num_labels=num_labels, trust_remote_code=True
    )
    kwargs = dict(config=config, trust_remote_code=True, ignore_mismatched_sizes=True)
    if load_dtype is not None:
        kwargs["torch_dtype"] = load_dtype
    if device_map is not None:
        kwargs["device_map"] = device_map
    if flash_attn:
        if _FLASH_ATTN_AVAILABLE:
            kwargs["attn_implementation"] = "flash_attention_2"
            log.info("  flash_attention_2 enabled")
        else:
            log.warning("  --flash-attn requested but flash_attn not installed — skipping")
    # XLMRobertaXLForSequenceClassification triggers modeling_layers.py → torchvision
    # which is broken in this env (operator mismatch).  Bypass the lazy-import chain by
    # importing XLMRobertaForSequenceClassification directly from its module file, then
    # loading the safetensors weights bypassing the class-name metadata check.
    if getattr(config, "model_type", "") in ("xlm-roberta-xl", "xlm-roberta-xxl"):
        log.info("  XL/XXL workaround: direct module import + safetensors load")
        import importlib
        from safetensors.torch import load_file as _st_load_file
        from huggingface_hub import hf_hub_download as _hf_download
        _mod = importlib.import_module(
            "transformers.models.xlm_roberta.modeling_xlm_roberta"
        )
        _Cls = _mod.XLMRobertaForSequenceClassification
        config.model_type = "xlm-roberta"  # needed so the class accepts the config
        model = _Cls(config)
        sf_path = _hf_download(model_id, "model.safetensors", local_files_only=True)
        state_dict = _st_load_file(sf_path)
        # XLMRobertaXL uses different LayerNorm key names than XLMRobertaForSequenceClassification.
        # Remap before loading so weights land in the right slots.
        import re as _re
        remapped = {}
        for k, v in state_dict.items():
            nk = k
            # attention.self_attn_layer_norm → attention.output.LayerNorm
            nk = _re.sub(
                r"(roberta\.encoder\.layer\.\d+)\.attention\.self_attn_layer_norm\.",
                r"\1.attention.output.LayerNorm.", nk,
            )
            # layer.N.LayerNorm → layer.N.output.LayerNorm  (post-FFN norm)
            nk = _re.sub(
                r"(roberta\.encoder\.layer\.\d+)\.LayerNorm\.",
                r"\1.output.LayerNorm.", nk,
            )
            # roberta.encoder.LayerNorm → roberta.embeddings.LayerNorm
            nk = nk.replace("roberta.encoder.LayerNorm.", "roberta.embeddings.LayerNorm.")
            remapped[nk] = v
        state_dict = remapped
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log.info(f"  weights: {len(state_dict)} tensors  missing={len(missing)}  unexpected={len(unexpected)}")
        if load_dtype is not None:
            model = model.to(load_dtype)
    else:
        try:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
        except Exception as e:
            log.warning(f"  safetensors load failed ({e}), retrying")
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_id, **kwargs, use_safetensors=False,
            )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"  ready in {_fmt(time.time()-t0)}  ({n_params:.0f}M params)")
    return model, tokenizer


def unload_model(model, log: logging.Logger):
    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("  GPU cache cleared")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_fold(
    train_rows:   list[dict],
    pred2idx:     dict,
    idx2pred:     dict,
    model_id:     str,
    model_tag:    str,
    fold_idx:     int,
    log:          logging.Logger,
    args,
    n_epochs:     Optional[int]   = None,
    max_steps:    Optional[int]   = None,
    eval_rows:    Optional[list]  = None,
    eval_labels:  Optional[list]  = None,
    tokenizer     = None,
    train_tokens: Optional[dict]  = None,
    amp_dtype:    Optional[torch.dtype] = None,
    scaler        = None,
    lr:           Optional[float] = None,
    load_dtype:   Optional[torch.dtype] = None,
    device_map:   Optional[str] = None,
    flash_attn:   bool = False,
    adam8bit:     bool = False,
):
    """
    Fine-tune model on train_rows (or pre-tokenized train_tokens).

    Every args.eval_every steps (and at the final step) runs inference on eval_rows
    (gold), records gold F1, and keeps the best-F1 model state in memory.
    Returns the best model (or final model if eval disabled).

    Returns: (model, tokenizer, device, eval_curve, best_step, best_f1)
      eval_curve : list of dicts {step, accuracy, precision, recall, f1, TP, FP, FN, TN}
      best_step  : step at which best gold F1 was achieved
      best_f1    : best gold F1 seen during training
    """
    model, tokenizer = load_model(model_id, len(pred2idx), log, tokenizer=tokenizer,
                                  load_dtype=load_dtype, device_map=device_map,
                                  flash_attn=flash_attn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_map is not None:
        device = torch.device("cuda:0")
    else:
        model.to(device)
        base_model = model
        if getattr(args, "grad_ckpt", False):
            base_model.gradient_checkpointing_enable()
            log.info("  gradient checkpointing enabled")
        if getattr(args, "compile", False):
            model = torch.compile(model)
            log.info("  torch.compile enabled")
        if torch.cuda.device_count() > 1:
            log.info(f"  DataParallel across {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

    if train_tokens is not None:
        dataset = PreTokenizedDataset(train_tokens)
        log.info(f"  train: {len(dataset):,} rows (pre-tokenized)")
    else:
        dataset = RCDataset(train_rows, pred2idx, tokenizer, args.max_length)
        log.info(
            f"  train: {len(dataset):,} rows "
            f"(dropped {len(train_rows)-len(dataset):,} with unknown predicates)"
        )
    loader = DataLoader(
        dataset, batch_size=args.train_batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )

    steps_per_epoch = len(loader)
    if max_steps is not None:
        total_steps  = max_steps
        budget_label = f"max_steps={max_steps}"
    else:
        total_steps  = n_epochs * steps_per_epoch
        budget_label = f"epochs={n_epochs}"

    effective_lr = lr if lr is not None else args.lr
    if adam8bit and _BNB_AVAILABLE:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=effective_lr, weight_decay=WEIGHT_DECAY
        )
    else:
        if adam8bit:
            log.warning("  --adam8bit requested but bitsandbytes not installed — using AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=effective_lr, weight_decay=WEIGHT_DECAY
        )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    do_eval = bool(eval_rows) and args.eval_every > 0
    log.info(
        f"  {budget_label}  steps_per_epoch={steps_per_epoch}  "
        f"total_steps={total_steps}  lr={effective_lr}  "
        f"eval_every={args.eval_every if do_eval else 'disabled'}"
    )

    t_start    = time.time()
    step       = 0
    epoch      = 0
    best_f1    = -1.0
    best_step  = 0
    best_state: Optional[dict] = None
    eval_curve: list[dict]     = []

    def _run_eval(current_step: int):
        nonlocal best_f1, best_step, best_state
        base = model.module if isinstance(model, nn.DataParallel) else model
        base.eval()
        preds = predict_predicates(
            eval_rows, idx2pred, base, tokenizer, device, log, args,
            amp_dtype=amp_dtype, desc="eval/gold",
        )
        binary = [
            "1" if preds[j] == eval_rows[j]["predicate"] else "0"
            for j in range(len(eval_rows))
        ]
        m = compute_metrics(binary, eval_labels)
        eval_curve.append({"step": current_step, **m})
        log.info(
            f"    [eval step={current_step}]  f1={m['f1']:.3f}  "
            f"acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  rec={m['recall']:.3f}  "
            f"TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}"
        )
        if m["f1"] > best_f1:
            best_f1   = m["f1"]
            best_step = current_step
            best_state = {k: v.cpu().clone() for k, v in base.state_dict().items()}
            log.info(f"    *** new best  step={best_step}  gold_f1={best_f1:.3f} ***")

    while step < total_steps:
        epoch += 1
        model.train()
        running_loss = 0.0
        epoch_steps  = 0

        for batch in tqdm(
            loader,
            desc=f"    ep{epoch} fold{fold_idx}/{model_tag}",
            file=TqdmToLogger(log), leave=False,
        ):
            if step >= total_steps:
                break
            labels = batch.pop("label").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with get_amp_ctx(amp_dtype):
                out    = model(**inputs)
                logits = out.logits if hasattr(out, "logits") else out[0]
                loss   = criterion(logits, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            epoch_steps  += 1
            step         += 1

            if step % LOG_EVERY_N == 0:
                log.info(
                    f"    step [{step}/{total_steps}]  ep{epoch}  "
                    f"loss={running_loss/epoch_steps:.4f}  "
                    f"elapsed={_fmt(time.time()-t_start)}"
                )

            if do_eval and step % args.eval_every == 0:
                _run_eval(step)
                model.train()

        log.info(
            f"  epoch {epoch}  steps {step}/{total_steps}  "
            f"avg_loss={running_loss/max(epoch_steps,1):.4f}  "
            f"({_fmt(time.time()-t_start)})"
        )

    # Final eval at end of training
    if do_eval:
        _run_eval(step)

    # Load best checkpoint (best gold F1 during training)
    base = model.module if isinstance(model, nn.DataParallel) else model
    if best_state is not None:
        base.load_state_dict(best_state)
        log.info(f"  loaded best checkpoint: step={best_step}  gold_f1={best_f1:.3f}")
    else:
        best_step = step
        best_f1   = eval_curve[-1]["f1"] if eval_curve else 0.0

    base.eval()
    return base, tokenizer, device, eval_curve, best_step, best_f1


# ---------------------------------------------------------------------------
# Inference  —  returns predicted predicate string per row (argmax)
# ---------------------------------------------------------------------------

def predict_predicates(
    rows:      list[dict],
    idx2pred:  dict,
    model,
    tokenizer,
    device:    torch.device,
    log:       logging.Logger,
    args,
    amp_dtype: Optional[torch.dtype] = None,
    desc:      str = "infer",
) -> list[str]:
    """Returns list of predicted predicate strings (argmax), one per row."""
    predictions: list[str] = []
    n = len(rows)

    for start in tqdm(range(0, n, args.eval_batch_size),
                      desc=f"    {desc}", file=TqdmToLogger(log), leave=False):
        batch = rows[start : start + args.eval_batch_size]
        enc = tokenizer(
            [f"{r['subject']} | {r['object']}" for r in batch],
            [r["text"] for r in batch],
            max_length=args.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad(), get_amp_ctx(amp_dtype):
            out    = model(**inputs)
            logits = out.logits if hasattr(out, "logits") else out[0]
        argmax = logits.argmax(dim=-1).cpu().tolist()
        predictions.extend(idx2pred.get(i, "UNKNOWN") for i in argmax)

    return predictions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: list[str], gold: list[str]) -> dict:
    """Binary classification metrics. preds and gold are strings '0'/'1'."""
    TP = FP = FN = TN = 0
    for p, g in zip(preds, gold):
        pos  = p == "1"
        gpos = g == "1"
        if   pos and gpos:     TP += 1
        elif pos and not gpos: FP += 1
        elif not pos and gpos: FN += 1
        else:                  TN += 1
    total = TP + FP + FN + TN
    prec  = TP / (TP + FP)        if (TP + FP)   else 0.0
    rec   = TP / (TP + FN)        if (TP + FN)   else 0.0
    f1    = 2*prec*rec/(prec+rec)  if (prec+rec)  else 0.0
    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "accuracy":  (TP+TN)/total if total else 0.0,
        "precision": prec, "recall": rec, "f1": f1,
    }


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------

def _metrics_row(m: dict) -> str:
    return (f"acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
            f"rec={m['recall']:.3f}  f1={m['f1']:.3f}  "
            f"TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}")


def write_summary(results: dict, summary_path: str, log: logging.Logger, total: float):
    """
    results: {model_tag: {K: {budget_label: {'fold_metrics': [...], 'ensemble': {...}}}}}
    """
    W = 145
    lines = ["=" * W,
             "CROSS-TRAINED RC — CLEANING SUMMARY",
             f"Total wall time: {_fmt(total)}",
             f"Vote threshold: {VOTE_THRESHOLD}",
             "=" * W]

    # Combined table: one row per (model, K, budget)
    hdr = (f"  {'model':<14} {'K':>3}  {'budget':<12}  "
           f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
           f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
           f"{'silver_pos%':>11}  {'time':>8}")
    lines += ["", "  ENSEMBLE GOLD METRICS  (majority vote across K fold-models)", hdr,
              "  " + "-" * (len(hdr) - 2)]

    for model_tag, k_dict in sorted(results.items()):
        for K, budget_dict in sorted(k_dict.items()):
            for budget_label, res in sorted(budget_dict.items()):
                m  = res["ensemble"]["metrics"]
                sp = res["silver_pos_pct"]
                t  = res.get("wall_time", "")
                lines.append(
                    f"  {model_tag:<14} {K:>3}  {budget_label:<12}  "
                    f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} "
                    f"{m['recall']:>6.3f} {m['f1']:>6.3f}  "
                    f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
                    f"{sp:>10.1f}%  {t:>8}"
                )

    # Per-model details
    for model_tag, k_dict in sorted(results.items()):
        lines += ["", "─" * W, f"Model: {model_tag}", "─" * W]
        for K, budget_dict in sorted(k_dict.items()):
            for budget_label, res in sorted(budget_dict.items()):
                lines += ["", f"  K={K}  budget={budget_label}"]
                lines.append(
                    f"  {'fold':<6}  "
                    f"{'gold_acc':>8} {'gold_prec':>9} {'gold_rec':>8} {'gold_f1':>7}  "
                    f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
                    f"{'best_step':>9} {'train_f1':>8}  {'time':>8}"
                )
                lines.append("  " + "-" * 100)
                for fr in res["fold_metrics"]:
                    m = fr["metrics"]
                    lines.append(
                        f"  {fr['fold_idx']:<6}  "
                        f"{m['accuracy']:>8.3f} {m['precision']:>9.3f} "
                        f"{m['recall']:>8.3f} {m['f1']:>7.3f}  "
                        f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
                        f"{str(fr.get('best_step','?')):>9} "
                        f"{fr.get('best_train_f1', 0.0):>8.3f}  "
                        f"{fr.get('time',''):>8}"
                    )
                m = res["ensemble"]["metrics"]
                lines += [
                    "",
                    f"  Ensemble (majority vote, thresh={VOTE_THRESHOLD}):  {_metrics_row(m)}",
                    f"  Silver cleaned pos rate: {res['silver_pos_pct']:.1f}%",
                ]

    lines += ["", "=" * W]
    text = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    for line in lines:
        log.info(line)


# ---------------------------------------------------------------------------
# Checkpoint (resume support)
# ---------------------------------------------------------------------------

def save_checkpoint(
    k_dir:           str,
    completed_folds: set,
    sil_correct:     np.ndarray,
    sil_total:       np.ndarray,
    gold_correct:    np.ndarray,
    fold_metrics:    list,
    fold_timing:     dict,
    log:             logging.Logger,
):
    np.savez_compressed(
        os.path.join(k_dir, "checkpoint.npz"),
        sil_correct=sil_correct,
        sil_total=sil_total,
        gold_correct=gold_correct,
    )
    with open(os.path.join(k_dir, "checkpoint.json"), "w", encoding="utf-8") as f:
        json.dump({
            "completed_folds": sorted(completed_folds),
            "fold_metrics":    fold_metrics,
            "fold_timing":     fold_timing,
        }, f, ensure_ascii=False, indent=2)
    log.info(f"  checkpoint saved  (folds done: {sorted(completed_folds)})")


def load_checkpoint(k_dir: str, log: logging.Logger) -> Optional[dict]:
    npz_path  = os.path.join(k_dir, "checkpoint.npz")
    json_path = os.path.join(k_dir, "checkpoint.json")
    if not (os.path.exists(npz_path) and os.path.exists(json_path)):
        return None
    arrays = np.load(npz_path)
    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)
    log.info(f"  checkpoint found  —  folds already done: {meta['completed_folds']}")
    return {
        "sil_correct":     arrays["sil_correct"],
        "sil_total":       arrays["sil_total"],
        "gold_correct":    arrays["gold_correct"],
        "completed_folds": set(meta["completed_folds"]),
        "fold_metrics":    meta["fold_metrics"],
        "fold_timing":     meta["fold_timing"],
    }


def delete_checkpoint(k_dir: str, log: logging.Logger):
    for fname in ("checkpoint.npz", "checkpoint.json"):
        p = os.path.join(k_dir, fname)
        if os.path.exists(p):
            os.remove(p)
    log.info("  checkpoint deleted (experiment complete)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-trained RC for silver-data cleaning"
    )
    parser.add_argument("--silver",       default=SILVER_FILE)
    parser.add_argument("--gold",         default=GOLD_FILE)
    parser.add_argument("--output-base",  default=OUTPUT_BASE)
    parser.add_argument(
        "--models", nargs="+",
        default=[tag for _, tag in MODEL_LIST],
        choices=[tag for _, tag in MODEL_LIST],
        help="Which models to run (default: all)",
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=K_VALUES,
        help="K values to evaluate (default: 3 5 7 10)",
    )
    parser.add_argument(
        "--max-silver", type=int, default=None,
        help="Cap silver rows loaded (default: all ~3.1M)",
    )
    parser.add_argument(
        "--epochs-list", type=int, nargs="+", default=TRAIN_EPOCHS_LIST,
        dest="epochs_list",
        help="Epoch counts to sweep (e.g. 1 2 3). Ignored if --steps-list is set.",
    )
    parser.add_argument(
        "--steps-list", type=int, nargs="+", default=TRAIN_STEPS_LIST,
        dest="steps_list",
        help="Step counts to sweep (e.g. 5000 10000 20000). Overrides --epochs-list.",
    )
    parser.add_argument(
        "--eval-every", type=int, default=EVAL_EVERY_N_STEPS, dest="eval_every",
        help="Evaluate on gold every N training steps; 0 = disable (default: 500).",
    )
    parser.add_argument("--train-batch",  type=int,   default=TRAIN_BATCH_SIZE,
                        dest="train_batch_size")
    parser.add_argument("--eval-batch",   type=int,   default=EVAL_BATCH_SIZE,
                        dest="eval_batch_size")
    parser.add_argument("--lr",           type=float, default=LEARNING_RATE,
                        help="Single LR (used when --lr-list is not set)")
    parser.add_argument("--lr-list",      type=float, nargs="+", default=None,
                        dest="lr_list",
                        help="Sweep multiple LRs (e.g. 1e-5 2e-5 5e-5). Overrides --lr.")
    parser.add_argument("--max-length",   type=int,   default=MAX_SEQ_LENGTH)
    parser.add_argument("--min-pred-freq", type=int,  default=MIN_PRED_FREQ)
    parser.add_argument(
        "--vote-threshold", type=float, default=VOTE_THRESHOLD,
        help="Fraction of correct votes required for cleaned_label=1 (default 0.5)",
    )
    parser.add_argument("--seed",         type=int,   default=RANDOM_SEED)
    parser.add_argument(
        "--debug", type=int, default=0,
        help="Use first N silver rows only; 0 = full dataset",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable automatic mixed precision (default: bf16 if available, else fp16)",
    )
    parser.add_argument(
        "--amp-dtype", choices=["bf16", "fp16"], default="bf16", dest="amp_dtype_str",
        help="AMP dtype when enabled (default: bf16)",
    )
    parser.add_argument(
        "--token-cache-dir", default="data/token_cache", dest="token_cache_dir",
        help="Directory for pre-tokenized silver tensor cache (default: data/token_cache)",
    )
    parser.add_argument(
        "--device-map", default=None, dest="device_map",
        help="HuggingFace device_map for model loading (e.g. 'auto' to split across GPUs)",
    )
    parser.add_argument(
        "--flash-attn", action="store_true", dest="flash_attn",
        help="Enable Flash Attention 2 (requires flash_attn package)",
    )
    parser.add_argument(
        "--adam8bit", action="store_true", dest="adam8bit",
        help="Use 8-bit AdamW optimizer via bitsandbytes (saves ~3x optimizer memory)",
    )
    parser.add_argument(
        "--grad-ckpt", action="store_true", dest="grad_ckpt",
        help="Enable gradient checkpointing (trades compute for activation memory)",
    )
    parser.add_argument(
        "--compile", action="store_true", dest="compile",
        help="Enable torch.compile for ~15-30%% speedup (adds ~2 min startup per fold)",
    )
    args = parser.parse_args()

    # Build budget sweep: cartesian product of (steps or epochs) × LR
    lr_values = sorted(args.lr_list) if args.lr_list else [args.lr]
    def _lr_str(v): return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e")
    if args.steps_list:
        step_budgets = [(f"steps{s}", {"max_steps": s}) for s in sorted(args.steps_list)]
    else:
        step_budgets = [(f"ep{e}", {"n_epochs": e}) for e in sorted(args.epochs_list)]
    budgets = [
        (f"{slabel}_lr{_lr_str(lr)}", {**skw, "lr": lr})
        for slabel, skw in step_budgets
        for lr in lr_values
    ]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    silver_path = os.path.join(base, args.silver)
    gold_path   = os.path.join(base, args.gold)
    out_base    = os.path.join(base, args.output_base)
    os.makedirs(out_base, exist_ok=True)

    log = setup_logger(os.path.join(out_base, "run.log"), "cross_train_rc")
    wall_start = time.time()

    # ── AMP setup ─────────────────────────────────────────────────────────────
    amp_dtype: Optional[torch.dtype] = None
    scaler = None
    if not args.no_amp and torch.cuda.is_available():
        if args.amp_dtype_str == "bf16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
            scaler    = torch.cuda.amp.GradScaler()

    token_cache_dir = os.path.join(base, args.token_cache_dir)

    log.info("=" * 70)
    log.info("cross_train_rc.py  started")
    log.info(f"  silver:         {silver_path}")
    log.info(f"  gold:           {gold_path}")
    log.info(f"  output:         {out_base}")
    log.info(f"  models:         {args.models}")
    log.info(f"  K values:       {args.k_values}")
    log.info(f"  budgets:        {[bl for bl, _ in budgets]}  lr={args.lr}")
    log.info(f"  eval_every:     {args.eval_every} steps  (0=disabled)")
    log.info(f"  max_length:     {args.max_length}")
    log.info(f"  vote_threshold: {args.vote_threshold}")
    log.info(f"  max_silver:     {args.max_silver or 'all'}")
    log.info(f"  debug:          {args.debug or 'off'}")
    log.info(f"  AMP:            {amp_dtype or 'disabled'}  scaler={scaler is not None}")
    log.info(f"  CUDA: {torch.cuda.is_available()}  GPUs: {torch.cuda.device_count()}")
    log.info("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    max_sil = args.debug if args.debug else args.max_silver
    log.info("[load] silver")
    silver_rows = load_csv(silver_path, max_sil, log)
    log.info("[load] gold")
    gold_rows   = load_csv(gold_path,   None,    log)

    gold_labels = [r["relation_present"] for r in gold_rows]
    n_pos = gold_labels.count("1")
    log.info(
        f"gold: {len(gold_rows)} rows  pos={n_pos}  neg={len(gold_rows)-n_pos}  "
        f"({100*n_pos/len(gold_rows):.1f}% pos)"
    )

    # ── Predicate vocabulary (shared across all experiments) ──────────────────
    log.info("[vocab]")
    pred2idx, idx2pred = build_pred_vocab(
        silver_rows, gold_rows, args.min_pred_freq, log
    )
    vocab_path = os.path.join(out_base, "pred2idx.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(pred2idx, f, ensure_ascii=False, indent=2)
    log.info(f"  vocab saved → {vocab_path}")

    # tag → model_id lookup
    tag2id = {tag: mid for mid, tag in MODEL_LIST}

    all_results: dict = {}  # model_tag → {K → {budget_label → result_dict}}

    # ── Outer loop: model ──────────────────────────────────────────────────────
    for model_tag in args.models:
        model_id   = tag2id[model_tag]
        model_dir  = os.path.join(out_base, model_tag)
        os.makedirs(model_dir, exist_ok=True)
        mlog = setup_logger(
            os.path.join(model_dir, "run.log"), f"cross_train_rc.{model_tag}"
        )
        mlog.info("=" * 70)
        mlog.info(f"MODEL: {model_tag}  ({model_id})")
        mlog.info("=" * 70)

        # Pre-tokenize silver once per model; cached to disk for reuse
        tokenizer  = load_tokenizer(model_id, mlog)
        sil_tokens = build_silver_tokens(
            silver_rows, pred2idx, tokenizer, args.max_length,
            model_tag, token_cache_dir, mlog,
        )
        # In debug mode the cache was built from the full dataset; restrict to loaded rows.
        if args.debug:
            mask = sil_tokens["orig_indices"] < len(silver_rows)
            sil_tokens = subset_tokens(sil_tokens, mask)
        # The cache may have been built with a different pred2idx (e.g., a full run with
        # silver predicates). Drop any rows whose cached label is out of range.
        label_mask = sil_tokens["labels"].numpy() < len(pred2idx)
        if not label_mask.all():
            n_dropped = int((~label_mask).sum())
            mlog.info(f"  [tokens] dropping {n_dropped} cached rows with out-of-range labels")
            sil_tokens = subset_tokens(sil_tokens, label_mask)

        model_results: dict = {}

        # ── K loop ────────────────────────────────────────────────────────────
        for K in args.k_values:
            mlog.info("─" * 70)
            mlog.info(f"K = {K}")
            mlog.info("─" * 70)

            k_dir = os.path.join(model_dir, f"k{K}")
            os.makedirs(k_dir, exist_ok=True)
            klog  = setup_logger(
                os.path.join(k_dir, "run.log"), f"cross_train_rc.{model_tag}.k{K}", mode="a"
            )

            # ── Stratified fold assignment (shared across all budgets for this K) ──
            fold_ids = np.zeros(len(silver_rows), dtype=int)
            rng = np.random.default_rng(args.seed)
            pred_indices: dict[str, list[int]] = defaultdict(list)
            for i, r in enumerate(silver_rows):
                pred_indices[r["predicate"]].append(i)
            for indices in pred_indices.values():
                arr = np.array(indices)
                rng.shuffle(arr)
                for rank, idx in enumerate(arr):
                    fold_ids[idx] = rank % K
            dist = "  ".join(f"f{i}={int((fold_ids == i).sum()):,}" for i in range(K))
            klog.info(f"  fold distribution: {dist}")

            model_results[K] = {}

            # ── Budget loop ───────────────────────────────────────────────────
            for budget_label, budget_kwargs in budgets:
                klog.info("·" * 50)
                klog.info(f"  budget = {budget_label}")
                klog.info("·" * 50)

                exp_dir = os.path.join(k_dir, budget_label)
                os.makedirs(exp_dir, exist_ok=True)
                t_exp = time.time()

                # ── Checkpoint: resume if partial run exists ───────────────────
                ckpt = load_checkpoint(exp_dir, klog)
                if ckpt:
                    sil_correct       = ckpt["sil_correct"]
                    sil_total         = ckpt["sil_total"]
                    gold_correct      = ckpt["gold_correct"]
                    fold_metrics_list = ckpt["fold_metrics"]
                    fold_timing       = ckpt["fold_timing"]
                    completed_folds   = ckpt["completed_folds"]
                else:
                    sil_correct       = np.zeros(len(silver_rows), dtype=int)
                    sil_total         = np.zeros(len(silver_rows), dtype=int)
                    gold_correct      = np.zeros(len(gold_rows),   dtype=int)
                    fold_metrics_list = []
                    fold_timing       = {}
                    completed_folds   = set()

                # ── Fold loop ─────────────────────────────────────────────────
                for fold_i in range(K):
                    if fold_i in completed_folds:
                        klog.info(f"[fold {fold_i+1}/{K}]  skipped (already done in checkpoint)")
                        continue

                    klog.info(f"[fold {fold_i+1}/{K}]  training on fold {fold_i}  budget={budget_label}")

                    fold_dir = os.path.join(exp_dir, f"fold_{fold_i}")
                    os.makedirs(fold_dir, exist_ok=True)
                    flog = setup_logger(
                        os.path.join(fold_dir, "train.log"),
                        f"cross_train_rc.{model_tag}.k{K}.{budget_label}.fold{fold_i}",
                        mode="a",
                    )

                    # Subset pre-tokenized silver: train fold / inference folds
                    valid_fold_ids = fold_ids[sil_tokens["orig_indices"]]
                    train_tok = subset_tokens(sil_tokens, valid_fold_ids == fold_i)
                    infer_tok = subset_tokens(sil_tokens, valid_fold_ids != fold_i)
                    klog.info(f"  training rows: {len(train_tok['labels']):,}  infer rows: {len(infer_tok['labels']):,}")

                    t_fold = time.time()
                    try:
                        model, tokenizer, device, eval_curve, best_step, best_train_f1 = train_one_fold(
                            [], pred2idx, idx2pred, model_id, model_tag,
                            fold_i, flog, args,
                            **{k: v for k, v in budget_kwargs.items() if k != "lr"},
                            lr=budget_kwargs.get("lr"),
                            eval_rows=gold_rows, eval_labels=gold_labels,
                            tokenizer=tokenizer, train_tokens=train_tok,
                            amp_dtype=amp_dtype, scaler=scaler,
                            load_dtype=amp_dtype, device_map=args.device_map,
                            flash_attn=args.flash_attn, adam8bit=args.adam8bit,
                        )
                    except Exception as exc:
                        klog.error(f"  fold {fold_i} training failed: {exc}")
                        fold_metrics_list.append({
                            "fold_idx":      fold_i,
                            "metrics":       {k: 0 for k in
                                              ["TP","FP","FN","TN","accuracy","precision","recall","f1"]},
                            "time":          "FAILED",
                            "best_step":     0,
                            "best_train_f1": 0.0,
                        })
                        continue
                    t_trained = time.time()

                    # Save eval curve
                    eval_curve_path = os.path.join(fold_dir, "eval_curve.csv")
                    with open(eval_curve_path, "w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=["step","accuracy","precision","recall","f1",
                                           "TP","FP","FN","TN"],
                        )
                        writer.writeheader()
                        writer.writerows(eval_curve)
                    klog.info(
                        f"  eval curve saved → {eval_curve_path}  "
                        f"best_step={best_step}  best_train_f1={best_train_f1:.3f}"
                    )

                    # ── Infer on all other silver folds (pre-tokenized) ───────
                    klog.info(f"  inferring on {len(infer_tok['labels']):,} silver rows (folds ≠ {fold_i})")
                    t_inf = time.time()
                    other_preds = predict_from_tokens(
                        infer_tok, idx2pred, model, device, flog,
                        args.eval_batch_size, amp_dtype,
                        desc=f"silver/{model_tag}/k{K}/{budget_label}/fold{fold_i}",
                    )
                    t_sil_done = time.time()
                    klog.info(f"  silver inference done  ({_fmt(t_sil_done - t_inf)})")

                    for local_i, global_i in enumerate(infer_tok["orig_indices"]):
                        correct = int(other_preds[local_i] == silver_rows[global_i]["predicate"])
                        sil_correct[global_i] += correct
                        sil_total[global_i]   += 1

                    # ── Infer on gold ─────────────────────────────────────────
                    klog.info(f"  inferring on {len(gold_rows)} gold rows")
                    t_gold = time.time()
                    gold_preds = predict_predicates(
                        gold_rows, idx2pred, model, tokenizer, device,
                        flog, args, amp_dtype=amp_dtype,
                        desc=f"gold/{model_tag}/k{K}/{budget_label}/fold{fold_i}",
                    )
                    t_gold_done = time.time()
                    klog.info(f"  gold inference done  ({_fmt(t_gold_done - t_gold)})")

                    for j, pred in enumerate(gold_preds):
                        gold_correct[j] += int(pred == gold_rows[j]["predicate"])

                    fold_binary = [
                        "1" if p == gold_rows[j]["predicate"] else "0"
                        for j, p in enumerate(gold_preds)
                    ]
                    fold_m = compute_metrics(fold_binary, gold_labels)
                    klog.info(
                        f"  gold (single fold): "
                        f"acc={fold_m['accuracy']:.3f}  prec={fold_m['precision']:.3f}  "
                        f"rec={fold_m['recall']:.3f}  f1={fold_m['f1']:.3f}  "
                        f"(TP={fold_m['TP']} FP={fold_m['FP']} "
                        f"FN={fold_m['FN']} TN={fold_m['TN']})"
                    )
                    fold_metrics_list.append({
                        "fold_idx":      fold_i,
                        "metrics":       fold_m,
                        "time":          _fmt(t_gold_done - t_fold),
                        "best_step":     best_step,
                        "best_train_f1": round(best_train_f1, 4),
                    })

                    fold_gold_path = os.path.join(fold_dir, "gold_preds.csv")
                    with open(fold_gold_path, "w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=list(gold_rows[0].keys()) + ["predicted_predicate", "vote"],
                        )
                        writer.writeheader()
                        for row, pred in zip(gold_rows, gold_preds):
                            writer.writerow({
                                **row,
                                "predicted_predicate": pred,
                                "vote": "1" if pred == row["predicate"] else "0",
                            })

                    unload_model(model, flog)
                    t_done = time.time()
                    fold_timing[str(fold_i)] = {
                        "train":        _fmt(t_trained   - t_fold),
                        "silver_infer": _fmt(t_sil_done  - t_trained),
                        "gold_infer":   _fmt(t_gold_done - t_sil_done),
                        "total":        _fmt(t_done      - t_fold),
                    }
                    klog.info(
                        f"  fold {fold_i} done  "
                        f"total={fold_timing[str(fold_i)]['total']}  "
                        f"(train={fold_timing[str(fold_i)]['train']}  "
                        f"silver_infer={fold_timing[str(fold_i)]['silver_infer']}  "
                        f"gold_infer={fold_timing[str(fold_i)]['gold_infer']})"
                    )
                    completed_folds.add(fold_i)
                    save_checkpoint(exp_dir, completed_folds, sil_correct, sil_total,
                                    gold_correct, fold_metrics_list, fold_timing, klog)

                # ── Experiment complete: timing + delete checkpoint ────────────
                delete_checkpoint(exp_dir, klog)
                timing = {
                    "model": model_tag, "K": K, "budget": budget_label,
                    "total_wall_time": _fmt(time.time() - t_exp),
                    "folds": fold_timing,
                }
                timing_path = os.path.join(exp_dir, "timing.json")
                with open(timing_path, "w", encoding="utf-8") as f:
                    json.dump(timing, f, ensure_ascii=False, indent=2)
                klog.info(f"  timing saved → {timing_path}")

                # ── Majority vote: silver ─────────────────────────────────────
                safe_total  = np.maximum(sil_total, 1)
                vote_frac   = sil_correct / safe_total
                cleaned_lbl = (vote_frac > args.vote_threshold).astype(int)

                n_pos_silver = int(cleaned_lbl.sum())
                pos_pct      = 100 * n_pos_silver / len(silver_rows)
                klog.info(
                    f"[silver cleaning]  "
                    f"relation_present=1: {n_pos_silver:,} / {len(silver_rows):,}  "
                    f"({pos_pct:.1f}%)  vote_threshold={args.vote_threshold}"
                )

                sil_clean_path = os.path.join(exp_dir, "silver_cleaned.csv")
                with open(sil_clean_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "docid", "subject", "predicate", "object",
                        "fold_id", "vote_correct", "vote_total",
                        "vote_fraction", "cleaned_label",
                    ])
                    for i, row in enumerate(silver_rows):
                        writer.writerow([
                            row.get("docid", ""), row["subject"],
                            row["predicate"],     row["object"],
                            int(fold_ids[i]),
                            int(sil_correct[i]), int(sil_total[i]),
                            f"{vote_frac[i]:.3f}",
                            int(cleaned_lbl[i]),
                        ])
                klog.info(f"  silver_cleaned.csv → {sil_clean_path}")

                # ── Majority vote: gold ───────────────────────────────────────
                gold_vote_frac = gold_correct / K
                gold_maj_preds = [
                    "1" if gold_vote_frac[j] > args.vote_threshold else "0"
                    for j in range(len(gold_rows))
                ]
                ens_m = compute_metrics(gold_maj_preds, gold_labels)
                klog.info(
                    f"[gold ensemble] majority vote (thresh={args.vote_threshold}): "
                    f"acc={ens_m['accuracy']:.3f}  prec={ens_m['precision']:.3f}  "
                    f"rec={ens_m['recall']:.3f}  f1={ens_m['f1']:.3f}  "
                    f"(TP={ens_m['TP']} FP={ens_m['FP']} "
                    f"FN={ens_m['FN']} TN={ens_m['TN']})"
                )

                gold_class_path = os.path.join(exp_dir, "gold_classified.csv")
                with open(gold_class_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=list(gold_rows[0].keys()) + [
                            "votes_correct", "vote_fraction", "predicted_relation_present"
                        ],
                    )
                    writer.writeheader()
                    for j, row in enumerate(gold_rows):
                        writer.writerow({
                            **row,
                            "votes_correct":              int(gold_correct[j]),
                            "vote_fraction":              f"{gold_vote_frac[j]:.3f}",
                            "predicted_relation_present": gold_maj_preds[j],
                        })
                klog.info(f"  gold_classified.csv → {gold_class_path}")

                model_results[K][budget_label] = {
                    "fold_metrics":   fold_metrics_list,
                    "ensemble":       {"metrics": ens_m},
                    "silver_pos_pct": pos_pct,
                    "wall_time":      _fmt(time.time() - t_exp),
                }

                # Per-experiment summary
                exp_summary_path = os.path.join(exp_dir, "summary.txt")
                write_summary(
                    {model_tag: {K: {budget_label: model_results[K][budget_label]}}},
                    exp_summary_path, klog, time.time() - wall_start,
                )

            # Per-K summary (across all budgets for this K)
            k_summary_path = os.path.join(k_dir, "summary.txt")
            write_summary(
                {model_tag: {K: model_results[K]}},
                k_summary_path, klog, time.time() - wall_start,
            )

        all_results[model_tag] = model_results

        # Per-model summary (across K values and budgets)
        m_summary_path = os.path.join(model_dir, "summary.txt")
        write_summary(
            {model_tag: model_results},
            m_summary_path, mlog, time.time() - wall_start,
        )

    # ── Combined summary ───────────────────────────────────────────────────────
    summary_path = os.path.join(out_base, "summary.txt")
    write_summary(all_results, summary_path, log, time.time() - wall_start)

    log.info("=" * 70)
    log.info(f"Done.  Total wall time: {_fmt(time.time()-wall_start)}")
    log.info(f"  summary → {summary_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()