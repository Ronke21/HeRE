"""
Classify silver data using cross-trained RC for two best configurations:
  • neodictabert K=3  (8000 steps, lr=2e-5)
  • mmbert       K=5  (8000 steps, lr=2e-5)

For each config:
  1. Stratified K-fold split of silver by predicate (seed=42).
  2. Train fold-i model on fold-i data; infer on all other folds.
  3. Accumulate per-row vote counts across all folds.
  4. Save silver_cleaned.csv:  docid, subject, predicate, object,
     fold_id, vote_correct, vote_total, vote_fraction, cleaned_label.
  5. Save gold_classified.csv for validation.

After both configs complete, combine into:
  outputs/silver_cross_train_rc/combined/silver_cleaned_combined.csv

Output layout:
  outputs/silver_cross_train_rc/
    run.log
    {config_name}/
      run.log
      silver_cleaned.csv
      gold_classified.csv
      summary.txt
      fold_{i}/
        train.log
        gold_preds.csv
        eval_curve.csv
    combined/
      silver_cleaned_combined.csv

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m scripts_silver_cleaning.classify_silver_cross_train_rc
  CUDA_VISIBLE_DEVICES=0 python -m scripts_silver_cleaning.classify_silver_cross_train_rc --config neodictabert_k3
  CUDA_VISIBLE_DEVICES=0 python -m scripts_silver_cleaning.classify_silver_cross_train_rc --config mmbert_k5 --no-amp

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
    import flash_attn  # noqa: F401
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Hardcoded configs: the two best (model, K) combos from the sweep
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "name":     "neodictabert_k3",
        "model_id": "dicta-il/neodictabert",
        "tag":      "neodictabert",
        "K":        3,
        "steps":    6500,   # fold-0 peak; flat 6500→8000
        "lr":       2e-5,
    },
    {
        "name":     "mmbert_k5",
        "model_id": "jhu-clsp/mmBERT-base",
        "tag":      "mmbert",
        "K":        5,
        "steps":    5500,   # fold-0 peak ~5000; small buffer
        "lr":       2e-5,
    },
]

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

SILVER_FILE      = "data/prepared_silver.parquet"
GOLD_FILE        = "data/prepared_gold_500.csv"
OUTPUT_BASE = "outputs/silver_cleaning/silver_cross_train_rc"
TOKEN_CACHE_DIR  = "data/token_cache"

TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE  = 1024
MAX_SEQ_LENGTH   = 256
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.1
MIN_PRED_FREQ    = 20
VOTE_THRESHOLD   = 0.5
RANDOM_SEED      = 42
EVAL_EVERY_N     = 500
LOG_EVERY_N      = 200


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PreTokenizedDataset(Dataset):
    """Dataset backed by pre-computed token tensors."""
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
    if amp_dtype is None or not torch.cuda.is_available():
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=amp_dtype)


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
    logger.propagate = False
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
# Token cache
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Inference from pre-tokenized tensors
# ---------------------------------------------------------------------------

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
# Inference from raw rows (for gold)
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
# Model
# ---------------------------------------------------------------------------

def load_tokenizer(model_id: str, log: logging.Logger):
    log.info(f"  loading tokenizer: {model_id}")
    return transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def load_model(
    model_id:   str,
    num_labels: int,
    log:        logging.Logger,
    tokenizer   = None,
    load_dtype: Optional[torch.dtype] = None,
    flash_attn: bool = False,
):
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
    if flash_attn:
        if _FLASH_ATTN_AVAILABLE:
            kwargs["attn_implementation"] = "flash_attention_2"
            log.info("  flash_attention_2 enabled")
        else:
            log.warning("  --flash-attn requested but flash_attn not installed — skipping")
    try:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id, **kwargs
        )
    except Exception as e:
        # Drop flash-attn if it was the culprit, then retry
        retry_kwargs = {k: v for k, v in kwargs.items() if k != "attn_implementation"}
        if "attn_implementation" in kwargs:
            log.warning(f"  flash_attention_2 not supported by this model ({e}), retrying without it")
        else:
            log.warning(f"  load failed ({e}), retrying without safetensors")
            retry_kwargs["use_safetensors"] = False
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id, **retry_kwargs
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
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: list[str], gold: list[str]) -> dict:
    TP = FP = FN = TN = 0
    for p, g in zip(preds, gold):
        pos  = p == "1"
        gpos = g == "1"
        if   pos and gpos:     TP += 1
        elif pos and not gpos: FP += 1
        elif not pos and gpos: FN += 1
        else:                  TN += 1
    total = TP + FP + FN + TN
    prec  = TP / (TP + FP)       if (TP + FP)  else 0.0
    rec   = TP / (TP + FN)       if (TP + FN)  else 0.0
    f1    = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "accuracy":  (TP+TN)/total if total else 0.0,
        "precision": prec, "recall": rec, "f1": f1,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_fold(
    pred2idx:     dict,
    idx2pred:     dict,
    model_id:     str,
    model_tag:    str,
    fold_idx:     int,
    log:          logging.Logger,
    args,
    max_steps:    int,
    lr:           float,
    gold_rows:    list,
    gold_labels:  list,
    tokenizer,
    train_tokens: dict,
    amp_dtype:    Optional[torch.dtype],
    scaler,
    flash_attn:   bool = False,
):
    """
    Fine-tune model on train_tokens (pre-tokenized).
    Evaluates on gold every args.eval_every steps.
    Returns (model, tokenizer, device, eval_curve, best_step, best_f1).
    """
    model, tokenizer = load_model(
        model_id, len(pred2idx), log, tokenizer=tokenizer,
        load_dtype=amp_dtype, flash_attn=flash_attn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if getattr(args, "grad_ckpt", False):
        model.gradient_checkpointing_enable()
        log.info("  gradient checkpointing enabled")
    if getattr(args, "compile", False):
        try:
            model = torch.compile(model)
            log.info("  torch.compile enabled")
        except Exception as e:
            log.warning(f"  torch.compile failed ({e}), continuing without")

    if torch.cuda.device_count() > 1:
        log.info(f"  DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    dataset = PreTokenizedDataset(train_tokens)
    log.info(f"  train: {len(dataset):,} rows (pre-tokenized)")
    loader = DataLoader(
        dataset, batch_size=args.train_batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )

    steps_per_epoch = len(loader)
    total_steps     = max_steps

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    do_eval = bool(gold_rows) and args.eval_every > 0
    log.info(
        f"  max_steps={total_steps}  steps_per_epoch={steps_per_epoch}  "
        f"lr={lr}  eval_every={args.eval_every if do_eval else 'disabled'}"
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
            gold_rows, idx2pred, base, tokenizer, device, log, args,
            amp_dtype=amp_dtype, desc="eval/gold",
        )
        binary = [
            "1" if preds[j] == gold_rows[j]["predicate"] else "0"
            for j in range(len(gold_rows))
        ]
        m = compute_metrics(binary, gold_labels)
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

    if do_eval:
        _run_eval(step)

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
# Checkpoint (fold-level resume)
# ---------------------------------------------------------------------------

def save_checkpoint(
    config_dir:      str,
    completed_folds: set,
    sil_correct:     np.ndarray,
    sil_total:       np.ndarray,
    gold_correct:    np.ndarray,
    fold_metrics:    list,
    fold_timing:     dict,
    log:             logging.Logger,
):
    np.savez_compressed(
        os.path.join(config_dir, "checkpoint.npz"),
        sil_correct=sil_correct,
        sil_total=sil_total,
        gold_correct=gold_correct,
    )
    with open(os.path.join(config_dir, "checkpoint.json"), "w", encoding="utf-8") as f:
        json.dump({
            "completed_folds": sorted(completed_folds),
            "fold_metrics":    fold_metrics,
            "fold_timing":     fold_timing,
        }, f, ensure_ascii=False, indent=2)
    log.info(f"  checkpoint saved  (folds done: {sorted(completed_folds)})")


def load_checkpoint(config_dir: str, log: logging.Logger) -> Optional[dict]:
    npz_path  = os.path.join(config_dir, "checkpoint.npz")
    json_path = os.path.join(config_dir, "checkpoint.json")
    if not (os.path.exists(npz_path) and os.path.exists(json_path)):
        return None
    arrays = np.load(npz_path)
    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)
    log.info(f"  checkpoint found  —  folds done: {meta['completed_folds']}")
    return {
        "sil_correct":     arrays["sil_correct"],
        "sil_total":       arrays["sil_total"],
        "gold_correct":    arrays["gold_correct"],
        "completed_folds": set(meta["completed_folds"]),
        "fold_metrics":    meta["fold_metrics"],
        "fold_timing":     meta["fold_timing"],
    }


def delete_checkpoint(config_dir: str, log: logging.Logger):
    for fname in ("checkpoint.npz", "checkpoint.json"):
        p = os.path.join(config_dir, fname)
        if os.path.exists(p):
            os.remove(p)
    log.info("  checkpoint deleted (config complete)")


# ---------------------------------------------------------------------------
# Run a single config
# ---------------------------------------------------------------------------

def run_config(
    cfg:         dict,
    silver_rows: list[dict],
    gold_rows:   list[dict],
    gold_labels: list[str],
    pred2idx:    dict,
    idx2pred:    dict,
    sil_tokens:  dict,
    config_dir:  str,
    args,
    amp_dtype:   Optional[torch.dtype],
    scaler,
    log:         logging.Logger,
) -> dict:
    """Run one (model, K) config. Returns result dict."""
    name     = cfg["name"]
    model_id = cfg["model_id"]
    model_tag = cfg["tag"]
    K        = cfg["K"]
    steps    = cfg["steps"]
    lr       = cfg["lr"]

    tokenizer = load_tokenizer(model_id, log)

    # Drop cache rows whose label is out of range (vocab mismatch)
    label_mask = sil_tokens["labels"].numpy() < len(pred2idx)
    if not label_mask.all():
        n_dropped = int((~label_mask).sum())
        log.info(f"  [tokens] dropping {n_dropped} cached rows with out-of-range labels")
        sil_tokens = subset_tokens(sil_tokens, label_mask)

    # Stratified fold assignment (seed=42, stratified by predicate)
    fold_ids = np.zeros(len(silver_rows), dtype=int)
    rng = np.random.default_rng(RANDOM_SEED)
    pred_indices: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(silver_rows):
        pred_indices[r["predicate"]].append(i)
    for indices in pred_indices.values():
        arr = np.array(indices)
        rng.shuffle(arr)
        for rank, idx in enumerate(arr):
            fold_ids[idx] = rank % K
    dist = "  ".join(f"f{i}={int((fold_ids == i).sum()):,}" for i in range(K))
    log.info(f"  fold distribution: {dist}")

    # Resume from checkpoint if exists
    ckpt = load_checkpoint(config_dir, log)
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

    t_config = time.time()

    # Fold loop
    for fold_i in range(K):
        if fold_i in completed_folds:
            log.info(f"[fold {fold_i+1}/{K}]  skipped (already done in checkpoint)")
            continue

        log.info(f"[fold {fold_i+1}/{K}]  training on fold {fold_i}  steps={steps}")

        fold_dir = os.path.join(config_dir, f"fold_{fold_i}")
        os.makedirs(fold_dir, exist_ok=True)
        flog = setup_logger(
            os.path.join(fold_dir, "train.log"),
            f"silver_cross_rc.{name}.fold{fold_i}",
            mode="a",
        )

        # Subset pre-tokenized silver
        valid_fold_ids = fold_ids[sil_tokens["orig_indices"]]
        train_tok = subset_tokens(sil_tokens, valid_fold_ids == fold_i)
        infer_tok = subset_tokens(sil_tokens, valid_fold_ids != fold_i)
        log.info(
            f"  training rows: {len(train_tok['labels']):,}  "
            f"infer rows: {len(infer_tok['labels']):,}"
        )

        # Clear GPU memory before loading next fold model to prevent fragmentation OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info(
                f"  pre-fold GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB alloc  "
                f"{torch.cuda.memory_reserved()/1e9:.2f}GB reserved"
            )

        t_fold = time.time()
        try:
            model, tokenizer, device, eval_curve, best_step, best_f1 = train_one_fold(
                pred2idx=pred2idx,
                idx2pred=idx2pred,
                model_id=model_id,
                model_tag=model_tag,
                fold_idx=fold_i,
                log=flog,
                args=args,
                max_steps=steps,
                lr=lr,
                gold_rows=gold_rows,
                gold_labels=gold_labels,
                tokenizer=tokenizer,
                train_tokens=train_tok,
                amp_dtype=amp_dtype,
                scaler=scaler,
                flash_attn=getattr(args, "flash_attn", False),
            )
        except Exception as exc:
            log.error(f"  fold {fold_i} FAILED: {exc}")
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
        log.info(
            f"  eval curve → {eval_curve_path}  "
            f"best_step={best_step}  best_f1={best_f1:.3f}"
        )

        # Infer on held-out silver folds
        log.info(f"  inferring on {len(infer_tok['labels']):,} silver rows (folds ≠ {fold_i})")
        t_inf = time.time()
        other_preds = predict_from_tokens(
            infer_tok, idx2pred, model, device, flog,
            args.eval_batch_size, amp_dtype,
            desc=f"silver/{name}/fold{fold_i}",
        )
        t_sil_done = time.time()
        log.info(f"  silver inference done  ({_fmt(t_sil_done - t_inf)})")

        for local_i, global_i in enumerate(infer_tok["orig_indices"]):
            correct = int(other_preds[local_i] == silver_rows[global_i]["predicate"])
            sil_correct[global_i] += correct
            sil_total[global_i]   += 1

        # Infer on gold
        log.info(f"  inferring on {len(gold_rows)} gold rows")
        t_gold = time.time()
        gold_preds = predict_predicates(
            gold_rows, idx2pred, model, tokenizer, device,
            flog, args, amp_dtype=amp_dtype,
            desc=f"gold/{name}/fold{fold_i}",
        )
        t_gold_done = time.time()
        log.info(f"  gold inference done  ({_fmt(t_gold_done - t_gold)})")

        for j, pred in enumerate(gold_preds):
            gold_correct[j] += int(pred == gold_rows[j]["predicate"])

        fold_binary = [
            "1" if p == gold_rows[j]["predicate"] else "0"
            for j, p in enumerate(gold_preds)
        ]
        fold_m = compute_metrics(fold_binary, gold_labels)
        log.info(
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
            "best_train_f1": round(best_f1, 4),
        })

        # Save fold gold preds
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
        log.info(
            f"  fold {fold_i} done  "
            f"total={fold_timing[str(fold_i)]['total']}  "
            f"(train={fold_timing[str(fold_i)]['train']}  "
            f"silver_infer={fold_timing[str(fold_i)]['silver_infer']}  "
            f"gold_infer={fold_timing[str(fold_i)]['gold_infer']})"
        )
        completed_folds.add(fold_i)
        save_checkpoint(
            config_dir, completed_folds, sil_correct, sil_total,
            gold_correct, fold_metrics_list, fold_timing, log,
        )

    # Only delete checkpoint if all folds succeeded; retain for resume if any failed
    n_failed = sum(1 for fm in fold_metrics_list if fm.get("time") == "FAILED")
    if n_failed == 0:
        delete_checkpoint(config_dir, log)
    else:
        log.warning(
            f"  {n_failed} fold(s) failed — checkpoint retained for resume "
            f"(re-run to complete missing folds)"
        )

    # Majority vote: silver
    safe_total  = np.maximum(sil_total, 1)
    vote_frac   = sil_correct / safe_total
    cleaned_lbl = (vote_frac > VOTE_THRESHOLD).astype(int)

    n_pos_silver = int(cleaned_lbl.sum())
    pos_pct      = 100 * n_pos_silver / len(silver_rows)
    log.info(
        f"[silver cleaning]  "
        f"relation_present=1: {n_pos_silver:,} / {len(silver_rows):,}  "
        f"({pos_pct:.1f}%)  vote_threshold={VOTE_THRESHOLD}"
    )

    sil_clean_path = os.path.join(config_dir, "silver_cleaned.csv")
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
                f"{vote_frac[i]:.4f}",
                int(cleaned_lbl[i]),
            ])
    log.info(f"  silver_cleaned.csv → {sil_clean_path}")

    # Majority vote: gold
    gold_vote_frac = gold_correct / K
    gold_maj_preds = [
        "1" if gold_vote_frac[j] > VOTE_THRESHOLD else "0"
        for j in range(len(gold_rows))
    ]
    ens_m = compute_metrics(gold_maj_preds, gold_labels)
    log.info(
        f"[gold ensemble] (thresh={VOTE_THRESHOLD}): "
        f"acc={ens_m['accuracy']:.3f}  prec={ens_m['precision']:.3f}  "
        f"rec={ens_m['recall']:.3f}  f1={ens_m['f1']:.3f}  "
        f"(TP={ens_m['TP']} FP={ens_m['FP']} "
        f"FN={ens_m['FN']} TN={ens_m['TN']})"
    )

    gold_class_path = os.path.join(config_dir, "gold_classified.csv")
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
    log.info(f"  gold_classified.csv → {gold_class_path}")

    # Summary
    wall_time = _fmt(time.time() - t_config)
    summary_lines = [
        "=" * 80,
        f"Config: {name}  (K={K}, steps={steps}, lr={lr})",
        f"Wall time: {wall_time}",
        f"Vote threshold: {VOTE_THRESHOLD}",
        "=" * 80,
        "",
        f"  Silver cleaned pos rate: {pos_pct:.1f}%",
        f"  Gold ensemble: acc={ens_m['accuracy']:.3f}  prec={ens_m['precision']:.3f}  "
        f"rec={ens_m['recall']:.3f}  f1={ens_m['f1']:.3f}  "
        f"TP={ens_m['TP']} FP={ens_m['FP']} FN={ens_m['FN']} TN={ens_m['TN']}",
        "",
        "  Per-fold metrics:",
        f"  {'fold':<6}  {'f1':>6}  {'prec':>6}  {'rec':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  {'best_step':>9}  time",
        "  " + "-" * 75,
    ]
    for fr in fold_metrics_list:
        m = fr["metrics"]
        summary_lines.append(
            f"  {fr['fold_idx']:<6}  {m['f1']:>6.3f}  {m['precision']:>6.3f}  "
            f"{m['recall']:>6.3f}  {m['TP']:>5} {m['FP']:>5} "
            f"{m['FN']:>5} {m['TN']:>5}  {str(fr.get('best_step','?')):>9}  {fr.get('time','')}"
        )
    summary_lines += ["", "=" * 80]
    summary_text = "\n".join(summary_lines)
    with open(os.path.join(config_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    for line in summary_lines:
        log.info(line)

    return {
        "name":          name,
        "ensemble":      ens_m,
        "silver_pos_pct": pos_pct,
        "wall_time":     wall_time,
        "fold_metrics":  fold_metrics_list,
    }


# ---------------------------------------------------------------------------
# Combine results
# ---------------------------------------------------------------------------

def combine_results(out_base: str, configs: list[dict], log: logging.Logger):
    """Merge all silver_cleaned.csv files into combined/silver_cleaned_combined.csv."""
    combined_dir = os.path.join(out_base, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    combined_path = os.path.join(combined_dir, "silver_cleaned_combined.csv")

    with open(combined_path, "w", encoding="utf-8", newline="") as fout:
        writer = None
        for cfg in configs:
            src = os.path.join(out_base, cfg["name"], "silver_cleaned.csv")
            if not os.path.exists(src):
                log.warning(f"  [combine] missing: {src} — skipping")
                continue
            with open(src, encoding="utf-8", newline="") as fin:
                reader = csv.DictReader(fin)
                if writer is None:
                    fieldnames = ["config"] + list(reader.fieldnames)
                    writer = csv.DictWriter(fout, fieldnames=fieldnames)
                    writer.writeheader()
                n = 0
                for row in reader:
                    writer.writerow({"config": cfg["name"], **row})
                    n += 1
            log.info(f"  [combine] {cfg['name']}: {n:,} rows")

    log.info(f"  combined → {combined_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Silver data cleaning via cross-trained RC (neodictabert K=3 + mmbert K=5)"
    )
    parser.add_argument(
        "--config", default="all",
        choices=["all"] + [c["name"] for c in CONFIGS],
        help="Which config to run (default: all)",
    )
    parser.add_argument("--silver",        default=SILVER_FILE)
    parser.add_argument("--gold",          default=GOLD_FILE)
    parser.add_argument("--output-base",   default=OUTPUT_BASE)
    parser.add_argument("--token-cache-dir", default=TOKEN_CACHE_DIR, dest="token_cache_dir")
    parser.add_argument(
        "--max-silver", type=int, default=None,
        help="Cap silver rows loaded (default: all)",
    )
    parser.add_argument(
        "--skip-combine", action="store_true",
        help="Skip the final combine step",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable automatic mixed precision (default: bf16)",
    )
    parser.add_argument(
        "--amp-dtype", choices=["bf16", "fp16"], default="bf16", dest="amp_dtype_str",
    )
    parser.add_argument(
        "--eval-every", type=int, default=EVAL_EVERY_N, dest="eval_every",
        help=f"Evaluate on gold every N steps (default: {EVAL_EVERY_N}; 0=disable)",
    )
    parser.add_argument("--train-batch",  type=int, default=TRAIN_BATCH_SIZE, dest="train_batch_size")
    parser.add_argument("--eval-batch",   type=int, default=EVAL_BATCH_SIZE,  dest="eval_batch_size")
    parser.add_argument("--max-length",   type=int, default=MAX_SEQ_LENGTH,   dest="max_length")
    parser.add_argument("--grad-ckpt",    action="store_true", dest="grad_ckpt")
    parser.add_argument("--compile",      action="store_true", dest="compile")
    parser.add_argument("--flash-attn",   action="store_true", dest="flash_attn",
                        help="Enable Flash Attention 2 (requires flash_attn package)")
    parser.add_argument(
        "--debug", type=int, default=0,
        help="Use first N silver rows only; 0 = full dataset",
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    base          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    silver_path   = os.path.join(base, args.silver)
    gold_path     = os.path.join(base, args.gold)
    out_base      = os.path.join(base, args.output_base)
    token_cache   = os.path.join(base, args.token_cache_dir)
    os.makedirs(out_base, exist_ok=True)

    log = setup_logger(os.path.join(out_base, "run.log"), "silver_cross_rc")
    wall_start = time.time()

    # AMP setup
    amp_dtype: Optional[torch.dtype] = None
    scaler = None
    if not args.no_amp and torch.cuda.is_available():
        if args.amp_dtype_str == "bf16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
            scaler    = torch.cuda.amp.GradScaler()

    log.info("=" * 70)
    log.info("classify_silver_cross_train_rc.py  started")
    log.info(f"  silver:     {silver_path}")
    log.info(f"  gold:       {gold_path}")
    log.info(f"  output:     {out_base}")
    log.info(f"  config:     {args.config}")
    log.info(f"  max_silver: {args.max_silver or 'all'}")
    log.info(f"  debug:      {args.debug or 'off'}")
    log.info(f"  AMP:        {amp_dtype or 'disabled'}  scaler={scaler is not None}")
    log.info(f"  CUDA: {torch.cuda.is_available()}  GPUs: {torch.cuda.device_count()}")
    log.info("=" * 70)

    # Load data
    max_sil = args.debug if args.debug else args.max_silver
    log.info("[load] silver")
    silver_rows = load_csv(silver_path, max_sil, log)
    log.info("[load] gold")
    gold_rows   = load_csv(gold_path, None, log)

    gold_labels = [r["relation_present"] for r in gold_rows]
    n_pos = gold_labels.count("1")
    log.info(
        f"gold: {len(gold_rows)} rows  pos={n_pos}  neg={len(gold_rows)-n_pos}  "
        f"({100*n_pos/len(gold_rows):.1f}% pos)"
    )

    # Predicate vocabulary (shared across configs)
    log.info("[vocab]")
    pred2idx, idx2pred = build_pred_vocab(silver_rows, gold_rows, MIN_PRED_FREQ, log)
    vocab_path = os.path.join(out_base, "pred2idx.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(pred2idx, f, ensure_ascii=False, indent=2)
    log.info(f"  vocab saved → {vocab_path}")

    # Select configs to run
    configs_to_run = CONFIGS if args.config == "all" else [
        c for c in CONFIGS if c["name"] == args.config
    ]
    log.info(f"[configs] running: {[c['name'] for c in configs_to_run]}")

    all_results = []

    for cfg in configs_to_run:
        name = cfg["name"]
        tag  = cfg["tag"]
        log.info("=" * 70)
        log.info(f"CONFIG: {name}  (K={cfg['K']}, steps={cfg['steps']}, lr={cfg['lr']})")
        log.info("=" * 70)

        config_dir = os.path.join(out_base, name)
        os.makedirs(config_dir, exist_ok=True)
        clog = setup_logger(
            os.path.join(config_dir, "run.log"),
            f"silver_cross_rc.{name}",
        )

        # Load token cache (built once per model by the original clean script)
        clog.info(f"[tokens] loading pre-tokenized silver for {tag}")
        sil_tokens = build_silver_tokens(
            silver_rows, pred2idx,
            load_tokenizer(cfg["model_id"], clog),
            MAX_SEQ_LENGTH, tag, token_cache, clog,
        )
        if args.debug:
            mask = sil_tokens["orig_indices"] < len(silver_rows)
            sil_tokens = subset_tokens(sil_tokens, mask)

        result = run_config(
            cfg=cfg,
            silver_rows=silver_rows,
            gold_rows=gold_rows,
            gold_labels=gold_labels,
            pred2idx=pred2idx,
            idx2pred=idx2pred,
            sil_tokens=sil_tokens,
            config_dir=config_dir,
            args=args,
            amp_dtype=amp_dtype,
            scaler=scaler,
            log=clog,
        )
        all_results.append(result)

        log.info(
            f"CONFIG {name} DONE  "
            f"gold_f1={result['ensemble']['f1']:.3f}  "
            f"silver_pos={result['silver_pos_pct']:.1f}%  "
            f"time={result['wall_time']}"
        )

    # Combine
    if not args.skip_combine and args.config == "all":
        log.info("[combine] merging silver_cleaned.csv files")
        combine_results(out_base, CONFIGS, log)

    log.info("=" * 70)
    log.info(f"Done.  Total wall time: {_fmt(time.time()-wall_start)}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
