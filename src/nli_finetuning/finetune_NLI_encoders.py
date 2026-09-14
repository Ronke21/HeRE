#!/usr/bin/env python3
"""
run_hebnli.py — Fine-tune and evaluate 6 models on HebNLI (Hebrew NLI).

Architecture routing:
  google/mt5-large (encoder-decoder)
      → mrl_eval subprocess pipeline: finetune → generate → evaluate
  All others (encoder-only: XLM-R, E5, mmBERT, NeoDictaBERT, AlephBERT)
      → AutoModelForSequenceClassification, 3-class, macro-F1

Conda environment: heb_nli_mrl_eval
    conda activate heb_nli_mrl_eval

Run from Hebrew_NLI/:
    cd /path/to/HEntailment-Hebrew-NLI-Repurposing-EACL27-Internal
    CUDA_VISIBLE_DEVICES=7 python finetune_heb_nli/finetune_NLI_encoders.py

Prerequisites:
    mrl_eval_data/hebnli/jsonl/{train,val,test}.jsonl   (run ingest scripts first)
    conda env: heb_nli_mrl_eval  (transformers 4.55, peft 0.17, torch 2.7+cu118)

Output layout under finetune_heb_nli/:
    logs/<model>.log        full training output per model
    encoder_checkpoints/<model>/        encoder-only checkpoints
    results/                per-model JSON + prediction files
    summary.json            final metrics table
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import ast
import datetime
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import traceback
from typing import Optional

# Parse --single-model arg early (used when launched via torchrun for FSDP)
_arg_parser = argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument("--single-model", default=None, dest="single_model")
_CLI, _ = _arg_parser.parse_known_args()

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import transformers
from torch.utils.data import Dataset

try:
    from sklearn.metrics import accuracy_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = pathlib.Path(__file__).parent.parent.resolve()
MRL_EVAL_ROOT  = SCRIPT_DIR.parent          # Hebrew_NLI/  (contains mrl_eval/ and mrl_eval_data/)
DATA_DIR       = MRL_EVAL_ROOT / "mrl_eval_data" / "hebnli" / "jsonl"
LOGS_DIR       = SCRIPT_DIR / "logs"
OUTPUTS_DIR    = SCRIPT_DIR / "encoder_checkpoints"
RESULTS_DIR    = SCRIPT_DIR / "results" / "encoders_singlerun"

for _d in (LOGS_DIR, OUTPUTS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────
DATASET = "hebnli"

MODELS = [
    "FacebookAI/xlm-roberta-large",
    "google/mt5-large",
    "intfloat/multilingual-e5-large",
    "jhu-clsp/mmBERT-base",
    "dicta-il/neodictabert",
    "onlplab/alephbert-base",
    "facebook/xlm-roberta-xl",
    "facebook/xlm-roberta-xxl",
    "google/mt5-xl",
]

# Per-model overrides for encoder-only models
MODEL_FLAGS: dict[str, dict] = {
    "dicta-il/neodictabert": {
        "trust_remote_code"  : True,
        "attn_implementation": None,
        "zero_token_type_ids": False,
    },
    "onlplab/alephbert-base": {
        "trust_remote_code"  : False,
        "attn_implementation": "eager",
        # type_vocab_size=1 → token_type_ids must stay 0 or CUDA asserts
        "zero_token_type_ids": True,
    },
    "facebook/xlm-roberta-xxl": {
        "trust_remote_code"  : False,
        "attn_implementation": None,
        "zero_token_type_ids": False,
        "torchrun_nproc"     : 2,  # FSDP across 2 GPUs — shards model+grad+optimizer → ~75GB each
    },
}

# HebNLI labels (entailment / contradiction / neutral in Hebrew)
LABELS    = ["היסק", "סתירה", "ניטרלי"]
LABEL2ID  = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for i, l in enumerate(LABELS)}

# Hyperparams for encoder-only classification
CLF = dict(
    max_length       = 250,
    max_steps        = 8000,
    batch_size       = 64,
    eval_batch_size  = 128,
    lr               = 2e-5,
    warmup_ratio     = 0.1,
    weight_decay     = 0.01,
    eval_steps       = 200,
    save_total_limit = 1,
    seed             = 1234,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(model_name: str) -> str:
    return model_name.replace("/", "__")


def make_logger(model_name: str) -> tuple[logging.Logger, pathlib.Path]:
    """Return a per-model logger that writes to both file and stdout."""
    log_path = LOGS_DIR / f"{_safe(model_name)}.log"
    log = logging.getLogger(model_name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log, log_path


# ─────────────────────────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run the mrl_eval ingest pipeline first:\n"
            "  bash mrl_eval/datasets/download_raw_data.sh\n"
            "  bash mrl_eval/datasets/ingest_all_datasets.sh"
        )
    return load_jsonl(path)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (manual fallback if sklearn unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def macro_f1(y_true: list, y_pred: list) -> float:
    if HAS_SKLEARN:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    classes = list(set(y_true))
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def accuracy(y_true: list, y_pred: list) -> float:
    if HAS_SKLEARN:
        return float(accuracy_score(y_true, y_pred))
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Trainer callback for rich logging
# ─────────────────────────────────────────────────────────────────────────────

class DetailedLoggingCallback(transformers.TrainerCallback):
    """Write all Trainer logs (steps, evals) to a model-specific logger."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info("step=%d | %s", state.global_step, logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.logger.info("EVAL step=%d | %s", state.global_step, metrics)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.logger.info("Epoch %.1f completed (global_step=%d)", state.epoch, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.logger.info("Training complete. best_model_checkpoint=%s best_metric=%.4f",
                         state.best_model_checkpoint, state.best_metric or 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# HF dataset for encoder-only classification
# ─────────────────────────────────────────────────────────────────────────────

class NLIDataset(Dataset):
    """Tokenised NLI pairs for sequence classification."""

    def __init__(self, examples: list[dict], tokenizer, max_length: int,
                 zero_token_type_ids: bool = False):
        self._data = []
        for ex in examples:
            enc = tokenizer(
                ex["translation1"],
                ex["translation2"],
                max_length=max_length,
                truncation=True,
            )
            item = {k: enc[k] for k in enc.keys()}
            if zero_token_type_ids and "token_type_ids" in item:
                item["token_type_ids"] = [0] * len(item["token_type_ids"])
            item["labels"] = LABEL2ID.get(ex.get("label_in_hebrew", ""), -1)
            self._data.append(item)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


# ─────────────────────────────────────────────────────────────────────────────
# compute_metrics for Trainer
# ─────────────────────────────────────────────────────────────────────────────

def _build_compute_metrics(tokenizer_or_none=None):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):  # enc-dec models return (logits, encoder_outputs, ...)
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        valid = labels != -1
        preds_v  = preds[valid].tolist()
        labels_v = labels[valid].tolist()
        return {
            "macro_f1": macro_f1(labels_v, preds_v),
            "accuracy": accuracy(labels_v, preds_v),
        }
    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Encoder-only: full inline training
# ─────────────────────────────────────────────────────────────────────────────

def run_encoder_only(
    model_name: str,
    log: logging.Logger,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
    zero_token_type_ids: bool = False,
    device_map: Optional[str] = None,
    optim: Optional[str] = None,
    gradient_checkpointing: bool = False,
    use_fsdp: bool = False,
) -> dict:
    safe = _safe(model_name)
    output_dir = OUTPUTS_DIR / safe

    log.info("Loading data from %s", DATA_DIR)
    train_data = load_split("train")
    val_data   = load_split("val")
    test_data  = load_split("test")
    log.info("Splits: train=%d  val=%d  test=%d", len(train_data), len(val_data), len(test_data))

    log.info("Loading tokenizer (trust_remote_code=%s)", trust_remote_code)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=trust_remote_code
    )

    log.info("Tokenising splits (max_length=%d, zero_token_type_ids=%s)",
             CLF["max_length"], zero_token_type_ids)
    train_ds = NLIDataset(train_data, tokenizer, CLF["max_length"], zero_token_type_ids)
    val_ds   = NLIDataset(val_data,   tokenizer, CLF["max_length"], zero_token_type_ids)
    test_ds  = NLIDataset(test_data,  tokenizer, CLF["max_length"], zero_token_type_ids)
    log.info("Tokenisation done")

    model_kwargs: dict = dict(
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        trust_remote_code=trust_remote_code,
    )
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if device_map:
        model_kwargs["device_map"] = device_map

    log.info("Loading model: %s  (num_labels=3, attn_implementation=%s, device_map=%s)",
             model_name, attn_implementation or "default", device_map or "single GPU")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name, **model_kwargs
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model loaded — parameters: %s", f"{n_params:,}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = transformers.TrainingArguments(
        output_dir                 = str(output_dir),
        max_steps                  = CLF["max_steps"],
        per_device_train_batch_size= 1 if use_fsdp else CLF["batch_size"],
        per_device_eval_batch_size = 1 if use_fsdp else CLF["eval_batch_size"],
        learning_rate              = CLF["lr"],
        warmup_ratio               = CLF["warmup_ratio"],
        weight_decay               = CLF["weight_decay"],
        eval_strategy              = "steps",
        eval_steps                 = CLF["eval_steps"],
        save_strategy              = "steps",
        save_steps                 = CLF["eval_steps"],
        save_total_limit           = CLF["save_total_limit"],
        load_best_model_at_end     = True,
        metric_for_best_model      = "macro_f1",
        greater_is_better          = True,
        logging_steps              = 50,
        report_to                  = [],
        seed                       = CLF["seed"],
        data_seed                  = CLF["seed"],
        bf16                       = use_bf16,
        fp16                       = use_fp16,
        auto_find_batch_size       = not use_fsdp,  # disabled for FSDP: probing batches triggers OOM before settling on 1
        optim                      = optim or "adamw_torch",
        gradient_checkpointing     = gradient_checkpointing or use_fsdp,
        fsdp                       = "full_shard" if use_fsdp else "",
        fsdp_config                = {"fsdp_min_num_params": 1e8} if use_fsdp else {},
    )
    log.info("TrainingArguments:\n%s", training_args.to_json_string())

    collator = transformers.DataCollatorWithPadding(tokenizer)

    trainer = transformers.Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        compute_metrics = _build_compute_metrics(),
        callbacks       = [DetailedLoggingCallback(log)],
    )

    log.info("Starting training …")
    trainer.train()

    best_ckpt   = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric
    log.info("Training done.  best_checkpoint=%s  dev_macro_f1=%.4f", best_ckpt, best_metric or 0.0)

    # ── dev evaluation ──────────────────────────────────────────────────────
    log.info("Final evaluation on dev set …")
    dev_result = trainer.evaluate(eval_dataset=val_ds)
    log.info("Dev metrics: %s", dev_result)

    # ── test evaluation ──────────────────────────────────────────────────────
    log.info("Generating predictions on test set …")
    test_pred = trainer.predict(test_ds)
    test_preds_int = np.argmax(test_pred.predictions, axis=-1)
    test_labels_int = test_pred.label_ids

    valid_mask    = test_labels_int != -1
    preds_str     = [ID2LABEL[int(p)] for p in test_preds_int[valid_mask]]
    gold_str      = [ID2LABEL[int(l)] for l in test_labels_int[valid_mask]]
    test_macro_f1 = macro_f1(gold_str, preds_str)
    test_acc      = accuracy(gold_str, preds_str)
    log.info("Test macro_f1=%.4f  accuracy=%.4f", test_macro_f1, test_acc)

    if _is_main_process():
        pred_path = RESULTS_DIR / f"{safe}_test_preds.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for ex, pred in zip(test_data, [ID2LABEL[int(p)] for p in test_preds_int]):
                f.write(json.dumps({"input": {"id": ex["id"]}, "prediction": pred},
                                   ensure_ascii=False) + "\n")
        log.info("Test predictions → %s", pred_path)

    results = {
        "model"           : model_name,
        "approach"        : "sequence_classification",
        "dev"             : {
            "macro_f1": round(dev_result.get("eval_macro_f1", best_metric or 0.0), 4),
            "accuracy": round(dev_result.get("eval_accuracy", 0.0), 4),
        },
        "test"            : {
            "macro_f1": round(test_macro_f1, 4),
            "accuracy": round(test_acc, 4),
        },
        "best_checkpoint" : best_ckpt,
    }
    if _is_main_process():
        _write_results(safe, results, log)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Encoder-decoder: subprocess pipeline (mrl_eval finetune → generate → evaluate)
# ─────────────────────────────────────────────────────────────────────────────

def _stream_cmd(cmd: list[str], log_path: pathlib.Path, env: dict,
                cwd: pathlib.Path, step_label: str) -> tuple[int, str]:
    """Run *cmd*, stream lines to log file + stdout, return (returncode, full_output)."""
    with open(log_path, "a", encoding="utf-8") as lf:
        header = f"\n{'='*72}\n[{step_label}] $ {' '.join(cmd)}\n{'='*72}\n"
        lf.write(header)
        print(header, end="", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            cwd=str(cwd),
            env=env,
        )
        lines: list[str] = []
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            print(line, end="", flush=True)
            lines.append(line)
        proc.wait()
        lf.write(f"\n[exit code: {proc.returncode}]\n")
    return proc.returncode, "".join(lines)


def run_encoder_decoder(model_name: str, log: logging.Logger, log_path: pathlib.Path,
                        device_map: Optional[str] = None) -> dict:
    safe = _safe(model_name)
    env  = os.environ.copy()
    pp   = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(MRL_EVAL_ROOT) + (":" + pp if pp else "")
    cwd  = MRL_EVAL_ROOT   # mrl_eval_data/ is relative to here

    # ── step 1: finetune ─────────────────────────────────────────────────────
    log.info("Step 1/3 — finetune  (mrl_eval.hf.finetune)")
    ft_cmd = [
        sys.executable, "-m", "mrl_eval.hf.finetune",
        "--dataset", DATASET,
        "--model", model_name,
    ]
    if device_map:
        ft_cmd += ["--device_map", device_map]
    rc, ft_out = _stream_cmd(ft_cmd, log_path, env, cwd, "FINETUNE")
    if rc != 0:
        raise RuntimeError(f"mrl_eval.hf.finetune exited with code {rc}")

    # parse best checkpoint (rich may line-wrap, so use \s+ before "with")
    m = re.search(r"Best checkpoint is saved at (.+?)\s+with", ft_out, re.DOTALL)
    if not m:
        raise RuntimeError("Could not parse best checkpoint path from finetune output")
    best_ckpt = m.group(1).strip()
    log.info("Best checkpoint: %s", best_ckpt)

    # parse best dev metric
    m2 = re.search(r"validation score of ([\d.eE+\-]+)", ft_out)
    dev_macro_f1 = float(m2.group(1)) if m2 else None
    log.info("Dev macro_f1 (best): %s", dev_macro_f1)

    # ── step 2: generate test predictions ────────────────────────────────────
    log.info("Step 2/3 — generate  (mrl_eval.hf.generate)")
    gen_cmd = [
        sys.executable, "-m", "mrl_eval.hf.generate",
        "--dataset", DATASET,
        "--checkpoint_path", best_ckpt,
    ]
    if device_map:
        gen_cmd += ["--device_map", device_map]
    rc, gen_out = _stream_cmd(gen_cmd, log_path, env, cwd, "GENERATE")
    if rc != 0:
        raise RuntimeError(f"mrl_eval.hf.generate exited with code {rc}")

    m3 = re.search(r"Generated responses saved to (.+\.jsonl)", gen_out)
    if not m3:
        raise RuntimeError("Could not parse predictions path from generate output")
    pred_file = m3.group(1).strip()
    log.info("Predictions: %s", pred_file)

    # ── step 3: evaluate test predictions via mrl_eval ───────────────────────
    log.info("Step 3/3 — evaluate  (mrl_eval.evaluation.evaluate)")
    eval_cmd = [
        sys.executable, "-m", "mrl_eval.evaluation.evaluate",
        "--dataset", DATASET,
        "--predictions_path", pred_file,
    ]
    rc, eval_out = _stream_cmd(eval_cmd, log_path, env, cwd, "EVALUATE")
    if rc != 0:
        log.warning("mrl_eval evaluate exited with code %d — falling back to inline eval", rc)
        test_scores = _eval_from_jsonl(pred_file, log)
    else:
        # parse dict printed by evaluate.py, e.g. {'accuracy': 0.9, 'macro_f1': 0.85}
        dm = re.search(r"\{[^}]+\}", eval_out)
        if dm:
            try:
                test_scores = ast.literal_eval(dm.group(0))
            except Exception:
                test_scores = _eval_from_jsonl(pred_file, log)
        else:
            test_scores = _eval_from_jsonl(pred_file, log)

    log.info("Test scores: %s", test_scores)

    # ── copy predictions to results dir ──────────────────────────────────────
    dest = RESULTS_DIR / f"{safe}_test_preds.jsonl"
    try:
        shutil.copy2(pred_file, dest)
        log.info("Predictions copied → %s", dest)
    except Exception as e:
        log.warning("Could not copy predictions: %s", e)

    results = {
        "model"           : model_name,
        "approach"        : "generative_seq2seq",
        "dev"             : {"macro_f1": round(dev_macro_f1, 4) if dev_macro_f1 else None},
        "test"            : {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in test_scores.items()},
        "best_checkpoint" : best_ckpt,
        "predictions_file": pred_file,
    }
    _write_results(safe, results, log)
    return results


def _eval_from_jsonl(pred_file: str, log: logging.Logger) -> dict:
    """Inline fallback evaluation using gold labels from test.jsonl."""
    try:
        gold_data  = load_split("test")
        gold_map   = {ex["id"]: ex.get("label_in_hebrew", "") for ex in gold_data}
        preds      = load_jsonl(pathlib.Path(pred_file))
        y_true, y_pred = [], []
        for p in preds:
            pid  = p["input"]["id"]
            gold = gold_map.get(pid, "")
            if gold:
                y_true.append(gold)
                y_pred.append(p["prediction"].strip())
        mf1 = macro_f1(y_true, y_pred)
        acc = accuracy(y_true, y_pred)
        log.info("Inline eval — macro_f1=%.4f  accuracy=%.4f", mf1, acc)
        return {"macro_f1": mf1, "accuracy": acc}
    except Exception as e:
        log.error("Inline eval failed: %s", e)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_results(safe: str, results: dict, log: logging.Logger) -> None:
    path = RESULTS_DIR / f"{safe}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log.info("Results written → %s", path)


def _is_encoder_decoder(model_name: str, trust_remote_code: bool = False) -> bool:
    cfg = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    return bool(cfg.is_encoder_decoder)


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _run_via_torchrun(
    model_name: str, nproc: int, log: logging.Logger, log_path: pathlib.Path
) -> dict:
    """Spawn this script via torchrun for FSDP multi-GPU training of a single model."""
    env = os.environ.copy()
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master_port", "29501",
        str(pathlib.Path(__file__).resolve()),
        "--single-model", model_name,
    ]
    rc, _ = _stream_cmd(cmd, log_path, env, SCRIPT_DIR.parent, "TORCHRUN")
    if rc != 0:
        raise RuntimeError(f"torchrun exited with code {rc}")
    result_file = RESULTS_DIR / f"{_safe(model_name)}_results.json"
    if not result_file.exists():
        raise RuntimeError("torchrun completed but result file not found")
    return json.load(open(result_file))


def _torchrun_single_model_main(model_name: str) -> None:
    """Entry point when this script is invoked by torchrun for a single model (FSDP mode)."""
    log, log_path = make_logger(model_name)
    log.info("torchrun FSDP mode — RANK=%s LOCAL_RANK=%s — model: %s",
             os.environ.get("RANK", "0"), os.environ.get("LOCAL_RANK", "0"), model_name)
    flags = MODEL_FLAGS.get(model_name, {})
    results = run_encoder_only(
        model_name, log,
        trust_remote_code   = flags.get("trust_remote_code", False),
        attn_implementation = flags.get("attn_implementation", None),
        zero_token_type_ids = flags.get("zero_token_type_ids", False),
        use_fsdp            = True,
    )
    if _is_main_process():
        log.info("FSDP training complete. results=%s", results)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # When launched by torchrun for a single model (FSDP mode), run only that model
    if _CLI.single_model:
        _torchrun_single_model_main(_CLI.single_model)
        return

    t0 = datetime.datetime.now()
    banner = (
        f"\n{'#'*72}\n"
        f"# HebNLI multi-model fine-tuning\n"
        f"# Started : {t0.isoformat()}\n"
        f"# Models  : {len(MODELS)}\n"
        f"# Data    : {DATA_DIR}\n"
        f"# Logs    : {LOGS_DIR}\n"
        f"{'#'*72}\n"
    )
    print(banner)

    # ── quick data-presence check ──────────────────────────────────────────
    for split in ("train", "val", "test"):
        p = DATA_DIR / f"{split}.jsonl"
        if not p.exists():
            print(
                f"[FATAL] Data file not found: {p}\n"
                "Run the mrl_eval data ingestion before this script:\n"
                "  cd /home/nlp/ronke21\n"
                "  bash mrl_eval/datasets/download_raw_data.sh\n"
                "  bash mrl_eval/datasets/ingest_all_datasets.sh\n"
                "Then re-run this script."
            )
            sys.exit(1)

    all_results: dict = {}

    for model_name in MODELS:
        sep = f"\n{'='*72}\n  MODEL: {model_name}\n{'='*72}"
        print(sep)

        # Skip if results already exist
        result_file = RESULTS_DIR / f"{_safe(model_name)}_results.json"
        if result_file.exists():
            print(f"  [SKIP] Results already exist: {result_file}")
            try:
                all_results[model_name] = {"status": "success",
                                           "results": json.load(open(result_file))}
            except Exception:
                pass
            continue

        log, log_path = make_logger(model_name)
        log.info("=" * 60)
        log.info("MODEL : %s", model_name)
        log.info("LOG   : %s", log_path)
        log.info("=" * 60)

        flags = MODEL_FLAGS.get(model_name, {})
        trust_remote_code      = flags.get("trust_remote_code", False)
        attn_implementation    = flags.get("attn_implementation", None)
        zero_token_type_ids    = flags.get("zero_token_type_ids", False)
        device_map             = flags.get("device_map", None)
        optim                  = flags.get("optim", None)
        gradient_checkpointing = flags.get("gradient_checkpointing", False)
        torchrun_nproc         = flags.get("torchrun_nproc", 0)
        log.info("model_flags: trust_remote_code=%s  attn_implementation=%s  zero_token_type_ids=%s  "
                 "device_map=%s  optim=%s  gradient_checkpointing=%s  torchrun_nproc=%s",
                 trust_remote_code, attn_implementation, zero_token_type_ids,
                 device_map, optim, gradient_checkpointing, torchrun_nproc)

        try:
            if torchrun_nproc:
                log.info("Launching via torchrun (FSDP, nproc=%d) …", torchrun_nproc)
                results = _run_via_torchrun(model_name, torchrun_nproc, log, log_path)
            else:
                log.info("Detecting architecture …")
                enc_dec = _is_encoder_decoder(model_name, trust_remote_code=trust_remote_code)
                log.info("Architecture: %s", "encoder-decoder" if enc_dec else "encoder-only")

                if enc_dec:
                    results = run_encoder_decoder(model_name, log, log_path, device_map=device_map)
                else:
                    results = run_encoder_only(
                        model_name, log,
                        trust_remote_code=trust_remote_code,
                        attn_implementation=attn_implementation,
                        zero_token_type_ids=zero_token_type_ids,
                        device_map=device_map,
                        optim=optim,
                        gradient_checkpointing=gradient_checkpointing,
                    )

            all_results[model_name] = {"status": "success", "results": results}
            log.info("FINISHED SUCCESSFULLY")

        except Exception as exc:
            tb = traceback.format_exc()
            log.error("FAILED: %s", exc)
            log.error("Traceback:\n%s", tb)
            print(f"\n[ERROR] {model_name} failed — see {log_path} for details.\n"
                  f"  Reason: {exc}\n"
                  "Continuing to next model …\n")
            all_results[model_name] = {
                "status"   : "failed",
                "error"    : str(exc),
                "traceback": tb,
            }

    # ── write summary JSON ─────────────────────────────────────────────────
    summary_path = SCRIPT_DIR / "results" / "summaries" / "encoder_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    # ── print summary table ────────────────────────────────────────────────
    elapsed = datetime.datetime.now() - t0
    print(f"\n\n{'#'*72}")
    print(f"# SUMMARY — HebNLI Fine-tuning  (metric: macro F1)")
    print(f"{'#'*72}")
    hdr = f"{'Model':<42} {'Status':<10} {'Dev F1':>8} {'Test F1':>9} {'Test Acc':>9}"
    print(hdr)
    print("-" * len(hdr))
    for model_name in MODELS:
        entry = all_results.get(model_name, {})
        if entry.get("status") == "success":
            r      = entry["results"]
            dev_f1 = r.get("dev", {}).get("macro_f1")
            t_f1   = r.get("test", {}).get("macro_f1")
            t_acc  = r.get("test", {}).get("accuracy")
            print(
                f"{model_name:<42} {'OK':<10}"
                f" {f'{dev_f1:.4f}' if dev_f1 is not None else 'N/A':>8}"
                f" {f'{t_f1:.4f}' if t_f1 is not None else 'N/A':>9}"
                f" {f'{t_acc:.4f}' if t_acc is not None else 'N/A':>9}"
            )
        else:
            err = entry.get("error", "?")
            print(f"{model_name:<42} {'FAILED':<10} {'N/A':>8} {'N/A':>9} {'N/A':>9}"
                  f"  ← {err[:60]}")

    print(f"\nTotal time  : {elapsed}")
    print(f"Summary JSON: {summary_path}")
    print(f"Logs dir    : {LOGS_DIR}")
    print(f"Results dir : {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()
