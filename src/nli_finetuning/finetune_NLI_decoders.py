#!/usr/bin/env python3
"""
finetune_NLI_decoders.py — Fine-tune and evaluate decoder LLMs on HebNLI.

All models are decoder-only instruct models fine-tuned with LoRA (r=16,
target_modules=all-linear, bf16) via the mrl_eval subprocess pipeline:
    finetune  →  generate (test)  →  evaluate (inline)

Models:
    google/gemma-2-9b
    google/gemma-2-9b-it
    google/gemma-3-12b-it
    google/gemma-3-12b-pt
    google/gemma-3-27b-it
    google/gemma-3-27b-pt
    Qwen/Qwen3-14B
    Qwen/Qwen3-14B-Base
    dicta-il/DictaLM-3.0-1.7B-Instruct
    dicta-il/DictaLM-3.0-1.7B-Thinking
    dicta-il/DictaLM-3.0-1.7B-Base
    dicta-il/DictaLM-3.0-24B-Base
    dicta-il/DictaLM-3.0-24B-Thinking
    CohereForAI/aya-expanse-32b
    mistralai/Mistral-Small-24B-Instruct-2501
    Qwen/Qwen3-30B-A3B-Instruct-2507
    Qwen/Qwen3.5-35B-A3B-Base
    google/gemma-4-26B-A4B
    google/gemma-4-26B-A4B-it
    google/gemma-4-31B
    google/gemma-4-31B-it

Conda environment: heb_nli_mrl_eval
    conda activate heb_nli_mrl_eval

Run from Hebrew_NLI/:
    cd /path/to/HEntailment-Hebrew-NLI-Repurposing-EACL27-Internal
    CUDA_VISIBLE_DEVICES=0,1 python finetune_heb_nli/finetune_NLI_decoders.py

Output layout under finetune_heb_nli/:
    logs/decoders/<model>.log     full training output per model
    results/decoders/             per-model JSON + prediction files
    decoder_summary.json          final metrics table
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
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

# ── third-party ───────────────────────────────────────────────────────────────
try:
    from sklearn.metrics import accuracy_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = pathlib.Path(__file__).parent.parent.resolve()
MRL_EVAL_ROOT = SCRIPT_DIR.parent           # Hebrew_NLI/  (contains mrl_eval/ and mrl_eval_data/)
DATA_DIR      = MRL_EVAL_ROOT / "mrl_eval_data" / "hebnli" / "jsonl"
LOGS_DIR      = SCRIPT_DIR / "logs" / "decoders"
RESULTS_DIR   = SCRIPT_DIR / "results" / "finetuned_singlerun"

for _d in (LOGS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────
DATASET = "hebnli"

MODELS = [
    "CohereForAI/aya-expanse-32b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "dicta-il/dictalm2.0",
    "dicta-il/dictalm2.0-instruct",
    "dicta-il/DictaLM-3.0-1.7B-Base",
    "dicta-il/DictaLM-3.0-1.7B-Instruct",
    "dicta-il/DictaLM-3.0-1.7B-Thinking",
    "dicta-il/DictaLM-3.0-24B-Base",
    "dicta-il/DictaLM-3.0-24B-Thinking",
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-12b-pt",
    "google/gemma-3-27b-it",
    "google/gemma-3-27b-pt",
    "google/gemma-4-26B-A4B",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B",
    "google/gemma-4-31B-it",
    "google/gemma-4-E4B",
    "mistralai/Mistral-Small-24B-Base-2501",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-14B-Base",
    "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3.5-35B-A3B-Base",
]

PER_MODEL_FLAGS: dict[str, list[str]] = {
    "google/gemma-3-12b-it": [
        "--attn_implementation", "eager",
    ],
    "google/gemma-3-12b-pt": [
        "--attn_implementation", "eager",
    ],
    "google/gemma-3-27b-it": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
    ],
    "google/gemma-3-27b-pt": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
        "--max_grad_norm", "0.3",
    ],
    "Qwen/Qwen3-14B": [
        "--attn_implementation", "eager",
    ],
    "Qwen/Qwen3-14B-Base": [
        "--attn_implementation", "eager",
    ],
    "dicta-il/dictalm2.0": [
        "--attn_implementation", "eager",
        "--trust_remote_code",
    ],
    "dicta-il/dictalm2.0-instruct": [
        "--attn_implementation", "eager",
        "--trust_remote_code",
    ],
    "dicta-il/DictaLM-3.0-1.7B-Instruct": [
        "--attn_implementation", "eager",
    ],
    "dicta-il/DictaLM-3.0-1.7B-Thinking": [
        "--attn_implementation", "eager",
        "--trust_remote_code",
    ],
    "dicta-il/DictaLM-3.0-1.7B-Base": [
        "--attn_implementation", "eager",
    ],
    "dicta-il/DictaLM-3.0-24B-Base": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
    ],
    "dicta-il/DictaLM-3.0-24B-Thinking": [
        "--attn_implementation", "eager",
        "--trust_remote_code",
        "--device_map", "auto",
    ],
    "CohereForAI/aya-expanse-32b": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
    ],
    "mistralai/Mistral-Small-24B-Base-2501": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
    ],
    "mistralai/Mistral-Small-24B-Instruct-2501": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
    ],
    "Qwen/Qwen3-4B": [
        "--attn_implementation", "eager",
    ],
    "Qwen/Qwen3-4B-Instruct-2507": [
        "--attn_implementation", "eager",
    ],
    "Qwen/Qwen3-4B-Thinking-2507": [
        "--attn_implementation", "eager",
    ],
    "Qwen/Qwen3-30B-A3B-Base": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
    ],
    "Qwen/Qwen3-30B-A3B-Instruct-2507": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
        "--save_strategy", "steps",
        "--save_steps", "200",
    ],
    "Qwen/Qwen3-30B-A3B-Thinking-2507": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
        "--save_strategy", "steps",
        "--save_steps", "200",
    ],
    "Qwen/Qwen3.5-35B-A3B-Base": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
    ],
    "google/gemma-4-E4B": [
        "--attn_implementation", "eager",
    ],
    "google/gemma-4-26B-A4B": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
    ],
    "google/gemma-4-26B-A4B-it": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
    ],
    "google/gemma-4-31B": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
    ],
    "google/gemma-4-31B-it": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
    ],
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": [
        "--attn_implementation", "eager",
        "--device_map", "auto",
    ],
}

# HebNLI labels in Hebrew (entailment / contradiction / neutral)
LABELS   = ["היסק", "סתירה", "ניטרלי"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _safe(model_name: str) -> str:
    return model_name.replace("/", "__")


def make_logger(model_name: str) -> tuple[logging.Logger, pathlib.Path]:
    log_path = LOGS_DIR / f"{_safe(model_name)}.log"
    log = logging.getLogger(f"dec.{model_name}")
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
# Data
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
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def macro_f1(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    if HAS_SKLEARN:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    classes = list(set(y_true))
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0)
    return float(sum(f1s) / len(f1s))


def acc(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    if HAS_SKLEARN:
        return float(accuracy_score(y_true, y_pred))
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction normalisation
# Decoder models sometimes emit extra whitespace, punctuation, or — for Qwen3
# in thinking mode — a <think>…</think> block before the answer.
# ─────────────────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def normalise_pred(pred: str) -> str:
    """Strip thinking blocks, then return the first matching Hebrew NLI label."""
    pred = _THINK_RE.sub("", pred).strip()
    # exact match first
    if pred in LABEL2ID:
        return pred
    # substring match (handles trailing punctuation / newlines)
    for label in LABELS:
        if label in pred:
            return label
    return pred   # return as-is so we can see what the model said


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess runner (streams output to log file + stdout simultaneously)
# ─────────────────────────────────────────────────────────────────────────────

def _stream_cmd(
    cmd: list[str],
    log_path: pathlib.Path,
    env: dict,
    cwd: pathlib.Path,
    step_label: str,
) -> tuple[int, str]:
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


# ─────────────────────────────────────────────────────────────────────────────
# Inline test evaluation (avoids TFRecord dependency of mrl_eval evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def eval_from_jsonl(pred_file: str, log: logging.Logger) -> dict:
    gold_data = load_split("test")
    gold_map  = {ex["id"]: ex.get("label_in_hebrew", "") for ex in gold_data}
    preds     = load_jsonl(pathlib.Path(pred_file))

    y_true, y_pred = [], []
    n_bad = 0
    for p in preds:
        pid  = p["input"]["id"]
        gold = gold_map.get(pid, "")
        if not gold:
            continue
        raw  = p["prediction"]
        norm = normalise_pred(raw)
        y_true.append(gold)
        y_pred.append(norm)
        if norm not in LABEL2ID:
            n_bad += 1
            log.warning("Unrecognised prediction for id=%s: %r → %r", pid, raw, norm)

    if n_bad:
        log.warning("%d / %d predictions could not be mapped to a label", n_bad, len(y_true))

    mf1 = macro_f1(y_true, y_pred)
    a   = acc(y_true, y_pred)
    log.info("Inline eval — macro_f1=%.4f  accuracy=%.4f  n=%d", mf1, a, len(y_true))
    return {"macro_f1": mf1, "accuracy": a, "n_evaluated": len(y_true), "n_bad_preds": n_bad}


# ─────────────────────────────────────────────────────────────────────────────
# Per-model training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_decoder(
    model_name: str,
    log: logging.Logger,
    log_path: pathlib.Path,
    extra_ft_flags: list[str] | None = None,
) -> dict:
    env = os.environ.copy()
    pp  = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(MRL_EVAL_ROOT) + (":" + pp if pp else "")
    cwd = MRL_EVAL_ROOT   # mrl_eval_data/ is relative to here

    # ── 1. Fine-tune with LoRA ───────────────────────────────────────────────
    log.info("Step 1/3 — LoRA fine-tune via mrl_eval.hf.finetune")
    ft_cmd = [
        sys.executable, "-m", "mrl_eval.hf.finetune",
        "--dataset", DATASET,
        "--model", model_name,
        "--lora_r", "16",
        "--decoder_max_steps", "8000",
        "--decoder_batch_size", "16",
    ]
    if extra_ft_flags:
        ft_cmd.extend(extra_ft_flags)

    # Auto-resume from latest checkpoint if one exists in the output dir.
    # Only applies when --save_strategy steps is in extra_ft_flags.
    if extra_ft_flags and "--save_strategy" in extra_ft_flags:
        safe_name = model_name.split("/")[-1]   # mrl_eval uses only the repo name
        ckpt_dir  = SCRIPT_DIR / "decoder_checkpoints" / f"{safe_name}_hebnli"
        if ckpt_dir.exists():
            ckpts = sorted(
                [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda d: int(d.name.split("-")[1]),
            )
            if ckpts:
                latest = ckpts[-1]
                log.info("Resuming from checkpoint: %s", latest)
                ft_cmd.extend(["--resume_from_checkpoint", str(latest)])

    log.info("ft_cmd: %s", " ".join(ft_cmd))
    rc, ft_out = _stream_cmd(ft_cmd, log_path, env, cwd, "FINETUNE")
    if rc != 0:
        raise RuntimeError(f"mrl_eval.hf.finetune exited with code {rc}")

    # parse best checkpoint path (rich may line-wrap, so use \s+ before "with")
    m = re.search(r"Best checkpoint is saved at (.+?)\s+with", ft_out, re.DOTALL)
    if not m:
        raise RuntimeError("Could not parse best checkpoint from finetune output")
    best_ckpt = m.group(1).strip()
    log.info("Best checkpoint: %s", best_ckpt)

    # parse best dev metric
    m2 = re.search(r"validation score of ([\d.eE+\-]+)", ft_out)
    dev_macro_f1 = float(m2.group(1)) if m2 else None
    log.info("Best dev macro_f1: %s", dev_macro_f1)

    # ── 2. Generate test predictions ────────────────────────────────────────
    log.info("Step 2/3 — generate test predictions via mrl_eval.hf.generate")
    gen_cmd = [
        sys.executable, "-m", "mrl_eval.hf.generate",
        "--dataset", DATASET,
        "--checkpoint_path", best_ckpt,
    ]
    if extra_ft_flags and "--device_map" in extra_ft_flags:
        idx = extra_ft_flags.index("--device_map")
        gen_cmd.extend(["--device_map", extra_ft_flags[idx + 1]])
    if extra_ft_flags and "--trust_remote_code" in extra_ft_flags:
        gen_cmd.append("--trust_remote_code")
    rc, gen_out = _stream_cmd(gen_cmd, log_path, env, cwd, "GENERATE")
    if rc != 0:
        raise RuntimeError(f"mrl_eval.hf.generate exited with code {rc}")

    m3 = re.search(r"Generated responses saved to (.+\.jsonl)", gen_out)
    if not m3:
        raise RuntimeError("Could not parse predictions path from generate output")
    pred_file = m3.group(1).strip()
    log.info("Predictions file: %s", pred_file)

    # ── 3. Evaluate (inline — avoids TFRecord dependency) ───────────────────
    log.info("Step 3/3 — inline evaluation against test.jsonl gold labels")
    test_scores = eval_from_jsonl(pred_file, log)
    log.info("Test scores: %s", test_scores)

    # copy predictions to results dir for later inspection
    dest = RESULTS_DIR / f"{_safe(model_name)}_test_preds.jsonl"
    try:
        shutil.copy2(pred_file, dest)
        log.info("Predictions copied → %s", dest)
    except Exception as e:
        log.warning("Could not copy predictions: %s", e)

    results = {
        "model"           : model_name,
        "approach"        : "decoder_lora",
        "dev"             : {"macro_f1": round(dev_macro_f1, 4) if dev_macro_f1 is not None else None},
        "test"            : {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in test_scores.items()},
        "best_checkpoint" : best_ckpt,
        "predictions_file": pred_file,
    }

    results_path = RESULTS_DIR / f"{_safe(model_name)}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log.info("Results written → %s", results_path)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--models", default=None,
                        help="Comma-separated subset of model IDs to run (default: all)")
    parser.add_argument("--single-model", default=None, dest="single_model",
                        help="Run exactly one model by ID")
    args, _ = parser.parse_known_args()

    run_models = MODELS
    if args.single_model:
        run_models = [args.single_model]
    elif args.models:
        requested = [m.strip() for m in args.models.split(",")]
        run_models = [m for m in MODELS if m in requested]
        unknown = set(requested) - set(MODELS)
        if unknown:
            print(f"[WARN] Unknown model(s) ignored: {unknown}")

    t0 = datetime.datetime.now()
    banner = (
        f"\n{'#'*72}\n"
        f"# HebNLI — Decoder LLM fine-tuning (LoRA)\n"
        f"# Started  : {t0.isoformat()}\n"
        f"# Models   : {len(run_models)}\n"
        f"# Data     : {DATA_DIR}\n"
        f"# Logs     : {LOGS_DIR}\n"
        f"{'#'*72}\n"
    )
    print(banner)

    # data-presence check
    for split in ("train", "val", "test"):
        p = DATA_DIR / f"{split}.jsonl"
        if not p.exists():
            print(
                f"[FATAL] Data file not found: {p}\n"
                "Run the mrl_eval data ingestion before this script:\n"
                "  cd /path/to/HEntailment-Hebrew-NLI-Repurposing-EACL27-Internal\n"
                "  bash mrl_eval/datasets/download_raw_data.sh\n"
                "  bash mrl_eval/datasets/ingest_all_datasets.sh\n"
            )
            sys.exit(1)

    all_results: dict = {}

    for model_name in run_models:
        print(f"\n{'='*72}\n  MODEL: {model_name}\n{'='*72}")
        log, log_path = make_logger(model_name)
        log.info("=" * 60)
        log.info("MODEL : %s", model_name)
        log.info("LOG   : %s", log_path)
        log.info("=" * 60)

        result_file = RESULTS_DIR / f"{_safe(model_name)}_results.json"
        if result_file.exists():
            print(f"  [SKIP] Results already exist: {result_file}")
            log.info("SKIP — results already exist at %s", result_file)
            try:
                all_results[model_name] = {"status": "success",
                                           "results": json.load(open(result_file))}
            except Exception:
                pass
            continue

        extra_flags = PER_MODEL_FLAGS.get(model_name, [])
        log.info("extra_flags : %s", extra_flags)

        try:
            results = run_decoder(model_name, log, log_path, extra_ft_flags=extra_flags)
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

    # ── write summary JSON ──────────────────────────────────────────────────
    summary_path = SCRIPT_DIR / "results" / "summaries" / "decoder_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    # ── print summary table ─────────────────────────────────────────────────
    elapsed = datetime.datetime.now() - t0
    print(f"\n\n{'#'*72}")
    print(f"# SUMMARY — HebNLI Decoder Fine-tuning  (metric: macro F1)")
    print(f"{'#'*72}")
    hdr = f"{'Model':<50} {'Status':<10} {'Dev F1':>8} {'Test F1':>9} {'Test Acc':>9}"
    print(hdr)
    print("-" * len(hdr))
    for model_name in run_models:
        entry = all_results.get(model_name, {})
        if entry.get("status") == "success":
            r      = entry["results"]
            dev_f1 = r.get("dev", {}).get("macro_f1")
            t_f1   = r.get("test", {}).get("macro_f1")
            t_acc  = r.get("test", {}).get("accuracy")
            print(
                f"{model_name:<50} {'OK':<10}"
                f" {f'{dev_f1:.4f}' if dev_f1 is not None else 'N/A':>8}"
                f" {f'{t_f1:.4f}' if t_f1 is not None else 'N/A':>9}"
                f" {f'{t_acc:.4f}' if t_acc is not None else 'N/A':>9}"
            )
        else:
            err = entry.get("error", "?")
            print(f"{model_name:<50} {'FAILED':<10} {'N/A':>8} {'N/A':>9} {'N/A':>9}"
                  f"  ← {err[:55]}")

    print(f"\nTotal time   : {elapsed}")
    print(f"Summary JSON : {summary_path}")
    print(f"Logs dir     : {LOGS_DIR}")
    print(f"Results dir  : {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()
