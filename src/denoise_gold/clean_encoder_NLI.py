"""
Run all fine-tuned NLI encoder models (from Hebrew_NLI project) on a CSV with
premise-hypothesis column pairs.  For each (model, hypothesis_col) combination adds:
  confidence_{tag}_{hyp_col}  — entailment probability

Evaluates each combination at multiple fixed thresholds AND with 5-fold CV
threshold optimisation against the gold label column, then writes:
  - per-combo metrics table
  - k-fold CV threshold search results (honest OOF F1 estimate)
  - ensemble columns
  - summary / error-analysis files

Models used:
  Encoders (AutoModelForSequenceClassification):
    xlmroberta   — FacebookAI/xlm-roberta-large  (checkpoint-8000, test macro-F1=0.876)
    me5large     — intfloat/multilingual-e5-large (checkpoint-6000, test macro-F1=0.850)
    mmbert       — jhu-clsp/mmBERT-base           (checkpoint-5800, test macro-F1=0.845)
    neodictabert — dicta-il/neodictabert          (checkpoint-5600, test macro-F1=0.878)
    alephbert    — onlplab/alephbert-base         (checkpoint-4600, test macro-F1=0.781)
    xlmrobertaxl — facebook/xlm-roberta-xl        (checkpoint-8000)
  Seq2Seq (MT5ForConditionalGeneration, first-token logit scoring):
    mt5large     — google/mt5-large  (checkpoint-7000)
    mt5xl        — google/mt5-xl     (checkpoint-6000)

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts_clean_data.clean_encoder_NLI
    CUDA_VISIBLE_DEVICES=0 python -m scripts_clean_data.clean_encoder_NLI --input data/prepared_gold_500.csv
    CUDA_VISIBLE_DEVICES=0 python -m scripts_clean_data.clean_encoder_NLI 2>&1 | tee nli_run_temp.txt
"""

import os
import csv
import time
import logging
import argparse

import numpy as np
import torch
import transformers
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths / macros
# ---------------------------------------------------------------------------

INPUT_FILE          = "data/prepared_gold_500.csv"
OUTPUT_FILE         = "outputs/encoder_NLI/classified.csv"
LOG_FILE            = "outputs/encoder_NLI/classify.log"
SUMMARY_FILE        = "outputs/encoder_NLI/summary.txt"
ERROR_ANALYSIS_FILE = "outputs/encoder_NLI/error_analysis.txt"

ANALYSIS_THRESHOLD = 0.7
LABEL_COL = "relation_present"

CV_N_SPLITS    = 5
THRESHOLD_GRID = [round(t / 100, 2) for t in range(5, 96, 5)]

# Absolute paths to Hebrew_NLI fine-tuned checkpoints
_HEB_NLI_ENC = "/path/to/Hebrew_NLI/finetune_heb_nli/outputs"
_HEB_NLI_LLM = "/path/to/Hebrew_NLI/output"

# Each entry: (checkpoint_path, tag, model_type)
# model_type "encoder"  → AutoModelForSequenceClassification  (3-class logits)
# model_type "seq2seq"  → AutoModelForSeq2SeqLM               (first-token logit scoring)
NLI_MODELS = [
    (f"{_HEB_NLI_ENC}/FacebookAI__xlm-roberta-large/checkpoint-8000",  "xlmroberta",   "encoder"),
    (f"{_HEB_NLI_ENC}/intfloat__multilingual-e5-large/checkpoint-6000", "me5large",    "encoder"),
    (f"{_HEB_NLI_ENC}/jhu-clsp__mmBERT-base/checkpoint-5800",          "mmbert",       "encoder"),
    (f"{_HEB_NLI_ENC}/dicta-il__neodictabert/checkpoint-5600",         "neodictabert", "encoder"),
    (f"{_HEB_NLI_ENC}/onlplab__alephbert-base/checkpoint-4600",        "alephbert",    "encoder"),
    (f"{_HEB_NLI_ENC}/facebook__xlm-roberta-xl/checkpoint-8000",       "xlmrobertaxl", "encoder"),
    (f"{_HEB_NLI_LLM}/mt5-large_hebnli/checkpoint-7000",               "mt5large",     "seq2seq"),
    (f"{_HEB_NLI_LLM}/mt5-xl_hebnli/checkpoint-6000",                  "mt5xl",        "seq2seq"),
]

PREMISE_HYPOTHESIS_PAIRS = [
    ("text", "basic_relation"),
    ("text", "template_relation"),
    ("text", "llm_relation"),
]

EVAL_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]

# Scoring formulas: (key, display_label, cv_grid, eval_thresholds)
#   pe  = P(entailment)                  — range [0, 1]
#   emc = P(entailment) − P(contradiction) — range [-1, 1]; separates e from c directly
#   enn = P(e) / (P(e)+P(c))             — range [0, 1]; fraction of non-neutral mass that is entailment
_EMC_GRID = [round(t / 100, 2) for t in range(-90, 91, 5)]
SCORING_FORMULAS = [
    ("pe",  "P(e)",            THRESHOLD_GRID, EVAL_THRESHOLDS),
    ("emc", "P(e)-P(c)",       _EMC_GRID,      [-0.5, 0.0, 0.3, 0.5, 0.7]),
    ("enn", "P(e)/(P(e)+P(c))", THRESHOLD_GRID, EVAL_THRESHOLDS),
]


def derive_scores(pe: list[float], pc: list[float], formula: str) -> list[float]:
    if formula == "pe":
        return pe
    if formula == "emc":
        return [e - c for e, c in zip(pe, pc)]
    if formula == "enn":
        return [e / max(e + c, 1e-8) for e, c in zip(pe, pc)]
    raise ValueError(f"Unknown formula: {formula}")

DEFAULT_BATCH_SIZE    = 64
LARGE_TEXT_THRESHOLD  = 256
LARGE_TEXT_BATCH_SIZE = 8
LOG_PROGRESS_EVERY    = 25


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TqdmToLogger:
    def __init__(self, logger):
        self._logger = logger

    def write(self, msg):
        msg = msg.strip()
        if msg:
            self._logger.info(msg)

    def flush(self):
        pass


def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("clean_encoder_NLI")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h:  return f"{h}h {m}m {s}s"
    if m:  return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(scores: list[float], labels: list[str], threshold: float) -> dict:
    TP = FP = FN = TN = 0
    for score, label in zip(scores, labels):
        pred = score >= threshold
        gold = label == "1"
        if   pred and gold:      TP += 1
        elif pred and not gold:  FP += 1
        elif not pred and gold:  FN += 1
        else:                    TN += 1

    total     = TP + FP + FN + TN
    accuracy  = (TP + TN) / total if total else 0
    precision = TP / (TP + FP)    if (TP + FP) else 0
    recall    = TP / (TP + FN)    if (TP + FN) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    sv     = sorted(scores)
    n      = len(sv)
    mean   = sum(scores) / n if n else 0
    median = sv[n // 2]      if n else 0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "score_mean": mean, "score_median": median,
        "score_min": sv[0] if n else 0, "score_max": sv[-1] if n else 0,
    }


# ---------------------------------------------------------------------------
# ROC / AUC helpers
# ---------------------------------------------------------------------------

def compute_roc_stats(scores: list[float], labels: list[str]) -> tuple[float, float, float]:
    """Return (auc, youden_threshold, f1_at_youden_threshold)."""
    labels_int = np.array([int(l) for l in labels])
    scores_arr = np.array(scores)
    auc        = float(roc_auc_score(labels_int, scores_arr))
    fpr, tpr, threshs = roc_curve(labels_int, scores_arr)
    best_idx   = int(np.argmax(tpr - fpr))
    opt_t      = float(threshs[best_idx])
    preds      = (scores_arr >= opt_t).astype(int)
    tp = int(((preds == 1) & (labels_int == 1)).sum())
    fp = int(((preds == 1) & (labels_int == 0)).sum())
    fn = int(((preds == 0) & (labels_int == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return auc, opt_t, f1


# ---------------------------------------------------------------------------
# K-fold CV threshold optimisation
# ---------------------------------------------------------------------------

def kfold_threshold_cv(
    scores: list[float],
    labels: list[str],
    n_splits: int,
    threshold_grid: list[float],
) -> tuple[float, float, list[float], list[float], float, list[float]]:
    """Returns (cv_mean_f1, cv_std_f1, fold_thresholds, fold_f1s, cv_mean_auc, fold_aucs)."""
    scores_arr = np.array(scores)
    labels_int = np.array([int(l) for l in labels])

    kf              = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_f1s:        list[float] = []
    fold_thresholds: list[float] = []
    fold_aucs:       list[float] = []

    for train_idx, val_idx in kf.split(scores_arr.reshape(-1, 1), labels_int):
        tr_s, tr_l = scores_arr[train_idx], labels_int[train_idx]
        va_s, va_l = scores_arr[val_idx],   labels_int[val_idx]

        best_t, best_train_f1 = threshold_grid[0], -1.0
        for t in threshold_grid:
            preds = (tr_s >= t).astype(int)
            tp = int(((preds == 1) & (tr_l == 1)).sum())
            fp = int(((preds == 1) & (tr_l == 0)).sum())
            fn = int(((preds == 0) & (tr_l == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best_train_f1:
                best_train_f1, best_t = f1, t

        fold_thresholds.append(best_t)

        val_preds = (va_s >= best_t).astype(int)
        tp = int(((val_preds == 1) & (va_l == 1)).sum())
        fp = int(((val_preds == 1) & (va_l == 0)).sum())
        fn = int(((val_preds == 0) & (va_l == 1)).sum())
        prec   = tp / (tp + fp) if (tp + fp) else 0.0
        rec    = tp / (tp + fn) if (tp + fn) else 0.0
        val_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        fold_f1s.append(val_f1)

        # Per-fold AUC on validation scores (threshold-free)
        fold_aucs.append(float(roc_auc_score(va_l, va_s)))

    f1_arr  = np.array(fold_f1s)
    auc_arr = np.array(fold_aucs)
    return (float(f1_arr.mean()), float(f1_arr.std()),
            fold_thresholds, fold_f1s,
            float(auc_arr.mean()), fold_aucs)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def compute_ensemble(rows: list[dict], conf_col_names: list[str]) -> list[float]:
    results = []
    for row in rows:
        vals = [float(row[col]) for col in conf_col_names if col in row]
        results.append(sum(vals) / len(vals) if vals else 0.0)
    return results


# ---------------------------------------------------------------------------
# Model loading / unloading
# ---------------------------------------------------------------------------

def load_model(model_path: str, log: logging.Logger):
    log.info(f"    path: {model_path}")
    log.info(f"    torch {torch.__version__}  CUDA build={torch.version.cuda}  "
             f"available={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Make sure PyTorch is installed with CUDA support "
            "and CUDA_VISIBLE_DEVICES is set correctly."
        )

    log.info(f"    visible GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config    = transformers.AutoConfig.from_pretrained(
        model_path, output_hidden_states=False, output_attentions=False, trust_remote_code=True
    )

    try:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path, config=config, trust_remote_code=True, use_safetensors=True
        )
    except Exception as e:
        log.warning(f"    safetensors load failed ({e}), trying normal load.")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path, config=config, trust_remote_code=True
        )

    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"    ready on {device} ({n_params:.0f}M params)")
    return model, tokenizer, device


def unload_model(model, log: logging.Logger):
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("    GPU cache cleared")


def load_seq2seq_model(model_path: str, log: logging.Logger):
    """Load an MT5ForConditionalGeneration checkpoint for first-token logit scoring."""
    log.info(f"    loading seq2seq model: {model_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16
    )
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"    ready on {device} ({n_params:.0f}M params)")
    return model, tokenizer, device


def _seq2seq_label_ids(tokenizer):
    """Return (entail_id, contra_id) — first subword token of each NLI label."""
    def first_id(word):
        ids = tokenizer.encode(word, add_special_tokens=False)
        return ids[0]
    return first_id("היסק"), first_id("סתירה")


def run_seq2seq_nli(
    model, tokenizer, device,
    texts: list[tuple[str, str]],
    log: logging.Logger,
    desc: str = "",
) -> dict:
    """
    Score (premise, hypothesis) pairs using an MT5 seq2seq model.
    Input format: "משפט 1: {premise} משפט 2: {hypothesis}"
    Scores the probability of the first generated token being היסק vs סתירה.
    """
    if not texts:
        return {"pe": [], "pc": []}

    entail_id, contra_id = _seq2seq_label_ids(tokenizer)
    batch_size = DEFAULT_BATCH_SIZE
    n_batches  = (len(texts) + batch_size - 1) // batch_size
    log_every  = max(1, n_batches * LOG_PROGRESS_EVERY // 100)
    log.info(f"    rows={len(texts)}, batch_size={batch_size}, n_batches={n_batches}")
    log.info(f"    entail_id={entail_id} ({tokenizer.decode([entail_id])})  "
             f"contra_id={contra_id} ({tokenizer.decode([contra_id])})")

    pe_list: list[float] = []
    pc_list: list[float] = []
    t_start = time.time()

    for batch_idx, start in enumerate(
        tqdm(range(0, len(texts), batch_size), desc=f"    {desc}",
             leave=False, file=TqdmToLogger(log)), 1
    ):
        batch    = texts[start : start + batch_size]
        prompts  = [f"משפט 1: {p} משפט 2: {h}" for p, h in batch]
        encoded  = tokenizer(
            prompts, return_tensors="pt", padding="longest",
            truncation=True, max_length=512,
        )
        for key in encoded:
            encoded[key] = encoded[key].to(device)

        # Decoder seed: one decoder-start token per example
        bsz = encoded["input_ids"].shape[0]
        dec_start = model.config.decoder_start_token_id or tokenizer.pad_token_id
        dec_ids   = torch.full((bsz, 1), dec_start, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(**encoded, decoder_input_ids=dec_ids, return_dict=True)

        logits = outputs.logits[:, 0, :].float()   # [B, vocab]
        probs  = logits.softmax(dim=-1)
        pe_list.extend(probs[:, entail_id].cpu().tolist())
        pc_list.extend(probs[:, contra_id].cpu().tolist())
        del outputs

        if batch_idx % log_every == 0 or batch_idx == n_batches:
            elapsed   = time.time() - t_start
            rows_done = min(start + batch_size, len(texts))
            log.info(
                f"    progress: {rows_done}/{len(texts)} rows "
                f"({100 * rows_done / len(texts):.0f}%)  {_fmt_duration(elapsed)}"
            )

    return {"pe": pe_list, "pc": pc_list}


# ---------------------------------------------------------------------------
# NLI inference
# ---------------------------------------------------------------------------

def _adaptive_batch_size(texts: list[tuple[str, str]]) -> int:
    if not texts:
        return DEFAULT_BATCH_SIZE
    max_len = max(len(p) + len(h) for p, h in texts)
    return LARGE_TEXT_BATCH_SIZE if max_len > LARGE_TEXT_THRESHOLD else DEFAULT_BATCH_SIZE


def run_nli(
    model, tokenizer, device,
    texts: list[tuple[str, str]],
    log: logging.Logger,
    desc: str = "",
    max_length: int | None = None,  # None → model's natural 512-token limit
) -> list[float]:
    if not texts:
        return []

    batch_size = _adaptive_batch_size(texts)
    n_batches  = (len(texts) + batch_size - 1) // batch_size
    log_every  = max(1, n_batches * LOG_PROGRESS_EVERY // 100)
    log.info(f"    rows={len(texts)}, batch_size={batch_size}, n_batches={n_batches}")

    result      = []
    t_inf_start = time.time()

    for batch_idx, start in enumerate(
        tqdm(range(0, len(texts), batch_size), desc=f"    {desc}",
             leave=False, file=TqdmToLogger(log)), 1
    ):
        batch   = texts[start : start + batch_size]
        tok_kwargs = dict(
            return_tensors="pt", add_special_tokens=True,
            padding="longest", return_token_type_ids=False, truncation=True,
        )
        if max_length is not None:
            tok_kwargs["max_length"] = max_length
        encoded = tokenizer(
            [p for p, _ in batch], [h for _, h in batch], **tok_kwargs
        )
        for key in encoded:
            encoded[key] = encoded[key].to(device)

        with torch.no_grad():
            outputs = model(**encoded, return_dict=True)

        result.append(outputs["logits"].softmax(dim=1))
        del outputs

        if batch_idx % log_every == 0 or batch_idx == n_batches:
            elapsed   = time.time() - t_inf_start
            rows_done = min(start + batch_size, len(texts))
            log.info(
                f"    progress: {rows_done}/{len(texts)} rows "
                f"({100 * rows_done / len(texts):.0f}%)  {_fmt_duration(elapsed)}"
            )

    logits = torch.cat(result)          # shape [N, 3] after softmax
    pe = logits[:, 0]                   # P(היסק / entailment)
    pc = logits[:, 1]                   # P(סתירה / contradiction)
    return {"pe": pe.cpu().tolist(), "pc": pc.cpu().tolist()}


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(
    run_stats: list[dict],
    summary_path: str,
    log: logging.Logger,
    total_time: float,
    ensemble_stats: list[dict] | None = None,
) -> None:
    W = 110
    lines: list[str] = []

    lines.append("=" * W)
    lines.append("ENCODER NLI CLASSIFICATION SUMMARY")
    lines.append(f"Total wall time : {_fmt_duration(total_time)}")
    lines.append(f"CV folds        : {CV_N_SPLITS}  |  thresh grid: 0.05–0.95 (pe/enn)  /  -0.90–0.90 (emc)")
    lines.append(f"Scoring formulas: " + "  |  ".join(f"{k}={lbl}" for k,lbl,_,_ in SCORING_FORMULAS))
    lines.append("Models:")
    for model_path, tag, mtype in NLI_MODELS:
        lines.append(f"  [{tag}]  {model_path}  ({mtype})")
    lines.append("=" * W)

    model_times: dict[str, float] = {}
    for s in run_stats:
        model_times.setdefault(s["model"], 0)
        model_times[s["model"]] += s["time"]
    lines.append("")
    lines.append("Per-model total inference time:")
    for model, t in model_times.items():
        lines.append(f"  {model:<16}  {_fmt_duration(t)}")

    lines.append("")
    lines.append("FIXED-THRESHOLD METRICS")
    hdr = (
        f"  {'model':<14} {'formula':<6} {'hypothesis':<22} {'thresh':>7}  "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
        f"{'mean_score':>10}  {'time':>8}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for s in run_stats:
        first = True
        for thresh, m in s["metrics_by_threshold"].items():
            time_str = _fmt_duration(s["time"]) if first else ""
            lines.append(
                f"  {s['model']:<14} {s['formula']:<6} {s['hypothesis']:<22} {thresh:>7.2f}  "
                f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
                f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
                f"{m['score_mean']:>10.4f}  {time_str:>8}"
            )
            first = False
        lines.append("  " + "-" * (len(hdr) - 2))

    lines.append("")
    lines.append(f"K-FOLD CV THRESHOLD OPTIMISATION  ({CV_N_SPLITS} folds, stratified, random_state=42)")
    lines.append("  NOTE: threshold found on train folds, evaluated on held-out fold — honest OOF F1.")
    lines.append("  AUC/roc_t/roc_f1 are computed on the full dataset (not held-out); cv_auc is mean fold AUC.")
    hdr_cv = (
        f"  {'model':<14} {'formula':<6} {'hypothesis':<22}  "
        f"{'cv_f1':>8} {'±std':>7}  {'mean_t':>7}  "
        f"{'auc':>7} {'cv_auc':>7} {'roc_t':>7} {'roc_f1':>7}  "
        f"fold_thresholds → fold_F1s → fold_AUCs"
    )
    lines.append(hdr_cv)
    lines.append("  " + "-" * 120)
    for fkey, _, _, _ in SCORING_FORMULAS:
        formula_stats = [s for s in run_stats if s["formula"] == fkey]
        if not formula_stats:
            continue
        lines.append(f"  --- {fkey} ---")
        for s in sorted(formula_stats, key=lambda x: -x["cv_mean_f1"]):
            mean_t     = sum(s["cv_thresholds"]) / len(s["cv_thresholds"])
            thresh_str = "  ".join(f"{t:.2f}" for t in s["cv_thresholds"])
            f1_str     = "  ".join(f"{f:.3f}" for f in s["cv_fold_f1s"])
            auc_str    = "  ".join(f"{a:.3f}" for a in s.get("cv_fold_aucs", []))
            lines.append(
                f"  {s['model']:<14} {s['formula']:<6} {s['hypothesis']:<22}  "
                f"{s['cv_mean_f1']:>8.4f} {s['cv_std_f1']:>7.4f}  {mean_t:>7.2f}  "
                f"{s.get('auc', 0):>7.4f} {s.get('cv_mean_auc', 0):>7.4f} "
                f"{s.get('roc_opt_thresh', 0):>7.3f} {s.get('roc_opt_f1', 0):>7.3f}  "
                f"[{thresh_str}] → [{f1_str}] → [{auc_str}]"
            )

    # AUC comparison across scoring formulas (suggestion 3):
    # pe/emc/enn are monotonic transforms → AUC should be equal; any divergence is noteworthy
    lines.append("")
    lines.append("AUC COMPARISON BY FORMULA  (full-dataset AUC — monotonic transforms should be equal)")
    hdr_auc = (
        f"  {'model':<14} {'hypothesis':<22}  "
        + "  ".join(f"{'auc_' + fkey:>10}" for fkey, *_ in SCORING_FORMULAS)
        + "  " + "  ".join(f"{'roc_t_' + fkey:>10}" for fkey, *_ in SCORING_FORMULAS)
    )
    lines.append(hdr_auc)
    lines.append("  " + "-" * (len(hdr_auc) - 2))
    models_hyps = list(dict.fromkeys(
        (s["model"], s["hypothesis"]) for s in run_stats
    ))
    for model, hyp in models_hyps:
        auc_vals   = []
        roc_t_vals = []
        for fkey, *_ in SCORING_FORMULAS:
            match = next((s for s in run_stats
                          if s["model"] == model and s["hypothesis"] == hyp
                          and s["formula"] == fkey), None)
            auc_vals.append(f"{match['auc']:>10.4f}" if match else f"{'N/A':>10}")
            roc_t_vals.append(f"{match['roc_opt_thresh']:>10.3f}" if match else f"{'N/A':>10}")
        lines.append(
            f"  {model:<14} {hyp:<22}  "
            + "  ".join(auc_vals) + "  " + "  ".join(roc_t_vals)
        )

    if ensemble_stats:
        lines.append("")
        lines.append("ENSEMBLE COLUMNS  (averaged confidence scores)")
        hdr2 = (
            f"  {'column':<36} {'thresh':>6}  "
            f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
            f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
            f"{'mean_conf':>9}  {'voters':>6}"
        )
        lines.append(hdr2)
        lines.append("  " + "-" * (len(hdr2) - 2))
        for s in ensemble_stats:
            first = True
            for thresh, m in s["metrics_by_threshold"].items():
                voters_str = str(s["n_voters"]) if first else ""
                lines.append(
                    f"  {s['col']:<36} {thresh:>6.1f}  "
                    f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
                    f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
                    f"{m['score_mean']:>9.4f}  {voters_str:>6}"
                )
                first = False
            lines.append("  " + "-" * (len(hdr2) - 2))

    lines.append("=" * W)
    text = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    log.info("=" * W)
    for line in lines:
        log.info(line)
    log.info("=" * W)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def _get_metrics_at(s: dict, thresh: float) -> dict:
    if thresh in s["metrics_by_threshold"]:
        return s["metrics_by_threshold"][thresh]
    closest = min(s["metrics_by_threshold"], key=lambda t: abs(t - thresh))
    return s["metrics_by_threshold"][closest]


def write_error_analysis(
    rows: list[dict],
    labels: list[str],
    run_stats: list[dict],
    analysis_path: str,
    log: logging.Logger,
) -> None:
    if not run_stats or not rows:
        log.warning("[error_analysis] No data to analyze.")
        return

    thresh      = ANALYSIS_THRESHOLD
    n_rows      = len(rows)
    hyp_cols    = list(dict.fromkeys(s["hypothesis"] for s in run_stats))
    premise_col = run_stats[0]["premise"] if run_stats else "text"

    # Error analysis uses pe formula only (primary baseline)
    pe_stats      = [s for s in run_stats if s["formula"] == "pe"]
    all_combos    = [(s["model"], s["hypothesis"]) for s in pe_stats]
    combo_scores: dict[tuple, list[float]] = {}
    combo_preds:  dict[tuple, list[bool]]  = {}
    for s in pe_stats:
        key            = (s["model"], s["hypothesis"])
        conf_col       = f"conf_pe_{s['model']}_{s['hypothesis']}"
        sc             = [float(r[conf_col]) for r in rows]
        combo_scores[key] = sc
        combo_preds[key]  = [v >= thresh for v in sc]

    gold_bool = [lbl == "1" for lbl in labels]
    n_pos     = sum(gold_bool)

    lines: list[str] = []

    def sep(title: str = "") -> None:
        lines.append("")
        lines.append("=" * 100)
        if title:
            lines.append(f"  {title}")
            lines.append("=" * 100)

    sep()
    lines.append("  ENCODER NLI ERROR ANALYSIS")
    lines.append(f"  Analysis threshold : {thresh}")
    lines.append(f"  Total rows         : {n_rows}")
    lines.append(f"  Positive / Negative: {n_pos} / {n_rows - n_pos}  "
                 f"({100 * n_pos / n_rows:.1f}% / {100 * (n_rows - n_pos) / n_rows:.1f}%)")
    lines.append("=" * 100)

    sep("SECTION 1 — BEST CONFIGURATION PER METRIC")
    for metric in ("f1", "accuracy", "precision", "recall"):
        best_val, best_key = -1.0, None
        for s in run_stats:
            for t, m in s["metrics_by_threshold"].items():
                if m[metric] > best_val:
                    best_val = m[metric]
                    best_key = (s["model"], s["hypothesis"], t)
        if best_key:
            lines.append(
                f"  {metric:<10}  {best_val:.4f}  →  "
                f"model={best_key[0]}  hyp={best_key[1]}  thresh={best_key[2]}"
            )

    sep(f"SECTION 2 — CV BEST THRESHOLDS  ({CV_N_SPLITS}-fold, metrics at mean CV threshold)")
    hdr2 = (
        f"  {'model':<14} {'hypothesis':<22}  "
        f"{'cv_f1':>8} {'±std':>7}  {'mean_t':>6}  "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1@t':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}"
    )
    lines.append(hdr2)
    lines.append("  " + "-" * (len(hdr2) - 2))
    for s in sorted(run_stats, key=lambda x: -x["cv_mean_f1"]):
        mean_t = sum(s["cv_thresholds"]) / len(s["cv_thresholds"])
        key    = (s["model"], s["hypothesis"])
        m      = compute_metrics(combo_scores[key], labels, mean_t)
        lines.append(
            f"  {s['model']:<14} {s['hypothesis']:<22}  "
            f"{s['cv_mean_f1']:>8.4f} {s['cv_std_f1']:>7.4f}  {mean_t:>6.2f}  "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
            f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}"
        )

    sep(f"SECTION 3 — PER-HYPOTHESIS RANKING  (thresh={thresh})")
    col_hdr = (
        f"  {'rank':<5} {'model':<14} "
        f"{'f1':>6} {'acc':>6} {'prec':>6} {'rec':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}"
    )
    for hyp_col in hyp_cols:
        lines.append(f"\n  Hypothesis: {hyp_col!r}")
        lines.append(col_hdr)
        lines.append("  " + "-" * (len(col_hdr) - 2))
        ranked = sorted(
            [s for s in run_stats if s["hypothesis"] == hyp_col],
            key=lambda s: _get_metrics_at(s, thresh)["f1"], reverse=True
        )
        for rank, s in enumerate(ranked, 1):
            m = _get_metrics_at(s, thresh)
            lines.append(
                f"  {rank:<5} {s['model']:<14} "
                f"{m['f1']:>6.3f} {m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f}  "
                f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}"
            )

    sep(f"SECTION 4 — MODEL AGREEMENT MATRIX  (thresh={thresh}, % same binary prediction)")
    for hyp_col in hyp_cols:
        avail_tags = [s["model"] for s in run_stats if s["hypothesis"] == hyp_col]
        lines.append(f"\n  Hypothesis: {hyp_col!r}")
        lines.append("  " + " " * 14 + "".join(f"{t:>14}" for t in avail_tags))
        for t1 in avail_tags:
            p1      = combo_preds[(t1, hyp_col)]
            row_str = f"  {t1:<14}"
            for t2 in avail_tags:
                p2    = combo_preds[(t2, hyp_col)]
                agree = sum(a == b for a, b in zip(p1, p2)) / n_rows
                row_str += f"  {agree:>10.3f}  "
            lines.append(row_str)

    sep(f"SECTION 5 — HARD EXAMPLES: ALL COMBOS WRONG  (thresh={thresh})")
    row_n_correct = [
        sum(1 for key in all_combos if combo_preds[key][i] == gold_bool[i])
        for i in range(n_rows)
    ]
    hard_idx = [i for i, nc in enumerate(row_n_correct) if nc == 0]
    lines.append(
        f"\n  {len(hard_idx)} rows ({len(hard_idx) / n_rows * 100:.1f}%) "
        "where every model+hypothesis prediction was wrong.  Showing up to 20."
    )
    for i in hard_idx[:20]:
        lines.append("")
        lines.append(f"  [row {i}]  gold={labels[i]}")
        lines.append(f"    premise  : {rows[i].get(premise_col, '')[:120]!r}")
        for hyp_col in hyp_cols:
            lines.append(f"    {hyp_col:<24}: {rows[i].get(hyp_col, '')!r}")
        for key in all_combos:
            tag, hyp_col = key
            lines.append(
                f"    [{tag}/{hyp_col}]  "
                f"conf={combo_scores[key][i]:.4f}  pred={'1' if combo_preds[key][i] else '0'}"
            )

    sep("SECTION 6 — CONFIDENT MISTAKES")
    CONF_FP_MIN = 0.85
    CONF_FN_MAX = 0.15
    fp_rows, fn_rows = [], []
    for i in range(n_rows):
        confs = {key: combo_scores[key][i] for key in all_combos}
        if not gold_bool[i]:
            worst_key = max(confs, key=confs.get)
            if confs[worst_key] >= CONF_FP_MIN:
                fp_rows.append((i, worst_key, confs[worst_key], confs))
        else:
            worst_key = min(confs, key=confs.get)
            if confs[worst_key] <= CONF_FN_MAX:
                fn_rows.append((i, worst_key, confs[worst_key], confs))
    fp_rows.sort(key=lambda x: -x[2])
    fn_rows.sort(key=lambda x:  x[2])

    lines.append(
        f"\n  False Positives (gold=0, max_conf >= {CONF_FP_MIN}): "
        f"{len(fp_rows)} — showing up to 15"
    )
    for i, worst_key, worst_conf, confs in fp_rows[:15]:
        lines.append("")
        lines.append(
            f"  [row {i}]  gold=0  "
            f"most-confident={worst_key[0]}/{worst_key[1]}  conf={worst_conf:.4f}"
        )
        lines.append(f"    premise  : {rows[i].get(premise_col, '')[:120]!r}")
        for hyp_col in hyp_cols:
            lines.append(f"    {hyp_col:<24}: {rows[i].get(hyp_col, '')!r}")
        lines.append("    scores   : " + "  ".join(
            f"{k[0]}/{k[1]}={v:.3f}" for k, v in confs.items()
        ))

    lines.append(
        f"\n  False Negatives (gold=1, min_conf <= {CONF_FN_MAX}): "
        f"{len(fn_rows)} — showing up to 15"
    )
    for i, worst_key, worst_conf, confs in fn_rows[:15]:
        lines.append("")
        lines.append(
            f"  [row {i}]  gold=1  "
            f"least-confident={worst_key[0]}/{worst_key[1]}  conf={worst_conf:.4f}"
        )
        lines.append(f"    premise  : {rows[i].get(premise_col, '')[:120]!r}")
        for hyp_col in hyp_cols:
            lines.append(f"    {hyp_col:<24}: {rows[i].get(hyp_col, '')!r}")
        lines.append("    scores   : " + "  ".join(
            f"{k[0]}/{k[1]}={v:.3f}" for k, v in confs.items()
        ))

    sep(f"SECTION 7 — MEAN CONFIDENCE BY OUTCOME CLASS  (thresh={thresh})")
    hdr7 = (
        f"  {'model':<14} {'hypothesis':<22}  "
        f"{'outcome':<6}  {'n':>5}  {'mean':>7}  {'median':>7}  {'min':>7}  {'max':>7}"
    )
    lines.append(hdr7)
    lines.append("  " + "-" * (len(hdr7) - 2))
    for s in run_stats:
        key = (s["model"], s["hypothesis"])
        outcome_confs: dict[str, list[float]] = {"TP": [], "FP": [], "FN": [], "TN": []}
        for i in range(n_rows):
            sc   = combo_scores[key][i]
            pred = combo_preds[key][i]
            gold = gold_bool[i]
            if   pred and gold:     outcome_confs["TP"].append(sc)
            elif pred and not gold: outcome_confs["FP"].append(sc)
            elif not pred and gold: outcome_confs["FN"].append(sc)
            else:                   outcome_confs["TN"].append(sc)
        first = True
        for outcome, vals in outcome_confs.items():
            if not vals:
                continue
            sv = sorted(vals)
            n  = len(sv)
            lines.append(
                f"  {s['model'] if first else '':<14} "
                f"{s['hypothesis'] if first else '':<22}  "
                f"{outcome:<6}  {n:>5}  "
                f"{sum(vals)/n:>7.4f}  {sv[n//2]:>7.4f}  {sv[0]:>7.4f}  {sv[-1]:>7.4f}"
            )
            first = False
        lines.append("  " + "-" * (len(hdr7) - 2))

    sep(f"SECTION 8 — TEXT LENGTH VS ACCURACY  (thresh={thresh})")
    length_bins = [(0, 100, "<100"), (100, 200, "100-200"), (200, 400, "200-400"), (400, 10**9, "400+")]
    lengths     = [len(rows[i].get(premise_col, "")) for i in range(n_rows)]
    bin_counts  = [sum(1 for l in lengths if lo <= l < hi) for lo, hi, _ in length_bins]
    lines.append("\n  Bins: " + "  ".join(
        f"{lb}(n={cnt})" for (_, _, lb), cnt in zip(length_bins, bin_counts)
    ))
    lines.append("")
    hdr8 = f"  {'model':<14} {'hypothesis':<22}  " + "  ".join(f"{lb:>10}" for _, _, lb in length_bins)
    lines.append(hdr8)
    lines.append("  " + "-" * (len(hdr8) - 2))
    for s in run_stats:
        key   = (s["model"], s["hypothesis"])
        cells = []
        for lo, hi, _ in length_bins:
            idx = [i for i, l in enumerate(lengths) if lo <= l < hi]
            if not idx:
                cells.append("       N/A")
                continue
            m = compute_metrics([combo_scores[key][i] for i in idx],
                                [labels[i] for i in idx], thresh)
            cells.append(f"{m['accuracy']:>10.3f}")
        lines.append(f"  {s['model']:<14} {s['hypothesis']:<22}  " + "  ".join(cells))

    sep()
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"[error_analysis]  report → {analysis_path}")

    tsv_path = os.path.splitext(analysis_path)[0] + "_predictions.tsv"
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        header = ["row_idx", "gold_label", "premise_preview"]
        for hyp_col in hyp_cols:
            header.append(f"hyp_{hyp_col}")
        for tag, hyp_col in all_combos:
            header += [f"conf_{tag}_{hyp_col}", f"pred_{tag}_{hyp_col}"]
        writer.writerow(header)
        for i, row in enumerate(rows):
            record = [i, labels[i], row.get(premise_col, "")[:120]]
            for hyp_col in hyp_cols:
                record.append(row.get(hyp_col, ""))
            for key in all_combos:
                record += [f"{combo_scores[key][i]:.4f}", "1" if combo_preds[key][i] else "0"]
            writer.writerow(record)
    log.info(f"[error_analysis]  predictions TSV → {tsv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NLI encoder classification using Hebrew_NLI fine-tuned models"
    )
    parser.add_argument("--input",          default=INPUT_FILE)
    parser.add_argument("--output",         default=None)
    parser.add_argument("--log",            default=None)
    parser.add_argument("--summary",        default=None)
    parser.add_argument("--error-analysis", default=None)
    parser.add_argument("--label-col",      default=LABEL_COL)
    args = parser.parse_args()

    base   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path   = os.path.join(base, args.input)
    output_path  = os.path.join(base, args.output  or "outputs/encoder_NLI/classified.csv")
    log_path     = os.path.join(base, args.log     or "outputs/encoder_NLI/classify.log")
    summary_path = os.path.join(base, args.summary or "outputs/encoder_NLI/summary.txt")
    error_path   = os.path.join(base, args.error_analysis or "outputs/encoder_NLI/error_analysis.txt")
    for p in (output_path, log_path, summary_path, error_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    log = setup_logger(log_path)
    log.info(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    log.info(f"torch {torch.__version__}  CUDA={torch.version.cuda}  "
             f"available={torch.cuda.is_available()}")

    wall_start = time.time()
    log.info("=" * 70)
    log.info("clean_encoder_NLI.py  started")
    log.info(f"  input      : {input_path}")
    log.info(f"  output     : {output_path}")
    log.info(f"  log        : {log_path}")
    log.info(f"  summary    : {summary_path}")
    log.info(f"  error      : {error_path}")
    log.info(f"  label col  : {args.label_col}")
    log.info(f"  max_length : 512 (model default)")
    log.info(f"  thresholds : {EVAL_THRESHOLDS}  (fixed)")
    log.info(f"  CV folds   : {CV_N_SPLITS}  |  thresh grid: "
             f"{THRESHOLD_GRID[0]:.2f}–{THRESHOLD_GRID[-1]:.2f} ({len(THRESHOLD_GRID)} values)")
    log.info(f"  models ({len(NLI_MODELS)}):")
    for ckpt_abs, tag, mtype in NLI_MODELS:
        log.info(f"    [{tag}]  {ckpt_abs}  ({mtype})")
    log.info(f"  pairs ({len(PREMISE_HYPOTHESIS_PAIRS)}):")
    for p, h in PREMISE_HYPOTHESIS_PAIRS:
        log.info(f"    {p!r} -> {h!r}")
    log.info("=" * 70)

    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    n_rows = len(rows)
    log.info(f"[load]  {n_rows} rows")

    available_cols = set(rows[0].keys())
    if args.label_col not in available_cols:
        raise ValueError(f"Label column '{args.label_col}' not found in {sorted(available_cols)}")

    # Filter pairs to only those whose columns are present in the input
    active_pairs = []
    for premise_col, hyp_col in PREMISE_HYPOTHESIS_PAIRS:
        if premise_col not in available_cols or hyp_col not in available_cols:
            log.warning(f"  skipping pair ({premise_col!r}, {hyp_col!r}) — column(s) missing in input")
        else:
            active_pairs.append((premise_col, hyp_col))
    if not active_pairs:
        raise ValueError("No valid premise-hypothesis pairs found in the input CSV.")
    log.info(f"  active pairs: {active_pairs}")

    n_pos = sum(1 for r in rows if r[args.label_col] == "1")
    log.info(f"[load]  pos={n_pos}  neg={n_rows - n_pos}  "
             f"({100 * n_pos / n_rows:.1f}% / {100 * (n_rows - n_pos) / n_rows:.1f}%)")
    labels = [r[args.label_col] for r in rows]

    # Resume: merge already-computed confidence columns from a previous run
    if os.path.exists(output_path):
        log.info(f"[resume]  checkpoint found: {output_path}")
        with open(output_path, encoding="utf-8-sig") as f:
            ckpt_rows = list(csv.DictReader(f))
        if len(ckpt_rows) == n_rows:
            ckpt_extra = set(ckpt_rows[0].keys()) - set(rows[0].keys())
            if ckpt_extra:
                for r, cr in zip(rows, ckpt_rows):
                    for col in ckpt_extra:
                        r[col] = cr[col]
                log.info(f"[resume]  merged {len(ckpt_extra)} columns — completed combos will be skipped")
            else:
                log.info("[resume]  checkpoint has no extra columns — starting fresh")
        else:
            log.warning(f"[resume]  row-count mismatch ({len(ckpt_rows)} vs {n_rows}) — ignoring checkpoint")

    run_stats: list[dict] = []

    for model_idx, (ckpt_abs, tag, mtype) in enumerate(NLI_MODELS, 1):
        log.info("-" * 70)
        log.info(f"[model {model_idx}/{len(NLI_MODELS)}]  {tag}  ({mtype})")

        # Skip model load if every pair for this model is already in the checkpoint
        _model_conf_cols = [f"conf_pe_{tag}_{hyp_col}" for _, hyp_col in active_pairs]
        if all(rows[0].get(c, "") != "" for c in _model_conf_cols):
            log.info(f"  all pairs in checkpoint — skipping model load")
            for premise_col, hyp_col in active_pairs:
                pe_list = [float(r[f"conf_pe_{tag}_{hyp_col}"]) for r in rows]
                pc_list = [float(r[f"conf_pc_{tag}_{hyp_col}"]) for r in rows]
                for fkey, _flabel, cv_grid, eval_thresh in SCORING_FORMULAS:
                    scores = derive_scores(pe_list, pc_list, fkey)
                    mbt    = {t: compute_metrics(scores, labels, t) for t in eval_thresh}
                    auc, roc_opt_t, roc_opt_f1 = compute_roc_stats(scores, labels)
                    cv_mean, cv_std, cv_thresholds, cv_fold_f1s, cv_mean_auc, cv_fold_aucs = \
                        kfold_threshold_cv(scores, labels, CV_N_SPLITS, cv_grid)
                    run_stats.append({
                        "model": tag, "premise": premise_col, "hypothesis": hyp_col,
                        "formula": fkey, "time": 0.0, "metrics_by_threshold": mbt,
                        "auc": auc, "roc_opt_thresh": roc_opt_t, "roc_opt_f1": roc_opt_f1,
                        "cv_mean_f1": cv_mean, "cv_std_f1": cv_std,
                        "cv_thresholds": cv_thresholds, "cv_fold_f1s": cv_fold_f1s,
                        "cv_mean_auc": cv_mean_auc, "cv_fold_aucs": cv_fold_aucs,
                    })
                    log.info(f"  [skip]  {tag}/{hyp_col}/{fkey}  "
                             f"AUC={auc:.4f}  roc_t={roc_opt_t:.3f}  roc_f1={roc_opt_f1:.3f}")
            continue

        t_model = time.time()
        if mtype == "seq2seq":
            model, tokenizer, device = load_seq2seq_model(ckpt_abs, log)
        else:
            model, tokenizer, device = load_model(ckpt_abs, log)
        log.info(f"    model loaded in {_fmt_duration(time.time() - t_model)}")

        for pair_idx, (premise_col, hyp_col) in enumerate(active_pairs, 1):
            pe_col = f"conf_pe_{tag}_{hyp_col}"
            pc_col = f"conf_pc_{tag}_{hyp_col}"
            desc   = f"{tag}/{hyp_col}"

            # Skip pair if already computed in checkpoint
            if rows[0].get(pe_col, "") != "":
                log.info(f"  [{pair_idx}/{len(active_pairs)}]  [skip]  {desc}  (already in checkpoint)")
                pe_list = [float(r[pe_col]) for r in rows]
                pc_list = [float(r[pc_col]) for r in rows]
                elapsed = 0.0
            else:
                log.info(f"  [{pair_idx}/{len(active_pairs)}]  "
                         f"premise={premise_col!r}  hypothesis={hyp_col!r}")

                texts = [(r[premise_col], r[hyp_col]) for r in rows]
                t0    = time.time()
                if mtype == "seq2seq":
                    raw = run_seq2seq_nli(model, tokenizer, device, texts, log, desc=desc)
                else:
                    raw = run_nli(model, tokenizer, device, texts, log, desc=desc)
                elapsed = time.time() - t0
                pe_list = raw["pe"]
                pc_list = raw["pc"]

                for r, pe_v, pc_v in zip(rows, pe_list, pc_list):
                    r[pe_col] = f"{pe_v:.4f}"
                    r[pc_col] = f"{pc_v:.4f}"

                log.info(f"    done  ({_fmt_duration(elapsed)}, {n_rows / elapsed:.1f} rows/s)")
                log.info(f"    P(e) mean={sum(pe_list)/len(pe_list):.4f}  "
                         f"P(c) mean={sum(pc_list)/len(pc_list):.4f}")

            for fkey, flabel, cv_grid, eval_thresh in SCORING_FORMULAS:
                scores = derive_scores(pe_list, pc_list, fkey)
                log.info(f"    [{fkey}]  score mean={sum(scores)/len(scores):.4f}")
                mbt: dict[float, dict] = {}
                for thresh in eval_thresh:
                    m = compute_metrics(scores, labels, thresh)
                    mbt[thresh] = m
                    log.info(
                        f"    [{fkey}] thresh={thresh:.2f}  "
                        f"TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}  "
                        f"F1={m['f1']:.3f}"
                    )

                auc, roc_opt_t, roc_opt_f1 = compute_roc_stats(scores, labels)
                log.info(f"    [{fkey}]  AUC={auc:.4f}  roc_opt_t={roc_opt_t:.3f}  "
                         f"roc_opt_f1={roc_opt_f1:.3f}")

                log.info(f"    [{fkey}]  [cv{CV_N_SPLITS}]  running threshold search …")
                cv_mean, cv_std, cv_thresholds, cv_fold_f1s, cv_mean_auc, cv_fold_aucs = \
                    kfold_threshold_cv(scores, labels, CV_N_SPLITS, cv_grid)
                mean_t = sum(cv_thresholds) / len(cv_thresholds)
                log.info(
                    f"    [{fkey}]  [cv{CV_N_SPLITS}]  mean_F1={cv_mean:.4f} ±{cv_std:.4f}  "
                    f"mean_thresh={mean_t:.2f}  mean_AUC={cv_mean_auc:.4f}"
                )

                run_stats.append({
                    "model":                tag,
                    "premise":              premise_col,
                    "hypothesis":           hyp_col,
                    "formula":              fkey,
                    "time":                 elapsed,
                    "metrics_by_threshold": mbt,
                    "auc":                  auc,
                    "roc_opt_thresh":       roc_opt_t,
                    "roc_opt_f1":           roc_opt_f1,
                    "cv_mean_f1":           cv_mean,
                    "cv_std_f1":            cv_std,
                    "cv_thresholds":        cv_thresholds,
                    "cv_fold_f1s":          cv_fold_f1s,
                    "cv_mean_auc":          cv_mean_auc,
                    "cv_fold_aucs":         cv_fold_aucs,
                })
                elapsed = 0.0  # only charge time to first formula

        unload_model(model, log)
        log.info(f"  [{tag}] total: {_fmt_duration(time.time() - t_model)}")

        # Checkpoint save after each model so a restart can resume from here
        fieldnames = list(dict.fromkeys(rows[0].keys()))
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"  [checkpoint]  {n_rows} rows → {output_path}")

    fieldnames = list(dict.fromkeys(rows[0].keys()))
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"[save]  {n_rows} rows → {output_path}")

    total = time.time() - wall_start
    write_summary(run_stats, summary_path, log, total)
    write_error_analysis(rows, labels, run_stats, error_path, log)

    log.info("=" * 70)
    log.info(f"Done.  Total wall time: {_fmt_duration(total)}")
    log.info(f"  {output_path}")
    log.info(f"  {summary_path}")
    log.info(f"  {error_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
