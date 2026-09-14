"""
Classify prepared_silver.csv (~2.5 M rows) using fine-tuned NLI encoder models.

Models (both AutoModelForSequenceClassification, 3-class NLI):
  xlmroberta   — FacebookAI/xlm-roberta-large   (checkpoint-8000)
  neodictabert — dicta-il/neodictabert          (checkpoint-5600)

Hypothesis used: template_relation (generated on-the-fly from PREDICATE_TEMPLATES).
Score:           P(entailment)  — index 0 from the 3-class softmax.
Thresholds:      xlmroberta=0.06   neodictabert=0.05
  (from 5-fold CV on gold-500; pe formula, template_relation hypothesis)

Pipeline:
  1. Run xlmroberta → writes pred_xlmroberta.txt (one float per line, resume-safe).
  2. Run neodictabert → writes pred_neodictabert.txt.
  3. Merge: stream silver CSV + both pred files → silver_encoder_nli.csv.
     Output columns: all original silver cols + score_* + pred_* for each model.
     This CSV is designed to receive additional model columns later
     (opensource LLM, finetuned LLM, cross-train RC, …).

Usage:
  conda activate hre_finetuned_nli
  CUDA_VISIBLE_DEVICES=0 python -m scripts_silver_cleaning.classify_silver_encoder_nli
  CUDA_VISIBLE_DEVICES=0 python -m scripts_silver_cleaning.classify_silver_encoder_nli --models xlmroberta
  CUDA_VISIBLE_DEVICES=0 python -m scripts_silver_cleaning.classify_silver_encoder_nli --max-rows 10000
  CUDA_VISIBLE_DEVICES=0 python -m scripts_silver_cleaning.classify_silver_encoder_nli --skip-merge
"""

import csv
import logging
import os
import time
import argparse
from pathlib import Path
from typing import Iterator

import torch
import transformers
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).resolve().parent.parent.parent
SILVER_CSV = ROOT / "data" / "prepared_silver.csv"
OUT_DIR    = ROOT / "outputs" / "silver_cleaning" / "silver_encoder_nli"
CLASS_DIR  = OUT_DIR / "classifications"
LOG_DIR    = OUT_DIR / "logs"
LOG_FILE   = OUT_DIR / "run.log"
SUMMARY    = OUT_DIR / "summary.txt"
OUT_CSV    = CLASS_DIR / "silver_encoder_nli.csv"

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

_ENC_BASE = "/path/to/Hebrew_NLI/finetune_heb_nli/outputs"

MODELS = {
    "xlmroberta": {
        "ckpt":      f"{_ENC_BASE}/FacebookAI__xlm-roberta-large/checkpoint-8000",
        "threshold": 0.06,
        "col_score": "score_xlmroberta",
        "col_pred":  "pred_xlmroberta",
        "dtype":     torch.bfloat16,
    },
    "neodictabert": {
        "ckpt":      f"{_ENC_BASE}/dicta-il__neodictabert/checkpoint-5600",
        "threshold": 0.05,
        "col_score": "score_neodictabert",
        "col_pred":  "pred_neodictabert",
        "dtype":     torch.float32,   # BF16 produces NaN for this architecture
    },
}

MODEL_ORDER = ["xlmroberta", "neodictabert"]

# Input silver columns (used as output base)
SILVER_COLS = ["docid", "title", "text", "subject", "predicate", "object"]

# ---------------------------------------------------------------------------
# Processing constants
# ---------------------------------------------------------------------------

CHUNK_SIZE          = 50_000
BATCH_SIZE          = 64
LARGE_BATCH_SIZE    = 64
LARGE_TEXT_THRESH   = 256    # chars; use smaller batch above this
LOG_EVERY_N_CHUNKS  = 1      # log every chunk = every 50K rows

# ---------------------------------------------------------------------------
# Predicate → template relation
# ---------------------------------------------------------------------------

PREDICATE_TEMPLATES = {
    "אב":                                  "האב של {subject} הוא {object}",
    "אזרחות":                              "האזרחות של {subject} היא {object}",
    "אחים ואחיות":                         "יש יחסי אחים בין {subject} ל-{object}",
    "אל של":                               "{subject} הוא האל של {object}",
    "אמן מבצע":                            "האמן המבצע של {subject} הוא {object}",
    "ארץ מקור":                            "ארץ המקור של {subject} היא {object}",
    "בירה של":                             "{subject} היא הבירה של {object}",
    "בסיס פעולה מרכזי":                    "בסיס הפעולה המרכזי של {subject} הוא {object}",
    "בעלים":                               "הבעלים של {subject} הוא {object}",
    "גובל עם":                             "{subject} גובל ב-{object}",
    "גוף תקינה":                           "גוף התקינה של {subject} הוא {object}",
    "דת":                                  "הדת של {subject} היא {object}",
    "ההפך מ־":                             "ההפך מ-{subject} הוא {object}",
    "המקום שמרכז התחבורה משרת":            "מרכז התחבורה {subject} משרת את {object}",
    "הנקודה הגבוהה ביותר":                 "הנקודה הגבוהה ביותר של {subject} היא {object}",
    "הקודם":                               "הקודם ל-{subject} הוא {object}",
    "השפה של היצירה או של השם":            "השפה של {subject} היא {object}",
    "השתתף ב־":                            "הייתה השתתפות של {subject} ב-{object}",
    "זוכה":                                "הזוכה ב-{subject} הוא {object}",
    "זרם אמנותי":                          "הזרם האמנותי של {subject} הוא {object}",
    "חבר בקבוצת ספורט":                    "{subject} משתייך לקבוצת הספורט {object}",
    "חברת תקליטים":                        "חברת התקליטים של {subject} היא {object}",
    "חלוקה משנית":                         "{object} הוא חלוקה משנית של {subject}",
    "חלק מהסדרה":                          "{subject} הוא חלק מהסדרה {object}",
    "חלק מתוך":                            "{subject} הוא חלק מתוך {object}",
    "יבשת":                                "המיקום של {subject} הוא ביבשת {object}",
    "יחידה מנהלית":                        "{subject} היא יחידה מנהלית של {object}",
    "יחסים דיפלומטיים":                    "יש יחסים דיפלומטיים בין {subject} ל-{object}",
    "יצרן":                                "היצרן של {subject} הוא {object}",
    "ליגה":                                "הליגה של {subject} היא {object}",
    "ליגה נמוכה יותר":                     "הליגה הנמוכה יותר של {subject} היא {object}",
    "מארגן":                               "המארגן של {subject} הוא {object}",
    "מדינה":                               "המיקום של {subject} הוא במדינת {object}",
    "מדינה בתחום של ספורט":                "הייצוג של {object} בספורט נעשה על ידי {subject}",
    "מדינות אגן הניקוז":                   "הזרימה של {subject} עוברת דרך {object}",
    "מועמד שנבחר":                         "המועמד שנבחר ב-{subject} הוא {object}",
    "מוענק על ידי":                        "{subject} מוענק על ידי {object}",
    "מופע של":                             "{subject} הוא סוג של {object}",
    "מוצג":                                "{subject} מציג את {object}",
    "מוצר":                                "{object} הוא מוצר של {subject}",
    "מוקד פעילות":                         "מוקד הפעילות של {subject} הוא {object}",
    "מותג":                                "{subject} שייך למותג {object}",
    "מחבר":                                "המחבר של {subject} הוא {object}",
    "מחבר המילים":                         "מחבר המילים של {subject} הוא {object}",
    "מטבח":                                "{subject} הוא מאכל ממטבח {object}",
    "מייסד":                               "המייסד של {subject} הוא {object}",
    "מיקום":                               "המיקום של {subject} הוא ב-{object}",
    "מיקום מטה הארגון":                    "מטה הארגון של {subject} ממוקם ב-{object}",
    "מכיל את החלק":                        "{subject} מכיל את {object}",
    "מכיל חלקים מסוג":                     "{subject} מכיל חלקים מסוג {object}",
    "ממוקם בגוף השמיימי":                  "{subject} ממוקם על {object}",
    "מעסיק":                               "המעסיק של {subject} הוא {object}",
    "מערכת תחבורה":                        "{subject} היא חלק ממערכת התחבורה {object}",
    "מפלגה":                               "{subject} משתייך למפלגת {object}",
    "מפעיל":                               "המפעיל של {subject} הוא {object}",
    "מקום לידה":                           "מקום הלידה של {subject} הוא {object}",
    "מקום לימודים":                        "מוסד הלימודים של {subject} הוא {object}",
    "מקום מוצא":                           "מקום המוצא של {subject} הוא {object}",
    "מקום פטירה":                          "מקום הפטירה של {subject} הוא {object}",
    "משמש לטיפול ב־":                      "{subject} משמש לטיפול ב-{object}",
    "נהרות יוצאים מהאגם":                  "{object} יוצא מתוך {subject}",
    "נושא היצירה":                         "הנושא של {subject} הוא {object}",
    "נושא המשרה":                          "נושא המשרה של {subject} הוא {object}",
    "נמצא בשימוש של":                      "{subject} נמצא בשימוש של {object}",
    "נמצא על שפת גוף מים":                 "{subject} נמצא על שפת {object}",
    "נקרא על שם":                          "{subject} נקרא על שם {object}",
    "נשפך ל":                              "{subject} נשפך אל {object}",
    "סוג יצירה":                           "סוג היצירה של {subject} הוא {object}",
    "סוגה":                                "הסוגה של {subject} היא {object}",
    "סמל מייצג":                           "הסמל המייצג של {subject} הוא {object}",
    "עונת ספורט של":                       "{subject} היא עונת ספורט של {object}",
    "עיסוק":                               "העיסוק של {subject} הוא {object}",
    "עיר בירה":                            "עיר הבירה של {subject} היא {object}",
    "ענף ספורט":                           "ענף הספורט של {subject} הוא {object}",
    "ערוץ שידור מקורי":                    "ערוץ השידור המקורי של {subject} הוא {object}",
    "פורסם ב־":                            "הפרסום של {subject} היה ב-{object}",
    "צאצא":                                "{object} הוא צאצא של {subject}",
    "צבע":                                 "הצבע של {subject} הוא {object}",
    "קבוצות משתתפות":                      "ישנה השתתפות של {object} ב-{subject}",
    "קבוצת כוכבים":                        "{subject} ממוקם בקבוצת הכוכבים {object}",
    "קו רכבת":                             "{subject} הוא חלק מקו הרכבת {object}",
    "קטגוריית כוכבי לכת מינוריים":         "{subject} משויך לקטגוריית {object}",
    "קיבל השראה מ־":                       "ההשראה עבור {subject} התקבלה מ-{object}",
    "רמה טקסונומית":                       "הרמה הטקסונומית של {subject} היא {object}",
    "שחקנים":                              "תפקיד המשחק ב-{subject} מבוצע על ידי {object}",
    "שטח שיפוט":                           "שטח השיפוט של {subject} הוא {object}",
    "שיטת כתב":                            "שיטת הכתב של {subject} היא {object}",
    "שימוש":                               "השימוש של {subject} הוא עבור {object}",
    "שפה מדוברת או נכתבת":                 "השפה בשימוש על ידי {subject} היא {object}",
    "שפה רשמית":                           "השפה הרשמית של {subject} היא {object}",
    "שפה שבשימוש":                         "השפה שבשימוש ב-{subject} היא {object}",
    "שפת אם":                              "שפת האם של {subject} היא {object}",
    "שפת הכתיבה":                          "שפת הכתיבה של {subject} היא {object}",
    "תעשייה":                              "התעשייה בה פועל {subject} היא {object}",
    "תפקיד":                               "התפקיד של {subject} הוא {object}",
    "תקופה":                               "התקופה אליה משויך {subject} היא {object}",
    "תת-קבוצה של":                         "{subject} הוא תת-קבוצה של {object}",
}


def make_template_relation(subject: str, predicate: str, obj: str) -> str:
    tpl = PREDICATE_TEMPLATES.get(predicate)
    if tpl is None:
        return f"{subject} {predicate} {obj}"
    return tpl.format(subject=subject, object=obj)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _fmt_dur(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h:  return f"{h}h {m}m {sec}s"
    if m:  return f"{m}m {sec}s"
    return f"{sec}s"


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("classify_silver_encoder_nli")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Streaming utilities
# ---------------------------------------------------------------------------

def count_silver_rows(max_rows: int | None = None) -> int:
    n = 0
    with open(SILVER_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for _ in reader:
            n += 1
            if max_rows and n >= max_rows:
                break
    return n


def count_pred_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def stream_silver(skip: int = 0, limit: int | None = None) -> Iterator[dict]:
    yielded = 0
    with open(SILVER_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < skip:
                continue
            yield row
            yielded += 1
            if limit and yielded >= limit:
                break


def chunked(it: Iterator, n: int):
    buf = []
    for item in it:
        buf.append(item)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def _pred_path(tag: str, suffix: str = "") -> Path:
    return OUT_DIR / f"pred_{tag}{suffix}.txt"


# ---------------------------------------------------------------------------
# Encoder model: load / infer / unload
# ---------------------------------------------------------------------------

def load_encoder(ckpt: str, log: logging.Logger, dtype=torch.bfloat16):
    log.info(f"    checkpoint : {ckpt}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    log.info(f"    GPU: {torch.cuda.get_device_name(0)}  dtype={dtype}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    config    = transformers.AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    try:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            ckpt, config=config, trust_remote_code=True, use_safetensors=True,
            torch_dtype=dtype,
        )
    except Exception as e:
        log.warning(f"    safetensors load failed ({e}), retrying")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            ckpt, config=config, trust_remote_code=True,
            torch_dtype=dtype,
        )
    device = torch.device("cuda:0")
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"    ready on {device}  ({n_params:.0f}M params)")
    return model, tokenizer, device


def unload_encoder(model, log: logging.Logger):
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("    GPU cache cleared")


def run_encoder_chunk(
    model, tokenizer, device,
    pairs: list[tuple[str, str]],
) -> list[float]:
    """Return P(entailment) for each (premise, hypothesis) pair."""
    max_chars = max((len(p) + len(h) for p, h in pairs), default=0)
    bs = LARGE_BATCH_SIZE if max_chars > LARGE_TEXT_THRESH else BATCH_SIZE
    all_pe: list[float] = []
    for start in range(0, len(pairs), bs):
        batch = pairs[start : start + bs]
        enc = tokenizer(
            [p for p, _ in batch], [h for _, h in batch],
            return_tensors="pt", padding="longest", truncation=True,
            add_special_tokens=True, return_token_type_ids=False,
        )
        for k in enc:
            enc[k] = enc[k].to(device)
        with torch.no_grad():
            logits = model(**enc, return_dict=True).logits.softmax(dim=1)
        all_pe.extend(logits[:, 0].cpu().tolist())
    return all_pe


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    tag: str,
    total_rows: int,
    log: logging.Logger,
    start_row: int | None = None,
    end_row: int | None = None,
    pred_suffix: str = "",
) -> dict:
    cfg       = MODELS[tag]
    pred_path = _pred_path(tag, pred_suffix)
    # base_skip: first silver row to process in this shard
    base_skip = start_row if start_row is not None else 0
    # already: rows written in a previous interrupted run of this shard
    already   = count_pred_rows(pred_path)
    skip      = base_skip + already
    end       = end_row if end_row is not None else total_rows
    limit     = end - skip

    if limit <= 0:
        log.info(f"[{tag}]  already done ({already:,} rows in {pred_path.name}) — skipping")
        return {"rows": already, "time": 0.0}

    log.info(f"[{tag}]  starting from row {skip:,}  (base={base_skip:,} + resumed={already:,})  end={end:,}  remaining={limit:,}")
    model, tokenizer, device = load_encoder(cfg["ckpt"], log, dtype=cfg.get("dtype", torch.bfloat16))
    try:
        model = torch.compile(model)
        log.info("    torch.compile: OK")
    except Exception as e:
        log.warning(f"    torch.compile: failed ({e}), continuing without")
    t0      = time.time()
    written = 0

    with open(pred_path, "a", encoding="utf-8") as fout:
        for chunk_idx, chunk in enumerate(
            chunked(stream_silver(skip=skip, limit=limit), CHUNK_SIZE), 1
        ):
            pairs = [
                (r["text"], make_template_relation(r["subject"], r["predicate"], r["object"]))
                for r in chunk
            ]
            scores = run_encoder_chunk(model, tokenizer, device, pairs)
            for s in scores:
                fout.write(f"{s:.6f}\n")
            fout.flush()  # NFS st_blksize=1MB > chunk size; flush explicitly
            written += len(chunk)

            if chunk_idx % LOG_EVERY_N_CHUNKS == 0 or skip + written >= end:
                elapsed = time.time() - t0
                rps     = written / elapsed if elapsed else 0
                eta     = (limit - written) / rps if rps else 0
                log.info(
                    f"[{tag}]  {skip + written:,}/{total_rows:,}  "
                    f"({100 * (skip + written) / total_rows:.1f}%)  "
                    f"{rps:.1f} rows/s  ETA {_fmt_dur(eta)}"
                )

    elapsed = time.time() - t0
    unload_encoder(model, log)
    log.info(f"[{tag}]  done  {_fmt_dur(elapsed)}")
    return {"rows": written, "time": elapsed}


# ---------------------------------------------------------------------------
# Final merge
# ---------------------------------------------------------------------------

def merge_predictions(models_run: list[str], log: logging.Logger) -> int:
    CLASS_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"[merge]  writing {OUT_CSV}")

    output_cols = SILVER_COLS + [
        col
        for tag in MODEL_ORDER
        if tag in models_run
        for col in (MODELS[tag]["col_score"], MODELS[tag]["col_pred"])
    ]

    pred_handles = {
        tag: open(_pred_path(tag), encoding="utf-8")
        for tag in models_run
        if _pred_path(tag).exists()
    }
    written = 0
    try:
        with (
            open(SILVER_CSV, encoding="utf-8-sig", newline="") as fin,
            open(OUT_CSV, "w", encoding="utf-8", newline="") as fout,
        ):
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=output_cols, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                out_row = {col: row.get(col, "") for col in SILVER_COLS}
                for tag in models_run:
                    cfg       = MODELS[tag]
                    score_str = pred_handles[tag].readline().strip() if tag in pred_handles else ""
                    if score_str:
                        score = float(score_str)
                        out_row[cfg["col_score"]] = f"{score:.6f}"
                        out_row[cfg["col_pred"]]  = "1" if score >= cfg["threshold"] else "0"
                    else:
                        out_row[cfg["col_score"]] = ""
                        out_row[cfg["col_pred"]]  = ""
                writer.writerow(out_row)
                written += 1
                if written % 500_000 == 0:
                    log.info(f"[merge]  {written:,} rows written")
    finally:
        for fh in pred_handles.values():
            fh.close()

    log.info(f"[merge]  {written:,} rows → {OUT_CSV}")
    return written


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(
    model_stats: dict,
    total_rows: int,
    total_time: float,
    models_run: list[str],
    log: logging.Logger,
) -> None:
    sep   = "=" * 80
    lines = [
        sep,
        "SILVER ENCODER NLI CLASSIFICATION SUMMARY",
        f"Date          : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total rows    : {total_rows:,}",
        f"Wall time     : {_fmt_dur(total_time)}",
        f"Input CSV     : {SILVER_CSV}",
        f"Output CSV    : {OUT_CSV}",
        f"Hypothesis    : template_relation (generated from PREDICATE_TEMPLATES)",
        f"Score formula : P(entailment)  — index 0 of 3-class NLI softmax",
        sep, "",
        "MODELS",
    ]
    for tag in MODEL_ORDER:
        cfg = MODELS[tag]
        if tag not in models_run:
            lines.append(f"  [{tag}]  SKIPPED")
            continue
        st = model_stats.get(tag, {})
        lines += [
            f"  [{tag}]",
            f"    checkpoint : {cfg['ckpt']}",
            f"    threshold  : {cfg['threshold']}",
            f"    score col  : {cfg['col_score']}",
            f"    pred col   : {cfg['col_pred']}",
            f"    rows done  : {st.get('rows', 0):,}",
            f"    time       : {_fmt_dur(st.get('time', 0))}",
        ]

    lines += ["", "PREDICTION DISTRIBUTIONS"]
    for tag in models_run:
        cfg = MODELS[tag]
        pp  = _pred_path(tag)
        if not pp.exists():
            continue
        pos = neg = total = 0
        with open(pp, encoding="utf-8") as f:
            for line in f:
                v = line.strip()
                if not v:
                    continue
                total += 1
                if float(v) >= cfg["threshold"]:
                    pos += 1
                else:
                    neg += 1
        pct = 100 * pos / total if total else 0
        lines.append(
            f"  [{tag:<16}]  "
            f"pos={pos:,} ({pct:.1f}%)  neg={neg:,} ({100-pct:.1f}%)  total={total:,}  "
            f"threshold={cfg['threshold']}"
        )

    lines += ["", "OUTPUT COLUMNS", f"  {', '.join(SILVER_COLS)}"]
    for tag in models_run:
        cfg = MODELS[tag]
        lines.append(f"  {cfg['col_score']}, {cfg['col_pred']}")
    lines += [
        "",
        "NOTE: output CSV is designed to receive additional classification columns",
        "      from other methods (finetuned LLM, opensource LLM, cross-train RC, …).",
        sep,
    ]

    text = "\n".join(lines)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    for line in lines:
        log.info(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Classify silver dataset with xlmroberta and neodictabert encoder NLI models"
    )
    parser.add_argument(
        "--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER,
        help="Which models to run (default: both)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Process only the first N rows (for testing)",
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Run inference only, skip final CSV merge step",
    )
    parser.add_argument(
        "--start-row", type=int, default=None,
        help="First silver row to process in this shard (for parallel runs)",
    )
    parser.add_argument(
        "--end-row", type=int, default=None,
        help="One-past-last silver row to process in this shard (for parallel runs)",
    )
    parser.add_argument(
        "--pred-suffix", default="",
        help="Suffix appended to pred_<tag><suffix>.txt for parallel runs (e.g. _part0)",
    )
    args = parser.parse_args()

    for d in (OUT_DIR, CLASS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log = setup_logger(LOG_FILE)
    log.info("=" * 70)
    log.info("classify_silver_encoder_nli.py  started")
    log.info(f"  models    : {args.models}")
    log.info(f"  max_rows  : {args.max_rows or 'all'}")
    log.info(f"  input     : {SILVER_CSV}")
    log.info(f"  output    : {OUT_CSV}")
    log.info(f"  summary   : {SUMMARY}")
    log.info("=" * 70)

    log.info("[count]  counting silver rows …")
    total_rows = count_silver_rows(max_rows=args.max_rows)
    log.info(f"[count]  total_rows = {total_rows:,}")

    wall_start  = time.time()
    model_stats = {}
    models_run  = []

    for tag in MODEL_ORDER:
        if tag not in args.models:
            log.info(f"[{tag}]  skipped (not in --models)")
            continue
        log.info(f"\n{'─' * 70}")
        log.info(f"[model]  {tag}")
        stats = run_model(tag, total_rows, log,
                          start_row=args.start_row,
                          end_row=args.end_row,
                          pred_suffix=args.pred_suffix)
        model_stats[tag] = stats
        models_run.append(tag)

    if args.skip_merge:
        log.info("[merge]  skipped (--skip-merge)")
    elif not models_run:
        log.info("[merge]  no models run — skipping")
    else:
        log.info(f"\n{'─' * 70}")
        merge_predictions(models_run, log)

    total_time = time.time() - wall_start
    write_summary(model_stats, total_rows, total_time, models_run, log)

    log.info("=" * 70)
    log.info(f"Done.  wall time: {_fmt_dur(total_time)}")
    log.info(f"  output  : {OUT_CSV}")
    log.info(f"  summary : {SUMMARY}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
