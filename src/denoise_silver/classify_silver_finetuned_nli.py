"""
Classify prepared_silver.csv (~2.5 M rows) using fine-tuned NLI decoder models
(LoRA adapters from Hebrew_NLI project, loaded via PEFT).

Models:
  dictalm24b_base_v2 — dicta-il/DictaLM-3.0-24B-Base  (ckpt DictaLM-3.0-24B-Base_hebnli/7920)
                       type=base, threshold=0.30
                       best finetuned-NLI result: template soft F1=0.872
  aya32b_v2          — CohereForAI/aya-expanse-32b     (ckpt aya-expanse-32b_hebnli/7920)
                       type=instruct, threshold=0.90
                       best finetuned-NLI result: template soft F1=0.873

Hypothesis used: template_relation (generated on-the-fly from PREDICATE_TEMPLATES).
NLI prompt:      "משפט 1: {text} משפט 2: {template_relation}"
Score:           P(היסק) / (P(היסק) + P(סתירה) + P(ניטרלי))  — from first-token logits.

Pipeline:
  1. Run dictalm24b_base_v2 → writes pred_dictalm24b_v2.txt (one float per line, resume-safe).
  2. Run aya32b_v2          → writes pred_aya32b_v2.txt.
  3. Merge: stream silver CSV + both pred files → silver_finetuned_nli.csv.
     Output columns: all original silver cols + score_* + pred_* for each model.
     Designed to join with silver_encoder_nli.csv on docid/subject/predicate/object.

Usage — recommended (handles env setup and GPU pinning automatically):
  bash run_finetuned_nli_tmux.sh                        # both models, GPU 7, detached tmux
  bash run_finetuned_nli_tmux.sh --models dictalm24b_base_v2
  bash run_finetuned_nli_tmux.sh --max-rows 5000        # smoke test

Usage — manual:
  conda activate hre_finetuned_nli
  export CUDA_VISIBLE_DEVICES=7                         # set AFTER conda activate — conda clears env vars
  export VLLM_WORKER_MULTIPROC_METHOD=spawn             # required: vLLM forks workers; spawn avoids CUDA re-init error
  cd /path/to/hebrew_RE
  python -m scripts_silver_cleaning.classify_silver_finetuned_nli [--models ...] [--max-rows N] [--skip-merge]

Resume: re-run the same command — pred_*.txt line count is checked at startup and already-done rows are skipped.

Backend (auto-selected, vLLM preferred):
  vLLM + LoRA (PunicaWrapperGPU):
    - Requires vllm, loaded via `from vllm import LLM, SamplingParams; from vllm.lora.request import LoRARequest`
    - Uses max_tokens=1 + logprobs=20 (vLLM 0.19.1 max) — only one forward pass per row, no generation loop (~10-30x faster than HF)
    - gpu_memory_utilization is computed dynamically from torch.cuda.mem_get_info(0) so it works even
      when the GPU is shared with other processes (avoids "Free memory < desired utilization" crash)
    - CRITICAL: do NOT call torch.cuda.device_count() or any CUDA function before constructing vLLM().
      Doing so initialises CUDA in the parent process; vLLM's multiprocessing workers then fail with
      "Cannot re-initialize CUDA in forked subprocess" even with spawn mode.
      GPU count is read from CUDA_VISIBLE_DEVICES env var instead.
  HF + PEFT fallback (automatic if vLLM fails):
    - Loads LoRA via peft_lib.PeftModel; uses model.generate() with output_scores=True
    - ~10-30x slower than vLLM; correct results but much longer wall time
    - device_map="auto" distributes across all visible GPUs

Model sizes & GPU requirements (bfloat16):
  dictalm24b_base_v2: DictaLM-3.0-24B-Base  ≈ 48 GB — fits on a single A100-80GB (tp=1)
  aya32b_v2:          aya-expanse-32b         ≈ 64 GB — fits on a single A100-80GB (tp=1)
"""

import csv
import gc
import logging
import math
import os
import queue as _queue
import threading
import time
import argparse
from pathlib import Path
from typing import Iterator

import torch
import transformers
import peft as peft_lib
from tqdm import tqdm

try:
    from vllm import LLM as vLLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).resolve().parent.parent.parent
SILVER_CSV = ROOT / "data" / "prepared_silver.csv"
OUT_DIR    = ROOT / "outputs" / "silver_cleaning" / "silver_finetuned_nli"
CLASS_DIR  = OUT_DIR / "classifications"
LOG_DIR    = OUT_DIR / "logs"
LOG_FILE   = OUT_DIR / "run.log"
SUMMARY    = OUT_DIR / "summary.txt"
OUT_CSV    = CLASS_DIR / "silver_finetuned_nli.csv"

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

_HEB_NLI_OUT = "/path/to/Hebrew_NLI/output"

MODELS = {
    "dictalm24b_base_v2": {
        "ckpt":        f"{_HEB_NLI_OUT}/DictaLM-3.0-24B-Base_hebnli/7920",
        "type":        "base",
        "user_prefix": "הוראה: קרא את המשפטים וקבע את היחס הלוגי. ענה רק במילה אחת: היסק, סתירה, או ניטרלי.\n\n",
        "user_suffix": "\nתשובה:",
        "system":      "",
        "threshold":   0.30,
        "col_score":   "score_dictalm24b_v2",
        "col_pred":    "pred_dictalm24b_v2",
    },
    "aya32b_v2": {
        "ckpt":        f"{_HEB_NLI_OUT}/aya-expanse-32b_hebnli/7920",
        "type":        "instruct",
        "user_prefix": "",
        "user_suffix": "\nהאם הטענה נובעת מהטקסט? ענה: היסק, סתירה, או ניטרלי.",
        "system":      "אתה מודל הסקה לוגית בעברית. ענה תמיד ורק במילה אחת מהאפשרויות: 'היסק', 'סתירה', או 'ניטרלי'.",
        "threshold":   0.90,
        "col_score":   "score_aya32b_v2",
        "col_pred":    "pred_aya32b_v2",
    },
    "gemma2_9b_v1": {
        "ckpt":        f"{_HEB_NLI_OUT}/gemma-2-9b-it_hebnli/7920",
        "type":        "instruct",
        "user_prefix": "",
        "user_suffix": "\nהאם הטענה נובעת מהטקסט? ענה: היסק, סתירה, או ניטרלי.",
        "system":      "אתה מודל הסקה לוגית בעברית. ענה תמיד ורק במילה אחת מהאפשרויות: 'היסק', 'סתירה', או 'ניטרלי'.",
        "threshold":   0.90,
        "col_score":   "score_gemma2_9b_v1",
        "col_pred":    "pred_gemma2_9b_v1",
    },
}

MODEL_ORDER = ["dictalm24b_base_v2", "aya32b_v2", "gemma2_9b_v1"]

# Input silver columns (output base)
SILVER_COLS = ["docid", "title", "text", "subject", "predicate", "object"]

# Hebrew NLI label tokens
NLI_ENTAIL  = "היסק"
NLI_CONTRA  = "סתירה"
NLI_NEUTRAL = "ניטרלי"

# ---------------------------------------------------------------------------
# Processing constants
# ---------------------------------------------------------------------------

CHUNK_SIZE         = 50_000
BATCH_SIZE         = 16
MAX_NEW_TOKENS     = 8       # NLI labels are short (≤5 tokens)
MAX_INPUT_LEN      = 512     # matches HebNLI fine-tuning context
LOG_EVERY_N_CHUNKS = 1

# vLLM settings
VLLM_CHUNK_SIZE   = 100_000  # larger chunks; continuous batching handles it
VLLM_MAX_LOGPROBS = 20       # vLLM 0.19.1 caps logprobs at 20; NLI label tokens dominate top-20 for a fine-tuned model

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
    logger = logging.getLogger("classify_silver_finetuned_nli")
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


def _pred_path(tag: str) -> Path:
    return OUT_DIR / f"pred_{tag}.txt"


# ---------------------------------------------------------------------------
# NLI token IDs
# ---------------------------------------------------------------------------

def get_nli_token_ids(tokenizer) -> tuple[list, list, list]:
    """Return (entail_ids, contra_ids, neutral_ids) — first token of each label."""
    entail_vars  = [NLI_ENTAIL,  " " + NLI_ENTAIL,  "▁" + NLI_ENTAIL]
    contra_vars  = [NLI_CONTRA,  " " + NLI_CONTRA,  "▁" + NLI_CONTRA]
    neutral_vars = [NLI_NEUTRAL, " " + NLI_NEUTRAL, "▁" + NLI_NEUTRAL]
    entail_ids, contra_ids, neutral_ids = set(), set(), set()
    for v in entail_vars:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if toks: entail_ids.add(toks[0])
    for v in contra_vars:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if toks: contra_ids.add(toks[0])
    for v in neutral_vars:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if toks: neutral_ids.add(toks[0])
    ambiguous = (entail_ids & contra_ids) | (entail_ids & neutral_ids) | (contra_ids & neutral_ids)
    entail_ids  -= ambiguous
    contra_ids  -= ambiguous
    neutral_ids -= ambiguous
    return sorted(entail_ids), sorted(contra_ids), sorted(neutral_ids)


def compute_soft_scores(logits, entail_ids: list, contra_ids: list, neutral_ids: list) -> list[float]:
    """Returns P(entail)/(P(entail)+P(contra)+P(neutral)) for each item in batch."""
    if not (entail_ids or contra_ids or neutral_ids):
        return [0.5] * logits.shape[0]
    probs = torch.softmax(logits.float(), dim=-1)

    def _sum(ids):
        if not ids:
            return torch.zeros(logits.shape[0], device=logits.device)
        t = torch.tensor(ids, device=logits.device)
        return probs[:, t].sum(dim=-1)

    p_e = _sum(entail_ids)
    p_c = _sum(contra_ids)
    p_n = _sum(neutral_ids)
    return ((p_e / (p_e + p_c + p_n + 1e-10)).cpu().tolist())


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(model_type: str, text: str, hypothesis: str, tokenizer,
                 system: str = "", user_prefix: str = "", user_suffix: str = "") -> str:
    nli_text = user_prefix + f"משפט 1: {text} משפט 2: {hypothesis}" + user_suffix
    if model_type == "instruct":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": nli_text})
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prefix = f"[SYSTEM]{system}[/SYSTEM]\n" if system else ""
            return prefix + nli_text + "\n"
    else:
        return nli_text


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

def _soft_vllm(logprobs_dict: dict, entail_ids: list, contra_ids: list, neutral_ids: list) -> float:
    def get_prob(ids):
        return sum(math.exp(logprobs_dict[tid].logprob) if tid in logprobs_dict else 0.0 for tid in ids)
    p_e = get_prob(entail_ids)
    p_c = get_prob(contra_ids)
    p_n = get_prob(neutral_ids)
    return p_e / (p_e + p_c + p_n + 1e-10)


def load_vllm_model(tag: str, log: logging.Logger):
    cfg      = MODELS[tag]
    ckpt     = cfg["ckpt"]
    config   = peft_lib.PeftConfig.from_pretrained(ckpt)
    base_id  = config.base_model_name_or_path
    # Parse GPU count from env — avoids calling torch.cuda before vLLM forks workers
    _cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tp  = len([x for x in _cv.split(",") if x.strip()]) if _cv else 1

    # Compute safe utilization from actual free memory (safe with spawn mode)
    torch.cuda.init()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    gpu_util = min(0.90, (free_bytes / total_bytes) * 0.92)
    log.info(f"    [vLLM] base={base_id}  lora={ckpt}  tp={tp}  gpu_util={gpu_util:.2f}  free={free_bytes/1e9:.1f}GB/{total_bytes/1e9:.1f}GB")

    llm = vLLM(
        model=base_id,
        tensor_parallel_size=tp,
        enable_lora=True,
        max_lora_rank=16,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
        max_model_len=MAX_INPUT_LEN + 4,
        max_num_seqs=4096,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lora_request = LoRARequest(tag, 1, str(ckpt))
    log.info("    [vLLM] ready")
    return llm, tokenizer, lora_request


def run_chunk_vllm(
    llm,
    tokenizer,
    lora_request,
    entail_ids: list,
    contra_ids: list,
    neutral_ids: list,
    rows: list[dict],
    cfg: dict,
    log: logging.Logger,
) -> list[float]:
    # Tokenize + left-truncate to MAX_INPUT_LEN tokens, then pass token IDs directly to
    # vLLM — avoids vLLM's internal re-tokenization ("Rendering prompts") pass entirely.
    token_ids_list = []
    for r in rows:
        prompt = build_prompt(
            cfg["type"],
            r["text"],
            make_template_relation(r["subject"], r["predicate"], r["object"]),
            tokenizer,
            system=cfg["system"], user_prefix=cfg["user_prefix"], user_suffix=cfg["user_suffix"],
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > MAX_INPUT_LEN:
            ids = ids[-MAX_INPUT_LEN:]  # left-truncate: keep end (hypothesis + answer cue)
        token_ids_list.append(ids)
    params  = SamplingParams(max_tokens=1, logprobs=VLLM_MAX_LOGPROBS)
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in token_ids_list],
        sampling_params=params, lora_request=lora_request,
    )
    scores  = []
    for out in outputs:
        lp = out.outputs[0].logprobs[0] if (out.outputs and out.outputs[0].logprobs) else {}
        scores.append(_soft_vllm(lp, entail_ids, contra_ids, neutral_ids))
    return scores


def unload_vllm(llm, log: logging.Logger):
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("    [vLLM] unloaded")


# ---------------------------------------------------------------------------
# HF Model load / unload
# ---------------------------------------------------------------------------

def load_model(tag: str, log: logging.Logger):
    cfg      = MODELS[tag]
    ckpt     = cfg["ckpt"]
    log.info(f"    checkpoint : {ckpt}")

    config  = peft_lib.PeftConfig.from_pretrained(ckpt)
    base_id = config.base_model_name_or_path
    log.info(f"    base model : {base_id}")

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 0:
        major, _ = torch.cuda.get_device_capability()
        pt_dtype = torch.bfloat16 if major >= 8 else torch.float16
        log.info(f"    GPUs={n_gpus}  dtype={pt_dtype}  GPU0={torch.cuda.get_device_name(0)}")
    else:
        pt_dtype = torch.float32
        log.warning("    No CUDA detected — running on CPU (very slow)")

    tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "left"

    try:
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            base_id, device_map="auto", torch_dtype=pt_dtype,
            trust_remote_code=True, attn_implementation="sdpa",
        )
        log.info("    sdpa enabled")
    except Exception as e:
        log.warning(f"    sdpa failed ({type(e).__name__}), retrying standard load")
        try:
            base_model = transformers.AutoModelForCausalLM.from_pretrained(
                base_id, device_map="auto", torch_dtype=pt_dtype, trust_remote_code=True,
            )
        except ValueError:
            log.info("    Falling back to Gemma4ForConditionalGeneration")
            from transformers import Gemma4ForConditionalGeneration
            try:
                base_model = Gemma4ForConditionalGeneration.from_pretrained(
                    base_id, device_map="auto", torch_dtype=pt_dtype,
                    trust_remote_code=True, attn_implementation="flash_attention_2",
                )
            except Exception:
                base_model = Gemma4ForConditionalGeneration.from_pretrained(
                    base_id, device_map="auto", torch_dtype=pt_dtype, trust_remote_code=True,
                )

    model = peft_lib.PeftModel.from_pretrained(base_model, ckpt, is_trainable=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info(f"    ready  ({n_params:.1f}B params, LoRA adapter applied)")
    return model, tokenizer


def unload_model(model, log: logging.Logger):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("    GPU cache cleared")


def _input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Chunk inference
# ---------------------------------------------------------------------------

def run_chunk(
    model,
    tokenizer,
    entail_ids: list,
    contra_ids: list,
    neutral_ids: list,
    rows: list[dict],
    cfg: dict,
    effective_batch: int,
    log: logging.Logger,
) -> tuple[list[float], int]:
    """
    Run NLI inference on a chunk of silver rows.
    Returns (soft_scores, final_effective_batch) — batch size may shrink on OOM.
    """
    model_type  = cfg["type"]
    system      = cfg["system"]
    user_prefix = cfg["user_prefix"]
    user_suffix = cfg["user_suffix"]
    device      = _input_device(model)

    prompts = [
        build_prompt(
            model_type,
            r["text"],
            make_template_relation(r["subject"], r["predicate"], r["object"]),
            tokenizer,
            system=system, user_prefix=user_prefix, user_suffix=user_suffix,
        )
        for r in rows
    ]

    # For base models stop at newline to get just the label
    hf_stop_kwargs: dict = {}
    if model_type == "base":
        nl_ids: set = set()
        for v in ["\n", NLI_ENTAIL, NLI_CONTRA, NLI_NEUTRAL]:
            toks = tokenizer.encode(v, add_special_tokens=False)
            if len(toks) == 1:
                nl_ids.add(toks[0])
        if nl_ids:
            hf_stop_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + list(nl_ids)

    all_scores: list[float] = []
    start = 0
    while start < len(prompts):
        sub = prompts[start : start + effective_batch]
        enc = tokenizer(
            sub,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
        )
        enc_dev = {k: v.to(device) for k, v in enc.items() if k != "token_type_ids"}
        try:
            with torch.no_grad():
                out = model.generate(
                    **enc_dev,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **hf_stop_kwargs,
                )
            scores = compute_soft_scores(out.scores[0], entail_ids, contra_ids, neutral_ids)
            all_scores.extend(scores)
            start += effective_batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            effective_batch = max(1, effective_batch // 2)
            log.warning(f"    OOM — reduced batch_size to {effective_batch}, retrying sub-batch")

    return all_scores, effective_batch


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(tag: str, total_rows: int, log: logging.Logger) -> dict:
    cfg       = MODELS[tag]
    pred_path = _pred_path(tag)
    done      = count_pred_rows(pred_path)

    if done >= total_rows:
        log.info(f"[{tag}]  {total_rows:,} rows already done — skipping")
        return {"rows": total_rows, "time": 0.0}

    log.info(f"[{tag}]  starting from row {done:,}  backend=auto")

    # ---- Try vLLM first (up to 3 attempts) ----
    # On shared GPUs, vLLM's init-time memory assertion can fire if another process
    # releases memory during the profiling pass.  A short wait + retry usually clears it.
    use_vllm = False
    if VLLM_AVAILABLE:
        for attempt in range(1, 4):
            try:
                llm, tokenizer, lora_request = load_vllm_model(tag, log)
                use_vllm    = True
                chunk_size  = VLLM_CHUNK_SIZE
                backend_tag = "vllm"
                break
            except Exception as e:
                log.warning(f"    vLLM load failed (attempt {attempt}/3): {e}")
                if attempt < 3:
                    import time as _time
                    _time.sleep(10)
        else:
            log.warning("    vLLM failed after 3 attempts, falling back to HF")

    if not use_vllm:
        model, tokenizer = load_model(tag, log)
        chunk_size  = CHUNK_SIZE
        backend_tag = "hf"

    entail_ids, contra_ids, neutral_ids = get_nli_token_ids(tokenizer)
    log.info(f"    backend={backend_tag}  chunk_size={chunk_size:,}")
    log.info(f"    entail_ids ({len(entail_ids)}): {entail_ids}")
    log.info(f"    contra_ids ({len(contra_ids)}): {contra_ids}")
    log.info(f"    neutral_ids ({len(neutral_ids)}): {neutral_ids}")

    example_hyp = make_template_relation("X", "מקום לידה", "Y")
    ex_prompt = build_prompt(
        cfg["type"], "X נולד בעיר Y.", example_hyp, tokenizer,
        system=cfg["system"], user_prefix=cfg["user_prefix"], user_suffix=cfg["user_suffix"],
    )
    log.info(f"    example prompt:\n{'─'*40}\n{ex_prompt[:500]}\n{'─'*40}")

    t0              = time.time()
    written         = done
    effective_batch = BATCH_SIZE

    # Prefetch: background thread reads the next CSV chunk while the GPU processes the current one.
    def _reader(skip, limit, q):
        try:
            for ch in chunked(stream_silver(skip=skip, limit=limit), chunk_size):
                q.put(ch)
        finally:
            q.put(None)

    fetch_q = _queue.Queue(maxsize=2)
    threading.Thread(target=_reader, args=(done, total_rows - done, fetch_q), daemon=True).start()

    chunk_idx = 0
    with open(pred_path, "a", encoding="utf-8") as fout:
        while True:
            chunk = fetch_q.get()
            if chunk is None:
                break
            chunk_idx += 1
            if use_vllm:
                scores = run_chunk_vllm(
                    llm, tokenizer, lora_request, entail_ids, contra_ids, neutral_ids,
                    chunk, cfg, log,
                )
            else:
                scores, effective_batch = run_chunk(
                    model, tokenizer, entail_ids, contra_ids, neutral_ids,
                    chunk, cfg, effective_batch, log,
                )
            for s in scores:
                fout.write(f"{s:.6f}\n")
            fout.flush()  # crash-safe: at most one chunk lost on failure
            written += len(chunk)

            if chunk_idx % LOG_EVERY_N_CHUNKS == 0 or written >= total_rows:
                elapsed = time.time() - t0
                rps     = (written - done) / elapsed if elapsed else 0
                eta     = (total_rows - written) / rps if rps else 0
                log.info(
                    f"[{tag}]  {written:,}/{total_rows:,}  "
                    f"({100 * written / total_rows:.1f}%)  "
                    f"{rps:.1f} rows/s  ETA {_fmt_dur(eta)}"
                )

    elapsed = time.time() - t0
    if use_vllm:
        unload_vllm(llm, log)
    else:
        unload_model(model, log)
    log.info(f"[{tag}]  done  backend={backend_tag}  {_fmt_dur(elapsed)}")
    return {"rows": written, "time": elapsed, "backend": backend_tag}


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
        "SILVER FINETUNED NLI CLASSIFICATION SUMMARY",
        f"Date          : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total rows    : {total_rows:,}",
        f"Wall time     : {_fmt_dur(total_time)}",
        f"Input CSV     : {SILVER_CSV}",
        f"Output CSV    : {OUT_CSV}",
        f"Hypothesis    : template_relation (generated from PREDICATE_TEMPLATES)",
        f"Score formula : P(היסק) / (P(היסק) + P(סתירה) + P(ניטרלי))  from first-token logits",
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
            f"    type       : {cfg['type']}",
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
            f"  [{tag:<22}]  "
            f"pos={pos:,} ({pct:.1f}%)  neg={neg:,} ({100-pct:.1f}%)  total={total:,}  "
            f"threshold={cfg['threshold']}"
        )

    lines += ["", "OUTPUT COLUMNS", f"  {', '.join(SILVER_COLS)}"]
    for tag in models_run:
        cfg = MODELS[tag]
        lines.append(f"  {cfg['col_score']}, {cfg['col_pred']}")
    lines += [
        "",
        "NOTE: output CSV is designed to join with silver_encoder_nli.csv and",
        "      silver_opensource_llm.csv on docid/subject/predicate/object.",
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
        description="Classify silver dataset with dictalm24b_base_v2 and aya32b_v2 finetuned NLI models"
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
    args = parser.parse_args()

    for d in (OUT_DIR, CLASS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log = setup_logger(LOG_FILE)
    log.info("=" * 70)
    log.info("classify_silver_finetuned_nli.py  started")
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
        stats = run_model(tag, total_rows, log)
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
