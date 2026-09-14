"""
Classify prepared_silver.csv (~2.5 M rows) using the best open-source LLM configs
selected from gold-500 evaluation (see outputs/opensource_LLM_clean/summary.txt).

Models and configs chosen:
  gemma4_31b_it — google/gemma-4-31B-it
    lang=he, rel=template, shot=2, no-CoT
    gold-500 soft F1=0.927  threshold=0.30
    STATUS: NOT RUNNABLE — Gemma4ForConditionalGeneration is not supported by vLLM 0.19.1;
            upgrade vLLM or use HF inference when a compatible version is available.
  dictalm3 — dicta-il/DictaLM-3.0-24B-Base
    lang=en, rel=template, shot=5, no-CoT
    gold-500 soft F1=0.836  threshold=0.50
    ~48 GB bfloat16 — fits on a single A100-80GB (tp=1) or two GPUs (tp=2)

Score: P(yes) / (P(yes) + P(no)) from first-token logits.
Hypothesis: template_relation generated on-the-fly from PREDICATE_TEMPLATES.

Pipeline:
  1. gemma4_31b_it  → pred_gemma4_31b_it.txt  (one float/line, resume-safe)  [currently skipped]
  2. dictalm3       → pred_dictalm3.txt
  3. Merge          → classifications/silver_opensource_llm.csv

Usage — recommended (handles env setup and GPU pinning automatically):
  bash run_opensource_llm_tmux.sh                        # GPU 4+5, detached tmux
  bash run_opensource_llm_tmux.sh --max-rows 5000        # smoke test

Usage — manual:
  conda activate hre_finetuned_nli
  export CUDA_VISIBLE_DEVICES=4,5                        # set AFTER conda activate — conda clears env vars
  export VLLM_WORKER_MULTIPROC_METHOD=spawn              # required: vLLM forks workers; spawn avoids CUDA re-init error
  cd /path/to/hebrew_RE
  python -m scripts_silver_cleaning.classify_silver_opensource_llm --models dictalm3 [--max-rows N]

Resume: re-run the same command — pred_*.txt line count is checked at startup and already-done rows are skipped.

Backend (vLLM only — no HF fallback in this script):
  - Uses max_tokens=1 + logprobs=20 (vLLM 0.19.1 max) — one forward pass per row, no generation loop (~10-30x faster than HF)
  - max_model_len=4096: intentionally larger than MAX_INPUT_LEN (2048) because vLLM tokenises prompts
    independently (does NOT apply the truncation the HF tokeniser does), so some prompts exceed
    MAX_INPUT_LEN + 64 after vLLM's own tokenisation. Setting 4096 gives headroom without waste.
  - gpu_memory_utilization is computed dynamically from torch.cuda.mem_get_info(0) so it works
    when the GPU is shared with other processes (avoids "Free memory < desired utilization" crash).
  - CRITICAL: do NOT call torch.cuda.device_count() or any CUDA function before constructing vLLM().
    Doing so initialises CUDA in the parent process; vLLM's multiprocessing workers then fail with
    "Cannot re-initialize CUDA in forked subprocess" even with spawn mode.
    GPU count is parsed from CUDA_VISIBLE_DEVICES env var instead.
"""

import gc
import math
import csv
import logging
import os
import re
import time
import argparse
from pathlib import Path
from typing import Iterator

import torch
import transformers

if not hasattr(transformers.PreTrainedTokenizerBase, "all_special_tokens_extended"):
    transformers.PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens
    )

try:
    from vllm import LLM as vLLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).resolve().parent.parent.parent
SILVER_CSV = ROOT / "data" / "prepared_silver.csv"
OUT_DIR    = ROOT / "outputs" / "silver_cleaning" / "silver_opensource_llm"
CLASS_DIR  = OUT_DIR / "classifications"
LOG_DIR    = OUT_DIR / "logs"
LOG_FILE   = OUT_DIR / "run.log"
SUMMARY    = OUT_DIR / "summary.txt"
OUT_CSV    = CLASS_DIR / "silver_opensource_llm.csv"

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODELS = {
    "gemma4_31b_it": {
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "model_id":      "google/gemma-4-31B-it",
        "type":          "instruct",
        "lang":          "he",
        "rel":           "template",
        "n_shot":        2,
        "threshold":     0.30,
        "col_score":     "score_gemma4_31b_it",
        "col_pred":      "pred_gemma4_31b_it",
    },
    "dictalm3": {
        "model_id":  "dicta-il/DictaLM-3.0-24B-Base",
        "type":      "base",
        "lang":      "en",
        "rel":       "template",
        "n_shot":    5,
        "threshold": 0.50,
        "col_score": "score_dictalm3",
        "col_pred":  "pred_dictalm3",
    },
    "gemma3_27b_it": {
        "model_id":  "google/gemma-3-27b-it",
        "type":      "instruct",
        "lang":      "he",
        "rel":       "template",
        "n_shot":    2,
        "threshold": 0.30,
        "col_score": "score_gemma3_27b_it",
        "col_pred":  "pred_gemma3_27b_it",
    },
}

MODEL_ORDER = ["gemma4_31b_it", "dictalm3", "gemma3_27b_it"]

SILVER_COLS = ["docid", "title", "text", "subject", "predicate", "object"]

# ---------------------------------------------------------------------------
# Processing constants
# ---------------------------------------------------------------------------

CHUNK_SIZE       = 50_000
BATCH_SIZE       = 16
MAX_NEW_TOKENS   = 32
MAX_INPUT_LEN    = 2048
LOG_EVERY_CHUNKS = 1

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
# Few-shot example pools  (same as gold-500 eval; indices [0,3] for 2-shot)
# ---------------------------------------------------------------------------

_POOL_HE_TEMPLATE = [
    # 0 — POSITIVE: birthplace
    {
        "text":      "ויליאם שייקספיר (1564–1616) נולד בסטרטפורד-אפון-אייבון, אנגליה. הוא נחשב לגדול המחזאים של כל הזמנים.",
        "statement": "מקום הלידה של שייקספיר הוא סטרטפורד-אפון-אייבון",
        "answer":    "כן",
    },
    # 1 — POSITIVE: founder
    {
        "text":      "מייקרוסופט נוסדה ב-1975 על ידי ביל גייטס ופול אלן. המטה ממוקם ברדמונד, וושינגטון.",
        "statement": "המייסד של מייקרוסופט הוא ביל גייטס",
        "answer":    "כן",
    },
    # 2 — POSITIVE: occupation
    {
        "text":      "מארי קירי הייתה פיזיקאית וכימאית פולנית-צרפתית. היא זכתה בפרס נובל פעמיים.",
        "statement": "העיסוק של מארי קירי הוא פיזיקאית",
        "answer":    "כן",
    },
    # 3 — NEGATIVE: official language (world-knowledge leakage)
    {
        "text":      'ברזיל היא המדינה הגדולה ביותר בדרום אמריקה, עם שטח של כ-8.5 מיליון קמ"ר ואוכלוסייה של כ-215 מיליון.',
        "statement": "השפה הרשמית של ברזיל היא פורטוגזית",
        "answer":    "לא",
    },
    # 4 — NEGATIVE: borders
    {
        "text":      "הנמר האמורי הוא תת-מין נדיר של נמר החי ביערות רוסיה ובצפון סין. אוכלוסייתו פחות ממאה פרטים.",
        "statement": "רוסיה גובלת ב-סין",
        "answer":    "לא",
    },
]

_POOL_EN_TEMPLATE = [
    {
        "text":      "William Shakespeare (1564–1616) was born in Stratford-upon-Avon, England. He is considered the greatest playwright of all time.",
        "statement": "The birthplace of Shakespeare is Stratford-upon-Avon",
        "answer":    "yes",
    },
    {
        "text":      "Microsoft was founded in 1975 by Bill Gates and Paul Allen. Its headquarters are in Redmond, Washington.",
        "statement": "The founder of Microsoft is Bill Gates",
        "answer":    "yes",
    },
    {
        "text":      "Marie Curie was a Polish-French physicist and chemist. She won the Nobel Prize twice.",
        "statement": "The occupation of Marie Curie is physicist",
        "answer":    "yes",
    },
    {
        "text":      "Brazil is the largest country in South America, with an area of about 8.5 million km² and a population of about 215 million.",
        "statement": "The official language of Brazil is Portuguese",
        "answer":    "no",
    },
    {
        "text":      "The Amur leopard is a rare subspecies of leopard living in the forests of Russia and northern China. Its population is estimated at fewer than one hundred individuals.",
        "statement": "Russia borders China",
        "answer":    "no",
    },
]

_SYSTEM_HE_TEMPLATE = (
    "אתה מסייע לזיהוי יחסים בטקסטים בעברית. "
    "יחס בין שתי ישויות מנוסח כמשפט. "
    "תפקידך לקרוא טקסט ולקבוע האם המשפט הנתון נובע מהטקסט או מופיע בו. "
    'ענה "כן" אם המשפט נובע מהטקסט, או "לא" אחרת. ענה במילה אחת בלבד.'
)

_SYSTEM_EN_TEMPLATE = (
    "You are a textual entailment assistant. "
    "A relation between two entities is expressed as a natural-language statement. "
    "Your task is to read a text and determine whether the given statement is entailed by or expressed in it. "
    'Answer "yes" if entailed, "no" if not. One word only.'
)


def _user_msg_he_template(text: str, statement: str) -> str:
    return (
        f"טקסט:\n{text}\n\n"
        f"משפט לבדיקה: {statement}\n"
        'חפש בטקסט האם המשפט הנ"ל נובע ממנו או מופיע בו.\n'
        'ענה "כן" או "לא" בלבד.'
    )


def _user_msg_en_template(text: str, statement: str) -> str:
    return (
        f"Text:\n{text}\n\n"
        f"Statement to check: {statement}\n"
        "Look in the text for whether the above statement is expressed or can be inferred from it.\n"
        'Answer "yes" or "no" only.'
    )


def build_prompt_gemma4(row: dict, tokenizer) -> str:
    """he + template + 2-shot instruct prompt for gemma4_31b_it."""
    shot_indices = [0, 3]
    messages = [{"role": "system", "content": _SYSTEM_HE_TEMPLATE}]
    for i in shot_indices:
        ex = _POOL_HE_TEMPLATE[i]
        messages.append({"role": "user",      "content": _user_msg_he_template(ex["text"], ex["statement"])})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": _user_msg_he_template(row["text"], row["template_relation"])})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt_dictalm3(row: dict) -> str:
    """en + template + 5-shot base completion prompt for dictalm3."""
    intro = (
        "Read the following text and answer whether the given statement is entailed by it.\n"
        "Answer 'yes' if entailed, 'no' if not."
    )
    parts = [intro]
    for ex in _POOL_EN_TEMPLATE:
        parts.append(
            f"Text: {ex['text']}\n"
            f"Statement: {ex['statement']}\n"
            f"Answer: {ex['answer']}"
        )
    parts.append(
        f"Text: {row['text']}\n"
        f"Statement: {row['template_relation']}\n"
        "Answer:"
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Yes/No token IDs
# ---------------------------------------------------------------------------

def get_yn_token_ids(tokenizer) -> tuple[list, list]:
    yes_vars = ["yes", "Yes", "YES", "כן", " yes", " Yes", " כן", "▁yes", "▁Yes"]
    no_vars  = ["no",  "No",  "NO",  "לא", " no",  " No",  " לא", "▁no",  "▁No"]
    yes_ids, no_ids = set(), set()
    for v in yes_vars:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if len(toks) == 1:
            yes_ids.add(toks[0])
    for v in no_vars:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if len(toks) == 1:
            no_ids.add(toks[0])
    return sorted(yes_ids), sorted(no_ids)


def _soft_yn(logits, yes_ids: list, no_ids: list) -> list[float]:
    if not yes_ids and not no_ids:
        return [0.5] * logits.shape[0]
    probs = logits.float().softmax(dim=-1)
    yes_t = torch.tensor(yes_ids, device=logits.device) if yes_ids else None
    no_t  = torch.tensor(no_ids,  device=logits.device) if no_ids  else None
    p_yes = probs[:, yes_t].sum(-1) if yes_t is not None else torch.zeros(logits.shape[0], device=logits.device)
    p_no  = probs[:, no_t ].sum(-1) if no_t  is not None else torch.zeros(logits.shape[0], device=logits.device)
    return (p_yes / (p_yes + p_no + 1e-10)).cpu().tolist()


def _soft_yn_vllm(first_lp: dict, yes_ids: list, no_ids: list) -> float:
    p_yes = p_no = 0.0
    for tid, lp_obj in first_lp.items():
        p = math.exp(lp_obj.logprob)
        if tid in yes_ids:   p_yes += p
        elif tid in no_ids:  p_no  += p
    denom = p_yes + p_no
    return p_yes / denom if denom > 1e-10 else 0.5


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _fmt_dur(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h:  return f"{h}h {m}m {sec}s"
    if m:  return f"{m}m {sec}s"
    return f"{sec}s"


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("classify_silver_opensource_llm")
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
# Model load / unload
# ---------------------------------------------------------------------------

def load_llm(model_id: str, log: logging.Logger, enforce_eager: bool = False,
             limit_mm_per_prompt: dict | None = None):
    """Load model; tries vLLM first, falls back to HF."""
    log.info(f"    model_id : {model_id}")
    # Parse GPU count from env — avoids calling torch.cuda before vLLM forks workers
    _cv    = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_gpus = len([x for x in _cv.split(",") if x.strip()]) if _cv else 0

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if VLLM_AVAILABLE and n_gpus > 0:
        try:
            tp  = n_gpus
            # Dynamic gpu_util: safe with spawn mode; avoids "Free memory < desired utilization"
            # when the GPU is shared with other processes.
            torch.cuda.init()
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            gpu_util = min(0.90, (free_bytes / total_bytes) * 0.92)
            log.info(f"    [vLLM] model={model_id}  tp={tp}  gpu_util={gpu_util:.2f}  free={free_bytes/1e9:.1f}GB/{total_bytes/1e9:.1f}GB  enforce_eager={enforce_eager}")
            vllm_kwargs = dict(
                model=model_id, dtype="bfloat16",
                tensor_parallel_size=tp, trust_remote_code=True,
                max_model_len=16384, gpu_memory_utilization=gpu_util,
                enforce_eager=enforce_eager,
            )
            if limit_mm_per_prompt is not None:
                vllm_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt
            llm = vLLM(**vllm_kwargs)
            log.info(f"    vLLM backend  (tensor_parallel_size={tp})")
            return llm, tokenizer, "vllm"
        except Exception as e:
            log.warning(f"    vLLM failed ({e}), falling back to HF")

    major = torch.cuda.get_device_capability()[0] if n_gpus > 0 else 0
    pt_dtype = torch.bfloat16 if major >= 8 else torch.float16
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=pt_dtype,
            trust_remote_code=True, attn_implementation="sdpa",
        )
    except Exception as e:
        log.warning(f"    sdpa/CausalLM failed ({type(e).__name__}), retrying")
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=pt_dtype, trust_remote_code=True,
            )
        except ValueError:
            from transformers import Gemma4ForConditionalGeneration
            try:
                model = Gemma4ForConditionalGeneration.from_pretrained(
                    model_id, device_map="auto", torch_dtype=pt_dtype,
                    trust_remote_code=True, attn_implementation="sdpa",
                )
            except Exception:
                model = Gemma4ForConditionalGeneration.from_pretrained(
                    model_id, device_map="auto", torch_dtype=pt_dtype, trust_remote_code=True,
                )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info(f"    HF backend  ({n_params:.1f}B params)")
    return model, tokenizer, "hf"


def unload_llm(model, backend: str, log: logging.Logger):
    if backend == "vllm":
        try:
            import ray; ray.shutdown()
        except Exception:
            pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("    model unloaded")


# ---------------------------------------------------------------------------
# Inference: HF batch
# ---------------------------------------------------------------------------

def _infer_hf_batch(
    model, tokenizer, prompts: list[str],
    yes_ids: list, no_ids: list,
    batch_size: int,
) -> list[float]:
    device = next(model.parameters()).device
    all_scores: list[float] = []
    effective_bs = batch_size

    for start in range(0, len(prompts), effective_bs):
        sub = prompts[start : start + effective_bs]
        while True:
            enc = tokenizer(
                sub, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_INPUT_LEN,
            )
            enc = {k: v.to(device) for k, v in enc.items() if k != "token_type_ids"}
            try:
                with torch.no_grad():
                    out = model.generate(
                        **enc, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False, temperature=None, top_p=None,
                        pad_token_id=tokenizer.pad_token_id,
                        output_scores=True, return_dict_in_generate=True,
                    )
                all_scores.extend(_soft_yn(out.scores[0], yes_ids, no_ids))
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                effective_bs = max(1, effective_bs // 2)
                sub = prompts[start : start + effective_bs]

    return all_scores


# ---------------------------------------------------------------------------
# Inference: vLLM batch
# ---------------------------------------------------------------------------

_MAX_VLLM_TOKENS = 16382  # max_model_len=16384 minus 1 for answer token minus 1 for BOS vLLM prepends internally


def _infer_vllm_batch(llm, prompts: list[str], yes_ids: list, no_ids: list, tokenizer=None) -> list[float]:
    if tokenizer is not None:
        safe = []
        for p in prompts:
            ids = tokenizer.encode(p, add_special_tokens=False)
            if len(ids) > _MAX_VLLM_TOKENS:
                ids = ids[-_MAX_VLLM_TOKENS:]  # left-truncate: drop oldest tokens
                p = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            safe.append(p)
        prompts = safe
    sampling = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, temperature=0, logprobs=20, stop=["\n"],
    )
    results = llm.generate(prompts, sampling)
    scores = []
    for res in results:
        try:
            first_lp = res.outputs[0].logprobs[0]
            scores.append(_soft_yn_vllm(first_lp, yes_ids, no_ids))
        except Exception:
            scores.append(0.5)
    return scores


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

    log.info(f"[{tag}]  starting from row {done:,}")
    model, tokenizer, backend = load_llm(cfg["model_id"], log,
                                          enforce_eager=cfg.get("enforce_eager", False),
                                          limit_mm_per_prompt=cfg.get("limit_mm_per_prompt"))
    yes_ids, no_ids = get_yn_token_ids(tokenizer)
    log.info(f"    yes_ids={yes_ids}  no_ids={no_ids}")
    t0      = time.time()
    written = done

    with open(pred_path, "a", encoding="utf-8") as fout:
        for chunk_idx, chunk in enumerate(
            chunked(stream_silver(skip=done, limit=total_rows - done), CHUNK_SIZE), 1
        ):
            for row in chunk:
                row["template_relation"] = make_template_relation(
                    row["subject"], row["predicate"], row["object"]
                )

            if tag == "gemma4_31b_it":
                prompts = [build_prompt_gemma4(row, tokenizer) for row in chunk]
            else:
                prompts = [build_prompt_dictalm3(row) for row in chunk]

            if backend == "vllm":
                scores = _infer_vllm_batch(model, prompts, yes_ids, no_ids, tokenizer)
            else:
                scores = _infer_hf_batch(model, tokenizer, prompts, yes_ids, no_ids, BATCH_SIZE)

            for s in scores:
                fout.write(f"{s:.6f}\n")
            fout.flush()  # crash-safe: at most one chunk lost on failure
            written += len(chunk)

            if chunk_idx % LOG_EVERY_CHUNKS == 0 or written >= total_rows:
                elapsed = time.time() - t0
                rps     = (written - done) / elapsed if elapsed else 0
                eta     = (total_rows - written) / rps if rps else 0
                log.info(
                    f"[{tag}]  {written:,}/{total_rows:,}  "
                    f"({100 * written / total_rows:.1f}%)  "
                    f"{rps:.1f} rows/s  ETA {_fmt_dur(eta)}"
                )

    elapsed = time.time() - t0
    unload_llm(model, backend, log)
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
        for tag in MODEL_ORDER if tag in models_run
        for col in (MODELS[tag]["col_score"], MODELS[tag]["col_pred"])
    ]

    pred_handles = {
        tag: open(_pred_path(tag), encoding="utf-8")
        for tag in models_run if _pred_path(tag).exists()
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
    model_stats: dict, total_rows: int, total_time: float,
    models_run: list[str], log: logging.Logger,
) -> None:
    sep   = "=" * 80
    lines = [
        sep,
        "SILVER OPENSOURCE LLM CLASSIFICATION SUMMARY",
        f"Date          : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total rows    : {total_rows:,}",
        f"Wall time     : {_fmt_dur(total_time)}",
        f"Input CSV     : {SILVER_CSV}",
        f"Output CSV    : {OUT_CSV}",
        sep, "",
        "MODELS",
    ]
    for tag in MODEL_ORDER:
        cfg = MODELS[tag]
        if tag not in models_run:
            lines.append(f"  [{tag}]  SKIPPED"); continue
        st = model_stats.get(tag, {})
        lines += [
            f"  [{tag}]",
            f"    model_id   : {cfg['model_id']}",
            f"    config     : lang={cfg['lang']}  rel={cfg['rel']}  shot={cfg['n_shot']}  no-CoT",
            f"    threshold  : {cfg['threshold']}  (soft P(yes)/(P(yes)+P(no)))",
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
                if not v: continue
                total += 1
                if float(v) >= cfg["threshold"]: pos += 1
                else: neg += 1
        pct = 100 * pos / total if total else 0
        lines.append(
            f"  [{tag:<18}]  pos={pos:,} ({pct:.1f}%)  neg={neg:,} ({100-pct:.1f}%)  "
            f"total={total:,}  threshold={cfg['threshold']}"
        )
    lines += ["", sep]
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
        description="Classify silver with best opensource LLM configs (gemma4_31b_it + dictalm3)"
    )
    parser.add_argument("--models",   nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    parser.add_argument("--max-rows", type=int,  default=None)
    parser.add_argument("--skip-merge", action="store_true")
    args = parser.parse_args()

    for d in (OUT_DIR, CLASS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log = setup_logger(LOG_FILE)
    log.info("=" * 70)
    log.info("classify_silver_opensource_llm.py  started")
    log.info(f"  models    : {args.models}")
    log.info(f"  max_rows  : {args.max_rows or 'all'}")
    log.info(f"  input     : {SILVER_CSV}")
    log.info(f"  output    : {OUT_CSV}")
    log.info("=" * 70)

    log.info("[count]  counting silver rows …")
    total_rows = count_silver_rows(max_rows=args.max_rows)
    log.info(f"[count]  total_rows = {total_rows:,}")

    wall_start  = time.time()
    model_stats = {}
    models_run  = []

    for tag in MODEL_ORDER:
        if tag not in args.models:
            log.info(f"[{tag}]  skipped (not in --models)"); continue
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
    log.info("=" * 70)


if __name__ == "__main__":
    main()
