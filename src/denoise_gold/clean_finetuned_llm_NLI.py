"""
Apply all successfully fine-tuned NLI decoder models (from Hebrew_NLI project) to the
RE dataset using the HebNLI entailment format.

Input format for each row:
  premise    = text passage
  hypothesis = basic_relation  OR  template_relation column

NLI prompt: "משפט 1: {premise} משפט 2: {hypothesis}"
  → model generates one of: היסק (entailment) | סתירה (contradiction) | ניטרלי (neutral)
  → entailment maps to relation_present=1; contradiction/neutral → 0

Soft score = P(היסק) / (P(היסק) + P(סתירה) + P(ניטרלי)) from first-token logits.

Models used (LoRA adapters from Hebrew_NLI/output/, loaded via PEFT):
  gemma2_9b       — google/gemma-2-9b-it           (dev macro-F1=87.3, test=89.9)
  gemma3_12b      — google/gemma-3-12b-it          (dev macro-F1=85.1, test=88.6)
  dictalm24b      — dicta-il/DictaLM-3.0-24B-Think (dev macro-F1=87.6, test=90.2)
  dictalm24b_base    — dicta-il/DictaLM-3.0-24B-Base  (dev macro-F1=88.4, test=92.1)
  dictalm24b_base_v2 — same model, explicit Hebrew instruction + תשובה: cue
  aya32b          — CohereForAI/aya-expanse-32b     (test=92.0)
  mistral24b      — Mistral-Small-24B-Instruct-2501 (dev macro-F1=88.6, test=90.5)
  qwen35base      — Qwen/Qwen3.5-35B-A3B-Base      (dev macro-F1=86.9, test=90.4)
  gemma4_31b      — google/gemma-4-31B             (dev macro-F1=87.0, test=90.5)
  gemma4_26b      — google/gemma-4-26B-A4B         (dev macro-F1=85.3, test=89.0)
  gemma3_27b      — google/gemma-3-27b-it          (dev macro-F1=85.7, test=89.4)

Output: outputs/finetuned_LLM_NLI/
  classified.csv         — input CSV enriched with NLI prediction columns
  classify.log           — full run log
  summary.txt            — metrics tables (hard-label + soft-score + ensemble)
  error_analysis.txt     — per-combo errors and hard examples
  predicate_analysis.txt — per-predicate stratified metrics

Usage:
    CUDA_VISIBLE_DEVICES=5,6 python -m scripts_clean_data.clean_finetuned_llm_NLI
    CUDA_VISIBLE_DEVICES=5,6 python -m scripts_clean_data.clean_finetuned_llm_NLI --debug 20
    CUDA_VISIBLE_DEVICES=5,6 python -m scripts_clean_data.clean_finetuned_llm_NLI --models gemma2_9b gemma3_12b

Conda env: hre_finetuned_nli
    conda activate hre_finetuned_nli
    (includes transformers>=5 for qwen3_5_moe support, peft, flash-attn)
"""

import gc
import os
import re
import csv
import time
import logging
import argparse
from collections import defaultdict

import torch
import transformers
import peft as peft_lib
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths / macros
# ---------------------------------------------------------------------------

INPUT_FILE    = "data/prepared_gold_500.csv"
OUTPUT_SUBDIR = "outputs/finetuned_LLM_NLI"
OUTPUT_FILE   = f"{OUTPUT_SUBDIR}/classified.csv"
LOG_FILE      = f"{OUTPUT_SUBDIR}/classify.log"
SUMMARY_FILE  = f"{OUTPUT_SUBDIR}/summary.txt"
ERROR_FILE    = f"{OUTPUT_SUBDIR}/error_analysis.txt"
PRED_FILE     = f"{OUTPUT_SUBDIR}/predicate_analysis.txt"

LABEL_COL     = "relation_present"
PREDICATE_COL = "predicate"

RELATION_COLS = {
    "basic":    "basic_relation",
    "template": "template_relation",
}

# Hebrew NLI label tokens
NLI_ENTAIL  = "היסק"
NLI_CONTRA  = "סתירה"
NLI_NEUTRAL = "ניטרלי"

# Base path for LoRA checkpoint dirs in Hebrew_NLI
_HEB_NLI_OUT = "/path/to/Hebrew_NLI/output"

LLM_MODELS = [
    {"tag": "gemma2_9b",   "type": "instruct",
     "ckpt": f"{_HEB_NLI_OUT}/gemma-2-9b-it_hebnli/7920"},
    {"tag": "gemma3_12b",  "type": "instruct",
     "ckpt": f"{_HEB_NLI_OUT}/gemma-3-12b-it_hebnli/5280"},
    {"tag": "dictalm24b",  "type": "instruct",
     "ckpt": f"{_HEB_NLI_OUT}/DictaLM-3.0-24B-Thinking_hebnli/5280"},
    {"tag": "dictalm24b_base", "type": "base",
     "ckpt": f"{_HEB_NLI_OUT}/DictaLM-3.0-24B-Base_hebnli/7920"},
    {"tag": "dictalm24b_base_v2", "type": "base",
     "ckpt": f"{_HEB_NLI_OUT}/DictaLM-3.0-24B-Base_hebnli/7920",
     "user_prefix": "הוראה: קרא את המשפטים וקבע את היחס הלוגי. ענה רק במילה אחת: היסק, סתירה, או ניטרלי.\n\n",
     "user_suffix": "\nתשובה:"},
    {"tag": "aya32b",      "type": "instruct",
     "ckpt": f"{_HEB_NLI_OUT}/aya-expanse-32b_hebnli/7920"},
    {"tag": "mistral24b",  "type": "instruct",
     "ckpt": f"{_HEB_NLI_OUT}/Mistral-Small-24B-Instruct-2501_hebnli/7920"},
    # v2 variants — alternative prompts for models that failed the default format
    {"tag": "dictalm24b_v2", "type": "instruct",
     "ckpt": f"{_HEB_NLI_OUT}/DictaLM-3.0-24B-Thinking_hebnli/5280",
     "system": "אתה מודל הסקה לוגית בעברית. ענה תמיד ורק במילה אחת מהאפשרויות: 'היסק', 'סתירה', או 'ניטרלי'.",
     "user_suffix": "\nהאם הטענה נובעת מהטקסט? ענה: היסק, סתירה, או ניטרלי.",
     "max_new_tokens": 300,
     "parse_last": True},
    {"tag": "aya32b_v2",    "type": "instruct",
     "ckpt": f"{_HEB_NLI_OUT}/aya-expanse-32b_hebnli/7920",
     "system": "אתה מודל הסקה לוגית בעברית. ענה תמיד ורק במילה אחת מהאפשרויות: 'היסק', 'סתירה', או 'ניטרלי'.",
     "user_suffix": "\nהאם הטענה נובעת מהטקסט? ענה: היסק, סתירה, או ניטרלי.",
     "parse_yesno": True},
    {"tag": "mistral24b_v2", "type": "instruct",
     "ckpt": f"{_HEB_NLI_OUT}/Mistral-Small-24B-Instruct-2501_hebnli/7920",
     "system": "אתה מודל הסקה לוגית בעברית. קרא את הטקסט והטענה, והחלט אם הטענה נובעת מהטקסט (היסק), סותרת אותו (סתירה), או לא קשורה (ניטרלי). ענה תמיד ורק במילה אחת.",
     "user_suffix": "\nהאם הטענה נובעת מהטקסט? ענה: היסק, סתירה, או ניטרלי."},
    # v3: embed instruction in user turn only — avoids Mistral's broken system-role handling
    {"tag": "mistral24b_v3", "type": "instruct",
     "ckpt": f"{_HEB_NLI_OUT}/Mistral-Small-24B-Instruct-2501_hebnli/7920",
     "user_prefix": "אתה מודל הסקה לוגית בעברית. קרא את הטקסט והטענה, והחלט אם הטענה נובעת מהטקסט (היסק), סותרת אותו (סתירה), או לא קשורה (ניטרלי). ענה תמיד ורק במילה אחת.\n\n",
     "user_suffix": "\nהאם הטענה נובעת מהטקסט? ענה: היסק, סתירה, או ניטרלי."},
    # v4: raw-text format matching fine-tuning (no chat template, ### Answer: separator)
    {"tag": "mistral24b_v4", "type": "base",
     "ckpt": f"{_HEB_NLI_OUT}/Mistral-Small-24B-Instruct-2501_hebnli/7920",
     "user_suffix": " ### Answer:"},
    {"tag": "qwen35base",  "type": "base",
     "ckpt": f"{_HEB_NLI_OUT}/Qwen3.5-35B-A3B-Base_hebnli/5280"},
    # v2 2026-08-26: default raw format yields truncated echoes (313-363/500
    # unparseable on gold). Same explicit-instruction + תשובה: cue that fixed
    # dictalm24b_base.
    {"tag": "qwen35base_v2", "type": "base",
     "ckpt": f"{_HEB_NLI_OUT}/Qwen3.5-35B-A3B-Base_hebnli/5280",
     "user_prefix": "הוראה: קרא את המשפטים וקבע את היחס הלוגי. ענה רק במילה אחת: היסק, סתירה, או ניטרלי.\n\n",
     "user_suffix": "\nתשובה:"},
    {"tag": "gemma4_31b",  "type": "instruct",
     "ckpt": f"{_HEB_NLI_OUT}/gemma-4-31B_hebnli/7920"},
    # v2 2026-08-26: default format yields free-form prose (483/500 unparseable
    # on gold). Instruction embedded in the user turn (Gemma chat templates
    # reject a system role — same reason as mistral24b_v3).
    {"tag": "gemma4_31b_v2", "type": "instruct",
     "ckpt": f"{_HEB_NLI_OUT}/gemma-4-31B_hebnli/7920",
     "user_prefix": "אתה מודל הסקה לוגית בעברית. קרא את הטקסט והטענה, והחלט אם הטענה נובעת מהטקסט (היסק), סותרת אותו (סתירה), או לא קשורה (ניטרלי). ענה תמיד ורק במילה אחת.\n\n",
     "user_suffix": "\nהאם הטענה נובעת מהטקסט? ענה: היסק, סתירה, או ניטרלי."},
    {"tag": "gemma4_26b",  "type": "instruct",
     "ckpt": f"{_HEB_NLI_OUT}/gemma-4-26B-A4B_hebnli/7920"},
    {"tag": "gemma3_27b",  "type": "instruct",
     "ckpt": f"{_HEB_NLI_OUT}/gemma-3-27b-it_hebnli/7920"},
    # newly added decoders fine-tuned on HebNLI
    {"tag": "dictalm17b",  "type": "instruct",
     "ckpt": f"{_HEB_NLI_OUT}/DictaLM-3.0-1.7B-Instruct_hebnli/7920"},
]

LLM_BATCH_SIZE   = 16
LLM_MAX_NEW_TOKENS = 8        # NLI labels are short (≤5 tokens)
LLM_MAX_INPUT_LEN  = 512      # matches HebNLI fine-tuning context

SOFT_THRESHOLDS   = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_PRED_EXAMPLES = 5


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
    logger = logging.getLogger("clean_finetuned_llm_NLI")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:  return f"{h}h {m}m {s}s"
    if m:  return f"{m}m {s}s"
    return f"{s}s"


def _col(prefix: str, tag: str, rel: str) -> str:
    return f"{prefix}_{tag}_{rel}"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _nli_input_text(text: str, hypothesis: str) -> str:
    return f"משפט 1: {text} משפט 2: {hypothesis}"


def build_prompt(model_type: str, text: str, hypothesis: str, tokenizer,
                 system: str = "", user_suffix: str = "", user_prefix: str = "") -> str:
    nli_text = user_prefix + _nli_input_text(text, hypothesis) + user_suffix
    if model_type == "instruct":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": nli_text})
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prefix = f"[SYSTEM]{system}[/SYSTEM]\n" if system else ""
            return prefix + nli_text + "\n"
    else:
        # Base model: plain completion prompt (same format used during fine-tuning)
        # Don't add "\n" when user_suffix already provides the completion cue (e.g. " ### Answer:")
        return nli_text if user_suffix else nli_text + "\n"


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_nli(raw: str, parse_last: bool = False, parse_yesno: bool = False) -> str:
    """Map generated text to binary RE label. Entailment → '1', else → '0'."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    if parse_last:
        # Thinking models: answer appears after the reasoning — scan from the end
        for chunk in [cleaned[i:] for i in range(len(cleaned) - 1, -1, -1)]:
            if NLI_ENTAIL in chunk:
                return "1"
            if NLI_CONTRA in chunk or NLI_NEUTRAL in chunk:
                return "0"
        return "unknown"

    if NLI_ENTAIL in cleaned:
        return "1"
    if NLI_CONTRA in cleaned or NLI_NEUTRAL in cleaned:
        return "0"
    # Fallback: partial matches
    if "היס" in cleaned:
        return "1"
    if "סתיר" in cleaned or "ניטר" in cleaned:
        return "0"
    # Fallback: yes/no Hebrew for models fine-tuned with that format
    if parse_yesno:
        low = cleaned.strip()
        if low.startswith("כן"):
            return "1"
        if low.startswith("לא"):
            return "0"
    return "unknown"


# ---------------------------------------------------------------------------
# Soft score helpers
# ---------------------------------------------------------------------------

def get_nli_token_ids(tokenizer) -> tuple[list, list, list]:
    """
    Return (entail_ids, contra_ids, neutral_ids).
    Uses the FIRST token of each label surface form as the discriminating signal —
    correct even when the full label is multi-token (as it is in most tokenizers).
    """
    entail_vars  = [NLI_ENTAIL,  " " + NLI_ENTAIL,  "▁" + NLI_ENTAIL]
    contra_vars  = [NLI_CONTRA,  " " + NLI_CONTRA,  "▁" + NLI_CONTRA]
    neutral_vars = [NLI_NEUTRAL, " " + NLI_NEUTRAL, "▁" + NLI_NEUTRAL]
    entail_ids = set()
    contra_ids = set()
    neutral_ids = set()
    for v in entail_vars:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if toks:
            entail_ids.add(toks[0])
    for v in contra_vars:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if toks:
            contra_ids.add(toks[0])
    for v in neutral_vars:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if toks:
            neutral_ids.add(toks[0])
    # Remove any token that appears in more than one label set (ambiguous)
    ambiguous = (entail_ids & contra_ids) | (entail_ids & neutral_ids) | (contra_ids & neutral_ids)
    entail_ids  -= ambiguous
    contra_ids  -= ambiguous
    neutral_ids -= ambiguous
    return sorted(entail_ids), sorted(contra_ids), sorted(neutral_ids)


def compute_soft_scores(logits, entail_ids: list, contra_ids: list, neutral_ids: list) -> list:
    """
    logits: (batch, vocab) float tensor — first-token logits.
    Returns list of floats in [0,1]: P(entailment) / (P(entailment) + P(contra) + P(neutral)).
    Falls back to 0.5 if no NLI token IDs found.
    """
    all_nli = entail_ids + contra_ids + neutral_ids
    if not all_nli:
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
    denom = p_e + p_c + p_n + 1e-10
    return (p_e / denom).cpu().tolist()


# ---------------------------------------------------------------------------
# Model loading / unloading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, log: logging.Logger):
    """Load base model + LoRA adapter from local checkpoint using PEFT."""
    log.info(f"    checkpoint : {ckpt_path}")

    config = peft_lib.PeftConfig.from_pretrained(ckpt_path)
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

    # Load tokenizer from checkpoint (includes chat_template.jinja if present)
    tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "left"

    # Load base model
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
            log.info("    Falling back to Gemma4ForConditionalGeneration (VLM model)")
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

    # Apply LoRA adapter
    model = peft_lib.PeftModel.from_pretrained(base_model, ckpt_path, is_trainable=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info(f"    ready  ({n_params:.1f}B params total, LoRA adapter applied)")
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
# Generation
# ---------------------------------------------------------------------------

def classify_rows(
    rows: list,
    model,
    tokenizer,
    model_type: str,
    rel_type: str,
    entail_ids: list,
    contra_ids: list,
    neutral_ids: list,
    batch_size: int,
    log: logging.Logger,
    desc: str,
    system: str = "",
    user_suffix: str = "",
    user_prefix: str = "",
    max_new_tokens: int = LLM_MAX_NEW_TOKENS,
    parse_last: bool = False,
    parse_yesno: bool = False,
) -> tuple[list, list, list]:
    """Returns (parsed_labels, raw_outputs, soft_scores)."""
    rel_col  = RELATION_COLS[rel_type]
    prompts  = [build_prompt(model_type, r["text"], r.get(rel_col, ""), tokenizer,
                             system=system, user_suffix=user_suffix,
                             user_prefix=user_prefix) for r in rows]
    n        = len(prompts)
    device   = _input_device(model)

    parsed_all, raws_all, soft_all = [], [], []
    log_every      = max(1, (n // batch_size) // 4)
    effective_batch = batch_size

    hf_stop_kwargs = {}
    if model_type == "base":
        # For base models, stop generation at the first newline to get just the label
        nl_ids = set()
        for v in ["\n", "היסק", "סתירה", "ניטרלי"]:
            toks = tokenizer.encode(v, add_special_tokens=False)
            if len(toks) == 1:
                nl_ids.add(toks[0])
        if nl_ids:
            hf_stop_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + list(nl_ids)

    for batch_idx, start in enumerate(
        tqdm(range(0, n, batch_size), desc=f"    {desc}", leave=False,
             file=TqdmToLogger(log)), 1
    ):
        batch_prompts = prompts[start : start + batch_size]

        sub_parsed, sub_raws, sub_soft = [], [], []
        sub_start = 0
        while sub_start < len(batch_prompts):
            sub = batch_prompts[sub_start : sub_start + effective_batch]
            encodings = tokenizer(
                sub,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=LLM_MAX_INPUT_LEN,
            )
            enc_on_device = {k: v.to(device) for k, v in encodings.items()
                             if k != "token_type_ids"}
            input_len = enc_on_device["input_ids"].shape[1]
            try:
                with torch.no_grad():
                    output = model.generate(
                        **enc_on_device,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.pad_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **hf_stop_kwargs,
                    )
                # Soft score from first-token logits
                sub_soft.extend(
                    compute_soft_scores(output.scores[0], entail_ids, contra_ids, neutral_ids)
                )
                for seq in output.sequences:
                    raw = tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
                    sub_raws.append(raw)
                    sub_parsed.append(parse_nli(raw, parse_last=parse_last,
                                                parse_yesno=parse_yesno))
                sub_start += effective_batch
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                effective_batch = max(1, effective_batch // 2)
                log.warning(f"    OOM — reduced batch_size to {effective_batch}, retrying")

        soft_all.extend(sub_soft)
        raws_all.extend(sub_raws)
        parsed_all.extend(sub_parsed)

        if batch_idx % log_every == 0 or (start + batch_size) >= n:
            done = min(start + batch_size, n)
            log.info(f"    progress: {done}/{n} ({100*done/n:.0f}%)")

    return parsed_all, raws_all, soft_all


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics_hard(labels: list, gold: list) -> dict:
    TP = FP = FN = TN = unknown = 0
    for pred, g in zip(labels, gold):
        if pred == "unknown":
            unknown += 1
        pos      = pred == "1"
        gold_pos = g == "1"
        if pos and gold_pos:       TP += 1
        elif pos and not gold_pos: FP += 1
        elif not pos and gold_pos: FN += 1
        else:                      TN += 1
    total     = TP + FP + FN + TN
    accuracy  = (TP + TN) / total          if total         else 0
    precision = TP / (TP + FP)            if (TP + FP)     else 0
    recall    = TP / (TP + FN)            if (TP + FN)     else 0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) else 0
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "unknown": unknown}


def compute_metrics_soft(scores: list, gold: list, threshold: float) -> dict:
    TP = FP = FN = TN = 0
    for sc, g in zip(scores, gold):
        pred     = sc >= threshold
        gold_pos = g == "1"
        if pred and gold_pos:       TP += 1
        elif pred and not gold_pos: FP += 1
        elif not pred and gold_pos: FN += 1
        else:                       TN += 1
    total     = TP + FP + FN + TN
    accuracy  = (TP + TN) / total          if total         else 0
    precision = TP / (TP + FP)            if (TP + FP)     else 0
    recall    = TP / (TP + FN)            if (TP + FN)     else 0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) else 0
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Ensemble voting
# ---------------------------------------------------------------------------

def compute_majority(rows: list, col_names: list) -> list:
    results = []
    for row in rows:
        votes = [row[c] for c in col_names if c in row and row[c] in ("1", "0")]
        ones  = votes.count("1")
        zeros = votes.count("0")
        if ones > zeros:   results.append("1")
        elif zeros > ones: results.append("0")
        else:              results.append("unknown")
    return results


def compute_soft_ensemble(rows: list, col_names: list) -> list:
    results = []
    for row in rows:
        vals = []
        for c in col_names:
            v = row.get(c)
            if v is not None:
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        results.append(sum(vals) / len(vals) if vals else 0.5)
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(run_stats: list, majority_stats: list, soft_ensemble_stats: list,
                  summary_path: str, log: logging.Logger, total: float):
    lines = []
    W = 120
    lines.append("=" * W)
    lines.append("FINETUNED NLI LLM CLASSIFICATION SUMMARY")
    lines.append(f"Total wall time: {_fmt_duration(total)}")
    lines.append("Models (LoRA fine-tuned on HebNLI):")
    for m in LLM_MODELS:
        lines.append(f"  [{m['tag']}]  {m['ckpt']}")
    lines.append("=" * W)

    model_times: dict = {}
    for s in run_stats:
        model_times.setdefault(s["model"], 0.0)
        model_times[s["model"]] += s["time"]
    lines.append("")
    lines.append("Per-model total inference time:")
    for m, t in model_times.items():
        r = sum(s["n_rows"] for s in run_stats if s["model"] == m)
        rps = r / t if t > 0 else float("inf")
        lines.append(f"  {m:<14}  {_fmt_duration(t)}  ({rps:.1f} rows/s)")

    lines.append("")
    lines.append("Hard-label metrics (NLI entailment → relation present):")
    hdr = (
        f"  {'model':<14} {'rel_type':<10}  "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
        f"{'unk':>4}  {'rows/s':>7}  {'time':>8}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for s in run_stats:
        m   = s["metrics"]
        rps = s["n_rows"] / s["time"] if s["time"] > 0 else float("inf")
        lines.append(
            f"  {s['model']:<14} {s['rel_type']:<10}  "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
            f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
            f"{m['unknown']:>4}  {rps:>7.1f}  {_fmt_duration(s['time']):>8}"
        )

    lines.append("")
    lines.append("Soft-score metrics (best F1 threshold per combo):")
    hdr2 = (
        f"  {'model':<14} {'rel_type':<10}  "
        f"{'best_thr':>8}  {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}"
    )
    lines.append(hdr2)
    lines.append("  " + "-" * (len(hdr2) - 2))
    for s in run_stats:
        if not s["soft_metrics"]:
            continue
        best_t, best_m = max(s["soft_metrics"].items(), key=lambda kv: kv[1]["f1"])
        lines.append(
            f"  {s['model']:<14} {s['rel_type']:<10}  "
            f"{best_t:>8.2f}  "
            f"{best_m['accuracy']:>6.3f} {best_m['precision']:>6.3f} "
            f"{best_m['recall']:>6.3f} {best_m['f1']:>6.3f}  "
            f"{best_m['TP']:>5} {best_m['FP']:>5} {best_m['FN']:>5} {best_m['TN']:>5}"
        )

    if majority_stats:
        lines.append("")
        lines.append("Majority-vote ensemble columns:")
        hdr3 = (
            f"  {'column':<45}  "
            f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
            f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
            f"{'unk':>4}  {'voters':>6}"
        )
        lines.append(hdr3)
        lines.append("  " + "-" * (len(hdr3) - 2))
        for s in majority_stats:
            m = s["metrics"]
            lines.append(
                f"  {s['col']:<45}  "
                f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
                f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
                f"{m['unknown']:>4}  {s['n_voters']:>6}"
            )

    if soft_ensemble_stats:
        lines.append("")
        lines.append("Soft-score ensemble columns (best F1 threshold):")
        hdr4 = (
            f"  {'column':<45}  {'best_thr':>8}  "
            f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
            f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  {'voters':>6}"
        )
        lines.append(hdr4)
        lines.append("  " + "-" * (len(hdr4) - 2))
        for s in soft_ensemble_stats:
            best_t, best_m = max(s["metrics_by_threshold"].items(), key=lambda kv: kv[1]["f1"])
            lines.append(
                f"  {s['col']:<45}  {best_t:>8.2f}  "
                f"{best_m['accuracy']:>6.3f} {best_m['precision']:>6.3f} "
                f"{best_m['recall']:>6.3f} {best_m['f1']:>6.3f}  "
                f"{best_m['TP']:>5} {best_m['FP']:>5} {best_m['FN']:>5} {best_m['TN']:>5}  "
                f"{s['n_voters']:>6}"
            )

    lines.append("=" * W)
    text = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    for line in lines:
        log.info(line)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def write_error_analysis(rows: list, run_stats: list, gold: list,
                         error_path: str, log: logging.Logger, max_examples: int = 10):
    W = 110
    n_rows = len(rows)
    n_pos  = gold.count("1")

    best_stat = max(run_stats, key=lambda s: s["metrics"]["f1"])
    best_clean_col = _col("nli_clean", best_stat["model"], best_stat["rel_type"])

    lines = []
    lines.append("=" * W)
    lines.append("  FINETUNED NLI LLM ERROR ANALYSIS")
    lines.append(f"  Total rows         : {n_rows}")
    lines.append(f"  Positive / Negative: {n_pos} / {n_rows-n_pos}  "
                 f"({100*n_pos/n_rows:.1f}% / {100*(n_rows-n_pos)/n_rows:.1f}%)")
    lines.append(f"  Best combo (F1={best_stat['metrics']['f1']:.3f}): "
                 f"{best_stat['model']} / {best_stat['rel_type']}")
    lines.append("=" * W)

    lines.append("")
    lines.append("=" * W)
    lines.append("  SECTION 1 — PER-COMBO ERROR COUNTS")
    lines.append("=" * W)
    hdr = f"  {'combo':<35} {'F1':>6} {'FP':>5} {'FN':>5} {'unk':>5}  FP-rate  FN-rate"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for s in sorted(run_stats, key=lambda x: x["metrics"]["f1"], reverse=True):
        m     = s["metrics"]
        combo = f"{s['model']}/{s['rel_type']}"
        fp_rate = m["FP"] / (m["FP"] + m["TN"]) if (m["FP"] + m["TN"]) else 0
        fn_rate = m["FN"] / (m["FN"] + m["TP"]) if (m["FN"] + m["TP"]) else 0
        lines.append(
            f"  {combo:<35} {m['f1']:>6.3f} {m['FP']:>5} {m['FN']:>5} {m['unknown']:>5}"
            f"  {fp_rate:.3f}    {fn_rate:.3f}"
        )

    lines.append("")
    lines.append("=" * W)
    lines.append("  SECTION 2 — RELATION TYPE COMPARISON")
    lines.append("=" * W)
    for rt in RELATION_COLS:
        rt_stats = [s for s in run_stats if s["rel_type"] == rt]
        mean_f1 = sum(s["metrics"]["f1"] for s in rt_stats) / len(rt_stats) if rt_stats else 0
        lines.append(f"  {rt:<12}  mean F1={mean_f1:.4f}  (across {len(rt_stats)} models)")

    for section_title, pred_val, gold_val in [
        ("SECTION 3 — FALSE POSITIVES  [best combo]", "1", "0"),
        ("SECTION 4 — FALSE NEGATIVES  [best combo]", "0", "1"),
    ]:
        examples = [(i, r) for i, r in enumerate(rows)
                    if r.get(best_clean_col) == pred_val and r[LABEL_COL] == gold_val]
        lines.append("")
        lines.append("=" * W)
        lines.append(f"  {section_title}  — {len(examples)} total")
        lines.append("=" * W)
        for _, (i, row) in enumerate(examples[:max_examples]):
            snip = row["text"].replace("\n", " ")[:150]
            rel  = row.get(RELATION_COLS[best_stat["rel_type"]], "")
            raw  = row.get(_col("nli_raw", best_stat["model"], best_stat["rel_type"]), "")
            lines.append(f"  [row {i}]")
            lines.append(f"    text      : {snip!r}")
            lines.append(f"    relation  : {rel!r}")
            lines.append(f"    raw output: {raw!r}")
        if len(examples) > max_examples:
            lines.append(f"  ... and {len(examples) - max_examples} more")

    nc_clean_cols = [_col("nli_clean", s["model"], s["rel_type"]) for s in run_stats]
    hard = [i for i, r in enumerate(rows)
            if all(r.get(c, "unknown") != gold[i] for c in nc_clean_cols if c in r)]
    lines.append("")
    lines.append("=" * W)
    lines.append(f"  SECTION 5 — HARD ROWS: all combos wrong — {len(hard)} ({100*len(hard)/n_rows:.1f}%)")
    lines.append("=" * W)
    for i in hard[:max_examples]:
        row  = rows[i]
        snip = row["text"].replace("\n", " ")[:150]
        lines.append(f"  [row {i}]  gold={gold[i]}")
        lines.append(f"    text: {snip!r}")
    if len(hard) > max_examples:
        lines.append(f"  ... and {len(hard) - max_examples} more")

    # Soft score distribution for best model
    best_soft = max(
        (s for s in run_stats if s["soft_metrics"]),
        key=lambda s: max(m["f1"] for m in s["soft_metrics"].values()),
        default=None,
    )
    if best_soft:
        score_col = _col("nli_score", best_soft["model"], best_soft["rel_type"])
        lines.append("")
        lines.append("=" * W)
        lines.append(f"  SECTION 6 — SOFT SCORE DISTRIBUTION  [{score_col}]")
        lines.append("=" * W)
        bins = [(0.0, 0.3, "<0.3"), (0.3, 0.5, "0.3-0.5"),
                (0.5, 0.7, "0.5-0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.01, "≥0.9")]
        for lo, hi, label in bins:
            pos_n = sum(1 for r, g in zip(rows, gold)
                        if lo <= float(r.get(score_col, 0.5)) < hi and g == "1")
            neg_n = sum(1 for r, g in zip(rows, gold)
                        if lo <= float(r.get(score_col, 0.5)) < hi and g == "0")
            lines.append(f"  [{label}]  pos={pos_n}  neg={neg_n}")

    lines.append("=" * W)
    with open(error_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Error analysis written to {error_path}")


# ---------------------------------------------------------------------------
# Predicate-stratified analysis
# ---------------------------------------------------------------------------

def write_predicate_stratified(rows: list, run_stats: list, gold: list,
                                pred_path: str, log: logging.Logger):
    W = 130
    n_rows = len(rows)

    pred_groups: dict = defaultdict(list)
    for i, row in enumerate(rows):
        pred_groups[row.get(PREDICATE_COL, "UNKNOWN")].append(i)

    top_combos = sorted(run_stats, key=lambda s: s["metrics"]["f1"], reverse=True)[:10]

    lines = []
    lines.append("=" * W)
    lines.append("  PREDICATE-STRATIFIED METRICS  (NLI fine-tuned decoders)")
    lines.append(f"  Predicates with >= {MIN_PRED_EXAMPLES} examples: "
                 f"{sum(1 for v in pred_groups.values() if len(v) >= MIN_PRED_EXAMPLES)}")
    lines.append("=" * W)

    lines.append("")
    lines.append("Predicate distribution (top 20 by count):")
    pred_pos: dict = {}
    for pred, idxs in pred_groups.items():
        pred_pos[pred] = sum(1 for i in idxs if gold[i] == "1")
    sorted_preds = sorted(pred_groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    hdr0 = f"  {'predicate':<35} {'n':>5} {'pos':>5} {'neg':>5} {'pos%':>6}"
    lines.append(hdr0)
    lines.append("  " + "-" * (len(hdr0) - 2))
    for p, idxs in sorted_preds[:20]:
        n  = len(idxs)
        ps = pred_pos[p]
        ns = n - ps
        lines.append(f"  {p:<35} {n:>5} {ps:>5} {ns:>5} {100*ps/n:>5.0f}%")

    lines.append("")
    lines.append("=" * W)
    lines.append(f"  PER-PREDICATE F1  (predicates with ≥ {MIN_PRED_EXAMPLES} examples, top combos)")
    lines.append("=" * W)
    combo_labels = [f"{s['model']}/{s['rel_type']}" for s in top_combos]
    hdr_parts = [f"  {'predicate':<35} {'n':>5}"]
    for lbl in combo_labels:
        hdr_parts.append(f" {lbl[:16]:>16}")
    lines.append("".join(hdr_parts))
    lines.append("  " + "-" * (len("".join(hdr_parts)) - 2))

    pred_f1_summary: list = []
    for pred, idxs in sorted_preds:
        if len(idxs) < MIN_PRED_EXAMPLES:
            continue
        pred_golds = [gold[i] for i in idxs]
        combo_f1s  = []
        row_parts  = [f"  {pred:<35} {len(idxs):>5}"]
        for s in top_combos:
            clean_col   = _col("nli_clean", s["model"], s["rel_type"])
            pred_labels = [rows[i].get(clean_col, "unknown") for i in idxs]
            m           = compute_metrics_hard(pred_labels, pred_golds)
            combo_f1s.append(m["f1"])
            row_parts.append(f" {m['f1']:>16.3f}")
        mean_f1 = sum(combo_f1s) / len(combo_f1s) if combo_f1s else 0
        pred_f1_summary.append((pred, mean_f1, len(idxs)))
        lines.append("".join(row_parts))

    lines.append("")
    lines.append("=" * W)
    lines.append("  EASIEST PREDICATES:")
    for pred, mean_f1, n in sorted(pred_f1_summary, key=lambda x: -x[1])[:10]:
        lines.append(f"    {pred:<35}  mean F1={mean_f1:.3f}  n={n}")
    lines.append("")
    lines.append("  HARDEST PREDICATES:")
    for pred, mean_f1, n in sorted(pred_f1_summary, key=lambda x: x[1])[:10]:
        lines.append(f"    {pred:<35}  mean F1={mean_f1:.3f}  n={n}")

    lines.append("=" * W)
    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Predicate analysis written to {pred_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RE classification using Hebrew_NLI fine-tuned NLI decoder models"
    )
    parser.add_argument("--input",      default=INPUT_FILE)
    parser.add_argument("--output",     default=OUTPUT_FILE)
    parser.add_argument("--log",        default=LOG_FILE)
    parser.add_argument("--summary",    default=SUMMARY_FILE)
    parser.add_argument("--error",      default=ERROR_FILE)
    parser.add_argument("--pred-file",  default=PRED_FILE)
    parser.add_argument("--label-col",  default=LABEL_COL)
    parser.add_argument("--batch-size", type=int, default=LLM_BATCH_SIZE)
    parser.add_argument("--debug",      type=int, default=0,
                        help="Run on first N rows only (0 = full dataset)")
    parser.add_argument("--models",     nargs="+", default=None,
                        help="Subset of model tags to run (default: all)")
    args = parser.parse_args()

    base        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path  = os.path.join(base, args.input)
    output_path = os.path.join(base, args.output)
    log_path    = os.path.join(base, args.log)
    sum_path    = os.path.join(base, args.summary)
    err_path    = os.path.join(base, args.error)
    pred_path   = os.path.join(base, args.pred_file)

    for p in (output_path, log_path, sum_path, err_path, pred_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    log = setup_logger(log_path)
    wall_start = time.time()

    active_models = [m for m in LLM_MODELS
                     if args.models is None or m["tag"] in args.models]
    if not active_models:
        raise ValueError(f"No models matched --models {args.models}. "
                         f"Available tags: {[m['tag'] for m in LLM_MODELS]}")

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log.info("=" * 70)
    log.info("clean_finetuned_llm_NLI.py  started")
    log.info(f"  input:      {input_path}")
    log.info(f"  output dir: {os.path.join(base, OUTPUT_SUBDIR)}")
    log.info(f"  label col:  {args.label_col}")
    log.info(f"  GPUs:       {n_gpus}")
    log.info(f"  batch size: {args.batch_size}")
    log.info(f"  debug rows: {args.debug if args.debug else 'all'}")
    log.info(f"  models ({len(active_models)}):")
    for m in active_models:
        log.info(f"    [{m['tag']}]  {m['ckpt']}  (type={m['type']})")
    log.info("=" * 70)

    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if args.debug:
        rows = rows[:args.debug]
        log.info(f"[debug]  truncated to first {args.debug} rows")
    n_rows = len(rows)
    gold   = [r[args.label_col] for r in rows]
    n_pos  = gold.count("1")
    log.info(f"[load]  {n_rows} rows  positive={n_pos}  negative={n_rows-n_pos}")

    available = set(rows[0].keys())
    if args.label_col not in available:
        raise ValueError(f"Label column '{args.label_col}' not in CSV.")
    if PREDICATE_COL not in available:
        log.warning(f"Predicate column '{PREDICATE_COL}' not found — predicate analysis will be skipped.")
    active_rel_types = [rt for rt, col in RELATION_COLS.items() if col in available]
    if not active_rel_types:
        raise ValueError(f"None of the relation columns {list(RELATION_COLS.values())} found in CSV.")
    log.info(f"  active relation types: {active_rel_types}")

    # Resume: merge previously-computed columns from output checkpoint
    if os.path.exists(output_path):
        log.info(f"[resume]  checkpoint found: {output_path}")
        with open(output_path, encoding="utf-8-sig") as f:
            ckpt_rows = list(csv.DictReader(f))
        if args.debug:
            ckpt_rows = ckpt_rows[:args.debug]
        if len(ckpt_rows) == n_rows:
            ckpt_extra = set(ckpt_rows[0].keys()) - set(rows[0].keys())
            if ckpt_extra:
                for r, cr in zip(rows, ckpt_rows):
                    for col in ckpt_extra:
                        r[col] = cr[col]
                log.info(f"[resume]  merged {len(ckpt_extra)} columns — will skip completed combos")
        else:
            log.warning(f"[resume]  row-count mismatch ({len(ckpt_rows)} vs {n_rows}) — ignoring checkpoint")

    run_stats: list = []

    for model_idx, model_cfg in enumerate(active_models, 1):
        tag            = model_cfg["tag"]
        model_type     = model_cfg["type"]
        ckpt_path      = model_cfg["ckpt"]
        m_system       = model_cfg.get("system", "")
        m_user_suffix  = model_cfg.get("user_suffix", "")
        m_user_prefix  = model_cfg.get("user_prefix", "")
        m_max_new_tok  = model_cfg.get("max_new_tokens", LLM_MAX_NEW_TOKENS)
        m_parse_last   = model_cfg.get("parse_last", False)
        m_parse_yesno  = model_cfg.get("parse_yesno", False)

        # Skip model entirely if all combos are already in checkpoint
        _combos = [_col("nli_clean", tag, rt) for rt in active_rel_types]
        if all(rows[0].get(c, "") != "" for c in _combos):
            log.info(f"[model {model_idx}/{len(active_models)}]  {tag}  — all combos in checkpoint, skipping")
            for rt in active_rel_types:
                clean_col = _col("nli_clean", tag, rt)
                score_col = _col("nli_score", tag, rt)
                p  = [r.get(clean_col, "unknown") if r.get(clean_col, "") in ("0", "1") else "unknown"
                      for r in rows]
                ss = [float(r[score_col]) if r.get(score_col, "") != "" else 0.5 for r in rows]
                m  = compute_metrics_hard(p, gold)
                sm = {thr: compute_metrics_soft(ss, gold, thr) for thr in SOFT_THRESHOLDS}
                run_stats.append({"model": tag, "rel_type": rt,
                                  "time": 0.0, "n_rows": n_rows, "metrics": m, "soft_metrics": sm})
                log.info(f"  [skip]  {tag}/{rt}  F1={m['f1']:.3f}")
            continue

        log.info("-" * 70)
        log.info(f"[model {model_idx}/{len(active_models)}]  {tag}  ({model_type})")
        t_model = time.time()
        model, tokenizer = load_model(ckpt_path, log)
        entail_ids, contra_ids, neutral_ids = get_nli_token_ids(tokenizer)
        log.info(f"    entail_ids ({len(entail_ids)}): {entail_ids}")
        log.info(f"    contra_ids ({len(contra_ids)}):  {contra_ids}")
        log.info(f"    neutral_ids ({len(neutral_ids)}): {neutral_ids}")

        for rt in active_rel_types:
            clean_col = _col("nli_clean", tag, rt)
            raw_col   = _col("nli_raw",   tag, rt)
            score_col = _col("nli_score", tag, rt)
            desc      = f"{tag}/{rt}"

            if rows[0].get(clean_col, "") != "":
                log.info(f"  [skip]  {desc}  (already in checkpoint)")
                p  = [r.get(clean_col, "unknown") if r.get(clean_col, "") in ("0", "1") else "unknown"
                      for r in rows]
                ss = [float(r[score_col]) if r.get(score_col, "") != "" else 0.5 for r in rows]
                m  = compute_metrics_hard(p, gold)
                sm = {thr: compute_metrics_soft(ss, gold, thr) for thr in SOFT_THRESHOLDS}
                run_stats.append({"model": tag, "rel_type": rt,
                                  "time": 0.0, "n_rows": n_rows, "metrics": m, "soft_metrics": sm})
                continue

            log.info("*" * 70)
            log.info(f"  COMBO: {desc}")
            log.info(f"    rel col: {RELATION_COLS[rt]}")
            log.info(f"    output cols: {clean_col} | {raw_col} | {score_col}")

            # Show example prompt
            ex_prompt = build_prompt(model_type, rows[0]["text"],
                                     rows[0].get(RELATION_COLS[rt], ""), tokenizer,
                                     system=m_system, user_suffix=m_user_suffix,
                                     user_prefix=m_user_prefix)
            log.info(f"    example prompt (row 0):\n{'-'*40}\n{ex_prompt[:500]}\n{'-'*40}")

            t0 = time.time()
            parsed, raws, soft_scores = classify_rows(
                rows, model, tokenizer, model_type, rt,
                entail_ids, contra_ids, neutral_ids,
                batch_size=args.batch_size, log=log, desc=desc,
                system=m_system, user_suffix=m_user_suffix, user_prefix=m_user_prefix,
                max_new_tokens=m_max_new_tok,
                parse_last=m_parse_last, parse_yesno=m_parse_yesno,
            )
            elapsed = time.time() - t0

            for r, label, raw, sc in zip(rows, parsed, raws, soft_scores):
                r[clean_col] = label
                r[raw_col]   = raw
                r[score_col] = f"{sc:.4f}"

            metrics = compute_metrics_hard(parsed, gold)
            soft_metrics = {thr: compute_metrics_soft(soft_scores, gold, thr)
                            for thr in SOFT_THRESHOLDS}
            rps = n_rows / elapsed if elapsed > 0 else float("inf")
            m   = metrics
            log.info(f"  {'='*66}")
            log.info(f"  RESULT  {desc}")
            log.info(f"  {'─'*66}")
            log.info(f"  acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
                     f"rec={m['recall']:.3f}  f1={m['f1']:.3f}")
            log.info(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}  "
                     f"unk={m['unknown']}  {rps:.1f} rows/s  {_fmt_duration(elapsed)}")
            best_soft_t, best_soft_m = max(soft_metrics.items(), key=lambda kv: kv[1]["f1"])
            log.info(f"  best soft: thresh={best_soft_t}  f1={best_soft_m['f1']:.3f}  "
                     f"prec={best_soft_m['precision']:.3f}  rec={best_soft_m['recall']:.3f}")
            log.info(f"  examples (first 3):")
            for r, label, raw, sc in zip(rows[:3], parsed[:3], raws[:3], soft_scores[:3]):
                log.info(f"    hyp={r.get(RELATION_COLS[rt], '')!r}")
                log.info(f"    raw={raw!r}  parsed={label}  score={sc:.3f}  gold={r[args.label_col]}")

            run_stats.append({"model": tag, "rel_type": rt,
                              "time": elapsed, "n_rows": n_rows,
                              "metrics": metrics, "soft_metrics": soft_metrics})

        model_total = time.time() - t_model
        unload_model(model, log)
        log.info(f"  [{tag}] total time: {_fmt_duration(model_total)}")

        # Checkpoint after each model
        fieldnames = list(dict.fromkeys(rows[0].keys()))
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"[checkpoint]  {n_rows} rows → {output_path}")

    # Ensemble columns
    log.info("=" * 70)
    log.info("[ensemble]  computing majority / soft-ensemble columns")
    majority_stats:      list = []
    soft_ensemble_stats: list = []

    all_clean_cols = [_col("nli_clean", s["model"], s["rel_type"]) for s in run_stats]
    all_score_cols = [_col("nli_score", s["model"], s["rel_type"]) for s in run_stats]

    # Per relation type: majority + soft ensemble across all models
    for rt in active_rel_types:
        rt_clean = [_col("nli_clean", s["model"], rt) for s in run_stats if s["rel_type"] == rt]
        rt_score = [_col("nli_score", s["model"], rt) for s in run_stats if s["rel_type"] == rt]

        maj_col = f"nli_majority_{rt}"
        maj_preds = compute_majority(rows, rt_clean)
        for r, v in zip(rows, maj_preds):
            r[maj_col] = v
        maj_m = compute_metrics_hard(maj_preds, gold)
        majority_stats.append({"col": maj_col, "n_voters": len(rt_clean), "metrics": maj_m})
        log.info(f"  {maj_col}:  F1={maj_m['f1']:.3f}")

        ens_col = f"nli_ens_soft_{rt}"
        ens_scores = compute_soft_ensemble(rows, rt_score)
        for r, sc in zip(rows, ens_scores):
            r[ens_col] = f"{sc:.4f}"
        ens_mbt = {thr: compute_metrics_soft(ens_scores, gold, thr) for thr in SOFT_THRESHOLDS}
        soft_ensemble_stats.append({"col": ens_col, "n_voters": len(rt_score),
                                    "metrics_by_threshold": ens_mbt})
        best_t, best_m = max(ens_mbt.items(), key=lambda kv: kv[1]["f1"])
        log.info(f"  {ens_col}:  best F1={best_m['f1']:.3f}  @ thresh={best_t}")

    # Global majority across all combos
    glob_maj_col = "nli_majority_all"
    glob_maj_preds = compute_majority(rows, all_clean_cols)
    for r, v in zip(rows, glob_maj_preds):
        r[glob_maj_col] = v
    glob_maj_m = compute_metrics_hard(glob_maj_preds, gold)
    majority_stats.append({"col": glob_maj_col, "n_voters": len(all_clean_cols), "metrics": glob_maj_m})
    log.info(f"  {glob_maj_col}:  F1={glob_maj_m['f1']:.3f}")

    # Global soft ensemble
    glob_ens_col = "nli_ens_soft_all"
    glob_ens_scores = compute_soft_ensemble(rows, all_score_cols)
    for r, sc in zip(rows, glob_ens_scores):
        r[glob_ens_col] = f"{sc:.4f}"
    glob_ens_mbt = {thr: compute_metrics_soft(glob_ens_scores, gold, thr) for thr in SOFT_THRESHOLDS}
    soft_ensemble_stats.append({"col": glob_ens_col, "n_voters": len(all_score_cols),
                                "metrics_by_threshold": glob_ens_mbt})
    best_t, best_m = max(glob_ens_mbt.items(), key=lambda kv: kv[1]["f1"])
    log.info(f"  {glob_ens_col}:  best F1={best_m['f1']:.3f}  @ thresh={best_t}")

    # Final save
    fieldnames = list(dict.fromkeys(rows[0].keys()))
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"[save]  {n_rows} rows → {output_path}")

    total = time.time() - wall_start
    write_summary(run_stats, majority_stats, soft_ensemble_stats, sum_path, log, total)

    write_error_analysis(rows, run_stats, gold, err_path, log)

    if PREDICATE_COL in available:
        write_predicate_stratified(rows, run_stats, gold, pred_path, log)
    else:
        log.info(f"[skip]  predicate analysis (column '{PREDICATE_COL}' not in CSV)")

    log.info("=" * 70)
    log.info(f"Done.  Total wall time: {_fmt_duration(total)}")
    log.info(f"  {output_path}")
    log.info(f"  {sum_path}")
    log.info(f"  {err_path}")
    log.info(f"  {pred_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
