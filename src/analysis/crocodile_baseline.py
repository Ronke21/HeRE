"""
Stage 1 (E): the CROCODILE baseline on the v3 gold-500.

The paper builds on CROCODILE, which "applies a single lightweight
XLM-RoBERTa NLI filter", but never reports what that filter scores on the
Hebrew gold set. Without it there is no answer to the obvious reviewer
question: how much better is a five-strategy study than the thing it replaces?

Reproduced faithfully from the pinned CROCODILE commit
(3c94898, `filter_relations.py`, recovered from .git/modules/crocodile):

  * model      joeddav/xlm-roberta-large-xnli, fp16 (`model.half()`)
  * premise    the article text
  * hypothesis " ".join([subject, predicate, object])  -- surface-form
               concatenation, which is exactly HeRE's `basic_relation` column
  * rule       P(entailment) > 0.75
               (the `argmax` branch above it is Korean-only; Hebrew takes this
               one)
  * tokenizer  max_length=256, truncation_strategy='only_first',
               padding='longest'

Two numbers are reported, and both belong in the paper:

  1. **as-published** -- the native 0.75 threshold. This is what a practitioner
     actually gets out of the box, and it is the honest baseline.
  2. **best-threshold** -- threshold swept like every other strategy in
     Table 3, which reports best-F1 per model. Reporting only (1) against
     tuned competitors would be an unfair comparison.

Also evaluates the template/llm hypothesis variants, so the paper can separate
"the filter is weak" from "the hypothesis format is weak".

Labels: adjudicated annotator1_Ron (open decision #3, settled 2026-08-16).

Usage:
    python -m post_rebuttal_and_camera_ready.analysis.crocodile_baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOLD_V3 = REPO / "post_rebuttal_and_camera_ready" / "data_v3" / "prepared_gold_500_v3.csv"
ADJUDICATED = REPO / "rebuttal" / "final_datasets" / "gold_validation_set.csv"
OUT = HERE / "stage0_results"

MODEL = "joeddav/xlm-roberta-large-xnli"
CROCODILE_THRESHOLD = 0.75
MAX_LEN = 256
LABEL_COL = "annotator1_Ron"
JOIN_KEYS = ["docid", "predicate", "object"]
HYPOTHESES = ["basic_relation", "template_relation", "llm_relation"]
N_BOOT = 10_000


def _f1(y, p):
    tp = float(np.sum((y == 1) & (p == 1)))
    fp = float(np.sum((y == 0) & (p == 1)))
    fn = float(np.sum((y == 1) & (p == 0)))
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d > 0 else 0.0


def _prf(y, p):
    tp = float(np.sum((y == 1) & (p == 1)))
    fp = float(np.sum((y == 0) & (p == 1)))
    fn = float(np.sum((y == 1) & (p == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, _f1(y, p), float(np.mean(y == p))


def bootstrap_ci(y, p, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    s = np.array([_f1(y[i], p[i]) for i in idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def macro_f1(df, y, p):
    out = []
    tmp = df.assign(_y=y, _p=p)
    for _, g in tmp.groupby("predicate"):
        out.append(_f1(g["_y"].to_numpy(), g["_p"].to_numpy()))
    return float(np.mean(out)) if out else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(GOLD_V3, encoding="utf-8-sig")

    # adjudicated label (same swap as stage0_ci_and_macro.py)
    adj = pd.read_csv(ADJUDICATED, encoding="utf-8-sig")
    adj = adj[adj[LABEL_COL].notna()]
    key_map = dict(zip(adj[JOIN_KEYS].astype(str).agg("||".join, axis=1),
                       adj[LABEL_COL].astype(int)))
    k = df[JOIN_KEYS].astype(str).agg("||".join, axis=1)
    mapped = k.map(key_map)
    print(f"[label] adjudicated: {mapped.notna().sum()}/{len(df)} joined, "
          f"{int((mapped.notna() & (mapped != df['relation_present'])).sum())} changed")
    df["label"] = mapped.fillna(df["relation_present"]).astype(int)
    y = df["label"].to_numpy()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — submit this to a GPU lane")

    print(f"[model] {MODEL}")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.cuda().eval().half()          # .half() matches CROCODILE
    ent_idx = next(v for kk, v in model.config.label2id.items()
                   if kk.lower() == "entailment")
    print(f"[model] entailment index = {ent_idx}  labels={model.config.label2id}")

    results, all_scores = [], {}
    for hyp in HYPOTHESES:
        pairs = list(zip(df["text"].astype(str), df[hyp].astype(str)))
        # CROCODILE's own adaptive batch size
        rl = 12 if max(len(a) for a, _ in pairs) > 256 else 64
        probs = []
        for i in range(0, len(pairs), rl):
            b = pairs[i:i + rl]
            enc = tok([a for a, _ in b], [c for _, c in b], return_tensors="pt",
                      add_special_tokens=True, max_length=MAX_LEN,
                      padding="longest", return_token_type_ids=False,
                      truncation=True)
            enc = {kk: v.cuda() for kk, v in enc.items()}
            with torch.no_grad():
                probs.append(model(**enc).logits.softmax(dim=1)[:, ent_idx].float().cpu())
        p_ent = torch.cat(probs).numpy()
        all_scores[hyp] = p_ent.tolist()

        # 1) as-published
        pred = (p_ent > CROCODILE_THRESHOLD).astype(int)
        prec, rec, f1, acc = _prf(y, pred)
        lo, hi = bootstrap_ci(y, pred)
        results.append({"hypothesis": hyp, "mode": "as-published (t=0.75)",
                        "threshold": CROCODILE_THRESHOLD, "f1": round(f1, 4),
                        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                        "precision": round(prec, 4), "recall": round(rec, 4),
                        "accuracy": round(acc, 4),
                        "macro_f1": round(macro_f1(df, y, pred), 4)})

        # 2) best threshold, matching Table 3's protocol
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
        best_t, best_f1 = max(((t, _f1(y, (p_ent > t).astype(int))) for t in grid),
                              key=lambda x: x[1])
        pred_b = (p_ent > best_t).astype(int)
        prec, rec, f1b, acc = _prf(y, pred_b)
        lo, hi = bootstrap_ci(y, pred_b)
        results.append({"hypothesis": hyp, "mode": "best-threshold",
                        "threshold": float(best_t), "f1": round(f1b, 4),
                        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                        "precision": round(prec, 4), "recall": round(rec, 4),
                        "accuracy": round(acc, 4),
                        "macro_f1": round(macro_f1(df, y, pred_b), 4)})
        print(f"  {hyp:20s} as-published F1={f1:.4f}  best(t={best_t:.2f}) F1={f1b:.4f}")

    res = pd.DataFrame(results)
    res.to_csv(OUT / "crocodile_baseline.csv", index=False)
    with open(OUT / "crocodile_baseline_scores.json", "w") as f:
        json.dump(all_scores, f)

    print("\n=== CROCODILE baseline (v3 gold-500, adjudicated labels) ===")
    print(res.to_string(index=False))
    print(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()
