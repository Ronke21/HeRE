"""
Strategy 1 (encoder NLI) re-benchmark on v3 against ADJUDICATED labels.

The v3 encoder scoring ran on 2026-08-16 (gold_benchmark_v3/encoder_nli/), but
its summary was computed against the original single-annotator label and it was
never folded into the bootstrap/macro analysis, so Strategy 1 was the one row of
Table 3 still carrying v1 numbers.

This reproduces the script's own protocol -- 5-fold stratified CV in which the
decision threshold is chosen on the training folds and applied to the held-out
fold (an honest out-of-fold estimate, not a threshold tuned on the test data) --
against the adjudicated label, and adds the bootstrap CI and macro-F1 that every
other strategy reports.

Scoring formulas, as in clean_encoder_NLI.py:
    pe  = P(entail)
    emc = P(entail) - P(contradict)
    enn = P(entail) / (P(entail) + P(contradict))

Output: analysis/stage0_results/encoder_nli_v3.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCORED = HERE.parent / "gold_benchmark_v3" / "encoder_nli" / "classified.csv"
ADJ = REPO / "rebuttal" / "final_datasets" / "gold_validation_set.csv"
OUT = HERE / "stage0_results" / "encoder_nli_v3.csv"

SEED = 42
N_FOLDS = 5
N_BOOT = 10_000
GRID_PE = np.arange(0.05, 0.96, 0.05)
GRID_EMC = np.arange(-0.90, 0.91, 0.05)


def cv_oof_predictions(scores: np.ndarray, y: np.ndarray, grid: np.ndarray):
    """Threshold chosen on train folds, applied to held-out fold."""
    oof = np.zeros(len(y), dtype=int)
    thresholds = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(scores.reshape(-1, 1), y):
        best_t = max(grid, key=lambda t: f1_score(y[tr], (scores[tr] >= t).astype(int)))
        oof[te] = (scores[te] >= best_t).astype(int)
        thresholds.append(float(best_t))
    return oof, thresholds


_BOOT_IDX = None  # shared resample matrix -- one draw reused by all configs


def bootstrap_ci(y: np.ndarray, pred: np.ndarray):
    """Vectorized percentile bootstrap (the per-call sklearn loop took >15 min
    for 72 configs; this runs in ~1 s per config)."""
    global _BOOT_IDX
    n = len(y)
    if _BOOT_IDX is None:
        _BOOT_IDX = np.random.default_rng(SEED).integers(0, n, (N_BOOT, n))
    yb, pb = y[_BOOT_IDX], pred[_BOOT_IDX]           # (N_BOOT, n)
    tp = ((yb == 1) & (pb == 1)).sum(axis=1).astype(float)
    fp = ((yb == 0) & (pb == 1)).sum(axis=1)
    fn = ((yb == 1) & (pb == 0)).sum(axis=1)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))


def main():
    df = pd.read_csv(SCORED)
    adj = pd.read_csv(ADJ)
    key = ["docid", "predicate", "object"]
    for c in key:
        df[c] = df[c].astype(str)
        adj[c] = adj[c].astype(str)
    m = df.merge(adj[key + ["annotator1_Ron"]], on=key, how="left")
    n_missing = int(m["annotator1_Ron"].isna().sum())
    y = m["annotator1_Ron"].fillna(m["relation_present"]).astype(int).to_numpy()
    print(f"[labels] adjudicated; {n_missing} row(s) fell back to the original label")

    models = sorted({c.split("conf_pe_")[1].rsplit("_", 2)[0]
                     for c in df.columns if c.startswith("conf_pe_")})
    hyps = ["basic_relation", "template_relation", "llm_relation"]

    rows = []
    for model in models:
        for hyp in hyps:
            pe_col, pc_col = f"conf_pe_{model}_{hyp}", f"conf_pc_{model}_{hyp}"
            if pe_col not in df.columns or pc_col not in df.columns:
                continue
            pe, pc = m[pe_col].to_numpy(), m[pc_col].to_numpy()
            for formula, s, grid in (("pe", pe, GRID_PE),
                                     ("emc", pe - pc, GRID_EMC),
                                     ("enn", pe / np.clip(pe + pc, 1e-9, None), GRID_PE)):
                oof, ths = cv_oof_predictions(s, y, grid)
                f1 = f1_score(y, oof, zero_division=0)
                lo, hi = bootstrap_ci(y, oof)
                rows.append({
                    "model": model, "hypothesis": hyp, "formula": formula,
                    "cv_oof_f1": round(f1, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                    "macro_f1": round(f1_score(y, oof, average="macro", zero_division=0), 4),
                    "mean_threshold": round(float(np.mean(ths)), 3),
                })
    res = pd.DataFrame(rows).sort_values("cv_oof_f1", ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print(f"\n[top 8 of {len(res)} configurations]")
    print(res.head(8).to_string(index=False))
    best_per_model = res.sort_values("cv_oof_f1", ascending=False).groupby("model").head(1)
    print("\n[best per model]")
    print(best_per_model.sort_values("cv_oof_f1", ascending=False).to_string(index=False))
    print(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()
