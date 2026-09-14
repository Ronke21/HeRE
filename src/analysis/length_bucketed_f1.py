"""
Length-bucketed F1 — the table promised to reviewers xwg9 and vvD7.

vvD7/W2: gold texts are much shorter than silver (79 vs 367 words), so gold F1
may not transfer to longer passages. The rebuttal built the instrument to test
this — the 500-row dual-annotated gold TEST set, whose expansion rows were
deliberately sampled at >=250 words — but the bucketed table itself was never
computed. This script computes it.

Predictions are recovered exactly as rebuttal/model_evaluation/
eval_silver_vs_gold_test_set.py does: every test row is literally a row of the
already-scored 2.56M silver corpus, so signals are joined from
outputs/silver_cleaning/silver_all_cleaned.parquet on
docid+silver_join_subject+predicate+object. No model is re-run.

Labels: adjudicated annotator1_Ron (decision #3). CPU-only.

Usage:
    python -m post_rebuttal_and_camera_ready.analysis.length_bucketed_f1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TEST = REPO / "rebuttal" / "final_datasets" / "gold_test_set.csv"
SILVER = REPO / "outputs" / "silver_cleaning" / "silver_all_cleaned.parquet"
OUT = HERE / "stage0_results" / "length_bucketed_f1.csv"

LABEL = "annotator1_Ron"
BUCKETS = [(0, 100), (100, 250), (250, 500), (500, 10_000)]
SIGNALS = {
    "gemma4_31b": "opensource_llm__gemma4_31b__pred",
    "gemma3_27b": "opensource_llm__gemma3_27b__pred",
    "dictalm24b_nli": "finetuned_nli__dictalm24b__pred",
    "agree2": "agree2__pred",
    "vote7": "vote7__pred",
}
N_BOOT = 10_000


def _f1(y, p):
    tp = ((y == 1) & (p == 1)).sum(); fp = ((y == 0) & (p == 1)).sum()
    fn = ((y == 1) & (p == 0)).sum(); d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def boot_ci(y, p, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    s = np.array([_f1(y[i], p[i]) for i in idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    gold = pd.read_csv(TEST, encoding="utf-8-sig")
    gold = gold[gold[LABEL].notna()].copy()
    gold["_key"] = (gold["docid"].astype(str) + "||" + gold["silver_join_subject"].astype(str)
                    + "||" + gold["predicate"].astype(str) + "||" + gold["object"].astype(str))
    print(f"[test set] {len(gold)} rows, word_count {gold.word_count.min()}-{gold.word_count.max()} "
          f"(mean {gold.word_count.mean():.0f})")

    cols = ["docid", "subject", "predicate", "object"] + list(SIGNALS.values())
    sv = pd.read_parquet(SILVER, columns=cols)
    sv["_key"] = (sv["docid"].astype(str) + "||" + sv["subject"].astype(str)
                  + "||" + sv["predicate"].astype(str) + "||" + sv["object"].astype(str))
    m = gold.merge(sv.drop_duplicates("_key"), on="_key", how="left", suffixes=("", "_sv"))
    n_matched = m[list(SIGNALS.values())[0]].notna().sum()
    print(f"[join] {n_matched}/{len(gold)} rows matched to silver predictions")

    rows = []
    for name, col in SIGNALS.items():
        sub_all = m[m[col].notna()]
        y_all = sub_all[LABEL].astype(int).to_numpy()
        p_all = sub_all[col].astype(int).to_numpy()
        for lo, hi in BUCKETS:
            b = sub_all[(sub_all.word_count >= lo) & (sub_all.word_count < hi)]
            if len(b) < 10:
                continue
            y = b[LABEL].astype(int).to_numpy(); p = b[col].astype(int).to_numpy()
            clo, chi = boot_ci(y, p)
            rows.append({"signal": name, "bucket": f"{lo}-{hi if hi<10000 else '+'}",
                         "n": len(b), "pos_rate": round(float(y.mean()), 3),
                         "f1": round(_f1(y, p), 4),
                         "ci_lo": round(clo, 4), "ci_hi": round(chi, 4)})
        rows.append({"signal": name, "bucket": "ALL", "n": len(sub_all),
                     "pos_rate": round(float(y_all.mean()), 3),
                     "f1": round(_f1(y_all, p_all), 4),
                     "ci_lo": round(boot_ci(y_all, p_all)[0], 4),
                     "ci_hi": round(boot_ci(y_all, p_all)[1], 4)})

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()
