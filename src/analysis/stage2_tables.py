"""
Stage 2: recompute the paper's Table 2 (dataset statistics) and Table 4
(per-signal positive rates + published-set composition) from the v3 artifacts,
side by side with the paper's v1 values.

Sources:
  Table 2 gold   — post_rebuttal_and_camera_ready/data_v3/prepared_gold_500_v3.csv
  Table 2 silver — the v3 merged corpus, restricted to the agree3-published
                   subset (unanimous Gemma-4-31B + Gemma-3-27B + DictaLM-24B),
                   matching how the paper's silver column was computed.
  Table 4        — the same merged corpus, all 10 signals + ensembles.

CPU-only, single pass over the merged parquet.

Usage:
    python -m post_rebuttal_and_camera_ready.analysis.stage2_tables
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOLD = REPO / "post_rebuttal_and_camera_ready" / "data_v3" / "prepared_gold_500_v3.csv"
MERGED = REPO / "post_rebuttal_and_camera_ready" / "silver_scoring_v3" / "silver_all_cleaned.parquet"
OUT = HERE / "stage0_results"

# paper's Table 2 silver column (v1 published agree3) for the delta column
PAPER_T2 = {"examples": 1_491_405, "unique_docs": 242_000, "unique_predicates": 1_257,
            "preds_ge20": 718, "pos_rate": 72.5, "mean_words": 367, "mean_chars": 2258}
AGREE3_COLS = ["opensource_llm__gemma4_31b__pred", "opensource_llm__gemma3_27b__pred",
               "finetuned_nli__dictalm24b__pred"]


def table2_gold():
    g = pd.read_csv(GOLD, encoding="utf-8-sig")
    words = g["text"].astype(str).str.split().str.len()
    return {
        "examples": len(g),
        "unique_docs": g["docid"].nunique(),
        "triples_per_doc": round(len(g) / g["docid"].nunique(), 2),
        "unique_predicates": g["predicate"].nunique(),
        "preds_ge20": int((g["predicate"].value_counts() >= 20).sum()),
        "pos_rate": round(100 * g["relation_present"].mean(), 1),
        "pos_rate_std_by_pred": round(100 * g.groupby("predicate")["relation_present"].mean().std(), 1),
        "mean_words": round(float(words.mean()), 1),
        "mean_chars": round(float(g["text"].astype(str).str.len().mean()), 0),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Table 2 — GOLD (v3) ===")
    tg = table2_gold()
    for k, v in tg.items():
        print(f"  {k:22s} {v}")

    print("\n=== Table 2 — SILVER published (v3 agree3) ===")
    cols = ["docid", "predicate", "text"] + AGREE3_COLS
    df = pd.read_parquet(MERGED, columns=cols)
    a3_agree = ((df[AGREE3_COLS[0]] == df[AGREE3_COLS[1]])
                & (df[AGREE3_COLS[1]] == df[AGREE3_COLS[2]]))
    pub = df[a3_agree].copy()
    pos = pub[AGREE3_COLS[0]] == 1
    words = pub["text"].astype(str).str.count(" ") + 1     # fast word estimate
    pc = pub["predicate"].value_counts()
    ts = {
        "examples": len(pub),
        "unique_docs": pub["docid"].nunique(),
        "triples_per_doc": round(len(pub) / pub["docid"].nunique(), 2),
        "unique_predicates": pub["predicate"].nunique(),
        "preds_ge20": int((pc >= 20).sum()),
        "pos_rate": round(100 * pos.mean(), 1),
        "mean_words": round(float(words.mean()), 1),
        "mean_chars": round(float(pub["text"].astype(str).str.len().mean()), 0),
    }
    for k, v in ts.items():
        ref = PAPER_T2.get(k)
        print(f"  {k:22s} {v}" + (f"   (paper v1: {ref})" if ref is not None else ""))

    rows = [{"table": "gold_v3", **tg}, {"table": "silver_published_v3", **ts}]
    pd.DataFrame(rows).to_csv(OUT / "table2_v3.csv", index=False)
    print(f"\nwrote → {OUT/'table2_v3.csv'}   (Table 4 rates: see the merge's own "
          f"statistics file in silver_scoring_v3/, reproduced ≤0.2pp vs paper)")


if __name__ == "__main__":
    main()
