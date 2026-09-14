"""
Smoke test for the rebuilt v3 corpus: did it change, and did it change safely?

Two halves, because "no markup left" and "no content lost" are different
questions and the second one is the one that bit us — the earlier 20k spot
checks all reported zero residuals while v3 was silently deleting whole
articles whose body was glued onto an image caption.

  A. CHANGED   — v1 vs v3 deltas, so the rebuild is demonstrably not a no-op.
  B. SAFE      — empties, large content loss, and residual markup.
  C. SAMPLES   — dumps N random passages for a human to read.

Usage:
    python -m post_rebuttal_and_camera_ready.pipeline.smoke_test_v3
    python -m post_rebuttal_and_camera_ready.pipeline.smoke_test_v3 --samples 30
"""

from __future__ import annotations

import argparse
import random
import re
import sys

import pandas as pd

from post_rebuttal_and_camera_ready.pipeline import paths as P
from scripts.prepare_data.text_cleaning import clean

RESIDUALS = {
    "wiki link [[ ]]": r"\[\[|\]\]",
    "pipe": r"\|",
    "line-initial katgoria": r"(?m)^\s*קטגוריה:",
    "footer header": r"(?m)^\s*(?:קישורים\s+חיצוניים|ראו\s+גם|הערות\s+שוליים|לקריאה\s+נוספת)\s*$",
    "bullet": r"(?m)^\s*[\*\#]",
    "image directive markup": r"ממוזער\s*\||\d+\s*x?\s*\d*\s*(?:px|פיקסלים)",
    "html tag": r"</?[a-zA-Z][^>\n]{0,40}>",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--rows", type=int, default=80000, help="raw rows for the v1-vs-v3 comparison")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    P.assert_repo_scripts()
    if not P.SILVER_V3_PARQUET.exists():
        raise SystemExit("v3 parquet missing — run prepare_v3_data first")

    # ---------- A. did it change? ----------
    print("=" * 78)
    print("A. CHANGED — v1 vs v3 on raw rows")
    print("=" * 78)
    raw = pd.read_csv(P.RAW_SILVER, nrows=args.rows)["text"].astype(str)
    v1 = [clean(t, "v1") for t in raw]
    v3 = [clean(t, "v3") for t in raw]
    changed = sum(a != b for a, b in zip(v1, v3))
    w1 = sum(len(t.split()) for t in v1) / len(v1)
    w3 = sum(len(t.split()) for t in v3) / len(v3)
    print(f"  rows compared        : {len(raw):,}")
    print(f"  rows changed by v3   : {changed:,} ({100*changed/len(raw):.1f}%)")
    print(f"  mean words v1 -> v3  : {w1:.1f} -> {w3:.1f}  ({100*(w3-w1)/w1:+.1f}%)")

    # ---------- B. is it safe? ----------
    print("\n" + "=" * 78)
    print("B. SAFE — content preservation and residual markup")
    print("=" * 78)
    empty = sum(not t.strip() for t in v3)
    lost = sum(1 for a, b in zip(v1, v3)
               if len(a.split()) >= 30 and len(b.split()) < 0.5 * len(a.split()))
    print(f"  empty after v3       : {empty}  ({100*empty/len(raw):.3f}%)")
    print(f"  lost >50% of words   : {lost}  ({100*lost/len(raw):.3f}%)")

    print("\n  residual markup in the BUILT corpus (300k sample):")
    df = pd.read_parquet(P.SILVER_V3_PARQUET, columns=["docid", "title", "text",
                                                       "subject", "predicate", "object"])
    print(f"  corpus rows          : {len(df):,}")
    s = df.sample(min(300000, len(df)), random_state=args.seed)
    t = s["text"].astype(str)
    for label, pat in RESIDUALS.items():
        n = int(t.str.contains(pat, regex=True, na=False).sum())
        flag = "" if n == 0 else "   <-- CHECK"
        print(f"    {label:26s} {n:>7,}{flag}")
    print(f"    {'empty text':26s} {int((t.str.strip()=='').sum()):>7,}")

    INVIS = {0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x202A, 0x202B,
             0x202C, 0x2060, 0xFEFF, 0x00AD, 0x00A0}
    print("\n  entity columns (full corpus):")
    for c in ("subject", "predicate", "object"):
        v = df[c].astype(str)
        nl = int(v.str.contains(r"\[\[|\]\]", regex=True, na=False).sum())
        npi = int(v.str.contains(r"\|", regex=True, na=False).sum())
        ni = int(v.map(lambda x: any(ord(ch) in INVIS for ch in x)).sum())
        print(f"    {c:10s} links={nl:>5,}  pipes={npi:>5,}  invisible={ni:>5,}")

    # ---------- C. samples to read ----------
    print("\n" + "=" * 78)
    print(f"C. SAMPLES — {args.samples} random passages")
    print("=" * 78)
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(df)), args.samples)
    for n, i in enumerate(idx, 1):
        r = df.iloc[i]
        txt = str(r["text"])
        words = len(txt.split())
        print(f"\n--- [{n}] docid={r['docid']} title={r['title']!r} words={words}")
        print(f"    triple: {r['subject']!r}  --{r['predicate']}--  {r['object']!r}")
        print(f"    subj in text: {str(r['subject']) in txt} | obj in text: {str(r['object']) in txt}")
        head = txt[:300].replace("\n", " ⏎ ")
        tail = txt[-160:].replace("\n", " ⏎ ") if len(txt) > 460 else ""
        print(f"    HEAD: {head}")
        if tail:
            print(f"    TAIL: {tail}")


if __name__ == "__main__":
    main()
