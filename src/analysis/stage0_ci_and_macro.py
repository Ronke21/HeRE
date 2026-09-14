"""
Stage 0.2 (D) + 0.3 (F): bootstrap confidence intervals and macro-F1 /
per-predicate breakdown for the v3 gold-500 benchmark.

Why both in one script: they consume exactly the same inputs (per-example
binary predictions + gold label + predicate) and the per-predicate table is
just a different grouping of the same bootstrap machinery.

D — the paper reports Table 3 F1 to three decimals on n=500 and draws
conclusions from gaps as small as 0.009 (GPT-5.4 0.936 vs Gemma-4-31B 0.927).
At that sample size those are very unlikely to be distinguishable. Reporting a
95% CI turns a weak claim ("within one point") into a strong one
("statistically indistinguishable from the frontier API").

F — gold-500 spans 97 predicates but only 4 have >=20 examples, and the
per-predicate positive rate has std 38.5%. A single micro-F1 over that
distribution is fragile; macro-F1 and a per-predicate table are the honest
companions, and were promised to reviewers xwg9 / vvD7 in the rebuttal.

CPU-only. Reads the v3 benchmark outputs; writes nothing outside analysis/.

Usage:
    python -m post_rebuttal_and_camera_ready.analysis.stage0_ci_and_macro
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOLD = HERE.parent / "gold_benchmark_v3"
OUT = HERE / "stage0_results"

# Open decision #3, settled 2026-08-16: the official gold label is the
# ADJUDICATED annotator1_Ron column, not the original single-annotator pass.
# Rationale: it is dual-annotated (answers reviewer xwg9's W1 directly), the
# headline barely moves (Gemma-4-31B 0.927 -> 0.922), and on all 12 rows where
# Ron and Roee still disagree the paper's original label already matches Ron,
# so adopting it needs no further adjudication.
#
# The gold-500 benchmark set and rebuttal/final_datasets/gold_validation_set.csv
# are the same 500 examples; the latter carries the adjudicated columns.
ADJUDICATED = REPO / "rebuttal" / "final_datasets" / "gold_validation_set.csv"
LABEL_COL = "annotator1_Ron"
JOIN_KEYS = ["docid", "predicate", "object"]

N_BOOT = 10_000
SEED = 42
MIN_PRED_N = 10          # per-predicate table: only predicates with >= this many examples

# family -> (subdir, how to find binary prediction columns)
FAMILIES = {
    "api_llm": "api_llm",
    "opensource_llm": "opensource_llm",
    "finetuned_llm_nli": "finetuned_llm_nli",
}


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def bootstrap_f1(y_true: np.ndarray, y_pred: np.ndarray, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap over examples. Returns (point, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        i = idx[b]
        stats[b] = _f1(y_true[i], y_pred[i])
    return _f1(y_true, y_pred), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_bootstrap_diff(y_true, pred_a, pred_b, n_boot=N_BOOT, seed=SEED):
    """P(model A better than B) under a paired bootstrap — the correct test when
    both models are scored on the *same* examples, which they are here."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.integers(0, n, size=(n_boot, n))
    wins = 0
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = idx[b]
        d = _f1(y_true[i], pred_a[i]) - _f1(y_true[i], pred_b[i])
        diffs[b] = d
        if d > 0:
            wins += 1
    return {
        "mean_diff": float(np.mean(diffs)),
        "lo95": float(np.percentile(diffs, 2.5)),
        "hi95": float(np.percentile(diffs, 97.5)),
        "p_a_better": wins / n_boot,
    }


def macro_f1(df: pd.DataFrame, label_col: str, pred_col: str, min_n=1):
    """Unweighted mean of per-predicate F1 over predicates with >= min_n examples."""
    scores = []
    for _, g in df.groupby("predicate"):
        if len(g) < min_n:
            continue
        scores.append(_f1(g[label_col].to_numpy(), g[pred_col].to_numpy()))
    return float(np.mean(scores)) if scores else float("nan")


def _adjudicated_labels():
    """Map join-key -> adjudicated annotator1_Ron label."""
    adj = pd.read_csv(ADJUDICATED, encoding="utf-8-sig")
    adj = adj[adj[LABEL_COL].notna()].copy()
    adj["_key"] = adj[JOIN_KEYS].astype(str).agg("||".join, axis=1)
    return dict(zip(adj["_key"], adj[LABEL_COL].astype(int)))


def load_family(subdir: str):
    """Load one family's per-example predictions.

    api_llm / opensource_llm / encoder_nli write a single classified.csv.
    finetuned_llm_nli (Strategy 2) instead writes one classified.csv per tag
    under finetuned_llm_nli/<tag>/ — originally skipped by this loader, which
    left all 17 Strategy-2 tags out of the CI/macro tables. Fixed 2026-08-21:
    per-tag CSVs are merged column-wise on the row order (all are the same
    500 gold rows in the same order; verified by docid equality below).
    """
    path = GOLD / subdir / "classified.csv"
    if path.exists():
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    else:
        tag_files = sorted((GOLD / subdir).glob("*/classified.csv"))
        if not tag_files:
            return None
        df = pd.read_csv(tag_files[0], encoding="utf-8-sig", low_memory=False)
        for f in tag_files[1:]:
            d2 = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
            assert (d2["docid"].astype(str).values == df["docid"].astype(str).values).all(),                 f"row order mismatch merging {f}"
            new_cols = [c for c in d2.columns if c not in df.columns]
            df = pd.concat([df, d2[new_cols]], axis=1)
    pred_cols = [c for c in df.columns if c.startswith(("llm_clean_", "nli_clean_"))]

    # Swap in the adjudicated label. Rows that fail to join keep the original
    # label and are counted, so a silent partial join can't pass unnoticed.
    labels = _adjudicated_labels()
    key = df[JOIN_KEYS].astype(str).agg("||".join, axis=1)
    mapped = key.map(labels)
    n_missing = int(mapped.isna().sum())
    n_changed = int((mapped.notna() & (mapped != df["relation_present"])).sum())
    df["relation_present"] = mapped.fillna(df["relation_present"]).astype(int)
    print(f"    label swap: {len(df)-n_missing}/{len(df)} joined, "
          f"{n_changed} labels changed, {n_missing} unmatched (kept original)")
    return df, pred_cols


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, per_pred_rows = [], []

    for fam, subdir in FAMILIES.items():
        loaded = load_family(subdir)
        if loaded is None:
            print(f"[skip] {fam}: no classified.csv yet")
            continue
        df, pred_cols = loaded
        if not pred_cols:
            print(f"[skip] {fam}: no binary prediction columns")
            continue
        df = df[df["relation_present"].notna()].copy()
        y = df["relation_present"].astype(int).to_numpy()
        print(f"[{fam}] {len(df)} rows, {len(pred_cols)} configurations")

        for c in pred_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.isna().all():
                continue
            # unparseable model output -> negative, matching the scripts' own convention
            yp = s.fillna(0).astype(int).to_numpy()
            point, lo, hi = bootstrap_f1(y, yp)
            rows.append({
                "family": fam,
                "config": c,
                "micro_f1": round(point, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "ci_halfwidth": round((hi - lo) / 2, 4),
                "macro_f1_all_preds": round(macro_f1(df.assign(_p=yp), "relation_present", "_p"), 4),
                "macro_f1_min10": round(macro_f1(df.assign(_p=yp), "relation_present", "_p", MIN_PRED_N), 4),
                "n_unparseable": int(s.isna().sum()),
            })

        # per-predicate detail for this family's single best configuration
        if rows:
            fam_rows = [r for r in rows if r["family"] == fam]
            best = max(fam_rows, key=lambda r: r["micro_f1"])
            bp = pd.to_numeric(df[best["config"]], errors="coerce").fillna(0).astype(int).to_numpy()
            tmp = df.assign(_p=bp)
            for pred, g in tmp.groupby("predicate"):
                if len(g) < MIN_PRED_N:
                    continue
                per_pred_rows.append({
                    "family": fam,
                    "best_config": best["config"],
                    "predicate": pred,
                    "n": len(g),
                    "pos_rate": round(float(g["relation_present"].mean()), 3),
                    "f1": round(_f1(g["relation_present"].to_numpy(), g["_p"].to_numpy()), 4),
                })

    if not rows:
        print("no results — gold benchmark outputs not present yet")
        return

    res = pd.DataFrame(rows).sort_values("micro_f1", ascending=False)
    res.to_csv(OUT / "bootstrap_ci_all_configs.csv", index=False)
    pd.DataFrame(per_pred_rows).sort_values(["family", "f1"]).to_csv(
        OUT / "per_predicate_best_config.csv", index=False)

    print(f"\n=== top 12 configurations by micro-F1 (95% bootstrap CI, n={N_BOOT:,}) ===")
    top = res.head(12)
    for _, r in top.iterrows():
        print(f"  {r['micro_f1']:.3f}  [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  "
              f"macro={r['macro_f1_min10']:.3f}  {r['config'][:62]}")

    # Is the top cluster separable? Paired bootstrap, best vs each of the next few.
    print("\n=== paired bootstrap: best vs next 5 (same examples) ===")
    best_row = res.iloc[0]
    fam_df = {f: load_family(s)[0] for f, s in FAMILIES.items() if load_family(s)}
    def preds_for(cfg):
        for d in fam_df.values():
            if cfg in d.columns:
                d2 = d[d["relation_present"].notna()]
                return (d2["relation_present"].astype(int).to_numpy(),
                        pd.to_numeric(d2[cfg], errors="coerce").fillna(0).astype(int).to_numpy())
        return None, None
    y_b, p_best = preds_for(best_row["config"])
    comparisons = []
    for _, r in res.iloc[1:6].iterrows():
        y_o, p_o = preds_for(r["config"])
        if p_o is None or len(p_o) != len(p_best):
            continue
        d = paired_bootstrap_diff(y_b, p_best, p_o)
        comparisons.append({"best": best_row["config"], "vs": r["config"], **d})
        sig = "SIGNIFICANT" if (d["lo95"] > 0 or d["hi95"] < 0) else "not distinguishable"
        print(f"  vs {r['config'][:50]:52s} Δ={d['mean_diff']:+.4f} "
              f"[{d['lo95']:+.4f},{d['hi95']:+.4f}]  {sig}")
    with open(OUT / "paired_bootstrap_top.json", "w") as f:
        json.dump(comparisons, f, indent=2)

    print(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()
