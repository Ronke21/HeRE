"""Rescore the 16 Strategy-3 grid cells (ensemble gold predictions) on the FINAL
adjudicated labels with the same join, bootstrap and macro-F1 as every other
strategy (stage0_ci_and_macro). Supersedes s3_grid_adjudicated.csv, which was
scored on the labels as they stood mid-adjudication (2026-08-20).
Output: analysis/stage0_results/s3_grid_final.csv"""
import sys, glob; sys.path.insert(0, ".")
import pandas as pd
from post_rebuttal_and_camera_ready.analysis.stage0_ci_and_macro import _adjudicated_labels, JOIN_KEYS, bootstrap_f1, macro_f1, MIN_PRED_N
labels = _adjudicated_labels()
grid = pd.read_csv("post_rebuttal_and_camera_ready/analysis/stage0_results/s3_grid_adjudicated.csv")
rows = []
for _, r in grid.iterrows():
    f = glob.glob(f"post_rebuttal_and_camera_ready/gold_benchmark_v3/cross_train_rc/{r.src}/**/gold_classified.csv", recursive=True)[0]
    d = pd.read_csv(f, encoding="utf-8-sig")
    key = d[JOIN_KEYS].astype(str).agg("||".join, axis=1); d["_y"] = key.map(labels).fillna(d.relation_present).astype(int)
    d["_p"] = d.predicted_relation_present.astype(int); f1, lo, hi = bootstrap_f1(d._y.to_numpy(), d._p.to_numpy())
    rows.append(dict(model=r.model, K=int(r.K), f1=round(f1, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3), macro=round(macro_f1(d, "_y", "_p", MIN_PRED_N), 3), f1_aug20=r.f1_adj, src=r.src))
out = pd.DataFrame(rows); out.to_csv("post_rebuttal_and_camera_ready/analysis/stage0_results/s3_grid_final.csv", index=False); print(out.to_string(index=False))
