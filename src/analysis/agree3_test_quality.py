"""Quality of the PUBLISHED label (agree3 = unanimous vote of Gemma-4-31B-it S4,
Gemma-3-27B-it S4, DictaLM-3.0-24B S2) on the held-out gold test set, plus the
overlap between the test set and the agree3 release (2026-09-14, audit item 4).
Output: analysis/stage0_results/agree3_test_quality.csv"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
SV = "post_rebuttal_and_camera_ready/silver_scoring_v3/silver_all_cleaned.parquet"
A3 = ["opensource_llm__gemma4_31b__pred", "opensource_llm__gemma3_27b__pred", "finetuned_nli__dictalm24b__pred"]
gold = pd.read_csv("rebuttal/final_datasets/gold_test_set.csv", encoding="utf-8-sig")
gold["_key"] = gold.docid.astype(str)+"||"+gold.silver_join_subject.astype(str)+"||"+gold.predicate.astype(str)+"||"+gold.object.astype(str)
sv = pd.read_parquet(SV, columns=["docid", "subject", "predicate", "object"] + A3)
sv["_key"] = sv.docid.astype(str)+"||"+sv.subject.astype(str)+"||"+sv.predicate.astype(str)+"||"+sv.object.astype(str)
for c in A3: sv[c] = (pd.to_numeric(sv[c], errors="coerce").fillna(0) > 0.5).astype(int)
sv["agree3"] = np.where(sv[A3].nunique(axis=1) == 1, sv[A3[0]], -1)
m = gold.merge(sv.drop_duplicates("_key"), on="_key", how="left")
y = m.annotator1_Ron.astype(int).to_numpy(); p = m.agree3.fillna(-1).astype(int).to_numpy()
def prf(y, p):
    tp = ((y == 1) & (p == 1)).sum(); fp = ((y == 0) & (p == 1)).sum(); fn = ((y == 1) & (p == 0)).sum()
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); return P, R, 2 * P * R / max(P + R, 1e-9)
rng = np.random.default_rng(42); rows = []
subsets = [("all", np.ones(len(y), bool))] + [(s, (m.selection_reason == s).to_numpy()) for s in ["silver_validation_200", "new_predicate", "depth_boost"]]
for name, sub in subsets:
    cov = (p >= 0) & sub; yy, pp = y[cov], p[cov]
    P, R, F = prf(yy, pp); idx = rng.integers(0, len(yy), (10000, len(yy)))
    f = [prf(yy[i], pp[i])[2] for i in idx]
    rows.append(dict(subset=name, n=int(sub.sum()), covered=int(cov.sum()), coverage=round(cov.sum() / sub.sum(), 3),
                     precision=round(P, 3), recall=round(R, 3), f1=round(F, 3), ci_lo=round(np.percentile(f, 2.5), 3), ci_hi=round(np.percentile(f, 97.5), 3),
                     n_pos_in_release=int(((p == 1) & sub).sum()), n_neg_in_release=int(((p == 0) & sub).sum())))
r = pd.DataFrame(rows); print("test rows matched to silver:", int(m[A3[0]].notna().sum()), "/", len(m)); print(r.to_string(index=False))
r.to_csv("post_rebuttal_and_camera_ready/analysis/stage0_results/agree3_test_quality.csv", index=False)
