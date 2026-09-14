"""Held-out TEST-set results for every denoising signal applied to the silver
corpus (the test rows come from the silver corpus, so all at-scale signals
already have predictions on them; no model is re-run). Test-set counterpart
of the validation-set main table, restricted to the deployed configurations.
Output: analysis/stage0_results/test_set_all_signals.csv"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.metrics import f1_score
from post_rebuttal_and_camera_ready.analysis.stage0_ci_and_macro import macro_f1, MIN_PRED_N
SV = "post_rebuttal_and_camera_ready/silver_scoring_v3/silver_all_cleaned.parquet"
gold = pd.read_csv("rebuttal/final_datasets/gold_test_set.csv", encoding="utf-8-sig")
gold["_key"] = gold.docid.astype(str)+"||"+gold.silver_join_subject.astype(str)+"||"+gold.predicate.astype(str)+"||"+gold.object.astype(str)
preds = [c for c in pq.ParquetFile(SV).schema_arrow.names if c.endswith("__pred")]
sv = pd.read_parquet(SV, columns=["docid","subject","predicate","object"]+preds)
sv["_key"] = sv.docid.astype(str)+"||"+sv.subject.astype(str)+"||"+sv.predicate.astype(str)+"||"+sv.object.astype(str)
m = gold.merge(sv.drop_duplicates("_key"), on="_key", how="left", suffixes=("","_sv"))
y = m.annotator1_Ron.astype(int).to_numpy()
print("matched:", int(m[preds[0]].notna().sum()), "/", len(m))
idx = np.random.default_rng(42).integers(0, len(y), (10000, len(y)))
def ci(p):
    yb,pb=y[idx],p[idx]; tp=((yb==1)&(pb==1)).sum(1); fp=((yb==0)&(pb==1)).sum(1); fn=((yb==1)&(pb==0)).sum(1)
    f=np.where(2*tp+fp+fn>0,2*tp/np.maximum(2*tp+fp+fn,1),0.0); return np.percentile(f,2.5),np.percentile(f,97.5)
rows=[]
for c in preds:
    p = (pd.to_numeric(m[c], errors="coerce").fillna(0).astype(float) > 0.5).astype(int).to_numpy()
    lo,hi = ci(p)
    rows.append(dict(signal=c.replace("__pred",""), f1=round(f1_score(y,p),3), ci_lo=round(lo,3), ci_hi=round(hi,3),
                     macro_f1=round(macro_f1(m.assign(_y=y,_p=p),"_y","_p",MIN_PRED_N),3), pos_rate=round(p.mean(),3)))
r = pd.DataFrame(rows).sort_values("f1", ascending=False)
r.to_csv("post_rebuttal_and_camera_ready/analysis/stage0_results/test_set_all_signals.csv", index=False)
print(r.to_string(index=False))
# Gemma-4-31B / Gemma-3-27B by test subset (v3-consistent generalization chain)
sub=[]
for sig in ["opensource_llm__gemma4_31b__pred","opensource_llm__gemma3_27b__pred"]:
    p=(pd.to_numeric(m[sig],errors="coerce").fillna(0).astype(float)>0.5).astype(int).to_numpy()
    for name,mask in [("all",np.ones(len(y),bool)),("new_predicate",(m.selection_reason=="new_predicate").to_numpy()),
                      ("depth_boost",(m.selection_reason=="depth_boost").to_numpy()),("silver_validation_200",(m.selection_reason=="silver_validation_200").to_numpy())]:
        sub.append(dict(signal=sig.replace("__pred",""),subset=name,n=int(mask.sum()),f1=round(f1_score(y[mask],p[mask]),3)))
s2=pd.DataFrame(sub); s2.to_csv("post_rebuttal_and_camera_ready/analysis/stage0_results/test_set_subsets.csv",index=False); print(s2.to_string(index=False))
