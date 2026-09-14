"""Build the public HeRE dataset files (Hugging Face `ronke21/HeRE`) from the
internal artifacts. The 499 gold-test rows that live in the silver corpus are
withheld from every silver/KG file. Usage: python -m post_rebuttal_and_camera_ready.analysis.build_hf_release <out_dir>"""
import sys, shutil
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from post_rebuttal_and_camera_ready.analysis.build_here_kg import test_row_mask, A3P
OUT = Path(sys.argv[1]); (OUT / "gold").mkdir(parents=True, exist_ok=True); (OUT / "silver").mkdir(exist_ok=True); (OUT / "kg").mkdir(exist_ok=True)
# ---- gold: adjudicated label = annotator1 column (identical to relation_present); annotator 2 = blind labels ----
for split, f in [("validation", "gold_validation_set_v3.csv"), ("test", "gold_test_set_v3.csv")]:
    d = pd.read_csv(f"rebuttal/final_datasets/{f}", encoding="utf-8-sig")
    assert (d.relation_present == d.annotator1_Ron).all()
    out = pd.DataFrame({"item_id": d.item_id, "docid": d.docid.astype(str), "title": d.title, "text": d.text,
                        "subject": d.subject, "predicate": d.predicate, "object": d.object,
                        "label": d.annotator1_Ron.astype(int), "annotator2_label": d.annotator2_Roee.astype(int),
                        "notes": d.roee_notes.fillna("") if "roee_notes" in d else "", "word_count": d.word_count})
    if "selection_reason" in d: out["selection_reason"] = d.selection_reason
    out.to_parquet(OUT / "gold" / f"{split}.parquet", index=False); print(split, len(out), "pos", out.label.mean().round(3))
# ---- silver ----
sv = pd.read_parquet("post_rebuttal_and_camera_ready/silver_scoring_v3/silver_all_cleaned.parquet")
sv = sv[~test_row_mask(sv)].reset_index(drop=True)
for c in [c for c in sv.columns if c.endswith("__pred")]:
    sv[c] = pd.to_numeric(sv[c], errors="coerce").astype("Int8")
una = (sv[A3P[0]] == sv[A3P[1]]) & (sv[A3P[1]] == sv[A3P[2]]) & sv[A3P[0]].notna()
sv["agree3_label"] = np.where(una, sv[A3P[0]].fillna(-1).astype(int), -1).astype("int8")
sv["docid"] = sv.docid.astype(str)
front = ["docid", "title", "text", "subject", "predicate", "object", "agree3_label"]
sv = sv[front + [c for c in sv.columns if c not in front]]
sv.to_parquet(OUT / "silver" / "silver_all_signals.parquet", index=False); print("all-signals rows", len(sv))
pub = sv[sv.agree3_label >= 0].rename(columns={"agree3_label": "label"})
pub.to_parquet(OUT / "silver" / "here_silver.parquet", index=False); print("published rows", len(pub), "pos", int((pub.label == 1).sum()), "neg", int((pub.label == 0).sum()))
# ---- KG (already rebuilt with test rows withheld) ----
shutil.copy("post_rebuttal_and_camera_ready/analysis/here_kg/here_kg_edges.tsv.gz", OUT / "kg" / "here_kg_edges.tsv.gz")
shutil.copy("post_rebuttal_and_camera_ready/analysis/here_kg/here_kg_stats.md", OUT / "kg" / "here_kg_stats.md")
print("done")
