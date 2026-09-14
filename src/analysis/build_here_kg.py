"""
Applications §1 — HeRE-KG: a provenance-grounded Hebrew knowledge graph.

Re-packages the agree3-denoised silver corpus as a first-class KG artifact:
every edge (subject, predicate, object) carries the Hebrew Wikipedia passage
that expresses it (docid + text) and the per-signal confidence scores, i.e. a
provenance- and confidence-annotated graph — which Wikidata itself does not
provide for Hebrew. Also emits the graph statistics table for the paper.

Outputs (analysis/here_kg/):
  here_kg_edges.tsv.gz   s, p, o, docid, n_signals_pos, mean_score
  here_kg_stats.md       entities / relations / degrees / components table

Heavy CPU (one pass over the 2.56M-row merged parquet + union-find over the
edge list) — run on a DGX host, not the A100 box or the login node.

Usage:
    python -m post_rebuttal_and_camera_ready.analysis.build_here_kg
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MERGED = REPO / "post_rebuttal_and_camera_ready" / "silver_scoring_v3" / "silver_all_cleaned.parquet"
OUTD = HERE / "here_kg"

A3P = ["opensource_llm__gemma4_31b__pred", "opensource_llm__gemma3_27b__pred",
       "finetuned_nli__dictalm24b__pred"]
SCORES = ["opensource_llm__gemma4_31b__score", "opensource_llm__gemma3_27b__score",
          "finetuned_nli__dictalm24b__score"]
PREDS_ALL = [c for c in []]  # filled at runtime from schema


TEST_CSV = REPO / "rebuttal" / "final_datasets" / "gold_test_set.csv"


def test_row_mask(df: pd.DataFrame) -> np.ndarray:
    """True for silver rows that are gold TEST-set rows (docid+subject+predicate
    +object). The test set was drawn from the silver corpus, so these rows are
    withheld from every released artifact (silver, HeRE-KG) and from the KGE
    training graphs; 499 of the 500 test rows match a silver row."""
    g = pd.read_csv(TEST_CSV, encoding="utf-8-sig")
    keys = set(g.docid.astype(str) + "||" + g.silver_join_subject.astype(str) + "||"
               + g.predicate.astype(str) + "||" + g.object.astype(str))
    k = (df["docid"].astype(str) + "||" + df["subject"].astype(str) + "||"
         + df["predicate"].astype(str) + "||" + df["object"].astype(str))
    m = k.isin(keys).to_numpy()
    print(f"[test-exclusion] {int(m.sum())} silver rows are gold test rows -> withheld")
    return m


def union_find_components(edges_a: np.ndarray, edges_b: np.ndarray, n: int):
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(edges_a, edges_b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(i) for i in range(n)])
    _, counts = np.unique(roots, return_counts=True)
    return counts


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    cols = ["docid", "subject", "predicate", "object"] + A3P + SCORES
    df = pd.read_parquet(MERGED, columns=cols)
    print(f"[load] {len(df):,} rows")
    df = df[~test_row_mask(df)].reset_index(drop=True)

    unanimous = (df[A3P[0]] == df[A3P[1]]) & (df[A3P[1]] == df[A3P[2]])
    kg = df[unanimous & (df[A3P[0]] == 1)].copy()          # agree3-positive edges
    print(f"[kg] {len(kg):,} positive provenance-grounded edges")

    kg["n_signals_pos"] = df.loc[kg.index, [c for c in df.columns if c.endswith("__pred")]].sum(axis=1)
    kg["mean_score"] = kg[SCORES].mean(axis=1).round(4)

    out_edges = kg[["subject", "predicate", "object", "docid", "n_signals_pos", "mean_score"]]
    out_edges.to_csv(OUTD / "here_kg_edges.tsv.gz", sep="\t", index=False, compression="gzip")

    # ---- statistics ----
    ents = pd.unique(pd.concat([kg["subject"], kg["object"]], ignore_index=True))
    ent_idx = {e: i for i, e in enumerate(ents)}
    ea = kg["subject"].map(ent_idx).to_numpy()
    eb = kg["object"].map(ent_idx).to_numpy()

    deg = np.zeros(len(ents), dtype=np.int64)
    np.add.at(deg, ea, 1); np.add.at(deg, eb, 1)
    comps = union_find_components(ea, eb, len(ents))
    dup = kg.duplicated(["subject", "predicate", "object"]).sum()
    uniq_triples = len(kg) - dup

    pc = kg["predicate"].value_counts()
    stats = f"""# HeRE-KG statistics (v3, agree3-positive, gold test rows withheld)

| Statistic | Value |
|---|---|
| Edges (evidence-linked mentions) | {len(kg):,} |
| Unique triples (s,p,o) | {uniq_triples:,} |
| Entities | {len(ents):,} |
| Relation types | {kg['predicate'].nunique():,} |
| Relations with >=100 edges | {(pc >= 100).sum():,} |
| Source documents | {kg['docid'].nunique():,} |
| Mean / median entity degree | {deg.mean():.1f} / {np.median(deg):.0f} |
| Max entity degree | {deg.max():,} |
| Connected components | {len(comps):,} |
| Largest component (share of entities) | {comps.max():,} ({100*comps.max()/len(ents):.1f}%) |
| Top-5 relations | {', '.join(f'{p} ({n:,})' for p, n in pc.head(5).items())} |
"""
    (OUTD / "here_kg_stats.md").write_text(stats)
    print(stats)
    print(f"wrote → {OUTD}")


if __name__ == "__main__":
    main()
