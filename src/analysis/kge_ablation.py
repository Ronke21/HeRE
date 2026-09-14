"""
Applications §2 — does denoising improve a downstream KG task?

Trains a TransE knowledge-graph embedding on three variants of the HeRE graph
and evaluates link prediction on ONE shared held-out test set:

  raw       all unique distant-supervision triples (no denoising)
  denoised  unique agree3-positive triples
  control   a uniform random sample of `raw`, size-matched to `denoised`
            — the essential control: it separates "denoising selected good
            edges" from "smaller data trained better". Note agree3 skews
            toward high-consensus examples (the G finding), which is exactly
            why this control cannot be skipped.
  weighted  the full raw graph, with training triples sampled proportionally
            to their confidence (max-over-mentions mean signal score) — soft
            denoising: keeps raw's coverage while down-weighting noise.

Two evaluations, both on the same held-out quality triples:
  * filtered link prediction (MRR, Hits@1/10) — coverage-oriented;
  * triple classification (accuracy/F1 vs corrupted negatives, global
    threshold tuned on a validation split) — precision-oriented, the metric
    that matches fact-validation use cases. Negatives and threshold protocol
    are identical across variants.

Test edges are sampled from the denoised graph (quality edges), removed from
the TRAINING SET OF EVERY VARIANT, and restricted to entities/relations seen
in all three training graphs so no variant is punished for unseen vocabulary.
Metric: filtered MRR and Hits@10 over full entity ranking (chunked matmul).

Self-contained TransE in plain torch (no pykeen dependency). One V100 per
variant; ~1-2h each.

Usage:
    python -m post_rebuttal_and_camera_ready.analysis.kge_ablation --variant denoised
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MERGED = REPO / "post_rebuttal_and_camera_ready" / "silver_scoring_v3" / "silver_all_cleaned.parquet"
OUTD = HERE / "kge_ablation"

A3P = ["opensource_llm__gemma4_31b__pred", "opensource_llm__gemma3_27b__pred",
       "finetuned_nli__dictalm24b__pred"]
SEED = 42
DIM = 128
MARGIN = 6.0
LR = 1e-3
EPOCHS = 50   # training is cheap (~4-10s/epoch on V100); 5 epochs was far from converged
BATCH = 4096
N_TEST = 20_000
EVAL_CHUNK = 1_000   # distance matrix is chunk x n_entities floats — keep it ~2-3GB on V100


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


SCORES = ["opensource_llm__gemma4_31b__score", "opensource_llm__gemma3_27b__score",
          "finetuned_nli__dictalm24b__score"]


def load_graphs(seed: int = SEED):
    df = pd.read_parquet(MERGED, columns=["docid", "subject", "predicate", "object"] + A3P + SCORES)
    df = df[~test_row_mask(df)].reset_index(drop=True)
    unanimous = (df[A3P[0]] == df[A3P[1]]) & (df[A3P[1]] == df[A3P[2]])
    den_mask = (unanimous & (df[A3P[0]] == 1)).to_numpy()

    # shared vocabulary over the raw graph (superset of all variants)
    ents = pd.unique(pd.concat([df["subject"], df["object"]], ignore_index=True))
    rels = pd.unique(df["predicate"])
    e_idx = {e: i for i, e in enumerate(ents)}
    r_idx = {r: i for i, r in enumerate(rels)}
    h = df["subject"].map(e_idx).to_numpy(np.int64)
    r = df["predicate"].map(r_idx).to_numpy(np.int64)
    t = df["object"].map(e_idx).to_numpy(np.int64)
    triples = np.stack([h, r, t], axis=1)

    # Deduplicate to unique (s,p,o) triples and split on TRIPLES, not mention
    # rows: the corpus holds multiple mentions of the same triple, so a
    # row-level split leaves other mentions of every test triple in training
    # (worst for `raw`, which has the most mentions per triple) — leakage.
    def triple_key(tr):
        return (tr[:, 0] * len(rels) + tr[:, 1]) * len(ents) + tr[:, 2]

    key = triple_key(triples)
    raw_u = triples[np.unique(key, return_index=True)[1]]
    den_rows = np.flatnonzero(den_mask)
    den_u = triples[den_rows[np.unique(key[den_mask], return_index=True)[1]]]

    # per-unique-triple confidence = max over its mentions of the mean signal
    # score (a triple is as credible as its best evidence)
    mean_score = df[SCORES].mean(axis=1).to_numpy()
    w_by_key = pd.Series(mean_score).groupby(pd.Series(key)).max()

    rng = np.random.default_rng(SEED)          # test split: fixed across seeds
    test_pick = rng.choice(len(den_u), size=N_TEST, replace=False)
    rng = np.random.default_rng(seed)          # control sample: varies with seed
    test_key_arr = np.unique(triple_key(den_u[test_pick]))

    def drop_test(tr):
        return tr[~np.isin(triple_key(tr), test_key_arr)]

    raw_train = drop_test(raw_u)
    den_train = drop_test(den_u)
    ctrl_train = raw_train[rng.choice(len(raw_train), size=len(den_train), replace=False)]
    w_raw = np.clip(w_by_key.loc[triple_key(raw_train)].to_numpy(), 1e-4, None)

    # test filtered to entities/relations present in ALL training variants
    def vocab(tr): return set(tr[:, 0]) | set(tr[:, 2]), set(tr[:, 1])
    e_ok = set.intersection(*[vocab(x)[0] for x in (raw_train, den_train, ctrl_train)])
    r_ok = set.intersection(*[vocab(x)[1] for x in (raw_train, den_train, ctrl_train)])
    test = den_u[test_pick]
    keep = np.array([(a in e_ok) and (b in r_ok) and (c in e_ok) for a, b, c in test])
    test = test[keep]
    print(f"[graphs] raw={len(raw_train):,} denoised={len(den_train):,} "
          f"control={len(ctrl_train):,} test={len(test):,} "
          f"entities={len(ents):,} relations={len(rels):,}")
    # (h,r) -> known true tails, for filtered ranking and negative validity
    hr2t = {}
    for a, b, c in triples:
        hr2t.setdefault((int(a), int(b)), set()).add(int(c))
    graphs = {"raw": raw_train, "denoised": den_train, "control": ctrl_train,
              "weighted": raw_train}
    weights = {"weighted": w_raw}
    return graphs, weights, test, len(ents), len(rels), hr2t


def train_transe(train: np.ndarray, n_ent: int, n_rel: int, dev,
                 sample_weights: np.ndarray | None = None, seed: int = SEED):
    torch.manual_seed(seed)
    E = torch.nn.Embedding(n_ent, DIM, max_norm=1.0).to(dev)
    R = torch.nn.Embedding(n_rel, DIM).to(dev)
    torch.nn.init.xavier_uniform_(E.weight); torch.nn.init.xavier_uniform_(R.weight)
    opt = torch.optim.Adam(list(E.parameters()) + list(R.parameters()), lr=LR)
    tr = torch.from_numpy(train).to(dev)
    w = None if sample_weights is None else torch.from_numpy(
        sample_weights.astype(np.float64)).to(dev)
    n = len(tr)
    for ep in range(EPOCHS):
        if w is None:
            perm = torch.randperm(n, device=dev)
        else:
            # confidence-proportional sampling with replacement (soft denoise)
            perm = torch.multinomial(w, n, replacement=True)
        tot = 0.0
        t0 = time.time()
        for i in range(0, n, BATCH):
            b = tr[perm[i:i + BATCH]]
            hneg = b.clone()
            corrupt_head = torch.rand(len(b), device=dev) < 0.5
            rand_e = torch.randint(0, n_ent, (len(b),), device=dev)
            hneg[corrupt_head, 0] = rand_e[corrupt_head]
            hneg[~corrupt_head, 2] = rand_e[~corrupt_head]
            def score(x):
                return (E(x[:, 0]) + R(x[:, 1]) - E(x[:, 2])).norm(p=1, dim=1)
            loss = torch.relu(MARGIN + score(b) - score(hneg)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        print(f"  epoch {ep+1}/{EPOCHS} loss={tot/n:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return E, R


@torch.no_grad()
def evaluate(E, R, test: np.ndarray, hr2t: dict, n_ent: int, dev):
    """Filtered tail ranking: a candidate tail that is itself a known-true
    triple (in the full raw graph) does not count against the gold tail."""
    ranks = []
    ew = E.weight                                     # (n_ent, dim)
    for i in range(0, len(test), EVAL_CHUNK):
        chunk = test[i:i + EVAL_CHUNK]
        h = torch.from_numpy(chunk[:, 0]).to(dev)
        r = torch.from_numpy(chunk[:, 1]).to(dev)
        t = torch.from_numpy(chunk[:, 2]).to(dev)
        q = E(h) + R(r)                               # (c, dim)
        d = torch.cdist(q, ew, p=1)                   # (c, n_ent) tail distances
        gold = d[torch.arange(len(chunk), device=dev), t]
        better = (d < gold.unsqueeze(1)).sum(dim=1)   # raw count ranked better
        for j, (hh, rr, tt) in enumerate(chunk):
            known = hr2t.get((int(hh), int(rr)), ())
            others = [e for e in known if e != int(tt)]
            n_filt = 0
            if others:
                idx = torch.tensor(others, device=dev)
                n_filt = int((d[j, idx] < gold[j]).sum())
            ranks.append(int(better[j]) - n_filt + 1)
        del d
    ranks = np.array(ranks)
    return {"mrr": float((1 / ranks).mean()), "hits@10": float((ranks <= 10).mean()),
            "hits@1": float((ranks <= 1).mean()), "n_test": int(len(ranks))}


@torch.no_grad()
def triple_classification(E, R, test: np.ndarray, hr2t: dict, n_ent: int, dev):
    """Precision-oriented eval: distinguish held-out true triples from
    tail-corrupted negatives (one per positive, resampled until not a known
    positive). Global distance threshold tuned on a 20% validation split;
    accuracy/F1 reported on the remaining 80%. Same seed => identical
    positives/negatives for every variant."""
    rng = np.random.default_rng(SEED + 1)
    neg = test.copy()
    for i in range(len(neg)):
        h, r = int(neg[i, 0]), int(neg[i, 1])
        known = hr2t.get((h, r), set())
        t = int(rng.integers(0, n_ent))
        while t in known:
            t = int(rng.integers(0, n_ent))
        neg[i, 2] = t

    def dist(tr):
        out = []
        for i in range(0, len(tr), 50_000):
            b = torch.from_numpy(tr[i:i + 50_000]).to(dev)
            out.append((E(b[:, 0]) + R(b[:, 1]) - E(b[:, 2]))
                       .norm(p=1, dim=1).cpu().numpy())
        return np.concatenate(out)

    d_pos, d_neg = dist(test), dist(neg)
    n_val = len(test) // 5
    scores = np.concatenate([d_pos[:n_val], d_neg[:n_val]])
    labels = np.concatenate([np.ones(n_val), np.zeros(n_val)])
    # tune: predict positive iff distance < thr; scan candidate thresholds
    cand = np.quantile(scores, np.linspace(0.01, 0.99, 199))
    accs = [( (scores < c).astype(int) == labels ).mean() for c in cand]
    thr = float(cand[int(np.argmax(accs))])

    pp, pn = d_pos[n_val:] < thr, d_neg[n_val:] < thr
    tp, fp = pp.sum(), pn.sum()
    fn = (~pp).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"tc_accuracy": float((pp.mean() + (~pn).mean()) / 2),
            "tc_f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
            "tc_threshold": thr, "tc_n_eval": int(len(test) - n_val)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["raw", "denoised", "control", "weighted"])
    ap.add_argument("--seed", type=int, default=SEED,
                    help="training/control-sample seed; the held-out split is fixed")
    args = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.variant}] device={dev}")

    graphs, weights, test, n_ent, n_rel, hr2t = load_graphs(args.seed)
    E, R = train_transe(graphs[args.variant], n_ent, n_rel, dev,
                        sample_weights=weights.get(args.variant), seed=args.seed)
    res = evaluate(E, R, test, hr2t, n_ent, dev)
    res.update(triple_classification(E, R, test, hr2t, n_ent, dev))
    res["variant"] = args.variant
    res["train_edges"] = int(len(graphs[args.variant]))
    res["seed"] = args.seed
    print(json.dumps(res, indent=2))
    with open(OUTD / f"result_{args.variant}_seed{args.seed}.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
