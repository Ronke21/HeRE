"""
Applications §3 (pilot, small dataset) — does denoised silver train a better
RE model than raw distant supervision?

Small-sample fine-tuning pilot: mmBERT-base predicate classifier trained on
50K silver examples, evaluated on the human-annotated gold test set.

  denoised  50K sampled from agree3-positive rows
  raw       50K sampled uniformly from all silver rows (size-matched control)

Input:  "subject [SEP] object [SEP] passage"  →  predicate (top-N classes).
Eval:   gold test rows the annotator confirmed (annotator1_Ron == 1) whose
        predicate is in the label set — accuracy + macro-F1.

Deliberately small (one GPU, ~1-2h per arm) — this is a pilot demonstrating
the corpus as RE training data, not a leaderboard entry.

Usage:
    python -m post_rebuttal_and_camera_ready.analysis.re_pilot --arm denoised
    python -m post_rebuttal_and_camera_ready.analysis.re_pilot --arm raw
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch._dynamo

# ModernBERT ships a torch.compile'd reference path; on hosts with old glibc
# (dgx01-03) triton's cached cuda_utils.so fails to import — run eager.
torch._dynamo.config.disable = True

from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MERGED = REPO / "post_rebuttal_and_camera_ready" / "silver_scoring_v3" / "silver_all_cleaned.parquet"
GOLD = REPO / "rebuttal" / "final_datasets" / "gold_test_set_v3.csv"
OUTD = HERE / "re_pilot"

A3P = ["opensource_llm__gemma4_31b__pred", "opensource_llm__gemma3_27b__pred",
       "finetuned_nli__dictalm24b__pred"]
MODEL = "jhu-clsp/mmBERT-base"
SEED = 42
N_TRAIN = 50_000
MIN_CLASS = 500          # predicate needs >=500 denoised examples to be a label
MAX_LEN = 256
BATCH = 64
EPOCHS = 2
LR = 2e-5


def build_data():
    df = pd.read_parquet(MERGED, columns=["subject", "predicate", "object", "text"] + A3P)
    unanimous = (df[A3P[0]] == df[A3P[1]]) & (df[A3P[1]] == df[A3P[2]])
    den = df[unanimous & (df[A3P[0]] == 1)]

    gold = pd.read_csv(GOLD)
    gold_pos = gold[gold["annotator1_Ron"] == 1]

    pc = den["predicate"].value_counts()
    labels = sorted(set(pc[pc >= MIN_CLASS].index) & set(gold_pos["predicate"]))
    l_idx = {p: i for i, p in enumerate(labels)}
    print(f"[labels] {len(labels)} predicates; gold-pos eval rows in label set: "
          f"{gold_pos['predicate'].isin(labels).sum()}/{len(gold_pos)}")

    rng = np.random.default_rng(SEED)

    def sample(pool):
        pool = pool[pool["predicate"].isin(labels)]
        idx = rng.choice(len(pool), size=min(N_TRAIN, len(pool)), replace=False)
        return pool.iloc[idx]

    arms = {"denoised": sample(den), "raw": sample(df)}
    ev = gold_pos[gold_pos["predicate"].isin(labels)]
    return arms, ev, labels, l_idx


def encode(tok, rows, l_idx, dev):
    texts = (rows["subject"].astype(str) + " [SEP] " + rows["object"].astype(str)
             + " [SEP] " + rows["text"].astype(str)).tolist()
    y = torch.tensor([l_idx[p] for p in rows["predicate"]], device=dev)
    return texts, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["denoised", "raw"])
    args = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)

    arms, ev, labels, l_idx = build_data()
    train = arms[args.arm]
    print(f"[{args.arm}] train={len(train):,} eval={len(ev)} classes={len(labels)} device={dev}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(labels), reference_compile=False).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    texts, y = encode(tok, train, l_idx, dev)
    n = len(texts)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.default_rng(SEED + ep).permutation(n)
        tot, t0 = 0.0, time.time()
        for i in range(0, n, BATCH):
            bi = perm[i:i + BATCH]
            enc = tok([texts[j] for j in bi], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(dev)
            loss = model(**enc, labels=y[bi]).loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(bi)
            if (i // BATCH) % 100 == 0:
                print(f"  ep{ep+1} step {i//BATCH}/{n//BATCH} avg_loss={tot/(i+len(bi)):.4f}", flush=True)
        print(f"  epoch {ep+1}/{EPOCHS} avg_loss={tot/n:.4f} ({time.time()-t0:.0f}s)", flush=True)

    model.eval()
    etexts, ey = encode(tok, ev, l_idx, dev)
    preds = []
    with torch.no_grad():
        for i in range(0, len(etexts), BATCH):
            enc = tok(etexts[i:i + BATCH], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(dev)
            preds.extend(model(**enc).logits.argmax(-1).tolist())
    yt = ey.cpu().numpy()
    res = {"arm": args.arm, "n_train": int(len(train)), "n_eval": int(len(ev)),
           "n_classes": len(labels),
           "accuracy": float(accuracy_score(yt, preds)),
           "macro_f1": float(f1_score(yt, preds, average="macro"))}
    print(json.dumps(res, indent=2))
    with open(OUTD / f"result_{args.arm}.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
