"""
Regenerate prepared_gold_500.csv with v3 text cleaning, without re-running the LLM.

`llm_relation` is generated from (subject, predicate, object) only — the passage
text is never part of that prompt (see `_build_user_message` in
prepare_dataset.py). `basic_relation` and `template_relation` are likewise pure
functions of the triple. So a cleaning-level change affects the `text` column and
nothing else, and the three relation columns can be carried over verbatim. This
turns what would be a GPU job into a few seconds of CSV work.

Reads : data/crocodile_heb25_gold_500.csv     (raw source text)
        data/prepared_gold_500.csv            (existing relation columns)
Writes: data/prepared_gold_500_v3.csv
        data/prepared_gold_500_v1.csv         (backup of the current file)

Usage:
    python -m scripts.prepare_data.regenerate_gold_v3
    python -m scripts.prepare_data.regenerate_gold_v3 --level v2
"""

import argparse
import os
import shutil

import pandas as pd

from scripts.prepare_data.text_cleaning import clean

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PATH  = os.path.join(BASE_DIR, "data", "crocodile_heb25_gold_500.csv")
PREP_PATH = os.path.join(BASE_DIR, "data", "prepared_gold_500.csv")

KEY = ["docid", "subject", "predicate", "object"]
RELATION_COLS = ["basic_relation", "template_relation", "llm_relation"]
OUT_COLS = ["docid", "title", "text", "subject", "predicate", "object",
            "relation_present"] + RELATION_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="v3", choices=["v1", "v2", "v3"])
    ap.add_argument("--raw", default=RAW_PATH)
    ap.add_argument("--prepared", default=PREP_PATH)
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    out_path = args.output or os.path.join(
        BASE_DIR, "data", f"prepared_gold_500_{args.level}.csv")

    raw  = pd.read_csv(args.raw)
    prep = pd.read_csv(args.prepared)

    if raw.duplicated(KEY).any() or prep.duplicated(KEY).any():
        raise SystemExit("join key is not unique — aborting")
    if set(map(tuple, raw[KEY].values)) != set(map(tuple, prep[KEY].values)):
        raise SystemExit("raw and prepared key sets differ — aborting")

    # Carry the triple-derived columns over; recompute only `text`.
    merged = raw.merge(prep[KEY + RELATION_COLS], on=KEY, how="left", validate="one_to_one")
    if merged[RELATION_COLS].isna().any().any():
        raise SystemExit("some relation columns did not join — aborting")

    old_text = prep.set_index(KEY)["text"]
    merged["text"] = merged["text"].astype(str).map(lambda t: clean(t, args.level))

    new_text = merged.set_index(KEY)["text"]
    aligned_old = old_text.reindex(new_text.index)
    n_changed = int((aligned_old != new_text).sum())
    w_old = aligned_old.str.split().str.len().mean()
    w_new = new_text.str.split().str.len().mean()

    if not args.no_backup:
        backup = os.path.join(BASE_DIR, "data", "prepared_gold_500_v1.csv")
        if not os.path.exists(backup):
            shutil.copy2(args.prepared, backup)
            print(f"backed up current prepared file -> {backup}")

    merged[OUT_COLS].to_csv(out_path, index=False)

    print(f"level                : {args.level}")
    print(f"rows                 : {len(merged)}")
    print(f"text rows changed    : {n_changed} ({100*n_changed/len(merged):.1f}%)")
    print(f"mean words           : {w_old:.1f} -> {w_new:.1f}")
    print(f"relation cols reused : {', '.join(RELATION_COLS)} (no LLM re-run)")
    print(f"output               : {out_path}")


if __name__ == "__main__":
    main()
