"""
Stage 0 — build every v3 dataset the re-run needs. CPU only, no GPU.

Applies v3 text cleaning AND clean_entity() to subject / predicate / object,
which the 2026-08-12 audit showed carry the same wiki markup as the passages
(per 200k silver rows: 496 subjects and 73 objects contain [[...]], 139
subjects contain a pipe, 81 contain nbsp). Those values are substring-matched
against the passage and embedded verbatim into the hypothesis templates, so
leaving them dirty puts markup straight into the model input.

Outputs (all under post_rebuttal_and_camera_ready/data_v3/):
    prepared_silver_v3.csv / .parquet     2,564,534 rows
    prepared_gold_500_v3.csv              500 rows
    gold_validation_set_v3.csv            500 rows, label = annotator1_Ron
    gold_test_set_v3.csv                  500 rows, label = annotator1_Ron

Resumable: each step writes a `.done` marker containing the output row count.
A step whose marker matches its expected row count is skipped. The silver step
writes through a `.partial` file and only renames on success, so an interrupted
run never leaves a half-written CSV that looks complete.

Usage:
    python -m post_rebuttal_and_camera_ready.pipeline.prepare_v3_data
    python -m post_rebuttal_and_camera_ready.pipeline.prepare_v3_data --only silver
    python -m post_rebuttal_and_camera_ready.pipeline.prepare_v3_data --force
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time

import pandas as pd

from post_rebuttal_and_camera_ready.pipeline import paths as P
from scripts.prepare_data.text_cleaning import clean, clean_entity

ENTITY_COLS = ("subject", "predicate", "object")
SILVER_FIELDS = ["docid", "title", "text", "subject", "predicate", "object"]

log = logging.getLogger("prepare_v3")


def setup_logging(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)


def _marker(out_path):
    return out_path.with_suffix(out_path.suffix + ".done")


def _is_done(out_path, expected_rows=None):
    m = _marker(out_path)
    if not (m.exists() and out_path.exists()):
        return False
    try:
        rows = json.loads(m.read_text())["rows"]
    except Exception:
        return False
    return expected_rows is None or rows == expected_rows


def _mark_done(out_path, rows):
    _marker(out_path).write_text(json.dumps({"rows": rows, "at": time.time()}))


def _clean_frame(df):
    """v3-clean the text column and normalise the entity columns."""
    df = df.copy()
    df["text"] = df["text"].astype(str).map(lambda t: clean(t, "v3"))
    for c in ENTITY_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).map(clean_entity)
    return df


# ---------------------------------------------------------------------------
# silver
# ---------------------------------------------------------------------------

def build_silver(force=False):
    out = P.SILVER_V3_CSV
    if not force and _is_done(out, P.SILVER_ROWS):
        log.info(f"silver: already complete ({P.SILVER_ROWS:,} rows) — skipping")
    else:
        partial = out.with_suffix(".csv.partial")
        log.info(f"silver: {P.RAW_SILVER}  ->  {out}")
        csv.field_size_limit(sys.maxsize)
        t0 = time.time()
        n = 0
        with open(P.RAW_SILVER, encoding="utf-8-sig", newline="") as fin, \
             open(partial, "w", encoding="utf-8", newline="") as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=SILVER_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                row["text"] = clean(row["text"], "v3")
                for c in ENTITY_COLS:
                    row[c] = clean_entity(row[c])
                writer.writerow(row)
                n += 1
                if n % 200_000 == 0:
                    el = time.time() - t0
                    log.info(f"  {n:,} rows  {el/60:.1f} min  {n/el:.0f} rows/s")
        os.replace(partial, out)
        log.info(f"silver: {n:,} rows in {(time.time()-t0)/60:.1f} min")
        _mark_done(out, n)

    pq = P.SILVER_V3_PARQUET
    if force or not _is_done(pq, P.SILVER_ROWS):
        log.info(f"silver: writing parquet -> {pq}")
        df = pd.read_csv(out, dtype=str, keep_default_na=False)
        df.to_parquet(pq, compression="snappy", index=False)
        _mark_done(pq, len(df))
        log.info(f"silver: parquet {len(df):,} rows")
    else:
        log.info("silver: parquet already complete — skipping")


# ---------------------------------------------------------------------------
# gold-500
# ---------------------------------------------------------------------------

def build_gold(force=False):
    out = P.GOLD_V3_CSV
    if not force and _is_done(out, 500):
        log.info("gold: already complete — skipping")
        return
    raw = pd.read_csv(P.RAW_GOLD)
    prep = pd.read_csv(P.PREPARED_GOLD_V1)
    key = ["docid", "subject", "predicate", "object"]
    rel = ["basic_relation", "template_relation", "llm_relation"]
    merged = raw.merge(prep[key + rel], on=key, how="left", validate="one_to_one")
    if merged[rel].isna().any().any():
        raise SystemExit("gold: relation columns did not join")

    # Relation columns are pure functions of the triple and never see the
    # passage, so they carry over. They do embed the entity strings, so they
    # get the same entity normalisation applied.
    merged = _clean_frame(merged)
    for c in rel:
        merged[c] = merged[c].astype(str).map(clean_entity)
    merged["word_count"] = merged["text"].str.split().str.len()
    merged.to_csv(out, index=False)
    _mark_done(out, len(merged))
    log.info(f"gold: {len(merged)} rows, mean words {merged['word_count'].mean():.1f}")


# ---------------------------------------------------------------------------
# validation / test evaluation sets
# ---------------------------------------------------------------------------

def _raw_text_for_test(test_df):
    """One streaming pass over the silver corpus to recover raw test text."""
    wanted = {}
    for _, r in test_df.iterrows():
        k = (str(r["docid"]), str(r["subject"]), str(r["predicate"]), str(r["object"]))
        wanted.setdefault(k, []).append(r["item_id"])
        alt = r.get("silver_join_subject")
        if isinstance(alt, str) and alt and alt != r["subject"]:
            k2 = (str(r["docid"]), alt, str(r["predicate"]), str(r["object"]))
            wanted.setdefault(k2, []).append(r["item_id"])
    found = {}
    csv.field_size_limit(sys.maxsize)
    t0 = time.time()
    with open(P.RAW_SILVER, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            k = (str(row["docid"]), str(row["subject"]),
                 str(row["predicate"]), str(row["object"]))
            if k in wanted:
                for item_id in wanted[k]:
                    found.setdefault(item_id, row["text"])
                if len(found) >= len(test_df):
                    break
            if i and i % 500_000 == 0:
                log.info(f"  scanned {i:,}, matched {len(found)}/{len(test_df)}"
                         f" ({time.time()-t0:.0f}s)")
    log.info(f"  matched {len(found)}/{len(test_df)} in {time.time()-t0:.0f}s")
    return found


def build_eval_sets(force=False):
    # validation — raw text comes from the gold file
    out = P.VALIDATION_V3_CSV
    if force or not _is_done(out, 500):
        val = pd.read_csv(P.REBUTTAL_VALIDATION)
        raw = pd.read_csv(P.RAW_GOLD)
        key = ["docid", "subject", "predicate", "object"]
        m = val.merge(raw[key + ["text"]], on=key, how="left", suffixes=("_old", "_raw"))
        if m["text_raw"].isna().any():
            raise SystemExit("validation: missing raw text")
        m["text"] = m["text_raw"]
        m = _clean_frame(m)
        m["relation_present"] = m["annotator1_Ron"].astype(int)
        m["word_count"] = m["text"].str.split().str.len()
        cols = ["item_id", "docid", "title", "subject", "predicate", "object", "text",
                "relation_present", "annotator1_Ron", "annotator2_Roee", "word_count"]
        m[[c for c in cols if c in m.columns]].to_csv(out, index=False)
        _mark_done(out, len(m))
        log.info(f"validation: {len(m)} rows, positive {m['relation_present'].mean():.3f}")
    else:
        log.info("validation: already complete — skipping")

    # test — raw text recovered from the silver corpus
    out = P.TEST_V3_CSV
    if force or not _is_done(out, 500):
        test = pd.read_csv(P.REBUTTAL_TEST)
        log.info("test: recovering raw text from the silver corpus ...")
        found = _raw_text_for_test(test)
        test["text"] = test["item_id"].map(found).fillna(test["text"])
        missing = int(test["item_id"].map(found).isna().sum())
        if missing:
            log.warning(f"test: {missing} rows fell back to their existing text")
        test = _clean_frame(test)
        test["relation_present"] = test["annotator1_Ron"].astype(int)
        test["word_count"] = test["text"].str.split().str.len()
        cols = ["item_id", "docid", "title", "subject", "predicate", "object", "text",
                "relation_present", "annotator1_Ron", "annotator2_Roee",
                "selection_reason", "word_count"]
        test[[c for c in cols if c in test.columns]].to_csv(out, index=False)
        _mark_done(out, len(test))
        log.info(f"test: {len(test)} rows, positive {test['relation_present'].mean():.3f}")
    else:
        log.info("test: already complete — skipping")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["silver", "gold", "eval"], default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    P.assert_repo_scripts()
    P.ensure_dirs()
    setup_logging(P.LOGS / "prepare_v3_data.log")
    log.info("=" * 70)
    log.info(f"prepare_v3_data  only={args.only or 'all'}  force={args.force}")
    log.info("=" * 70)

    if args.only in (None, "gold"):
        build_gold(args.force)
    if args.only in (None, "eval"):
        build_eval_sets(args.force)
    if args.only in (None, "silver"):
        build_silver(args.force)
    log.info("prepare_v3_data: done")


if __name__ == "__main__":
    main()
