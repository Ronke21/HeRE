"""
Preprocess the full silver dataset (text cleaning only).

Text cleaning is shared with prepare_dataset.py via
scripts/prepare_data/text_cleaning.py — see that module for the rule list.

Reads : data/crocodile_heb25_full_without_gold_and_duplicates_2564.csv
Writes: data/prepared_silver_v3.csv

Usage:
    python -m scripts.prepare_data.prepare_silver
    python -m scripts.prepare_data.prepare_silver --max-rows 100000   # quick test
"""

import csv
import os

import time
import logging
import argparse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_FILE  = "data/crocodile_heb25_full_without_gold_and_duplicates_2564.csv"
OUTPUT_FILE = "data/prepared_silver_v3.csv"
LOG_FILE    = "outputs/prepare_silver_v3.log"
LOG_EVERY_N = 200_000

FIELDNAMES  = ["docid", "title", "text", "subject", "predicate", "object"]

# ---------------------------------------------------------------------------
# Text preprocessing
#
# Delegated to scripts/prepare_data/text_cleaning.py, shared with
# prepare_dataset.py so the gold and silver paths cannot drift apart again
# (they did between 2026-07-10 and 2026-08-12 — see that module's docstring).
# ---------------------------------------------------------------------------

from scripts.prepare_data.text_cleaning import clean, preprocess_text  # noqa: F401


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _fmt(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, s   = divmod(rem, 60)
    if h:  return f"{h}h {m}m {s}s"
    if m:  return f"{m}m {s}s"
    return f"{s}s"


def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("prepare_silver")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default=INPUT_FILE)
    parser.add_argument("--output",    default=OUTPUT_FILE)
    parser.add_argument("--log",       default=LOG_FILE)
    parser.add_argument("--max-rows",  type=int, default=None,
                        help="Stop after this many rows (for testing)")
    args = parser.parse_args()

    base        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path  = os.path.join(base, args.input)
    output_path = os.path.join(base, args.output)
    log_path    = os.path.join(base, args.log)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log = setup_logger(log_path)
    log.info("=" * 60)
    log.info("prepare_silver.py  started")
    log.info(f"  input : {input_path}")
    log.info(f"  output: {output_path}")
    log.info("=" * 60)

    t_start = time.time()
    n_written = 0

    with open(input_path,  encoding="utf-8-sig", newline="") as fin, \
         open(output_path, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for i, row in enumerate(reader):
            if args.max_rows and i >= args.max_rows:
                break
            row["text"] = preprocess_text(row["text"])
            writer.writerow(row)
            n_written += 1

            if n_written % LOG_EVERY_N == 0:
                elapsed = time.time() - t_start
                rps     = n_written / elapsed if elapsed > 0 else 0
                log.info(f"  {n_written:,} rows written  "
                         f"elapsed={_fmt(elapsed)}  speed={rps:.0f} rows/s")

    elapsed = time.time() - t_start
    rps     = n_written / elapsed if elapsed > 0 else 0
    log.info("=" * 60)
    log.info(f"  done: {n_written:,} rows  elapsed={_fmt(elapsed)}  speed={rps:.0f} rows/s")
    log.info(f"  output: {output_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()