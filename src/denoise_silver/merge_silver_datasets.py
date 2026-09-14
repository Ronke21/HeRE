"""
Merge the three per-method silver classification CSVs into one unified file.

Input CSVs (each produced by its own classify_silver_*.py script):
  outputs/silver_encoder_nli/classifications/silver_encoder_nli.csv
    → score_xlmroberta, pred_xlmroberta, score_neodictabert, pred_neodictabert
  outputs/silver_opensource_llm/classifications/silver_opensource_llm.csv
    → score_gemma4_31b_it, pred_gemma4_31b_it, score_dictalm3, pred_dictalm3
  outputs/silver_finetuned_nli/classifications/silver_finetuned_nli.csv
    → score_dictalm24b_v2, pred_dictalm24b_v2, score_aya32b_v2, pred_aya32b_v2

Output:
  outputs/silver_combined/silver_combined.csv
    SILVER_COLS + all 12 model columns (score + pred per model)

Row alignment: all input CSVs are derived from prepared_silver.csv in the same row
order, so merging is done by position (zip). Alignment is verified via
docid/subject/predicate/object on every row — mismatch raises an error.

The script tolerates partially-completed runs: if a source CSV does not exist yet,
its columns are written as empty strings.

Usage:
  conda activate hre_finetuned_nli
  python -m scripts_silver_cleaning.merge_silver_datasets
  python -m scripts_silver_cleaning.merge_silver_datasets --skip-verify
"""

import csv
import logging
import time
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent

SOURCES = [
    {
        "name":    "encoder_nli",
        "csv":     ROOT / "outputs" / "silver_encoder_nli" / "classifications" / "silver_encoder_nli.csv",
        "cols":    ["score_xlmroberta", "pred_xlmroberta", "score_neodictabert", "pred_neodictabert"],
    },
    {
        "name":    "opensource_llm",
        "csv":     ROOT / "outputs" / "silver_opensource_llm" / "classifications" / "silver_opensource_llm.csv",
        "cols":    ["score_gemma4_31b_it", "pred_gemma4_31b_it", "score_dictalm3", "pred_dictalm3"],
    },
    {
        "name":    "finetuned_nli",
        "csv":     ROOT / "outputs" / "silver_finetuned_nli" / "classifications" / "silver_finetuned_nli.csv",
        "cols":    ["score_dictalm24b_v2", "pred_dictalm24b_v2", "score_aya32b_v2", "pred_aya32b_v2"],
    },
]

OUT_DIR  = ROOT / "outputs" / "silver_combined"
OUT_CSV  = OUT_DIR / "silver_combined.csv"
LOG_FILE = OUT_DIR / "merge.log"
SUMMARY  = OUT_DIR / "summary.txt"

SILVER_COLS  = ["docid", "title", "text", "subject", "predicate", "object"]
ALIGN_KEYS   = ["docid", "subject", "predicate", "object"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _fmt_dur(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h:  return f"{h}h {m}m {sec}s"
    if m:  return f"{m}m {sec}s"
    return f"{sec}s"


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("merge_silver_datasets")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_key(row: dict) -> tuple:
    return tuple(row.get(k, "") for k in ALIGN_KEYS)


def _count_rows(path: Path) -> int:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return sum(1 for _ in reader)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(verify: bool, log: logging.Logger) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Open available source files; absent ones get a sentinel None
    handles = []
    readers = []
    available = []
    for src in SOURCES:
        if src["csv"].exists():
            fh = open(src["csv"], encoding="utf-8-sig", newline="")
            handles.append(fh)
            readers.append(csv.DictReader(fh))
            available.append(src)
            log.info(f"  [found]    {src['name']}  →  {src['csv']}")
        else:
            handles.append(None)
            readers.append(None)
            available.append(None)
            log.warning(f"  [missing]  {src['name']}  →  {src['csv']}  (columns will be empty)")

    # Build output column list in fixed order
    extra_cols = [col for src in SOURCES for col in src["cols"]]
    output_cols = SILVER_COLS + extra_cols

    written = 0
    mismatches = 0

    try:
        with open(OUT_CSV, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=output_cols)
            writer.writeheader()

            # Use first available source as the row driver
            primary_idx = next(i for i, r in enumerate(readers) if r is not None)
            primary_reader = readers[primary_idx]

            for primary_row in primary_reader:
                out_row = {col: primary_row.get(col, "") for col in SILVER_COLS}
                primary_key = _align_key(primary_row)

                # Add extra cols from primary source
                for col in SOURCES[primary_idx]["cols"]:
                    out_row[col] = primary_row.get(col, "")

                # Read one row from each other source
                for i, (src, reader) in enumerate(zip(SOURCES, readers)):
                    if i == primary_idx or reader is None:
                        if available[i] is None:
                            for col in SOURCES[i]["cols"]:
                                out_row[col] = ""
                        continue
                    try:
                        other_row = next(reader)
                    except StopIteration:
                        for col in src["cols"]:
                            out_row[col] = ""
                        continue

                    if verify:
                        other_key = _align_key(other_row)
                        if other_key != primary_key:
                            mismatches += 1
                            if mismatches <= 5:
                                log.error(
                                    f"  [mismatch row {written}]  "
                                    f"primary={primary_key}  {src['name']}={other_key}"
                                )
                            if mismatches == 1:
                                raise RuntimeError(
                                    f"Row alignment mismatch at row {written} between "
                                    f"{SOURCES[primary_idx]['name']} and {src['name']}. "
                                    f"Re-run without --skip-verify to confirm or check source CSVs."
                                )

                    for col in src["cols"]:
                        out_row[col] = other_row.get(col, "")

                writer.writerow(out_row)
                written += 1
                if written % 500_000 == 0:
                    log.info(f"[merge]  {written:,} rows written")

    finally:
        for fh in handles:
            if fh is not None:
                fh.close()

    log.info(f"[merge]  {written:,} rows → {OUT_CSV}")
    return written


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(
    written: int,
    total_time: float,
    log: logging.Logger,
) -> None:
    sep = "=" * 80
    lines = [
        sep,
        "SILVER COMBINED DATASET SUMMARY",
        f"Date       : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total rows : {written:,}",
        f"Wall time  : {_fmt_dur(total_time)}",
        f"Output     : {OUT_CSV}",
        sep, "",
        "SOURCE FILES",
    ]
    for src in SOURCES:
        exists = src["csv"].exists()
        status = "OK" if exists else "MISSING (columns empty)"
        rows   = _count_rows(src["csv"]) if exists else 0
        lines += [
            f"  [{src['name']}]  {status}",
            f"    file : {src['csv']}",
            f"    rows : {rows:,}" if exists else "",
            f"    cols : {', '.join(src['cols'])}",
        ]

    lines += ["", "OUTPUT COLUMNS"]
    lines.append(f"  base   : {', '.join(SILVER_COLS)}")
    for src in SOURCES:
        lines.append(f"  {src['name']:<16} : {', '.join(src['cols'])}")
    lines += [
        "",
        f"  Total columns: {len(SILVER_COLS) + sum(len(s['cols']) for s in SOURCES)}",
        sep,
    ]

    text = "\n".join(l for l in lines if l is not None)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    for line in lines:
        if line is not None:
            log.info(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge per-method silver CSVs into one unified silver_combined.csv"
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip row-alignment verification (faster but unsafe if sources are misaligned)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logger(LOG_FILE)
    log.info("=" * 70)
    log.info("merge_silver_datasets.py  started")
    log.info(f"  output : {OUT_CSV}")
    log.info(f"  verify : {not args.skip_verify}")
    log.info("=" * 70)

    for src in SOURCES:
        if src["csv"].exists():
            log.info(f"  [found]   {src['name']}  {src['csv']}")
        else:
            log.warning(f"  [missing] {src['name']}  {src['csv']}")

    t0      = time.time()
    written = merge(verify=not args.skip_verify, log=log)
    elapsed = time.time() - t0

    write_summary(written, elapsed, log)

    log.info("=" * 70)
    log.info(f"Done.  {written:,} rows  {_fmt_dur(elapsed)}")
    log.info(f"  output  : {OUT_CSV}")
    log.info(f"  summary : {SUMMARY}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
