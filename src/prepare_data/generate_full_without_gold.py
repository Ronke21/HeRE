"""
Generate a version of the full dataset with gold rows removed.
Uses streaming to handle the large (12 GB) full dataset.


python -m scripts.prepare_data.generate_full_without_gold 2>&1 | tee generate_full_without_gold.log

"""
import csv
import os
import time

# Repo root, not this file's directory — the data lives in <repo>/data/.
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR  = os.path.join(BASE_DIR, "data")
FULL_PATH = os.path.join(DATA_DIR, "crocodile_heb25_full_dataset_3124k.csv")
GOLD_PATH = os.path.join(DATA_DIR, "crocodile_heb25_gold_500.csv")
OUT_PATH  = os.path.join(DATA_DIR, "crocodile_heb25_full_without_gold.csv")

KEY_COLS = ["docid", "subject", "predicate", "object"]
PRINT_EVERY = 100_000


def make_key(row):
    return tuple(row[c] for c in KEY_COLS)


def main():
    # 1. Load gold keys
    gold_keys = set()
    with open(GOLD_PATH, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gold_keys.add(make_key(row))

    print(f"Gold rows:             {len(gold_keys):>10,}")

    # 2. Stream full dataset, skip gold rows and duplicates
    kept = gold_skipped = dup_skipped = 0
    seen_keys = set()
    start = time.time()

    with open(FULL_PATH, newline="", encoding="utf-8-sig") as fin, \
         open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            key = make_key(row)
            if key in gold_keys:
                gold_skipped += 1
            elif key in seen_keys:
                dup_skipped += 1
            else:
                seen_keys.add(key)
                writer.writerow(row)
                kept += 1

            total = kept + gold_skipped + dup_skipped
            if total % PRINT_EVERY == 0:
                elapsed = time.time() - start
                rate = total / elapsed if elapsed else 0
                print(f"  processed {total:>10,} rows  |  {elapsed:6.1f}s  |  {rate:,.0f} rows/s",
                      flush=True)

    total = kept + gold_skipped + dup_skipped
    print(f"Full dataset rows:          {total:>10,}")
    print(f"  - gold rows removed:      {gold_skipped:>10,}")
    print(f"  - duplicates removed:     {dup_skipped:>10,}")
    print(f"  - rows kept:              {kept:>10,}")
    print(f"\nOutput written to: {OUT_PATH}")


if __name__ == "__main__":
    main()