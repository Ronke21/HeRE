"""
Convert a silver/gold CSV to Parquet for faster loading.

Usage:
    python -m scripts.prepare_data.csv_to_parquet                        # convert default silver
    python -m scripts.prepare_data.csv_to_parquet --input data/prepared_gold_500.csv
    python -m scripts.prepare_data.csv_to_parquet --input data/prepared_silver.csv --output data/prepared_silver.parquet
"""

import argparse
import os
import time

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/prepared_silver.csv")
    parser.add_argument("--output", default=None,
                        help="Output path (default: same as input with .parquet extension)")
    parser.add_argument("--compression", default="snappy", choices=["snappy", "gzip", "zstd", "none"])
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path  = os.path.join(base, args.input)
    output_path = args.output or os.path.splitext(input_path)[0] + ".parquet"

    print(f"Reading  {input_path} ...")
    t0 = time.time()
    df = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
    read_time = time.time() - t0
    print(f"  {len(df):,} rows  {len(df.columns)} cols  ({read_time:.1f}s)")

    compression = None if args.compression == "none" else args.compression
    print(f"Writing  {output_path}  (compression={compression}) ...")
    t1 = time.time()
    df.to_parquet(output_path, index=False, compression=compression)
    write_time = time.time() - t1

    size_csv = os.path.getsize(input_path)  / 1e6
    size_pq  = os.path.getsize(output_path) / 1e6
    print(f"  done in {write_time:.1f}s  |  {size_csv:.0f} MB → {size_pq:.0f} MB  ({size_pq/size_csv*100:.0f}%)")

    # Quick round-trip check
    print("Verifying round-trip ...")
    t2 = time.time()
    df2 = pd.read_parquet(output_path)
    reload_time = time.time() - t2
    assert len(df2) == len(df), "Row count mismatch!"
    print(f"  parquet reload: {reload_time:.1f}s  (vs CSV {read_time:.1f}s  →  {read_time/reload_time:.1f}× faster)")
    print("Done.")


if __name__ == "__main__":
    main()
