"""
Merge all silver cleaning signals into a single file.

Sources (positional unless noted):
  base            : data/prepared_silver.parquet
  opensource LLM  : outputs/silver_opensource_llm/pred_*.txt          (float per line)
  finetuned NLI   : outputs/silver_finetuned_nli/pred_*.txt            (float per line)
  encoder NLI     : outputs/silver_encoder_nli/classifications/silver_encoder_nli.csv
  cross-train RC  : outputs/silver_cross_train_rc/{model}/silver_cleaned.csv  (key join)

Output columns (source__model__score|pred):
  docid, title, text, subject, predicate, object,
  opensource_llm__gemma4_31b__score/pred   (thresh 0.30)
  opensource_llm__gemma3_27b__score/pred   (thresh 0.30)
  opensource_llm__dictalm3__score/pred     (thresh 0.50)
  finetuned_nli__dictalm24b__score/pred    (thresh 0.30)
  finetuned_nli__aya32b__score/pred        (thresh 0.90)
  finetuned_nli__gemma2_9b__score/pred     (thresh 0.90)
  encoder_nli__xlmroberta__score/pred
  encoder_nli__neodictabert__score/pred
  cross_rc__neodictabert_k3__score/pred
  cross_rc__mmbert_k5__score/pred
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger()

ROOT    = Path(__file__).resolve().parent.parent.parent
DATA    = ROOT / "data"
OUT_LLM = ROOT / "outputs" / "silver_cleaning" / "silver_opensource_llm"
OUT_NLI = ROOT / "outputs" / "silver_cleaning" / "silver_finetuned_nli"
OUT_ENC = ROOT / "outputs" / "silver_cleaning" / "silver_encoder_nli" / "classifications"
OUT_RC  = ROOT / "outputs" / "silver_cleaning" / "silver_cross_train_rc"
OUT_DIR = ROOT / "outputs" / "silver_cleaning"

# ---------------------------------------------------------------------------
# Config: (output_col_prefix, pred_file, threshold)
# ---------------------------------------------------------------------------
LLM_MODELS = [
    ("opensource_llm__gemma4_31b", OUT_LLM / "pred_gemma4_31b_it.txt",       0.30),
    ("opensource_llm__gemma3_27b", OUT_LLM / "pred_gemma3_27b_it.txt",       0.30),
    ("opensource_llm__dictalm3",   OUT_LLM / "pred_dictalm3.txt",            0.50),
]
NLI_MODELS = [
    ("finetuned_nli__dictalm24b",  OUT_NLI / "pred_dictalm24b_base_v2.txt",  0.30),
    ("finetuned_nli__aya32b",      OUT_NLI / "pred_aya32b_v2.txt",           0.90),
    ("finetuned_nli__gemma2_9b",   OUT_NLI / "pred_gemma2_9b_v1.txt",        0.90),
]
ENC_MODELS = [
    # (output_prefix, source_score_col, source_pred_col)
    ("encoder_nli__xlmroberta",    "score_xlmroberta",    "pred_xlmroberta"),
    ("encoder_nli__neodictabert",  "score_neodictabert",  "pred_neodictabert"),
]
RC_MODELS = [
    # (output_prefix, silver_cleaned_csv)
    ("cross_rc__neodictabert_k3",  OUT_RC / "neodictabert_k3" / "silver_cleaned.csv"),
    ("cross_rc__mmbert_k5",        OUT_RC / "mmbert_k5"        / "silver_cleaned.csv"),
]

# ---------------------------------------------------------------------------

def load_pred_file(path: Path) -> np.ndarray:
    log.info(f"  loading {path.name}")
    return np.fromiter((float(line) for line in open(path) if line.strip()),
                       dtype=np.float32)


def main():
    t0 = time.time()

    # --- base ---------------------------------------------------------------
    log.info("Loading base parquet …")
    df = pd.read_parquet(DATA / "prepared_silver.parquet")
    n  = len(df)
    log.info(f"  {n:,} rows, columns: {list(df.columns)}")

    # --- positional pred files (LLM + finetuned NLI) -----------------------
    for prefix, path, thresh in LLM_MODELS + NLI_MODELS:
        scores = load_pred_file(path)
        assert len(scores) == n, f"{path.name}: {len(scores)} rows != {n}"
        df[f"{prefix}__score"] = scores
        df[f"{prefix}__pred"]  = (scores >= thresh).astype(np.int8)

    # --- encoder NLI CSV (positional) ---------------------------------------
    log.info("Loading encoder NLI CSV …")
    enc = pd.read_csv(
        OUT_ENC / "silver_encoder_nli.csv",
        usecols=["docid"] + [c for _, c, _ in ENC_MODELS] + [c for _, _, c in ENC_MODELS],
    )
    assert len(enc) == n, f"encoder NLI: {len(enc)} rows != {n}"
    # verify docid alignment
    assert (enc["docid"].values == df["docid"].values).all(), \
        "encoder NLI docid order mismatch — cannot use positional join"
    for prefix, score_col, pred_col in ENC_MODELS:
        df[f"{prefix}__score"] = enc[score_col].values.astype(np.float32)
        df[f"{prefix}__pred"]  = enc[pred_col].values.astype(np.int8)

    # --- cross-train RC (key join) ------------------------------------------
    KEY = ["docid", "subject", "predicate", "object"]
    for prefix, csv_path in RC_MODELS:
        log.info(f"  joining {csv_path.name} …")
        rc = pd.read_csv(csv_path, usecols=KEY + ["vote_fraction", "cleaned_label"])
        rc = rc.rename(columns={
            "vote_fraction":  f"{prefix}__score",
            "cleaned_label":  f"{prefix}__pred",
        })
        rc[f"{prefix}__pred"] = rc[f"{prefix}__pred"].astype(np.int8)
        df = df.merge(rc[KEY + [f"{prefix}__score", f"{prefix}__pred"]],
                      on=KEY, how="left")
        missing = df[f"{prefix}__pred"].isna().sum()
        if missing:
            log.warning(f"  {missing:,} rows without {prefix} prediction (filled 0/NaN)")

    # --- ensemble columns ---------------------------------------------------
    log.info("Computing ensemble columns …")
    pred_cols = [c for c in df.columns if c.endswith("__pred")]

    # 1. majority vote of ALL models
    all_pred = df[pred_cols].fillna(0).astype(np.int8)
    df["vote_all__pred"] = (all_pred.sum(axis=1) > len(pred_cols) / 2).astype(np.int8)

    # 2. majority vote of selected 7 models
    VOTE7 = [
        "opensource_llm__gemma4_31b__pred",
        "opensource_llm__gemma3_27b__pred",
        "opensource_llm__dictalm3__pred",
        "finetuned_nli__dictalm24b__pred",
        "encoder_nli__neodictabert__pred",
        "cross_rc__neodictabert_k3__pred",
        "cross_rc__mmbert_k5__pred",
    ]
    vote7_pred = df[VOTE7].fillna(0).astype(np.int8)
    df["vote7__pred"] = (vote7_pred.sum(axis=1) > len(VOTE7) / 2).astype(np.int8)

    # 3. unanimous agreement of 4 core models: 1=all-1, 0=all-0, -5=split
    AGREE4 = [
        "opensource_llm__gemma3_27b__pred",
        "finetuned_nli__dictalm24b__pred",
        "encoder_nli__neodictabert__pred",
        "cross_rc__neodictabert_k3__pred",
    ]
    agree4 = df[AGREE4].fillna(0).astype(np.int8)
    agree4_sum = agree4.sum(axis=1)
    df["agree4__pred"] = np.where(agree4_sum == len(AGREE4), 1,
                          np.where(agree4_sum == 0, 0, -5)).astype(np.int8)

    # 4. unanimous agreement of 2 models: gemma3_27b + dictalm24b
    AGREE2 = [
        "opensource_llm__gemma3_27b__pred",
        "finetuned_nli__dictalm24b__pred",
    ]
    agree2 = df[AGREE2].fillna(0).astype(np.int8)
    agree2_sum = agree2.sum(axis=1)
    df["agree2__pred"] = np.where(agree2_sum == len(AGREE2), 1,
                          np.where(agree2_sum == 0, 0, -5)).astype(np.int8)

    # --- write --------------------------------------------------------------
    out_parquet = OUT_DIR / "silver_all_cleaned.parquet"
    out_csv     = OUT_DIR / "silver_all_cleaned.csv"

    log.info(f"Writing parquet → {out_parquet} …")
    df.to_parquet(out_parquet, index=False)

    log.info(f"Writing CSV → {out_csv} …")
    df.to_csv(out_csv, index=False)

    elapsed = time.time() - t0
    log.info(f"Done in {elapsed/60:.1f} min.  Output shape: {df.shape}")
    log.info(f"Columns: {list(df.columns)}")

    # --- statistics summary -------------------------------------------------
    all_pred_cols = [c for c in df.columns if c.endswith("__pred")]
    rows = []
    for col in all_pred_cols:
        counts = df[col].value_counts().to_dict()
        n_pos   = int(counts.get(1,  0))
        n_neg   = int(counts.get(0,  0))
        n_split = int(counts.get(-5, 0))
        n_na    = int(df[col].isna().sum())
        n_total = n_pos + n_neg + n_split + n_na
        rows.append({
            "column":      col,
            "n_total":     n_total,
            "n_pos (1)":   n_pos,
            "n_neg (0)":   n_neg,
            "n_split (-5)": n_split,
            "n_na":        n_na,
            "pct_pos":     round(n_pos / n * 100, 2),
            "pct_neg":     round(n_neg / n * 100, 2),
            "pct_split":   round(n_split / n * 100, 2),
        })
    stats_df = pd.DataFrame(rows)
    out_stats = OUT_DIR / "silver_all_cleaned_stats.csv"
    stats_df.to_csv(out_stats, index=False)
    log.info(f"\nStatistics written → {out_stats}")
    log.info("\n" + stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
