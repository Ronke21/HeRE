"""
Stage 3 — merge the v3 silver signals into one table, into the new tree.

Same shim approach as run_classifier: `scripts/clean_silver/build_silver_all_cleaned.py`
hardcodes its input and output roots, so its module attributes are rebound
before main() rather than editing the file. The old script keeps working
against the old paths for anyone who runs it directly.

Note it reads `DATA / "prepared_silver.parquet"` by name inside main(), so DATA
is repointed at data_v3/ and the v3 parquet is exposed under the expected
filename via a symlink created here (a copy would be another 1.2 GB for no
reason).

Usage:
    python -m post_rebuttal_and_camera_ready.pipeline.build_silver_v3
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from post_rebuttal_and_camera_ready.pipeline import paths as P


def main():
    P.ensure_dirs()
    if not P.SILVER_V3_PARQUET.exists():
        raise SystemExit(f"missing {P.SILVER_V3_PARQUET} — run prepare_v3_data first")

    # The old script looks for exactly this basename inside its DATA dir.
    alias = P.DATA_V3 / "prepared_silver.parquet"
    if not alias.exists():
        os.symlink(P.SILVER_V3_PARQUET.name, alias)

    mod = importlib.import_module("scripts.clean_silver.build_silver_all_cleaned")
    mod.DATA = P.DATA_V3
    mod.OUT_LLM = P.SILVER_SCORES / "silver_opensource_llm"
    mod.OUT_NLI = P.SILVER_SCORES / "silver_finetuned_nli"
    mod.OUT_ENC = P.SILVER_SCORES / "silver_encoder_nli" / "classifications"
    mod.OUT_RC = P.SILVER_SCORES / "silver_cross_train_rc"
    mod.OUT_DIR = P.SILVER_SCORES

    old_root = str(P.ROOT / "outputs")
    for attr in ("OUT_LLM", "OUT_NLI", "OUT_ENC", "OUT_RC", "OUT_DIR"):
        if str(getattr(mod, attr)).startswith(old_root):
            raise SystemExit(f"shim failed: {attr} still points into outputs/")

    # The frozen script positionally joins each family CSV against the base
    # parquet and asserts docid equality first. The v3 parquet stores docid as
    # *string* while pandas infers int64 when reading the family CSVs, so the
    # elementwise == is all-False and the assert fires ("docid order mismatch")
    # even though the order is byte-identical — verified 2026-08-18 on the full
    # 2,564,534 rows. Force docid to str on every read_csv the module performs
    # rather than editing the frozen file. Patching mod.pd.read_csv patches
    # pandas process-wide, which is fine: run_unit runs this module and exits.
    _orig_read_csv = mod.pd.read_csv
    def _read_csv_docid_str(*a, **k):
        d = k.get("dtype")
        if d is None:
            k["dtype"] = {"docid": str}
        elif isinstance(d, dict):
            d.setdefault("docid", str)
        return _orig_read_csv(*a, **k)
    mod.pd.read_csv = _read_csv_docid_str

    print(f"[build_silver_v3] input  = {P.SILVER_V3_PARQUET}")
    print(f"[build_silver_v3] output = {mod.OUT_DIR}")
    old_argv, sys.argv = sys.argv, ["build_silver_all_cleaned"]
    try:
        mod.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
