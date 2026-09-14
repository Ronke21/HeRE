"""Canonical paths for the post-rebuttal v3 re-run.

Everything this pipeline produces lives under
`post_rebuttal_and_camera_ready/`. Nothing here writes to `data/`,
`outputs/`, or `rebuttal/` — the pre-rebuttal artifacts stay exactly as they
are, so the paper's existing numbers remain reproducible.

Read-only inputs from the old tree are listed explicitly in READ_ONLY_INPUTS.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POST = ROOT / "post_rebuttal_and_camera_ready"

# --- outputs (all new) -----------------------------------------------------
DATA_V3 = POST / "data_v3"
RUNS = POST / "runs"
LOGS = RUNS / "logs"
STATE_FILE = RUNS / "state.json"
SLURM_LOGS = LOGS / "slurm"
A100_LOGS = LOGS / "a100"

SILVER_V3_CSV = DATA_V3 / "prepared_silver_v3.csv"
SILVER_V3_PARQUET = DATA_V3 / "prepared_silver_v3.parquet"
GOLD_V3_CSV = DATA_V3 / "prepared_gold_500_v3.csv"
VALIDATION_V3_CSV = DATA_V3 / "gold_validation_set_v3.csv"
TEST_V3_CSV = DATA_V3 / "gold_test_set_v3.csv"

SILVER_SCORES = POST / "silver_scoring_v3"
GOLD_SCORES = POST / "gold_benchmark_v3"
ANALYSIS = POST / "analysis_v3"

# --- read-only inputs from the existing tree -------------------------------
RAW_SILVER = ROOT / "data" / "crocodile_heb25_full_without_gold_and_duplicates_2564.csv"
RAW_GOLD = ROOT / "data" / "crocodile_heb25_gold_500.csv"
PREPARED_GOLD_V1 = ROOT / "data" / "prepared_gold_500.csv"   # for relation columns
REBUTTAL_VALIDATION = ROOT / "rebuttal" / "final_datasets" / "gold_validation_set.csv"
REBUTTAL_TEST = ROOT / "rebuttal" / "final_datasets" / "gold_test_set.csv"
TOKEN_CACHE = ROOT / "data" / "token_cache"

READ_ONLY_INPUTS = [RAW_SILVER, RAW_GOLD, PREPARED_GOLD_V1,
                    REBUTTAL_VALIDATION, REBUTTAL_TEST]

# Total rows in the silver corpus, used for completion checks.
SILVER_ROWS = 2_564_534

# Host that provides the continuously-held A100 lane.
A100_HOST = os.environ.get("HERE_A100_HOST", "dsinlp01")
A100_FREE_MIB = 5000     # a GPU counts as free below this much used memory


def assert_repo_scripts() -> None:
    """Fail loudly if `scripts` resolves outside this repo.

    The `heb_relation_extraction` env ships an unrelated `scripts` package in
    site-packages. Because a regular package anywhere on sys.path beats a
    namespace package, it shadows this repo's `scripts/` no matter how
    PYTHONPATH is ordered — and an import of `scripts.clean_silver...` would
    fail confusingly, or worse, resolve to something else entirely. Use
    `hre_finetuned_nli`, which has no such package.
    """
    import scripts
    where = getattr(scripts, "__file__", None) or list(scripts.__path__)[0]
    if not str(where).startswith(str(ROOT)):
        raise SystemExit(
            f"'scripts' resolves to {where}, outside the repo.\n"
            f"Use {ROOT}-compatible interpreter: "
            f"/path/to/miniconda3/envs/hre_finetuned_nli/bin/python")


def ensure_dirs() -> None:
    for d in (DATA_V3, RUNS, LOGS, SLURM_LOGS, A100_LOGS,
              SILVER_SCORES, GOLD_SCORES, ANALYSIS):
        d.mkdir(parents=True, exist_ok=True)
