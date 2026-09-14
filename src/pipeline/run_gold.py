"""
Thin CLI driver for scripts/clean_gold/*.py.

These scripts already take real --input/--output/--log/... flags, unlike the
silver family, so they need no OUT_DIR-style shim for paths. Two of them
(clean_encoder_NLI, clean_finetuned_llm_NLI) still bake stale Hebrew_NLI
checkpoint paths into module-level constants at import time — this driver
patches those via shim.remap_gold_checkpoints() before calling main(), then
forwards every other CLI arg unchanged. clean_dataset_with_opensource_llm has
no such dependency and passes straight through.

Usage:
    python -m post_rebuttal_and_camera_ready.pipeline.run_gold \
        scripts.clean_gold.clean_encoder_NLI --input ... --output ...
"""
from __future__ import annotations

import importlib
import sys

from post_rebuttal_and_camera_ready.pipeline import paths as P
from post_rebuttal_and_camera_ready.pipeline import shim as S

NEEDS_CHECKPOINT_REMAP = {
    "scripts.clean_gold.clean_encoder_NLI",
    "scripts.clean_gold.clean_finetuned_llm_NLI",
}

# See shim.force_gpu_only_device_map()'s docstring: without this, a vLLM
# engine-init failure partway through this script's 7-model sequence can
# silently fall back to running a 24B model on CPU for hours instead of
# erroring. Not needed for clean_finetuned_llm_NLI (same device_map="auto"
# pattern, but that script's units already finished before this was found).
NEEDS_GPU_ONLY_DEVICE_MAP = {
    "scripts.clean_gold.clean_dataset_with_opensource_llm",
}

# User call 2026-08-16: skip mistral_small24b entirely for Strategy 4 rather
# than chase the CPU-fallback issue further — finish the other 6 models.
DROP_MODELS = {
    "scripts.clean_gold.clean_dataset_with_opensource_llm": ("LLM_MODELS", {"mistral_small24b"}),
}


def main():
    P.assert_repo_scripts()
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_gold.py <module.path> [args...]")
    mod_name, argv = sys.argv[1], sys.argv[2:]
    # Applied unconditionally, every script, every host: see
    # shim.disable_torch_compile()'s docstring — a compile failure on
    # dgx01/dgx03's older glibc is a hard crash, not the graceful fallback
    # these scripts' own try/except implies, and compilation wasn't likely
    # paying for itself on A100 either for this kind of workload.
    S.disable_torch_compile()
    if mod_name in NEEDS_GPU_ONLY_DEVICE_MAP:
        S.force_gpu_only_device_map()
    mod = importlib.import_module(mod_name)
    if mod_name in NEEDS_CHECKPOINT_REMAP:
        S.remap_gold_checkpoints(mod)
    if mod_name in DROP_MODELS:
        list_attr, tags = DROP_MODELS[mod_name]
        S.drop_models(mod, list_attr, tags)
    old_argv = sys.argv
    sys.argv = [mod_name] + argv
    try:
        mod.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
