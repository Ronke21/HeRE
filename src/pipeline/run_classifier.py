"""
Run one silver classifier against the v3 data, writing into the v3 output tree.

Thin entry point: resolve the redirect via shim.py, then hand the remaining
argv to the original script's main(). Used as the command in every stage-2 unit
so that the redirect is applied in exactly one place.

Usage:
    python -m post_rebuttal_and_camera_ready.pipeline.run_classifier \
        --family opensource_llm -- --models dictalm3 --skip-merge
"""

from __future__ import annotations

import argparse
import sys

from post_rebuttal_and_camera_ready.pipeline import paths as P
from post_rebuttal_and_camera_ready.pipeline import shim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(shim.FAMILIES))
    ap.add_argument("--silver", default=None,
                    help="override the v3 silver CSV (default: data_v3/prepared_silver_v3.csv)")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="args after '--' are passed through to the classifier")
    args = ap.parse_args()

    passthrough = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest

    P.ensure_dirs()
    mod = shim.load_redirected(args.family, silver_csv=args.silver)

    # Only validate the models this invocation will actually load.
    wanted = None
    if "--models" in passthrough:
        i = passthrough.index("--models") + 1
        wanted = []
        while i < len(passthrough) and not passthrough[i].startswith("-"):
            wanted.append(passthrough[i])
            i += 1
    shim.assert_checkpoints_exist(mod, tags=wanted)

    print(f"[run_classifier] family={args.family}", flush=True)
    print(f"[run_classifier] input   = {mod.SILVER_CSV}", flush=True)
    print(f"[run_classifier] outdir  = {mod.OUT_DIR}", flush=True)
    print(f"[run_classifier] args    = {passthrough}", flush=True)

    old = sys.argv
    sys.argv = [mod.__name__] + passthrough
    try:
        mod.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
