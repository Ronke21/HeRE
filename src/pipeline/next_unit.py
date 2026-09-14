"""
Print the id of the next eligible unit for a lane, or nothing if there is none.

Exists so the A100 lane script can keep its GPU and pull its own next unit
without re-implementing the scheduling rules that live in watcher.next_unit.

Usage:
    python -m post_rebuttal_and_camera_ready.pipeline.next_unit --lane a100
"""

from __future__ import annotations

import argparse

from post_rebuttal_and_camera_ready.pipeline import paths as P
from post_rebuttal_and_camera_ready.pipeline import units as U
from post_rebuttal_and_camera_ready.pipeline import watcher as W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True, choices=["slurm", "a100"])
    args = ap.parse_args()

    P.ensure_dirs()
    unit = W.next_unit(U.build_units(), lane=args.lane)
    if unit is not None:
        print(unit.id)


if __name__ == "__main__":
    main()
