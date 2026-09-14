"""
Print how many rows a unit has scored. Used by the lane scripts to tell a real
failure ("died and produced nothing") from a normal interruption ("wall clock
killed it mid-stream, but rows landed").

Usage:
    python -m post_rebuttal_and_camera_ready.pipeline.unit_progress <unit_id>
"""

from __future__ import annotations

import sys

from post_rebuttal_and_camera_ready.pipeline import units as U


def main():
    if len(sys.argv) < 2:
        print(0)
        return
    unit = U.by_id().get(sys.argv[1])
    print(unit.done_rows() if unit else 0)


if __name__ == "__main__":
    main()
