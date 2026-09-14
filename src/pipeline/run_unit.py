"""
Run a single unit by id. This is what both lanes actually execute.

Keeping the lane scripts dumb (they just call this with a unit id) means the
command for a unit is defined in exactly one place — units.py — so the SLURM
job and the A100 job can never drift apart.

Writes a per-unit log under runs/logs/units/<id>.log in addition to whatever the
underlying script logs, and drops runs/done/<id>.done on a clean exit for units
that have no pred file of their own.

Usage:
    python -m post_rebuttal_and_camera_ready.pipeline.run_unit <unit_id>
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime

from post_rebuttal_and_camera_ready.pipeline import paths as P
from post_rebuttal_and_camera_ready.pipeline import units as U


def _rc_failure_reason(unit_id: str) -> str | None:
    """Return a reason string if a cross_train_rc unit visibly failed, else None.

    `clean_cross_train_rc.py` catches per-fold exceptions, marks the fold
    FAILED, keeps going, prints a summary and exits 0. Four separate incidents
    rode that behaviour into a false `.done` marker — torch-2.11 NaN, a
    poisoned resume checkpoint, batch-256 OOM on A100, and torch-2.6 on a B200
    ("no kernel image"). Every one produced a correctly-shaped output tree with
    an all-zero ensemble, and each was only caught by reading the loss by hand.

    So: judge these units by their results, never by their exit code. Checked
    only for gold_rc_*/silver_rc_* — other units have their own row-count
    completion test via pred_file.
    """
    if "_rc_" not in unit_id:
        return None
    tag = unit_id.split("_rc_", 1)[1]
    run_log = P.GOLD_SCORES / "cross_train_rc" / tag / "run.log"
    if not run_log.exists():
        run_log = P.SILVER_SCORES / "silver_cross_train_rc" / tag / "run.log"
    if not run_log.exists():
        return "no run.log produced"
    text = run_log.read_text(errors="replace")
    if "FAILED" in text:
        return f"{text.count('FAILED')} fold(s) marked FAILED"
    for line in reversed(text.splitlines()):
        if "Ensemble (majority vote" in line and "f1=0.000" in line:
            return "ensemble gold F1 = 0.000"
    if "avg_loss=nan" in text:
        return "avg_loss=nan during training"
    return None


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_unit.py <unit_id>")
    unit_id = sys.argv[1]

    P.assert_repo_scripts()
    P.ensure_dirs()
    unit = U.by_id().get(unit_id)
    if unit is None:
        raise SystemExit(f"unknown unit: {unit_id}")

    if unit.is_done():
        print(f"[run_unit] {unit_id} already complete ({unit.progress()}) — nothing to do")
        return

    log_dir = P.LOGS / "units"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{unit_id}.log"

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["HERE_UNIT"] = unit_id
    env.update(unit.env)

    start = time.time()
    header = (f"\n{'='*70}\n[run_unit] {unit_id}  start {datetime.now():%Y-%m-%d %H:%M:%S}\n"
              f"[run_unit] host={os.uname().nodename} "
              f"gpu={env.get('CUDA_VISIBLE_DEVICES','?')}\n"
              f"[run_unit] progress before: {unit.progress()}\n"
              f"[run_unit] cmd: {' '.join(unit.cmd)}\n{'='*70}\n")
    print(header, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(header)

    with open(log_path, "a", encoding="utf-8") as f:
        proc = subprocess.Popen(unit.cmd, cwd=str(P.ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = proc.wait()

    mins = (time.time() - start) / 60
    unit = U.by_id()[unit_id]          # re-read progress from disk
    tail = (f"[run_unit] {unit_id} exit={rc} after {mins:.1f} min; "
            f"progress now: {unit.progress()}\n")
    print(tail, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(tail)

    if rc == 0 and unit.pred_file is None:
        bad = _rc_failure_reason(unit_id)
        if bad:
            msg = (f"[run_unit] REFUSING .done for {unit_id}: {bad}\n"
                   f"[run_unit] exit code was 0 but the run did not produce a valid model.\n")
            print(msg, flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg)
            sys.exit(1)
        done_dir = P.RUNS / "done"
        done_dir.mkdir(parents=True, exist_ok=True)
        (done_dir / f"{unit_id}.done").write_text(datetime.now().isoformat())

    # A non-zero exit is not necessarily failure: the SLURM lane kills the job at
    # the wall limit mid-unit, which is expected and resumed next submission.
    sys.exit(0 if rc == 0 else rc)


if __name__ == "__main__":
    main()
