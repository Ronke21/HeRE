"""
Orchestrator for the v3 re-run: keeps two SLURM jobs (one per permitted
partition) and two A100 GPUs busy until every unit is finished.

The concurrency limits here — 2 SLURM jobs, 2 A100 lanes — are what the user
authorised. They are not tuning parameters. See SKILL.md section 6b: raising
them unasked once took a shared host down for the whole lab.

Two lanes, deliberately different because the resources behave differently:

  SLURM lane — one job at a time on p_b200_tsarfaty / p_b200_nlp. Hard 4h wall
    limit, and the tsarfaty QOS caps the whole group at 1 GPU, so the job may sit
    PENDING behind a colleague. The watcher submits a job for the next eligible
    unit, waits for it to end (completed, timed out or preempted), then submits
    again. Because every unit resumes from its own pred file, a 13-hour unit
    simply takes four or five 4-hour jobs. Nothing is lost on a timeout beyond
    the current chunk.

  A100 lane — one GPU on a shared, unscheduled host. There is no queue: a GPU is
    either free or someone else has it. The watcher claims one free GPU and then
    HOLDS it, running units back-to-back in a single long-lived ssh session, so
    the GPU is never handed back mid-campaign. If no GPU is free it keeps polling
    and the campaign proceeds on the SLURM lane alone.

State lives in runs/state.json and is rewritten atomically after every poll, so
the watcher itself is restartable: kill it, start it again, and it picks up from
whatever is on disk. Progress is always re-derived from the pred files rather
than trusted from state, so a stale state file cannot cause double-counting.

Usage:
    python -m post_rebuttal_and_camera_ready.pipeline.watcher --status
    python -m post_rebuttal_and_camera_ready.pipeline.watcher --dry-run
    nohup python -u -m post_rebuttal_and_camera_ready.pipeline.watcher \
        --loop --interval 120 >> post_rebuttal_and_camera_ready/runs/logs/watcher.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

from post_rebuttal_and_camera_ready.pipeline import paths as P
from post_rebuttal_and_camera_ready.pipeline import units as U

# p_b200_nlp dropped 2026-08-14 at the user's request ("when you finish current
# run on p_b200_nlp leave it, continue with 2 a100 GPUs and the p_b200_tsarfaty").
# Not in this list means no NEW job is submitted there — any job already running
# on it is left alone by tick_slurm's general squeue scan (which tracks every
# JOB_PREFIX job regardless of partition) and simply is not resubmitted once it
# ends. Do not add it back without being asked.
SLURM_PARTITIONS = [("p_b200_tsarfaty", 1)]
ACCOUNT = "ug_tsarfaty"
JOB_PREFIX = "herev3_"
SBATCH = P.POST / "pipeline" / "slurm_unit.sbatch"
A100_LANE_SH = P.POST / "pipeline" / "a100_lane.sh"
# After this many consecutive unreachable polls, release the A100 assignment so
# the unit can fall back to SLURM instead of stalling forever. main() recomputes
# this from --interval to keep the wall-clock detection window at ~30 min
# regardless of poll cadence; the value here is only the default for direct
# import / testing (matches the original 2min x 15 = 30min).
A100_UNREACHABLE_LIMIT = 15
# TWO GPUs, raised from one at the user's explicit request on 2026-08-14
# ("2 SLURM and 2 a100. not more please"). This is an authorisation, not a
# tuning knob — do not raise it again without being asked.
#
# On 2026-08-13 it was set to 4 on my own initiative and a probe bug stacked it
# to 5+ concurrent vLLM engines. Each engine's prompt-rendering phase is
# CPU-bound, so ~2,450 threads landed on 128 cores, load hit 96+, and colleagues
# could not get a shell on dsinlp01 at all. GPU memory was never the limit — the
# CPUs were. The thread caps in a100_lane.sh exist because of that.
A100_MAX_LANES = 2


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"{now()}  {msg}", flush=True)


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

def load_state():
    if P.STATE_FILE.exists():
        try:
            return json.loads(P.STATE_FILE.read_text())
        except Exception:
            log("state.json unreadable — starting fresh")
    return {"slurm": {}, "a100": {}, "history": [], "a100_unreachable": 0}


def save_state(state):
    P.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = P.STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    os.replace(tmp, P.STATE_FILE)


def record(state, msg):
    state.setdefault("history", []).append({"t": now(), "msg": msg})
    state["history"] = state["history"][-500:]
    log(msg)


# ---------------------------------------------------------------------------
# SLURM lane
# ---------------------------------------------------------------------------

def squeue_jobs():
    """job-name -> state for our jobs."""
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "ronke21"),
             "--format=%j|%T|%i|%P", "--noheader"],
            capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        log(f"squeue failed: {e}")
        return None                      # None = unknown, do not act
    jobs = {}
    for line in out.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 4 and parts[0].startswith(JOB_PREFIX):
            jobs[parts[0]] = {"state": parts[1], "id": parts[2], "part": parts[3]}
    return jobs


def submit_slurm(unit, partition):
    name = JOB_PREFIX + unit.id
    cmd = ["sbatch", f"--job-name={name}", "-p", partition, "-A", ACCOUNT,
           f"--export=ALL,HERE_UNIT={unit.id}", str(SBATCH)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:
        log(f"sbatch failed for {unit.id}: {e}")
        return None
    if r.returncode != 0:
        log(f"sbatch rejected {unit.id}: {r.stderr.strip()}")
        return None
    return r.stdout.strip()


def tick_slurm(state, units, dry_run):
    jobs = squeue_jobs()
    if jobs is None:
        return

    # A unit can become A100-eligible after its SLURM job was already queued.
    # Two lanes appending to one pred file would corrupt it, so the queued job
    # loses — the A100 lane is already running and holding a GPU.
    a100_units = set(state.get("a100_lanes", {}))
    for name, j in list(jobs.items()):
        uid = name[len(JOB_PREFIX):]
        if uid in a100_units and j["state"] in ("PENDING", "CONFIGURING"):
            record(state, f"slurm: cancelling {name} — {uid} is running on the A100 lane")
            if not dry_run:
                subprocess.run(["scancel", j["id"]], capture_output=True, timeout=60)
            jobs.pop(name, None)
    # One job per permitted partition, held concurrently — p_b200_nlp and
    # p_b200_tsarfaty. Both land on dgx-b200-01, and tsarfaty's QOS caps the
    # whole group at 1 GPU, so its job often sits PENDING behind a colleague.
    # That is fine: it costs nothing to hold the queue slot.
    tracked = state.setdefault("slurm_jobs", {})   # unit_id -> {partition, rows_at_submit}

    active_units = set()
    per_partition = {p: 0 for p, _ in SLURM_PARTITIONS}
    for name, j in jobs.items():
        if j["state"] not in ("RUNNING", "PENDING", "CONFIGURING"):
            continue
        uid = name[len(JOB_PREFIX):]
        active_units.add(uid)
        per_partition[j["part"]] = per_partition.get(j["part"], 0) + 1
        log(f"slurm: {uid} {j['state']} ({j['part']}, job {j['id']})")

    # Adopt live jobs we are not tracking (watcher restart, state reset). squeue
    # is the truth; without this the unit is not treated as claimed and gets
    # submitted a second time to the other partition.
    for uid in active_units - set(tracked):
        unit = U.by_id(units).get(uid)
        record(state, f"slurm: adopting already-queued {uid}")
        tracked[uid] = {"partition": None, "at": now(),
                        "rows_at_submit": unit.done_rows() if unit else 0}

    # Anything we were tracking that is no longer queued or running has ended.
    # Judge it by rows produced, not exit status — slurm_unit.sbatch exits 0 on a
    # wall-clock kill, so a crash and a normal interruption look identical.
    for uid in [u for u in tracked if u not in active_units]:
        info = tracked.pop(uid)
        unit = U.by_id(units).get(uid)
        if unit is None:
            continue
        after = unit.done_rows()
        before = info.get("rows_at_submit", 0)
        if after > before:
            record(state, f"slurm: {uid} advanced {before:,} -> {after:,} rows")
        else:
            record(state, f"slurm: {uid} produced no rows this attempt")
        record_attempt(state, uid, before, after)

    for partition, limit in SLURM_PARTITIONS:
        while per_partition.get(partition, 0) < limit:
            unit = next_unit(units, lane="slurm", state=state)
            if unit is None:
                return                       # nothing eligible left to submit
            if dry_run:
                log(f"[dry-run] would submit {unit.id} to {partition}")
                tracked[unit.id] = {"partition": partition, "at": now(),
                                    "rows_at_submit": unit.done_rows()}
                per_partition[partition] = per_partition.get(partition, 0) + 1
                continue
            res = submit_slurm(unit, partition)
            if not res:
                log(f"slurm: {partition} rejected {unit.id}")
                break                        # try the next partition
            record(state, f"slurm: submitted {unit.id} to {partition} — {res}")
            tracked[unit.id] = {"partition": partition, "at": now(),
                                "rows_at_submit": unit.done_rows()}
            per_partition[partition] = per_partition.get(partition, 0) + 1


# ---------------------------------------------------------------------------
# A100 lane
# ---------------------------------------------------------------------------

def ssh(cmd, timeout=90):
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", P.A100_HOST, cmd],
            capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def a100_free_gpus():
    """Every genuinely idle GPU index, plus whether the host answered at all."""
    out = ssh("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader")
    if out is None:
        return [], False                 # unreachable
    free = []
    for line in out.splitlines():
        try:
            idx, used = line.split(",")
            if int(used.strip().split()[0]) < P.A100_FREE_MIB:
                free.append(int(idx.strip()))
        except Exception:
            continue

    return free, True


def a100_running_units():
    """{unit_id: gpu_index} currently executing on the A100 host.

    Returns None (not {}) when the host could not be probed. That distinction is
    the whole point: this used to return an empty set both for "nothing running"
    and for "ssh failed", and grep exits non-zero when it matches nothing, so a
    transient failure was indistinguishable from every lane having finished.
    The watcher duly relaunched them all, every poll, stacking vLLM engines
    until the machine was unusable. An unknown state must never be read as idle.

    Reads the per-GPU marker files a100_lane.sh writes
    (runs/logs/a100/gpu<N>.current, "<unit_id> <pid>"), cross-checked with
    `kill -0` on that pid, rather than grepping ps for a literal HERE_UNIT=
    string. That string only ever appears in the *original* ssh launch
    command for the whole a100_lane.sh wrapper — once its own loop moves on
    to a second, third, ... unit (env vars set via subprocess.Popen(env=...),
    as run_unit.py does for the actual work, never appear in `ps args`), the
    process table still showed the *first* unit forever. Found 2026-08-16:
    the probe kept reporting a long-finished gold_llm_nli_aya32b as occupying
    GPU 1 while that GPU had actually moved on to gold_opensource_llm, so
    state["a100_lanes"] never learned that unit was A100-claimed and the
    SLURM lane submitted a duplicate of it, running for 47 minutes before
    caught by hand.

    An earlier version of this function also had to start parsing
    CUDA_VISIBLE_DEVICES, not just HERE_UNIT=: an *adopted* unit (one already
    running before the watcher noticed it) was recorded with gpu=None —
    busy_gpus in tick_a100 couldn't exclude a GPU it didn't have an index
    for, so a second lane got launched onto the SAME physical GPU as
    neodictabert_k3 on 2026-08-15. The marker filename itself (gpu<N>.current)
    carries the index now, so that class of bug can't recur here.
    """
    marker = "___A100_PROBE_OK___"
    state_dir = shlex.quote(str(P.A100_LOGS))
    out = ssh(f"echo {marker}; "
              f"cd {state_dir} 2>/dev/null && for f in gpu*.current; do "
              f"[ -f \"$f\" ] || continue; "
              f"read -r u p < \"$f\"; "
              f"kill -0 \"$p\" 2>/dev/null && echo \"FILE=$f UNIT=$u PID=$p\"; "
              f"done; true")
    if out is None or marker not in out:
        return None                      # could not probe — assume nothing changed
    if out.strip() == marker:
        return {}                        # probe worked, genuinely nothing running
    running = {}
    for line in out.splitlines():
        if not line.startswith("FILE="):
            continue
        fields = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
        fname, unit_id = fields.get("FILE"), fields.get("UNIT")
        if not fname or not unit_id:
            continue
        try:
            gpu = int(fname[len("gpu"):-len(".current")])
        except ValueError:
            continue
        running[unit_id] = gpu
    return running


def tick_a100(state, units, dry_run):
    """Keep up to A100_MAX_LANES units running, one per genuinely idle GPU.

    Capped rather than greedy: dsinlp01 is shared and unscheduled, so claiming
    every idle card would leave colleagues nothing. "Idle" stays strict — under
    A100_FREE_MIB used — which is what makes co-tenancy unnecessary.
    """
    lanes = state.setdefault("a100_lanes", {})       # unit_id -> {gpu, rows_at_launch}
    running = a100_running_units()
    if running is None:
        state["a100_unreachable"] = state.get("a100_unreachable", 0) + 1
        log(f"a100: probe failed ({state['a100_unreachable']}) — holding, not relaunching")
        return

    # Adopt anything running that we are not already tracking — after a state
    # reset, or a watcher restart, the process table is the truth and state is
    # empty. Without this the unit stays absent from `lanes`, so next_unit does
    # not treat it as claimed and the SLURM lane happily queues the same unit.
    # Recording the real gpu index (not None) is what lets busy_gpus below
    # actually exclude it — see a100_running_units' docstring for the incident
    # this fixes.
    for unit_id, gpu in running.items():
        if unit_id in lanes:
            # Backfill a gpu index recorded as None by an older watcher
            # version, or by a probe that couldn't parse CUDA_VISIBLE_DEVICES
            # at adoption time. A stale None here is exactly what let a second
            # lane land on the same physical GPU on 2026-08-15 — busy_gpus
            # cannot exclude a GPU it was never told about.
            if lanes[unit_id].get("gpu") is None and gpu is not None:
                lanes[unit_id]["gpu"] = gpu
                record(state, f"a100: backfilled gpu={gpu} for {unit_id}")
            continue
        unit = U.by_id(units).get(unit_id)
        record(state, f"a100: adopting already-running {unit_id} (gpu={gpu})")
        lanes[unit_id] = {"gpu": gpu, "at": now(),
                          "rows_at_launch": unit.done_rows() if unit else 0}

    # Reconcile: any lane we thought was running but isn't has ended.
    for unit_id in [u for u in lanes if u not in running]:
        unit = U.by_id(units).get(unit_id)
        info = lanes.pop(unit_id, {})
        if unit is not None:
            after = unit.done_rows()
            before = info.get("rows_at_launch", 0)
            if after > before:
                record(state, f"a100: {unit_id} advanced {before:,} -> {after:,} rows")
            else:
                record(state, f"a100: {unit_id} produced no rows this attempt")
            record_attempt(state, unit_id, before, after)

    free, reachable = a100_free_gpus()
    if not reachable:
        state["a100_unreachable"] = state.get("a100_unreachable", 0) + 1
        n = state["a100_unreachable"]
        log(f"a100: host unreachable ({n} consecutive)")
        if n >= A100_UNREACHABLE_LIMIT and lanes:
            record(state, f"a100: releasing {sorted(lanes)} back to the pool")
            state["a100_lanes"] = {}
        return
    state["a100_unreachable"] = 0

    if running:
        log(f"a100: {len(running)} lane(s) busy: {sorted(running)}")

    busy_gpus = {info.get("gpu") for info in lanes.values()}
    slots = A100_MAX_LANES - len(running)
    if slots <= 0:
        return
    if not free:
        log(f"a100: {len(running)}/{A100_MAX_LANES} lanes, no idle GPU to add")
        return

    for gpu in free:
        if slots <= 0:
            break
        if gpu in busy_gpus:
            continue
        unit = next_unit(units, lane="a100", state=state)
        if unit is None:
            return
        if dry_run:
            log(f"[dry-run] would launch {unit.id} on {P.A100_HOST} GPU {gpu}")
            lanes[unit.id] = {"gpu": gpu, "at": now(), "rows_at_launch": unit.done_rows()}
            slots -= 1
            busy_gpus.add(gpu)
            continue

        logfile = P.A100_LOGS / f"{unit.id}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        # < /dev/null on the backgrounded process matters: without it, the
        # child inherits the ssh session's stdin pipe, and ssh then waits for
        # that pipe to close — which never happens on its own — instead of
        # returning once the remote shell has forked and detached. That made
        # every launch block for the full 180s timeout despite succeeding in
        # milliseconds, and because the loop only records a lane on a clean
        # "launched:" reply, a timed-out-but-actually-running launch fell
        # through un-recorded into the same tick's tick_slurm(), which had no
        # way to see it as claimed and submitted the same unit to SLURM too.
        # Found 2026-08-16 when gold_llm_nli_aya32b was caught running on both
        # lanes at once (the SLURM copy was still PENDING, cancelled before
        # any write happened).
        remote = (f"cd {shlex.quote(str(P.ROOT))} && "
                  f"HERE_UNIT={unit.id} CUDA_VISIBLE_DEVICES={gpu} "
                  f"HERE_MAX_LANES={A100_MAX_LANES} "
                  f"nohup bash {shlex.quote(str(A100_LANE_SH))} {unit.id} "
                  f">> {shlex.quote(str(logfile))} 2>&1 < /dev/null & echo launched:$!")
        out = ssh(remote, timeout=180)
        if out and "launched:" in out:
            record(state, f"a100: launched {unit.id} on GPU {gpu}")
            lanes[unit.id] = {"gpu": gpu, "at": now(),
                              "rows_at_launch": unit.done_rows()}
            slots -= 1
            busy_gpus.add(gpu)
        else:
            log(f"a100: launch of {unit.id} returned {out!r} — will verify next poll")
            return          # don't spray launches if ssh is misbehaving


# ---------------------------------------------------------------------------
# scheduling
# ---------------------------------------------------------------------------

MAX_UNIT_FAILURES = 3


def _blocked_units(state):
    """Units that failed repeatedly with zero progress — stop retrying them.

    A unit that dies instantly (missing dependency, OOM) is otherwise handed
    back forever, because "not done" is indistinguishable from "not started".
    """
    return {uid for uid, n in state.get("failures", {}).items()
            if n >= MAX_UNIT_FAILURES}


def record_attempt(state, unit_id, rows_before, rows_after):
    """Count a zero-progress attempt as a failure; any progress clears it."""
    fails = state.setdefault("failures", {})
    if rows_after > rows_before:
        fails.pop(unit_id, None)
        return
    fails[unit_id] = fails.get(unit_id, 0) + 1
    if fails[unit_id] >= MAX_UNIT_FAILURES:
        record(state, f"!!! {unit_id} failed {fails[unit_id]}x with no progress — "
                      f"BLOCKED. Check runs/logs/units/{unit_id}.log")


def next_unit(units, lane, state=None):
    """First eligible unit for a lane: prerequisites met, not done, not claimed.

    `state` must be the caller's LIVE state when launching several lanes in one
    pass. Re-reading it from disk here made claims made earlier in the same loop
    invisible, so the multi-lane A100 tick handed the same unit to every idle
    GPU — three processes appending to one pred file.
    """
    done = {u.id for u in units if u.is_done()}
    claimed = set()
    if state is None:
        state = load_state()
    blocked = _blocked_units(state)
    claimed.update(state.get("slurm_jobs", {}).keys())
    claimed.update(state.get("a100_lanes", {}).keys())
    for u in sorted(units, key=lambda x: (x.stage, -x.est_hours)):
        if u.is_done() or u.id in claimed or u.id in blocked:
            continue
        if u.lane == "cpu":
            continue
        if u.lane != "any" and u.lane != lane:
            continue
        if any(n not in done for n in u.needs):
            continue
        return u
    return None


def tick(dry_run=False):
    P.ensure_dirs()
    units = U.build_units()
    state = load_state()

    remaining = [u for u in units if not u.is_done()]
    if not remaining:
        record(state, "ALL UNITS COMPLETE")
        save_state(state)
        return True

    # clear finished claims so the lanes free up
    for uid in [u for u in state.get("slurm_jobs", {})
                if U.by_id(units).get(u) and U.by_id(units)[u].is_done()]:
        record(state, f"slurm: {uid} complete")
        state["slurm_jobs"].pop(uid, None)
    for uid in [u for u in state.get("a100_lanes", {})
                if (U.by_id(units).get(u) or None) and U.by_id(units)[u].is_done()]:
        record(state, f"a100: {uid} complete")
        state["a100_lanes"].pop(uid, None)

    # A100 first, deliberately. It discovers what is running from the remote
    # process table, so running it first means tick_slurm sees those claims and
    # will not queue a SLURM job for a unit already on the A100 lane. (The
    # supersede-and-cancel check in tick_slurm remains as the backstop for when
    # the A100 probe fails.)
    tick_a100(state, units, dry_run)
    tick_slurm(state, units, dry_run)

    # A dry run must leave no trace. It records simulated lane claims so its own
    # output is honest about which unit goes to which GPU, and persisting those
    # made the next real tick believe three units had run and produced nothing —
    # complete with failure counters against them.
    if not dry_run:
        save_state(state)
    log(f"remaining: {len(remaining)} units, "
        f"{sum(u.est_hours for u in remaining if u.lane != 'cpu'):.1f} GPU-hours")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    P.ensure_dirs()
    if args.status:
        print(U.summary())
        if P.STATE_FILE.exists():
            print("\nstate:", P.STATE_FILE.read_text()[:2000])
        return

    if not args.loop:
        tick(args.dry_run)
        return

    global A100_UNREACHABLE_LIMIT
    # ceil(1800 / interval): keeps the "unreachable -> release" wall-clock time
    # at ~30 min regardless of poll cadence, floored at 2 so a single blip can
    # never trigger it.
    A100_UNREACHABLE_LIMIT = max(2, -(-1800 // args.interval))

    log(f"watcher starting (interval {args.interval}s, dry_run={args.dry_run}, "
        f"a100_unreachable_limit={A100_UNREACHABLE_LIMIT} "
        f"[~{A100_UNREACHABLE_LIMIT * args.interval // 60}min])")
    while True:
        try:
            if tick(args.dry_run):
                log("watcher exiting — everything finished")
                return
        except KeyboardInterrupt:
            log("watcher interrupted")
            return
        except Exception as e:
            log(f"watcher error (continuing): {e!r}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
