"""
The work-unit registry for the v3 re-run.

A *unit* is the smallest thing worth scheduling: one model over one dataset. It
must be resumable, because the SLURM lane is capped at 4 hours and several units
need far longer than that.

Resume mechanism: every silver classifier appends one float per line to
`pred_<tag>.txt` and flushes each chunk, then on startup counts the lines
already there and skips that many rows. So "resume" is just "run the same
command again" — and `done_rows()` below reads the same file to report progress.

Lane assignment
---------------
`lane` is a hint, not a hard constraint, and exists because the two lanes have
different memory:

  * SLURM / B200 (~180 GB) takes the large decoders — 31B/32B/24B in bf16 are
    60-70 GB of weights before activations and KV cache.
  * A100 (80 GB) takes the 9B decoder, the encoders and the cross-trained
    classifiers, which fit with room to spare.

A unit marked "slurm" will not be sent to the A100 lane. A unit marked "any"
goes wherever a slot opens first.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from post_rebuttal_and_camera_ready.pipeline import paths as P


@dataclass
class Unit:
    id: str
    stage: int
    cmd: list[str]                 # argv, run from the repo root
    lane: str = "any"              # "any" | "slurm" | "a100" | "cpu"
    est_hours: float = 1.0
    needs: list[str] = field(default_factory=list)
    pred_file: Path | None = None  # progress/completion marker
    expect_rows: int | None = None
    env: dict = field(default_factory=dict)

    def done_rows(self) -> int:
        if self.pred_file is None or not self.pred_file.exists():
            return 0
        n = 0
        with open(self.pred_file, "rb") as f:
            for _ in f:
                n += 1
        return n

    def is_done(self) -> bool:
        if self.pred_file is None:
            return (P.RUNS / "done" / f"{self.id}.done").exists()
        if self.expect_rows is None:
            return self.pred_file.exists()
        return self.done_rows() >= self.expect_rows

    def progress(self) -> str:
        if self.pred_file is None or self.expect_rows is None:
            return "done" if self.is_done() else "pending"
        d = self.done_rows()
        return f"{d:,}/{self.expect_rows:,} ({100*d/self.expect_rows:.1f}%)"


# The interpreter running this module, NOT a bare "python". A bare name resolves
# through PATH to the system python3.9 on both the B200 nodes and the A100 host,
# which has neither peft nor the rest of the stack — every unit died in 15
# seconds with ModuleNotFoundError while the lanes happily reported success.
# sys.executable guarantees the child inherits the environment that was verified
# to work.
PY = sys.executable

# torch 2.6.0+cu124. Every cross_train_rc run on torch 2.11 (PY) produced
# avg_loss=nan -- xlmroberta_k10 from step 1, neodictabert_k7 partway through
# fold 1, and silver_rc_neodictabert_k3 across all 6 folds (which silently
# shipped a 0.0%-positive signal). The same models on torch 2.6 train cleanly
# (xlmroberta_k3 3.49, me5large_k3 3.99, me5large_k5 3.97). mmbert happens to
# survive 2.11 (mmbert_k3 1.58, mmbert_k5 1.65, 53.5% positive vs the paper's
# 54.4%) but is pinned here too, so Strategy 3 is internally consistent.
TORCH26_PY = os.environ.get(
    # HERE_RC_PY lets a caller override the cross_train_rc interpreter without
    # editing this file. Needed for SLURM/B200: torch 2.6/cu124 has no sm_100
    # kernels, so a B200 run must use torch 2.11 -- which is only safe for
    # mmBERT (the sole model proven healthy on 2.11). Setting PY on the sbatch
    # is NOT enough: run_unit rebuilds the command from this registry, so the
    # override has to happen here. That mistake cost a B200 job on 2026-08-19.
    "HERE_RC_PY",
    "/path/to/miniconda3/envs/dicta_nemotron/bin/python")
PIPE = "post_rebuttal_and_camera_ready.pipeline"


def _silver_pred(family_dir: str, tag: str) -> Path:
    return P.SILVER_SCORES / family_dir / f"pred_{tag}.txt"


def _load_env_file(path: Path) -> dict:
    """Parse KEY=VALUE lines from a gitignored .env file, if it exists.

    Only used to source OPENROUTER_API_KEY into one unit's env dict at build
    time — the value never appears as a literal in this (tracked) file, only
    the path to the (untracked, chmod 600) file that holds it.
    """
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_units() -> list[Unit]:
    U: list[Unit] = []

    # ---------------- stage 0: data preparation (CPU) ----------------------
    # Completion is keyed on the marker files prepare_v3_data writes itself, not
    # on runs/done/. Stage 0 is normally run by hand (it is CPU-only and needs no
    # scheduling), so a marker written by run_unit would never appear and every
    # stage-2 unit would stay blocked forever.
    U.append(Unit("prep_gold_eval", 0, [PY, "-m", f"{PIPE}.prepare_v3_data", "--only", "gold"],
                  lane="cpu", est_hours=0.1,
                  pred_file=Path(str(P.GOLD_V3_CSV) + ".done")))
    U.append(Unit("prep_eval_sets", 0, [PY, "-m", f"{PIPE}.prepare_v3_data", "--only", "eval"],
                  lane="cpu", est_hours=0.2, needs=["prep_gold_eval"],
                  pred_file=Path(str(P.TEST_V3_CSV) + ".done")))
    U.append(Unit("prep_silver", 0, [PY, "-m", f"{PIPE}.prepare_v3_data", "--only", "silver"],
                  lane="cpu", est_hours=0.6,
                  pred_file=Path(str(P.SILVER_V3_PARQUET) + ".done")))

    # ---------------- stage 2: silver re-scoring (GPU) ---------------------
    # Only the two largest are B200-only. vLLM claims ~78 GB of an 80 GB A100
    # even for a 9B model, so 31B (~62 GB of weights) and 32B (~64 GB) would be
    # left with almost no KV cache and would OOM-loop on a shared host. The
    # 24B (~48 GB) and 27B (~54 GB) models fit with room to spare.
    # mistral_small24b was added 2026-08-16 (user request: extend the silver
    # opensource_llm signal beyond its original curated 3 models), injected
    # via shim.EXTRA_SILVER_MODELS — then reverted the same day: dropped back
    # to the original 3-model curated set to keep the silver corpus on the
    # paper's original models, after a duplicate-launch incident (this unit
    # ended up running on two A100 GPUs at once — see README) prompted a
    # re-think. Its partial output (pred_mistral_small24b.txt, ~300K/2.56M
    # rows) was deleted, not kept as a partial artifact.
    for tag, hours, lane in [("gemma4_31b_it", 11.0, "slurm"),
                             ("gemma3_27b_it", 10.0, "any"),
                             ("dictalm3", 10.0, "any")]:
        U.append(Unit(
            f"silver_llm_{tag}", 2,
            [PY, "-m", f"{PIPE}.run_classifier", "--family", "opensource_llm",
             "--", "--models", tag, "--skip-merge"],
            lane=lane, est_hours=hours, needs=["prep_silver"],
            pred_file=_silver_pred("silver_opensource_llm", tag),
            expect_rows=P.SILVER_ROWS))

    # mistral24b: same addition/revert story as mistral_small24b above, same
    # day. Its completed output (pred_mistral24b.txt, full 2.56M rows) was
    # also deleted on revert, not merged into the corpus.
    for tag, hours, lane in [("dictalm24b_base_v2", 13.0, "any"),
                             ("aya32b_v2", 13.0, "slurm"),
                             ("gemma2_9b_v1", 6.0, "any")]:
        U.append(Unit(
            f"silver_nli_{tag}", 2,
            [PY, "-m", f"{PIPE}.run_classifier", "--family", "finetuned_nli",
             "--", "--models", tag, "--skip-merge"],
            lane=lane, est_hours=hours, needs=["prep_silver"],
            pred_file=_silver_pred("silver_finetuned_nli", tag),
            expect_rows=P.SILVER_ROWS))

    # neodictabert's checkpoint pins transformers_version=4.55.4 in its own
    # config.json. Under this repo's usual interpreter (hre_finetuned_nli,
    # transformers==5.8.1 as of 2026-08-15) its custom modeling_neobert.py
    # architecture — trust_remote_code, calls torch's SDPA primitive directly
    # rather than going through the library's compat layer — produces 100%
    # literal "nan" for every row. Reproduced on pure CPU with eager attention
    # and torch.compile disabled, ruling out GPU/kernel causes; the checkpoint's
    # 174 weight tensors were scanned directly and contain zero NaN/Inf, ruling
    # out a corrupted checkpoint. dicta_nemotron has transformers==4.55.4, an
    # exact version match, and was verified end-to-end (2000 real rows, 0 nan,
    # sane 55% positive rate) before wiring in. xlmroberta has no such issue and
    # stays on the normal interpreter.
    # LARGE_BATCH_SIZE bug (was == BATCH_SIZE, a no-op) and torch.compile
    # (crashes on dgx01/dgx03's older glibc, lazily, past the script's own
    # try/except) both fixed in shim.py's _fix_large_text_batch_size() —
    # 2026-08-16, see that function's comments for the full story. neodictabert
    # moved to lane="cpu" (run by hand, same "next_unit() must never
    # auto-schedule this" convention as the other manually-launched units)
    # once it was running for real on dgx03 GPU2 — it was still lane="any"
    # up to that point, which would otherwise have let the watcher schedule
    # a second, A100/SLURM copy of it once a lane freed.
    ENCODER_PY = {"neodictabert": "/path/to/miniconda3/envs/dicta_nemotron/bin/python"}
    ENCODER_LANE = {"neodictabert": "cpu"}
    for tag, hours in [("neodictabert", 3.0), ("xlmroberta", 3.0)]:
        U.append(Unit(
            f"silver_enc_{tag}", 2,
            [ENCODER_PY.get(tag, PY), "-m", f"{PIPE}.run_classifier",
             "--family", "encoder_nli", "--", "--models", tag, "--skip-merge"],
            lane=ENCODER_LANE.get(tag, "any"), est_hours=hours, needs=["prep_silver"],
            pred_file=_silver_pred("silver_encoder_nli", tag),
            expect_rows=P.SILVER_ROWS))

    # Cross-trained RC takes real CLI flags, so it needs no shim. The script's
    # only selector is --config <name> (choices: all, neodictabert_k3, mmbert_k5)
    # — a --models/--k-folds pair was invented here and never existed in the
    # script's argparse, so both units failed with "unrecognized arguments" on
    # every attempt since 2026-08-12 (0.3 min no-op each time).
    for tag, hours in [("neodictabert_k3", 6.0), ("mmbert_k5", 4.0)]:
        U.append(Unit(
            f"silver_rc_{tag}", 2,
            [TORCH26_PY, "-m", "scripts.clean_silver.classify_silver_cross_train_rc",
             "--config", tag,
             # --train-batch 32: NeoBERT's custom attention materializes the
             # full (batch, heads, seq, seq) tensor. Under torch 2.6 at the
             # script's default batch 256 this allocates ~78 GB before the
             # first optimizer step and OOMs even an empty 80 GB A100
             # (2026-08-18; same root cause as the 48 GB inference OOM fixed
             # in shim._fix_large_text_batch_size). mmbert tolerates 256;
             # neodictabert does not. 32 is uniform with every other RC run.
             "--train-batch", "32",
             "--silver", str(P.SILVER_V3_PARQUET),
             "--gold", str(P.GOLD_V3_CSV),
             "--output-base", str(P.SILVER_SCORES / "silver_cross_train_rc")],
            lane="any", est_hours=hours, needs=["prep_silver", "prep_gold_eval"]))

    # ---------------- stage 2b: gold-500 benchmark re-run (GPU) -------------
    # Strategies 1/2/4 from the paper, re-scored on the v3-cleaned 500-row
    # gold set. Unlike the silver classifiers these scripts already take real
    # --input/--output/--log/... flags, so run_gold.py needs no OUT_DIR shim,
    # only a checkpoint remap (shim.remap_gold_checkpoints) for the two that
    # load Hebrew_NLI fine-tunes. Every one of them joins --input/--output
    # against os.path.dirname(os.path.dirname(__file__)) — which for a script
    # living in scripts/clean_gold/ resolves to <repo>/scripts, not <repo> —
    # so every path passed below must be absolute or it silently lands under
    # scripts/scripts/clean_gold/... instead of post_rebuttal_and_camera_ready/.
    # Confirmed by smoke test on 2026-08-15: a relative --input raised
    # FileNotFoundError pointing at exactly that extra "scripts/" segment.
    GOLD_PY = "post_rebuttal_and_camera_ready.pipeline.run_gold"

    def _gold_out(subdir: str) -> Path:
        return P.GOLD_SCORES / subdir

    # Strategy 1: 8 encoder/seq2seq models, all in one process (the script has
    # no --models selector). Same neodictabert config.json/transformers
    # version pin as the silver side, so this needs the dicta_nemotron
    # interpreter too, and — since that one process also carries the other 7
    # models — ALL of clean_encoder_NLI.py runs under it, not just neodictabert.
    # lane="a100" is load-bearing, not a hint here: dicta_nemotron's
    # torch==2.6.0+cu124 build has no compiled kernels for B200 (Blackwell,
    # sm_100) — "CUDA error: no kernel image is available for execution on
    # the device" on the very first forward pass, 3x in a row, BLOCKED, until
    # traced 2026-08-16. It was only ever verified on A100 (sm_80), which is
    # where silver_enc_neodictabert already runs successfully under the same
    # interpreter.
    enc_out = _gold_out("encoder_nli")
    U.append(Unit(
        "gold_encoder_nli", 2,
        ["/path/to/miniconda3/envs/dicta_nemotron/bin/python", "-m", GOLD_PY,
         "scripts.clean_gold.clean_encoder_NLI",
         "--input", str(P.GOLD_V3_CSV),
         "--output", str(enc_out / "classified.csv"),
         "--log", str(enc_out / "classify.log"),
         "--summary", str(enc_out / "summary.txt"),
         "--error-analysis", str(enc_out / "error_analysis.txt")],
        lane="a100", est_hours=1.5, needs=["prep_gold_eval"]))

    # Strategy 2: 17 fine-tuned-LLM-NLI tags (same set the pre-rebuttal v1
    # gold run used — see outputs/gold_cleaning/finetuned_llm_nli/summary.txt
    # — including the v2/v3/v4 alternative-prompt variants, since the paper's
    # Table 3/7 report the best F1 across those variants per base model).
    # Split one unit per tag, unlike the single-process Strategy 1 script,
    # because clean_finetuned_llm_NLI.py's own --models flag makes that free,
    # and it caps the blast radius of any one crash to ~10-30 min instead of
    # the whole family.
    # Originally pinned to slurm for the 32B/31B/35B tags, matching the same
    # sizing caution used for silver's biggest models. Switched to "any" on
    # 2026-08-16: this script loads plain HuggingFace checkpoints (not vLLM,
    # which is what actually made the silver-side 31B/32B models B200-only —
    # vLLM reserves ~78/80GB on an A100 even for a 9B model), and the one
    # full opensource_llm gold run on record (up to 32B, HF backend) finished
    # in 21m50s with no OOM. Both A100 GPUs were sitting idle with nothing
    # "any"-eligible queued while these 5 units serialized behind the single
    # SLURM ticket — user approved letting A100 pick some of them up too.
    LLM_NLI_BIG = {"aya32b", "aya32b_v2", "gemma4_31b", "gemma4_31b_v2",
                   "qwen35base", "qwen35base_v2"}
    llm_nli_tags = [
        "gemma2_9b", "gemma3_12b", "dictalm24b", "dictalm24b_base",
        "dictalm24b_base_v2", "aya32b", "mistral24b", "dictalm24b_v2",
        "aya32b_v2", "mistral24b_v2", "mistral24b_v3", "mistral24b_v4",
        "qwen35base", "gemma4_31b", "gemma4_26b", "gemma3_27b", "dictalm17b",
        # v2 reruns 2026-08-26: the originals emit free-form prose the parser
        # cannot read (483 and 313-363 of 500 rows unparseable) -- constrained
        # prompt variants, same fix that repaired aya32b/dictalm24b_base.
        "gemma4_31b_v2", "qwen35base_v2",
    ]
    for tag in llm_nli_tags:
        out = _gold_out(f"finetuned_llm_nli/{tag}")
        U.append(Unit(
            f"gold_llm_nli_{tag}", 2,
            [PY, "-m", GOLD_PY, "scripts.clean_gold.clean_finetuned_llm_NLI",
             "--input", str(P.GOLD_V3_CSV),
             "--output", str(out / "classified.csv"),
             "--log", str(out / "classify.log"),
             "--summary", str(out / "summary.txt"),
             "--error", str(out / "error_analysis.txt"),
             "--pred-file", str(out / "predicate_analysis.txt"),
             "--models", tag],
            lane="any",
            est_hours=1.0 if tag in LLM_NLI_BIG else 0.5,
            needs=["prep_gold_eval"]))

    # Strategy 4: 7 open-source LLMs few-shot, all in one process (no
    # --models selector; internally sweeps langs/shots/models). No Hebrew_NLI
    # dependency — confirmed via grep — so no checkpoint remap needed. The
    # pre-rebuttal v1 gold run of this exact script completed in 21m50s
    # end-to-end (outputs/gold_cleaning/opensource_llm/summary.txt) despite
    # loading models up to 32B (cohere_aya32b) sequentially in one process —
    # HF backend, not vLLM, so it doesn't reserve ~78/80GB up front the way
    # vLLM does. Originally pinned to "slurm" out of caution; switched to
    # "any" on 2026-08-16 once that reference timing was on hand and both
    # A100 GPUs were sitting idle with nothing else queued for them.
    #
    # Moved to lane="cpu" later the same day (the "run by hand" convention,
    # not actually CPU-only — see other units with this comment) after a
    # real incident: mistral_small24b (model 6/7) silently fell back to CPU
    # (vLLM engine-init failure -> HF device_map="auto" -> Accelerate put the
    # whole 24B model in CPU RAM with no error) and per user direction was
    # dropped entirely rather than chased further (shim.drop_models(), see
    # run_gold.py's DROP_MODELS). The unit was then finished by hand on
    # dgx01 GPU7. Left as lane="any" during that manual run, the watcher
    # couldn't see it was already claimed and submitted a live duplicate to
    # SLURM once a slot freed (job 23159655, caught PENDING before it could
    # start). lane="cpu" is the fix — same "watcher must never auto-schedule
    # this" contract already used for every other manually-launched unit.
    os_out = _gold_out("opensource_llm")
    U.append(Unit(
        "gold_opensource_llm", 2,
        [PY, "-m", GOLD_PY, "scripts.clean_gold.clean_dataset_with_opensource_llm",
         "--input", str(P.GOLD_V3_CSV),
         "--output", str(os_out / "classified.csv"),
         "--log", str(os_out / "classify.log"),
         "--summary", str(os_out / "summary.txt"),
         "--error", str(os_out / "error_analysis.txt"),
         "--pred-file", str(os_out / "predicate_analysis.txt"),
         "--no-archive"],
        lane="cpu", est_hours=1.0, needs=["prep_gold_eval"]))

    # Strategy 5: 3 frontier API models via OpenRouter (gpt-5.4, gemini-3-
    # flash-preview, claude-haiku-4-5) x 2 langs x 2 shot configs, pure HTTP —
    # no GPU, no checkpoint. Needs OPENROUTER_API_KEY, which lives only in
    # the gitignored post_rebuttal_and_camera_ready/.env (never in this
    # tracked file) — user-provided 2026-08-16, scoped to this project only.
    # lane="cpu" is deliberate: next_unit() skips "cpu" units entirely (see
    # module docstring — same as the stage-0 prep units), so this is meant to
    # be run by hand via run_unit.py, not through the SLURM/A100 schedulers
    # that don't apply to it. Smoke-tested at --debug 3: all three models
    # responded correctly, ~$0.0047 for 3 rows — extrapolates to roughly
    # $1.50-2 for the full 500-row x 2-lang x 2-shot sweep.
    _env_vars = _load_env_file(P.POST / ".env")
    if "OPENROUTER_API_KEY" in _env_vars:
        api_out = _gold_out("api_llm")
        U.append(Unit(
            "gold_api_llm", 2,
            [PY, "-m", GOLD_PY, "scripts.clean_gold.clean_dataset_with_api_llm",
             "--input", str(P.GOLD_V3_CSV),
             "--output", str(api_out / "classified.csv"),
             "--log", str(api_out / "classify.log"),
             "--summary", str(api_out / "summary.txt"),
             "--error", str(api_out / "error_analysis.txt"),
             "--pred-file", str(api_out / "predicate_analysis.txt"),
             "--prompts", str(api_out / "prompts.jsonl"),
             "--no-archive"],
            lane="cpu", est_hours=0.2, needs=["prep_gold_eval"],
            env={"OPENROUTER_API_KEY": _env_vars["OPENROUTER_API_KEY"]}))

    # Strategy 3 remainder: scripts/clean_gold/clean_cross_train_rc.py sweeps
    # 6 base encoders x K in {3,5,7,10} = 24 configs; 2 (neodictabert_k3,
    # mmbert_k5) are already covered as a byproduct of the silver campaign's
    # own cross-train script. Each remaining config is a full K-fold TRAINING
    # run, not inference against an existing checkpoint like Strategies 1/2/4
    # — categorically heavier, flagged as out of scope in the 2026-08-16
    # README write-up. Smoke-tested 2026-08-16 (10 steps, 200 debug rows,
    # 1m20s, correct output) before committing to real runs. Real runs use
    # the script's own default budget (10000 steps, lr=2e-5) — no shortcuts,
    # since a smaller step budget would not be the config the paper wants.
    # Starting with 4 of the 22 remaining configs, smallest models first, per
    # user direction 2026-08-16 ("use 1 SLURM and 2 A100 to finish all, DGX
    # for the smaller models if needed") — more can be added the same way as
    # capacity allows; this is deliberately not an attempt to queue all 22.
    #
    # xlmroberta_k3/neodictabert_k7/me5large_k3 moved to lane="cpu" 2026-08-16:
    # not actually CPU-only, but "cpu" is this registry's existing convention
    # for "next_unit() must never auto-schedule this — run by hand" (same as
    # gold_api_llm and the stage-0 prep units). They run on dgx01/dgx03
    # (shared V100-32GB hosts, reached directly by ssh, no SLURM/A100 involved)
    # under the `dicta_nemotron` env — the repo's usual `hre_finetuned_nli` env
    # needs a newer CUDA driver than dgx01/03 have; dicta_nemotron's
    # torch==2.6.0+cu124 was verified working there. dgx02 avoided entirely
    # (other users' load average ~76/80 cores when checked). Launched directly
    # via ssh + nohup + logfile + captured PID, same pattern as a100_lane.sh
    # (stdin redirected, thread pools capped) — NOT a bare `timeout ssh`, which
    # orphaned two duplicate processes earlier this session (see README).
    #
    # --train-batch 32 (script default 256) is load-bearing for these three,
    # not a tuning choice: smoke-tested me5large (560M params — same scale as
    # xlmroberta-large) at the default batch size on a V100-32GB and it OOM'd
    # on every fold, even at --debug 200/--steps-list 10 — "31.73 GiB memory
    # in use" before the first optimizer step completed, so more silver rows
    # or more real steps would not have changed the outcome. Retested at
    # --train-batch 32: completed a full fold cleanly at 20.7 GiB. Worth
    # flagging in the eventual results write-up that these three configs ran
    # at a smaller batch size than mmbert_k3 (A100, default 256) for a
    # hardware reason, not a methodology choice.
    # me5large_k5 added 2026-08-16 to match the paper's Table 3 (mE5-large K=5,
    # F1=0.818). The K=3 configs above are NOT paper configs — §5.3 sweeps
    # K ∈ {3,5,10} and Table 3 reports XLM-R K=10, mmBERT K=5, mE5 K=5,
    # NeoDictaBERT K=3. Of those four, NeoDictaBERT K=3 and mmBERT K=5 are
    # already covered as byproducts of the silver campaign, so only XLM-R K=10
    # (below, on A100) and mE5 K=5 (here) were missing. User chose to keep the
    # non-paper K=3/K=7 runs as extra data points rather than cancel them.
    DGX_PY = TORCH26_PY

    # Full paper sweep completed 2026-08-18 on user request: all 4 Strategy-3
    # models x K in {3,5,7,10}. Per-model train batch is evidence-based, not
    # tuned: 256 is proven stable for mmbert/neodictabert (mmbert_k3/k5,
    # neodictabert_k7 all healthy at 256); 32 for xlmroberta/me5large (256
    # OOMs a V100 outright and was never validated on these two -- 32 is the
    # setting every healthy run of them used). eval-batch 4096 everywhere
    # (pure inference, no optimizer state; cut k7's silver-infer pass ~4x).
    # All lane="cpu": run by hand in two sequential A100 chains -- SLURM/B200
    # is off the table because TORCH26_PY (cu124) has no sm_100 kernels, the
    # exact failure gold_encoder_nli hit there.
    #
    # CORRECTION 2026-08-18: neodictabert was first listed here at 256 on the
    # claim "k7 proven at 256" -- false. The 256 k7 run is the quarantined
    # torch-2.11 NaN one; the *successful* Aug-18 k7 rerun trained at 32
    # (its unit cmd carried --train-batch 32). At 256 + eval 4096 the process
    # self-allocates 78.3 GiB and OOMs fold 0 on an empty A100 -- k5 and k10
    # burned through all folds as FAILED in ~2 min each, exited 0, and wrote
    # false .done markers (stubs quarantined, markers cleared).
    # CORRECTED CONFIG 2026-08-20 — matches the v1 runs that produced the
    # paper's Table 3 (steps8000_lr2e-5, train batch 256, eval batch 1024).
    #
    # Why the previous grid was wrong: --train-batch 32 was chosen to survive a
    # V100's 32 GB, then never revisited after everything moved to A100. Because
    # the script trains for a FIXED max_steps, batch size sets how much data the
    # model sees: 10000x32 = 320K samples vs v1's 8000x256 = 2.05M — an 8x cut.
    # Final loss tracked it exactly (718-way task, random = 6.58): mmBERT at
    # batch 256 reached 1.58 and F1 0.72-0.74, while mE5-large at batch 32 sat
    # at 4.0-4.3 (barely above chance) and F1 0.51-0.56. The threshold protocol
    # was NOT the cause — v1 mmbert_k5 scores 0.820 both at t=0.5 and at its
    # best threshold, so the paper already used plain majority vote.
    #
    # eval_batch: 4096 is proven for mmBERT only. neodictabert OOM'd on an A100
    # at train256+eval4096 (self-allocated 78.3 GiB), so the three larger models
    # use the script's default 1024 — exactly what v1 used.
    RC_GOLD_CONFIGS = [
        # model, K, est_hours, train_batch, eval_batch
        ("mmbert",       3,  4.0, 256, 4096),
        ("mmbert",       5,  6.0, 256, 4096),
        ("mmbert",       7,  8.0, 256, 4096),
        ("mmbert",      10, 11.0, 256, 4096),
        ("xlmroberta",   3,  8.0, 256, 1024),
        ("xlmroberta",   5, 13.0, 256, 1024),
        ("xlmroberta",   7, 18.0, 256, 1024),
        ("xlmroberta",  10, 26.0, 256, 1024),
        ("me5large",     3,  8.0, 256, 1024),
        ("me5large",     5, 13.0, 256, 1024),
        ("me5large",     7, 18.0, 256, 1024),
        ("me5large",    10, 26.0, 256, 1024),
        # neodictabert: ModernBERT-style arch with no flash-attn path under
        # torch 2.6 on A100 (mmBERT gets flash via torch 2.11 on B200, but
        # 2.11 NaNs neodicta) -- train-batch 256 self-allocates 78GB and OOMs
        # at step 0 (failed identically 2026-08-18 and 2026-08-23; stubs in
        # runs/quarantine_2026-08-23/). 128 x 16000 steps keeps the grid's
        # data volume identical (2.048M samples = 256 x 8000).
        ("neodictabert", 3,  9.0, 128, 1024),
        ("neodictabert", 5, 15.0, 128, 1024),
        ("neodictabert", 7, 21.0, 128, 1024),
        ("neodictabert",10, 30.0, 128, 1024),
    ]
    # XLM-R-large K=10 — the paper's Table 3 config (F1=0.817), and the single
    # longest job in the campaign: 10 fold-cycles, each a full 10k-step training
    # run plus a cross-fold inference pass over ~2.3M rows. Placed on A100 with
    # the script's own default batch (256) and a raised eval batch, for the same
    # reason neodictabert_k7 was moved there — the V100 measured ~1.73 s/step,
    # which at K=10 would run into multiple days. This is the 2nd and last A100
    # GPU for this campaign (neodictabert_k7 holds the other); everything else
    # goes to DGX, per the standing 2-A100 cap.
    # DGX_PY (torch 2.6.0+cu124), NOT PY (torch 2.11.0+cu130): XLM-R-large
    # produces avg_loss=nan from step 1 under torch 2.11 on A100, at both
    # --train-batch 256 and 32, while the *same model* trains healthily
    # (loss 3.49) under torch 2.6 on the DGX, and neodictabert_k7 trains
    # healthily under torch 2.11 on the same A100. So it is an
    # XLM-R-large x torch-2.11 interaction, not batch size and not hardware
    # -- the same class of environment drift as the neodictabert /
    # transformers-4.55.4 pin documented above. ~5 GPU-h lost to two wrong
    # hypotheses before comparing interpreters (2026-08-17).

    # neodictabert_k7 pulled out of the shared DGX loop 2026-08-16: after
    # fold 0 alone took ~4.2h training + a projected ~9.5h cross-fold
    # inference pass on the V100 (K=7 means 7 of these cycles — a multi-day
    # total, discovered only once fold 0 reached the inference phase), moved
    # to A100 for real speed instead of just tolerating it. Two changes from
    # the DGX default: --train-batch back up to the script's own default 256
    # (already proven safe at that value for mmbert_k3 on this same A100
    # class), and --eval-batch raised 1024 -> 4096 since inference carries no
    # optimizer state and the V100 run was only using half its 32GB at 1024
    # (16.6/32.8 GiB) — same headroom argument scaled to the A100's 80GB
    # leaves a wide margin. Restarted from scratch: the script's fold-level
    # checkpoint only saves after a fold's full train+inference cycle, and
    # fold 0 hadn't reached that point yet, so the ~5h13m already spent on
    # the V100 couldn't be preserved. Still lane="cpu" (watcher must not
    # auto-schedule it) even though it now runs via ssh to dsinlp01, not a
    # DGX host — the interpreter is the normal hre_finetuned_nli, not
    # DGX_PY, since A100 doesn't have DGX's older-glibc/driver constraints.
    # DGX_PY, same reason as xlmroberta_k10 below: this run was healthy at
    # 20 min (loss 5.09) but had degraded to avg_loss=nan by 1h40 under
    # torch 2.11. Every cross_train_rc job on torch 2.11 NaNs (xlmroberta_k10
    # immediately, neodictabert_k7 partway through fold 1); every one on
    # torch 2.6 stays healthy (xlmroberta_k3 3.49, me5large_k3 3.99,
    # me5large_k5 3.97). Pin the whole strategy to torch 2.6.

    for model, k, hours, tb, eb in RC_GOLD_CONFIGS:
        rc_out = _gold_out(f"cross_train_rc/{model}_k{k}")
        # constant data volume across the grid: steps x batch = 2.048M samples
        steps = 8000 * 256 // tb
        cmd = [TORCH26_PY, "-m", "scripts.clean_gold.clean_cross_train_rc",
               "--silver", str(P.SILVER_V3_PARQUET),
               "--gold", str(P.GOLD_V3_CSV),
               "--output-base", str(rc_out),
               "--models", model,
               "--k-values", str(k),
               "--steps-list", str(steps),
               "--train-batch", str(tb),
               "--eval-batch", str(eb)]
        U.append(Unit(
            f"gold_rc_{model}_k{k}", 2, cmd,
            lane="cpu", est_hours=hours, needs=["prep_silver", "prep_gold_eval"]))

    # ---------------- stage 3: merge + publish (CPU) -----------------------
    silver_ids = [u.id for u in U if u.stage == 2
                  and not u.id.startswith(("gold_encoder_nli", "gold_llm_nli_",
                                            "gold_opensource_llm", "gold_api_llm", "gold_rc_"))]
    U.append(Unit("merge_silver_v3", 3,
                  [PY, "-m", f"{PIPE}.build_silver_v3"],
                  lane="cpu", est_hours=0.5, needs=silver_ids))

    return U


def by_id(units=None) -> dict[str, Unit]:
    return {u.id: u for u in (units or build_units())}


def summary(units=None) -> str:
    units = units or build_units()
    lines = [f"{'unit':32s} {'stage':>5s} {'lane':>6s} {'est_h':>6s}  progress"]
    for u in units:
        lines.append(f"{u.id:32s} {u.stage:5d} {u.lane:>6s} {u.est_hours:6.1f}  {u.progress()}")
    total = sum(u.est_hours for u in units if u.lane != "cpu" and not u.is_done())
    lines.append(f"\nremaining GPU hours (single lane): {total:.1f}")
    return "\n".join(lines)


if __name__ == "__main__":
    P.ensure_dirs()
    print(summary())
