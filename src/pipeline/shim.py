"""
Redirect the existing silver classifiers at the v3 data and the new output tree
WITHOUT editing a single line of them.

`scripts/clean_silver/classify_silver_*.py` hardcode two roots:

    SILVER_CSV = ROOT/"data"/"prepared_silver.csv"
    OUT_DIR    = ROOT/"outputs"/"silver_cleaning"/<family>

and derive CLASS_DIR / LOG_DIR / LOG_FILE / OUT_CSV from them. Every function
resolves these as module globals at call time — `_pred_path(tag)` returns
`OUT_DIR / f"pred_{tag}.txt"` when called, not when defined — so rebinding the
module attributes before invoking `main()` redirects all of it.

Editing those scripts would have been the obvious alternative, but the brief for
this re-run is that the pre-rebuttal material stays untouched, and a shim keeps
that guarantee absolute: the old scripts still default to the old paths for
anyone who runs them directly.

`classify_silver_cross_train_rc.py` already exposes --silver/--gold/
--output-base, so it needs no patching and is driven by flags instead.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from post_rebuttal_and_camera_ready.pipeline import paths as P

# family -> (module, output subdirectory, merged-csv basename)
FAMILIES = {
    "opensource_llm": ("scripts.clean_silver.classify_silver_opensource_llm",
                       "silver_opensource_llm", "silver_opensource_llm.csv"),
    "finetuned_nli": ("scripts.clean_silver.classify_silver_finetuned_nli",
                      "silver_finetuned_nli", "silver_finetuned_nli.csv"),
    "encoder_nli": ("scripts.clean_silver.classify_silver_encoder_nli",
                    "silver_encoder_nli", "silver_encoder_nli.csv"),
}


def load_redirected(family: str, silver_csv: Path = None, out_root: Path = None):
    """Import a classifier module and repoint its paths into the v3 tree."""
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; expected one of {list(FAMILIES)}")
    mod_name, subdir, out_csv_name = FAMILIES[family]
    mod = importlib.import_module(mod_name)

    silver_csv = Path(silver_csv or P.SILVER_V3_CSV)
    out_dir = Path(out_root or P.SILVER_SCORES) / subdir

    if not silver_csv.exists():
        raise SystemExit(f"input not found: {silver_csv} — run prepare_v3_data first")

    old_out_dir = Path(getattr(mod, "OUT_DIR"))

    mod.SILVER_CSV = silver_csv
    mod.OUT_DIR = out_dir
    mod.CLASS_DIR = out_dir / "classifications"
    mod.LOG_DIR = out_dir / "logs"
    mod.LOG_FILE = out_dir / "run.log"
    mod.OUT_CSV = mod.CLASS_DIR / out_csv_name

    # Rebind every OTHER module-level Path that was derived from the original
    # OUT_DIR at import time. Hand-listing the constants missed SUMMARY
    # (= OUT_DIR / "summary.txt"), which kept pointing into the old tree and
    # would have written there at the end of a run. Anything shaped like
    # "<old OUT_DIR>/x" is remapped to "<new OUT_DIR>/x" generically, so a
    # constant added to those scripts later cannot silently escape.
    _remap_derived_paths(mod, old_out_dir, out_dir)
    _remap_checkpoints(mod)
    _inject_extra_models(mod, family)
    _fix_large_text_batch_size(mod, family)

    for d in (mod.OUT_DIR, mod.CLASS_DIR, mod.LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    _assert_redirected(mod, out_dir)
    return mod


# The Hebrew_NLI project — which holds every fine-tuned checkpoint these
# classifiers load — was renamed and restructured, exactly as this repo was:
#
#   Hebrew_NLI/output                      -> HEntailment-.../finetune_heb_nli/decoder_checkpoints
#   Hebrew_NLI/finetune_heb_nli/outputs    -> HEntailment-.../finetune_heb_nli/encoder_checkpoints
#
# The old paths are baked into MODELS[tag]["ckpt"] as f-strings evaluated at
# import, so rebinding _HEB_NLI_OUT / _ENC_BASE alone is not enough — the
# already-built strings have to be rewritten too.
_HEB_NLI_NEW = "/path/to/HEntailment-Hebrew-NLI-Repurposing-EACL27-Internal"
CHECKPOINT_REMAP = [
    ("/path/to/Hebrew_NLI/finetune_heb_nli/outputs",
     f"{_HEB_NLI_NEW}/finetune_heb_nli/encoder_checkpoints"),
    ("/path/to/Hebrew_NLI/output",
     f"{_HEB_NLI_NEW}/finetune_heb_nli/decoder_checkpoints"),
]


def _remap_one(value: str) -> str:
    for old, new in CHECKPOINT_REMAP:
        if value.startswith(old):
            return new + value[len(old):]
    return value


def _remap_checkpoints(mod):
    """Point every model checkpoint at the renamed Hebrew_NLI project."""
    for attr in ("_HEB_NLI_OUT", "_ENC_BASE"):
        if isinstance(getattr(mod, attr, None), str):
            setattr(mod, attr, _remap_one(getattr(mod, attr)))

    models = getattr(mod, "MODELS", None)
    if not isinstance(models, dict):
        return
    for tag, cfg in models.items():
        if isinstance(cfg, dict) and isinstance(cfg.get("ckpt"), str):
            cfg["ckpt"] = _remap_one(cfg["ckpt"])


# Mechanism for adding a model to the silver campaign's curated 3-per-family
# subset without editing the frozen scripts (MODELS in each is a literal
# 3-entry dict). Used once, 2026-08-16: added Mistral-Small-24B-Instruct-2501
# to both families per user request, then reverted the same day back to the
# original 3-model curated set to keep the silver corpus on the paper's
# original models (prompted by a duplicate-launch incident — see README).
# Both its pred files were deleted, not merged. Left empty rather than
# removed — the injection point (load_redirected -> _inject_extra_models) is
# generic and ready to reuse if a model needs adding again.
EXTRA_SILVER_MODELS: dict = {}


def _inject_extra_models(mod, family: str):
    extra = EXTRA_SILVER_MODELS.get(family)
    models = getattr(mod, "MODELS", None)
    if not extra or not isinstance(models, dict):
        return
    for tag, cfg in extra.items():
        models[tag] = dict(cfg)
    order = getattr(mod, "MODEL_ORDER", None)
    if isinstance(order, list):
        order.extend(t for t in extra if t not in order)


# classify_silver_encoder_nli.py's own "use a smaller batch for long
# documents" guard is a no-op as written: LARGE_BATCH_SIZE == BATCH_SIZE
# (both 64), so run_encoder_chunk()'s `bs = LARGE_BATCH_SIZE if long else
# BATCH_SIZE` never actually shrinks anything. neodictabert hit a genuine
# "CUDA out of memory: tried to allocate 48.00 GiB" on one silver document —
# the (64, 12, s0, s0) attention tensor in the traceback backs out to
# s0 ≈ 4096 tokens.
#
# First fix (2026-08-16, same day) only touched LARGE_BATCH_SIZE (64→8) and
# left LARGE_TEXT_THRESH at the script's original 256 chars. Measured against
# a 200K-row sample of the real corpus, 99.2% of rows exceed 256 chars — so
# that threshold routes nearly the *entire* 2.56M-row corpus onto the slow
# batch=8 path, not just the one outlier document. Real throughput came back
# at 29.2 rows/s, a ~24h ETA instead of the ~3h estimate.
#
# Second attempt raised LARGE_TEXT_THRESH to 8000 chars (comfortably above
# p99 of 7,506, well below the actual outlier's 34,553) expecting that to fix
# it — it didn't move the needle at all (29.8 rows/s after). Root cause is
# one level deeper than either constant: run_encoder_chunk() computes
# max_chars = max(...) over the ENTIRE ~50,000-row chunk (CHUNK_SIZE) and
# picks ONE batch size for the whole chunk. With chunks that large, raising
# the per-row threshold doesn't help — at any realistic threshold, a 50k-row
# chunk from a 2.56M-row corpus almost certainly contains at least one row
# past it, so nearly every chunk still gets flagged "large" in its entirety.
# Fixed by replacing run_encoder_chunk itself (monkey-patched onto the
# module, not edited in the frozen file) with a version that decides batch
# size per mini-batch — using a BATCH_SIZE-sized lookahead window — instead
# of once for the whole chunk. Long documents still get the safe small
# batch; short documents on either side of them still get the fast one.
def _fix_large_text_batch_size(mod, family: str):
    if family != "encoder_nli":
        return
    if hasattr(mod, "LARGE_BATCH_SIZE"):
        mod.LARGE_BATCH_SIZE = 8
    if hasattr(mod, "LARGE_TEXT_THRESH"):
        mod.LARGE_TEXT_THRESH = 8000
    if hasattr(mod, "run_encoder_chunk"):
        mod.run_encoder_chunk = _make_per_minibatch_run_encoder_chunk(mod)
    # Regression 2026-08-16: this call was dropped when the torch.compile
    # no-op below was extracted into the standalone disable_torch_compile()
    # (to also be reusable from run_gold.py) — left this function calling
    # nothing, so the very next relaunch crashed on dgx03 with the exact
    # GLIBC/Triton error this was already fixed for. Caught within 2 minutes
    # via the run_unit crash-loop's exit=1, no real time lost.
    disable_torch_compile()


def _make_per_minibatch_run_encoder_chunk(mod):
    import torch as _torch

    def run_encoder_chunk(model, tokenizer, device, pairs):
        all_pe: list[float] = []
        i, n = 0, len(pairs)
        while i < n:
            window = pairs[i:i + mod.BATCH_SIZE]
            max_chars = max((len(p) + len(h) for p, h in window), default=0)
            bs = mod.LARGE_BATCH_SIZE if max_chars > mod.LARGE_TEXT_THRESH else mod.BATCH_SIZE
            batch = pairs[i:i + bs]
            enc = tokenizer(
                [p for p, _ in batch], [h for _, h in batch],
                return_tensors="pt", padding="longest", truncation=True,
                add_special_tokens=True, return_token_type_ids=False,
            )
            for k in enc:
                enc[k] = enc[k].to(device)
            with _torch.no_grad():
                logits = model(**enc, return_dict=True).logits.softmax(dim=1)
            all_pe.extend(logits[:, 0].cpu().tolist())
            i += bs
        return all_pe

    return run_encoder_chunk

    # Separate issue, found 2026-08-16 trying to run this same unit on the
    # dgx01/dgx03 V100 hosts: torch.compile(model) (line ~373) is wrapped in
    # its own try/except, but that only guards the wrap call — actual
    # compilation is lazy and happens on the first real forward pass, outside
    # that guard, so a compile failure there is a hard crash, not the
    # graceful "continuing without" fallback the script's log message implies.
    # Hit exactly that on dgx03: Triton's precompiled CUDA utils need a newer
    # glibc than that host has (GLIBC_2.34 not found).
    #
    # First attempt was torch._dynamo.config.suppress_errors = True (the fix
    # the error message itself suggests) — it does stop the crash, but dynamo
    # retries compilation on every new input shape and fails the same way
    # every time, so a 2.56M-row run with varying document lengths would
    # spend the whole run re-attempting and re-failing compilation, each
    # failure dumping a full multi-line traceback to the log. Cleaner fix:
    # make torch.compile() itself a no-op, so the script's call still runs
    # (nothing to edit in the frozen file) but never attempts compilation at
    # all. Applied unconditionally, not just for the DGX hosts — encoder
    # inference here is a single forward pass per batch, not a tight
    # repeated-shape loop, so compilation was unlikely to be paying for
    # itself even on A100, and this removes an entire class of host
    # dependency for a cost that should be in the noise.
    disable_torch_compile()


def disable_torch_compile():
    """Make torch.compile() a no-op for the rest of this process.

    Shared fix for two independent scripts hitting the same underlying issue
    on dgx01/dgx03 (older glibc than Triton's precompiled CUDA utils need):
    classify_silver_encoder_nli.py and clean_dataset_with_opensource_llm.py
    each wrap their own torch.compile(model) call in a try/except that only
    guards the wrap itself — actual compilation is lazy, on first forward
    pass, outside that guard — so a compile failure there is a hard crash on
    those hosts, not the graceful fallback the scripts' log messages imply.
    Making the call itself a no-op sidesteps this entirely, on any host.
    """
    import torch
    torch.compile = lambda model, *a, **k: model


def assert_checkpoints_exist(mod, tags=None):
    """Fail in seconds if a checkpoint is missing, not hours into a GPU booking.

    A stale checkpoint path previously cost ~6 hours: the unit crashed on model
    load every time, and with no precondition check the lanes simply kept
    relaunching it.
    """
    models = getattr(mod, "MODELS", None)
    if not isinstance(models, dict):
        return
    missing = []
    for tag, cfg in models.items():
        if tags and tag not in tags:
            continue
        ckpt = cfg.get("ckpt") if isinstance(cfg, dict) else None
        if not ckpt:
            continue
        p = Path(ckpt)
        if not p.exists():
            missing.append(f"{tag}: {ckpt} (not found)")
        elif not (p / "adapter_config.json").exists() and not (p / "config.json").exists():
            missing.append(f"{tag}: {ckpt} (no adapter_config.json / config.json)")
    if missing:
        raise SystemExit("checkpoint precondition failed:\n  " + "\n  ".join(missing))


# neodictabert NaN bug (2026-08-15): scores are 100% literal "nan" for every
# row, reproduced even on pure CPU with eager attention and torch.compile
# skipped entirely — ruling out GPU kernels, SDPA backend selection, and
# compilation as the cause. The checkpoint's 174 weight tensors were scanned
# directly via safetensors and contain zero NaN/Inf, ruling out a corrupted
# checkpoint. That leaves the custom modeling_neobert.py forward pass itself,
# likely incompatible with the environment's transformers==5.8.1 (the
# checkpoint and diagnose_neodictabert.py predate that version by months).
# Not patchable from the shim — see post_rebuttal_and_camera_ready/README.md
# for the open decision on how to proceed (skip the model / pin an older
# transformers version for this one process / debug modeling_neobert.py).


def _remap_derived_paths(mod, old_root: Path, new_root: Path):
    """Repoint any module attribute that still lives under the old output root."""
    for name in dir(mod):
        if name.startswith("__"):
            continue
        val = getattr(mod, name, None)
        if not isinstance(val, Path):
            continue
        try:
            rel = val.relative_to(old_root)
        except ValueError:
            continue
        setattr(mod, name, new_root / rel)


def _assert_redirected(mod, out_dir: Path):
    """Fail loudly rather than silently writing into the old tree.

    Checks every Path attribute on the module, not a curated list — the curated
    list is exactly how SUMMARY slipped through.
    """
    old = str(P.ROOT / "outputs")
    offenders = []
    for name in dir(mod):
        if name.startswith("__"):
            continue
        val = getattr(mod, name, None)
        if isinstance(val, Path) and str(val).startswith(old):
            offenders.append(f"{name}={val}")
    if offenders:
        raise SystemExit(f"shim failed: {mod.__name__} still points into outputs/: "
                         + ", ".join(offenders))
    if hasattr(mod, "_pred_path"):
        probe = str(mod._pred_path("probe"))
        if not probe.startswith(str(out_dir)):
            raise SystemExit(f"shim failed: _pred_path resolves to {probe}")


def run(family: str, argv: list[str]):
    """Invoke a redirected classifier's main() with the given CLI args."""
    mod = load_redirected(family)
    old_argv = sys.argv
    sys.argv = [mod.__name__] + list(argv)
    try:
        mod.main()
    finally:
        sys.argv = old_argv


# --- scripts/clean_gold/* --------------------------------------------------
# Unlike the silver family, every I/O path on these three scripts (input,
# output, log, summary, ...) is already a real --flag with an argparse
# default, so they need no OUT_DIR-style path shim at all — run_gold.py just
# passes the v3 paths on the command line.
#
# What they DO still bake in at import time, same as the silver scripts, are
# Hebrew_NLI checkpoint paths built from module-level string constants via
# f-strings. clean_encoder_NLI.py and clean_finetuned_llm_NLI.py each do this
# with a different container shape than _remap_checkpoints() above expects
# (MODELS: dict of dicts with a "ckpt" key) — NLI_MODELS is a list of
# (ckpt, tag, type) tuples, LLM_MODELS is a list of dicts — so they get their
# own remap function rather than overloading that one.
# clean_dataset_with_opensource_llm.py has no Hebrew_NLI dependency (few-shot
# prompting of base pretrained models) and needs no remap at all.

def remap_gold_checkpoints(mod):
    """Point clean_encoder_NLI / clean_finetuned_llm_NLI at the renamed
    Hebrew_NLI project. Rebinding _HEB_NLI_ENC/_HEB_NLI_LLM/_HEB_NLI_OUT alone
    is not enough — NLI_MODELS/LLM_MODELS were already built from the old
    values via f-strings at import time, so each entry's ckpt string is
    rewritten directly, same as CHECKPOINT_REMAP already does on the silver
    side.
    """
    for attr in ("_HEB_NLI_ENC", "_HEB_NLI_LLM", "_HEB_NLI_OUT"):
        if isinstance(getattr(mod, attr, None), str):
            setattr(mod, attr, _remap_one(getattr(mod, attr)))

    nli_models = getattr(mod, "NLI_MODELS", None)
    if isinstance(nli_models, list):
        mod.NLI_MODELS = [
            (_remap_one(ckpt) if isinstance(ckpt, str) else ckpt, tag, mtype)
            for ckpt, tag, mtype in nli_models
        ]

    llm_models = getattr(mod, "LLM_MODELS", None)
    if isinstance(llm_models, list):
        for cfg in llm_models:
            if isinstance(cfg, dict) and isinstance(cfg.get("ckpt"), str):
                cfg["ckpt"] = _remap_one(cfg["ckpt"])


def assert_gold_checkpoints_exist(mod, tags=None):
    """Same fail-in-seconds precondition as assert_checkpoints_exist(), for
    NLI_MODELS/LLM_MODELS instead of MODELS."""
    missing = []
    for ckpt, tag, _mtype in getattr(mod, "NLI_MODELS", None) or []:
        if tags and tag not in tags:
            continue
        if not Path(ckpt).exists():
            missing.append(f"{tag}: {ckpt} (not found)")
    for cfg in getattr(mod, "LLM_MODELS", None) or []:
        tag, ckpt = cfg.get("tag"), cfg.get("ckpt")
        if tags and tag not in tags:
            continue
        if ckpt and not Path(ckpt).exists():
            missing.append(f"{tag}: {ckpt} (not found)")
    if missing:
        raise SystemExit("checkpoint precondition failed:\n  " + "\n  ".join(missing))


# clean_dataset_with_opensource_llm.py's load_llm() tries vLLM first, and
# falls back to transformers.AutoModelForCausalLM.from_pretrained(
# device_map="auto") on failure — with no max_memory constraint. Found
# 2026-08-16: vLLM's engine failed to init for the 6th model loaded in one
# process (mistral_small24b, 24B) — plausibly GPU memory fragmentation left
# by 5 prior vLLM engines not tearing down perfectly cleanly — and the HF
# fallback's device_map="auto" then silently put the *entire* model on CPU
# instead of GPU. No error, no warning: just 215-225s/batch generation
# instead of what should be single-digit seconds, discovered only by
# noticing GPU memory (8.8GB) and utilization (0%) were both far too low for
# an active 24B-parameter run — confirmed by the process's own RSS (~49GB,
# matching a 24B bf16 model resident in CPU RAM).
#
# Fixed by forcing an explicit max_memory on every device_map="auto" call for
# AutoModelForCausalLM / Gemma4ForConditionalGeneration for the rest of this
# process's lifetime: monkey-patch .from_pretrained so device_map="auto"
# calls also get max_memory={0: "<cap>GiB", "cpu": "0GiB"} unless the caller
# already specified one. This does not touch the vLLM path (untouched, tries
# first as before) — it only changes what happens on the HF fallback: either
# the model now fits on GPU and runs at full speed, or accelerate raises a
# loud CUDA OOM instead of a silent multi-hour CPU crawl. 75GiB cap (of the
# A100's 80GB) leaves headroom for KV cache / activations on top of weights.
def force_gpu_only_device_map(max_gpu_gib: int = 75):
    import transformers

    def _wrap(cls):
        orig = cls.from_pretrained.__func__ if isinstance(cls.from_pretrained, classmethod) else cls.from_pretrained
        if getattr(orig, "_here_gpu_forced", False):
            return  # already wrapped (e.g. two classes sharing a bound method)

        def patched(model_id, *args, **kwargs):
            if kwargs.get("device_map") == "auto" and "max_memory" not in kwargs:
                kwargs["max_memory"] = {0: f"{max_gpu_gib}GiB", "cpu": "0GiB"}
            return orig(model_id, *args, **kwargs)
        patched._here_gpu_forced = True
        cls.from_pretrained = classmethod(lambda c, model_id, *a, **k: patched(model_id, *a, **k))

    _wrap(transformers.AutoModelForCausalLM)
    try:
        _wrap(transformers.Gemma4ForConditionalGeneration)
    except AttributeError:
        pass  # not every transformers version ships this class


def drop_models(mod, list_attr: str, tags: set[str], tag_key: str = "tag"):
    """Remove entries from a module-level list-of-dicts model config by tag,
    so main() never attempts them — no --models selector on this script
    (clean_dataset_with_opensource_llm.py) to do this via CLI. User call
    2026-08-16: skip mistral_small24b entirely rather than chase the
    CPU-fallback issue further; finish the other 6 models. Its 8/16 batches
    of partial progress on the killed run are simply not resumed — the
    script's own checkpoint logic works at whole-model granularity, and this
    model was never fully checkpointed."""
    models = getattr(mod, list_attr, None)
    if not isinstance(models, list):
        return
    models[:] = [m for m in models if m.get(tag_key) not in tags]
