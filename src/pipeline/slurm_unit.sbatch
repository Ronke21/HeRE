#!/bin/bash
# Run one unit of the v3 re-run on the B200 cluster.
#
# The watcher submits this with --export=ALL,HERE_UNIT=<unit_id> and picks the
# partition. Both permitted partitions cap wall time at 4h, so this asks for
# 3h55m and lets the unit be killed mid-way: every unit resumes from its own
# pred file, and the watcher resubmits until the unit reports complete.
#
# Manual submission (normally the watcher does this):
#   sbatch -p p_b200_nlp -A ug_tsarfaty --export=ALL,HERE_UNIT=silver_llm_dictalm3 \
#          post_rebuttal_and_camera_ready/pipeline/slurm_unit.sbatch

#SBATCH -A ug_tsarfaty
#SBATCH -t 03:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --output=/path/to/HeRE-Hebrew-RE-Benchmark-EMNLP2026-Internal/post_rebuttal_and_camera_ready/runs/logs/slurm/%x_%j.out
#SBATCH --error=/path/to/HeRE-Hebrew-RE-Benchmark-EMNLP2026-Internal/post_rebuttal_and_camera_ready/runs/logs/slurm/%x_%j.err
#SBATCH --mail-user=ronke21@gmail.com
#SBATCH --mail-type=FAIL

set -uo pipefail

: "${HERE_UNIT:?HERE_UNIT must be set via --export=ALL,HERE_UNIT=<unit_id>}"

ROOT=/path/to/HeRE-Hebrew-RE-Benchmark-EMNLP2026-Internal
ENV_NAME=hre_finetuned_nli
ENV_LIB=/path/to/miniconda3/envs/${ENV_NAME}/lib/python3.10/site-packages

echo "JobID=${SLURM_JOB_ID:-?}  Node=$(hostname)  Unit=${HERE_UNIT}  Start=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Mamba-installed conda envs need explicit CUDA library paths on B200, or they
# silently fall back to CPU.
export LD_LIBRARY_PATH=${ENV_LIB}/nvidia/cuda_runtime/lib:${ENV_LIB}/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/path/to/.cache/huggingface}

cd "$ROOT"

/path/to/miniconda3/envs/${ENV_NAME}/bin/python -u \
    -m post_rebuttal_and_camera_ready.pipeline.run_unit "${HERE_UNIT}"
rc=$?

echo "Unit=${HERE_UNIT}  exit=${rc}  End=$(date)"
# Exit 0 even on a wall-clock kill: an unfinished unit is normal here and the
# watcher decides what happens next from the unit's own progress on disk.
exit 0
