#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_weight_diff_conflict.sh \
    --student ../models/Qwen3-4B \
    --teacher math=../models/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
    --teacher code=../models/Qwen3-4B-Non-Thinking-RL-Code-Step300 \
    --output-jsonl audit/weight_diff_conflict/pairs.jsonl \
    --teacher-jsonl audit/weight_diff_conflict/teachers.jsonl \
    --layer-jsonl audit/weight_diff_conflict/layers.jsonl \
    --output-md audit/weight_diff_conflict/summary.md

Environment:
  MOPD_REMOTE_CONDA_ENV=/root/miniconda3/envs/mopd-verl
  MOPD_REMOTE_CONDA_ROOT=/root/miniconda3
  VERL_RUNTIME_DIR=<repo>/third_party/verl
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-${CODE_DIR}/third_party/verl}"
MOPD_REMOTE_CONDA_ENV="${MOPD_REMOTE_CONDA_ENV:-/root/miniconda3/envs/mopd-verl}"
MOPD_REMOTE_CONDA_ROOT="${MOPD_REMOTE_CONDA_ROOT:-/root/miniconda3}"

if [[ -d "${MOPD_REMOTE_CONDA_ENV}/bin" ]]; then
  export PATH="${MOPD_REMOTE_CONDA_ENV}/bin:${MOPD_REMOTE_CONDA_ROOT}/bin:${PATH:-}"
fi

export PYTHONPATH="${CODE_DIR}:${VERL_RUNTIME_DIR}:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

cd "${CODE_DIR}"
exec "${PYTHON_BIN}" -m mopd_verl.weight_diff_conflict "$@"
