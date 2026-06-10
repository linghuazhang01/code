#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="$(cd "${EVAL_DIR}/.." && pwd)"

if [[ -f "${CODE_DIR}/logs/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CODE_DIR}/logs/env.sh"
fi

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
ENV_NAME="${ENV_NAME:-mopd-verl}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"

cd "${CODE_DIR}"
exec "${PYTHON_BIN}" -m eval.official_runner "$@"
