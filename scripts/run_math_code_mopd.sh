#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${MOPD_CONFIG:-${CODE_DIR}/configs/mopd_math_code.yaml}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="python3"
  fi
fi
VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-${CODE_DIR}/third_party/verl}"

if [[ ! -f "${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" ]]; then
  echo "Vendored verl runtime not found at '${VERL_RUNTIME_DIR}'." >&2
  echo "Expected '${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py'." >&2
  echo "Sync or restore third_party/verl before launching training." >&2
  exit 2
fi
export PYTHONPATH="${CODE_DIR}:${VERL_RUNTIME_DIR}:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"

if ! "${PYTHON_BIN}" -c "import yaml" >/dev/null 2>&1; then
  echo "Python interpreter '${PYTHON_BIN}' cannot import yaml. Install requirements.txt or set PYTHON_BIN to the training environment." >&2
  exit 2
fi

ARGS=(--config "${CONFIG_PATH}")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

exec "${PYTHON_BIN}" -m mopd_verl.launch "${ARGS[@]}" "$@"
