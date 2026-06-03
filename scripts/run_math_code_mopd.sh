#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${MOPD_CONFIG:-${CODE_DIR}/configs/mopd_math_code.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"

ARGS=(--config "${CONFIG_PATH}")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

exec "${PYTHON_BIN}" -m mopd_verl.launch "${ARGS[@]}" "$@"
