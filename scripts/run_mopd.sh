#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_mopd.sh [config] [--dry-run] [-- <hydra overrides...>]

Examples:
  scripts/run_mopd.sh configs/mopd_formal_audit_all_2gpu.yaml --dry-run

  scripts/run_mopd.sh configs/mopd_formal_audit_off_2gpu.yaml -- \
    trainer.experiment_name=mopd_audit_off_manual

Environment:
  MOPD_CONFIG=<default config when config arg is omitted>
  VERL_RUNTIME_DIR=<vendored verl runtime dir>
  MOPD_LAUNCH_PYTHON=<python executable for this launcher, default: python3>
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${CODE_DIR}/configs/mopd_formal_audit_all_2gpu.yaml"
CONFIG_PATH="${MOPD_CONFIG:-${DEFAULT_CONFIG}}"
VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-${CODE_DIR}/third_party/verl}"
DRY_RUN_FLAG=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN_FLAG=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    -*)
      echo "Unknown script option: $1" >&2
      echo "Put Hydra overrides after '--'." >&2
      exit 2
      ;;
    *)
      if [[ "${CONFIG_PATH}" != "${MOPD_CONFIG:-${DEFAULT_CONFIG}}" ]]; then
        echo "Only one config path is allowed." >&2
        exit 2
      fi
      CONFIG_PATH="$1"
      shift
      ;;
  esac
done

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${CODE_DIR}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

if [[ ! -f "${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" ]]; then
  echo "Vendored verl runtime not found at '${VERL_RUNTIME_DIR}'." >&2
  echo "Expected '${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py'." >&2
  echo "Sync or restore third_party/verl before launching training." >&2
  exit 2
fi

export PYTHONPATH="${CODE_DIR}:${VERL_RUNTIME_DIR}:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"
cd "${CODE_DIR}"

ARGS=(--config "${CONFIG_PATH}")
if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN_FLAG}" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  ARGS+=(-- "${EXTRA_ARGS[@]}")
fi

exec "${MOPD_LAUNCH_PYTHON:-python3}" -m mopd_verl.launch "${ARGS[@]}"
