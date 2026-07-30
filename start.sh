#!/usr/bin/env bash
set -euo pipefail
echo "=== start.sh begin ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
DEFAULT_CONFIG_PATH="${SCRIPT_DIR}/configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_topk32.yaml"
MOPD_SEED="${MOPD_SEED:-42}"

usage() {
  cat <<'USAGE'
Usage:
  bash start.sh [--config|-c <config[::profile]>] [launcher args...]
  bash start.sh [<config[::profile]>] [launcher args...]

Examples:
  bash start.sh
  bash start.sh --config configs/mopd_formal_audit_all_8gpu.yaml
  bash start.sh test_grad_configs/mopd_domain_weighting_qwen0p6b_8b_matrix.yaml::gradnorm --dry-run

If no config is selected, start.sh uses the original Top-32 config.
USAGE
}

CONFIG_REFERENCE="${MOPD_CONFIG:-${DEFAULT_CONFIG_PATH}}"
case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  -c|--config)
    if [[ $# -lt 2 ]]; then
      echo "$1 requires a config path." >&2
      exit 2
    fi
    CONFIG_REFERENCE="$2"
    shift 2
    ;;
  --config=*)
    CONFIG_REFERENCE="${1#*=}"
    shift
    ;;
  *.yaml|*.yml|*::*)
    CONFIG_REFERENCE="$1"
    shift
    ;;
esac

PROFILE_SUFFIX=""
CONFIG_FILE="${CONFIG_REFERENCE}"
if [[ "${CONFIG_REFERENCE}" == *::* ]]; then
  PROFILE_SUFFIX="::${CONFIG_REFERENCE##*::}"
  CONFIG_FILE="${CONFIG_REFERENCE%::*}"
  if [[ "${PROFILE_SUFFIX}" == "::" ]]; then
    echo "Config profile cannot be empty: ${CONFIG_REFERENCE}" >&2
    exit 2
  fi
fi

if [[ "${CONFIG_FILE}" != /* && -f "${CONFIG_FILE}" ]]; then
  CONFIG_FILE="$(cd "$(dirname "${CONFIG_FILE}")" && pwd -P)/$(basename "${CONFIG_FILE}")"
elif [[ "${CONFIG_FILE}" != /* && -f "${SCRIPT_DIR}/${CONFIG_FILE}" ]]; then
  CONFIG_FILE="$(cd "$(dirname "${SCRIPT_DIR}/${CONFIG_FILE}")" && pwd -P)/$(basename "${CONFIG_FILE}")"
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config not found: ${CONFIG_FILE}" >&2
  exit 2
fi
CONFIG_REFERENCE="${CONFIG_FILE}${PROFILE_SUFFIX}"

# These variables must be set before Python, Ray, Torch, or vLLM starts.
export MOPD_GLOBAL_SEED="${MOPD_SEED}"
export PYTHONHASHSEED="${MOPD_SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export FLASH_ATTENTION_DETERMINISTIC="${FLASH_ATTENTION_DETERMINISTIC:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
echo "CONFIG_REFERENCE=${CONFIG_REFERENCE}"
echo "=== start.sh: exec run_local_mopd_training.sh ==="

# Temporarily disable set -e so we can capture the exit code
set +e
bash "${SCRIPT_DIR}/scripts/run_local_mopd_training.sh" \
  "${CONFIG_REFERENCE}" \
  "$@" \
  -- \
  "++data.seed=${MOPD_SEED}" \
  "++actor_rollout_ref.rollout.seed=${MOPD_SEED}" \
  "++trainer.seed=${MOPD_SEED}"
rc=$?
set -e
echo "=== start.sh: run_local_mopd_training.sh exited with rc=${rc} ==="
exit ${rc}
