#!/usr/bin/env bash
# start.sh — Launch MOPD training locally or via Slurm.
#
# Local mode (default):
#   bash start.sh [--config|-c <config>] [--foreground] [-- <hydra overrides>]
#
# Slurm mode:
#   bash start.sh [--config|-c <config>] --slurm [--slurm-args <sbatch directive>]... [-- <hydra overrides>]
#
# Environment variables:
#   GPU_IDS       Comma-separated physical GPU indices (local mode only).
#   MOPD_SEED     Random seed (default: 42).
#   SLURM_MEM     Memory per node for Slurm jobs (default: 700G).
#   SLURM_TIME    Wall-time limit for Slurm jobs (default: 72:00:00).
set -euo pipefail
echo "=== start.sh begin ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
DEFAULT_CONFIG_PATH="${SCRIPT_DIR}/configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_topk32.yaml"
MOPD_SEED="${MOPD_SEED:-42}"

usage() {
  cat <<'USAGE'
Usage:
  bash start.sh [--config|-c <config[::profile]>] [--foreground] [-- <hydra overrides>]
  bash start.sh [--config|-c <config[::profile]>] --slurm [--slurm-args <sbatch directive>]... [-- <hydra overrides>]

Examples:
  bash start.sh
  bash start.sh --config configs/mopd_formal_audit_all_8gpu.yaml
  bash start.sh -c configs/xxx.yaml --foreground
  bash start.sh -c configs/xxx.yaml --slurm
  bash start.sh -c configs/xxx.yaml --slurm --slurm-args "--partition=gpu" --slurm-args "--time=48:00:00"

If no config is selected, start.sh uses the original Top-32 config.
USAGE
}

CONFIG_REFERENCE="${MOPD_CONFIG:-${DEFAULT_CONFIG_PATH}}"
SLURM_FLAG=0
SLURM_EXTRA_DIRECTIVES=()
LAUNCHER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --slurm)
      SLURM_FLAG=1
      shift
      ;;
    --slurm-args)
      if [[ $# -lt 2 ]]; then
        echo "--slurm-args requires a value" >&2
        exit 2
      fi
      SLURM_EXTRA_DIRECTIVES+=("$2")
      shift 2
      ;;
    --)
      shift
      LAUNCHER_ARGS+=("$@")
      break
      ;;
    *.yaml|*.yml|*::*)
      CONFIG_REFERENCE="$1"
      shift
      ;;
    *)
      # Everything else (--foreground, --dry-run, --run-id, --tail, etc.)
      # is passed through to the launcher script.
      LAUNCHER_ARGS+=("$1")
      shift
      ;;
  esac
done

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

echo "CONFIG_REFERENCE=${CONFIG_REFERENCE}"

if [[ "${SLURM_FLAG}" == "1" ]]; then
  # ── Slurm mode ──────────────────────────────────────────────────────────
  # Hand off to run_mopd.sh which auto-detects GPU/CPU requirements from the
  # config and generates & submits an sbatch script.
  SLURM_MEM="${SLURM_MEM:-700G}"
  SLURM_TIME="${SLURM_TIME:-72:00:00}"

  echo "=== start.sh: submitting via Slurm ==="
  echo "Default resources: --mem=${SLURM_MEM} --time=${SLURM_TIME}"
  if [[ "${#SLURM_EXTRA_DIRECTIVES[@]}" -gt 0 ]]; then
    echo "Extra sbatch directives: ${SLURM_EXTRA_DIRECTIVES[*]}"
  fi

  # Pass default and user-specified sbatch directives via SLURM_EXTRA_ENV.
  # run_mopd.sh reads this env var and appends them as #SBATCH lines.
  export SLURM_EXTRA_ENV="--mem=${SLURM_MEM} --time=${SLURM_TIME}"
  if [[ "${#SLURM_EXTRA_DIRECTIVES[@]}" -gt 0 ]]; then
    SLURM_EXTRA_ENV="${SLURM_EXTRA_ENV} ${SLURM_EXTRA_DIRECTIVES[*]}"
  fi

  set +e
  bash "${SCRIPT_DIR}/scripts/run_mopd.sh" \
    "${CONFIG_REFERENCE}" \
    --slurm \
    -- "${LAUNCHER_ARGS[@]}"
  rc=$?
  set -e
  echo "=== start.sh: run_mopd.sh exited with rc=${rc} ==="
  exit ${rc}
fi

# ── Local mode ─────────────────────────────────────────────────────────────
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
echo "=== start.sh: exec run_local_mopd_training.sh ==="

set +e
bash "${SCRIPT_DIR}/scripts/run_local_mopd_training.sh" \
  "${CONFIG_REFERENCE}" \
  "${LAUNCHER_ARGS[@]}" \
  -- \
  "++data.seed=${MOPD_SEED}" \
  "++actor_rollout_ref.rollout.seed=${MOPD_SEED}" \
  "++trainer.seed=${MOPD_SEED}"
rc=$?
set -e
echo "=== start.sh: run_local_mopd_training.sh exited with rc=${rc} ==="
exit ${rc}
