#!/usr/bin/env bash
# start.sh — Launch MOPD training locally or via Slurm.
#
# Local mode is the safe default. Use --slurm to submit, or set
# MOPD_LAUNCH_MODE=auto to select Slurm when sbatch is available.
#
# Environment variables:
#   GPU_IDS       Comma-separated physical GPU indices (local mode only).
#   MOPD_SEED     Random seed (default: 42).
#   SLURM_MEM     Memory per node for Slurm jobs (default: 700G).
#   SLURM_TIME    Wall-time limit for Slurm jobs (default: 72:00:00).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Evaluation is Slurm-only and has its own resource/protocol parser. Keep this
# dispatch before the training parser so existing training semantics stay intact.
if [[ "${1:-}" == "--eval" ]]; then
  shift
  [[ "${1:-}" != "--slurm" ]] || shift
  exec bash "${SCRIPT_DIR}/slurm_parallel_eval.sh" "$@"
fi
if [[ "${1:-}" == "--slurm" && "${2:-}" == "--eval" ]]; then
  shift 2
  exec bash "${SCRIPT_DIR}/slurm_parallel_eval.sh" "$@"
fi

DEFAULT_CONFIG_PATH="${SCRIPT_DIR}/configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_topk32.yaml"
MOPD_SEED="${MOPD_SEED:-42}"

usage() {
  cat <<'USAGE'
Usage:
  bash start.sh [--config|-c <config[::profile]>] [--local|--slurm] [--dry-run] [-- <hydra overrides>]
  bash start.sh --slurm --eval --model_path PATH [evaluation options]

Examples:
  bash start.sh
  bash start.sh --config configs/mopd_formal_audit_all_8gpu.yaml
  bash start.sh -c configs/mopd_formal_audit_all_2gpu.yaml --local --foreground
  bash start.sh -c configs/mopd_formal_audit_all_8gpu.yaml --slurm
  bash start.sh -c configs/mopd_formal_audit_all_8gpu.yaml --slurm --slurm-args "--partition=gpu"
  bash start.sh --slurm --eval --model_path checkpoints/model --gpus 4 --dry_run

The default mode is local. MOPD_LAUNCH_MODE=auto selects Slurm when sbatch is
available. If no config is selected, start.sh uses the original Top-32 config.
USAGE
}

CONFIG_REFERENCE="${MOPD_CONFIG:-${DEFAULT_CONFIG_PATH}}"
LAUNCH_MODE="${MOPD_LAUNCH_MODE:-local}"
SLURM_EXTRA_DIRECTIVES=()
LOCAL_ONLY_ARGS=()
HYDRA_OVERRIDES=()
DRY_RUN_FLAG=0
CONFIG_ARG_SEEN=0
CLI_LAUNCH_MODE=""

if [[ "${LAUNCH_MODE}" != "auto" && "${LAUNCH_MODE}" != "local" && "${LAUNCH_MODE}" != "slurm" ]]; then
  echo "MOPD_LAUNCH_MODE must be one of: auto, local, slurm." >&2
  exit 2
fi

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
      if [[ "${CONFIG_ARG_SEEN}" == "1" ]]; then
        echo "Only one config path is allowed." >&2
        exit 2
      fi
      CONFIG_REFERENCE="$2"
      CONFIG_ARG_SEEN=1
      shift 2
      ;;
    --config=*)
      if [[ "${CONFIG_ARG_SEEN}" == "1" ]]; then
        echo "Only one config path is allowed." >&2
        exit 2
      fi
      CONFIG_REFERENCE="${1#*=}"
      CONFIG_ARG_SEEN=1
      shift
      ;;
    --slurm)
      if [[ -n "${CLI_LAUNCH_MODE}" && "${CLI_LAUNCH_MODE}" != "slurm" ]]; then
        echo "--local and --slurm cannot be used together." >&2
        exit 2
      fi
      CLI_LAUNCH_MODE="slurm"
      LAUNCH_MODE="slurm"
      shift
      ;;
    --local)
      if [[ -n "${CLI_LAUNCH_MODE}" && "${CLI_LAUNCH_MODE}" != "local" ]]; then
        echo "--local and --slurm cannot be used together." >&2
        exit 2
      fi
      CLI_LAUNCH_MODE="local"
      LAUNCH_MODE="local"
      shift
      ;;
    --slurm-args)
      if [[ $# -lt 2 ]]; then
        echo "--slurm-args requires a value." >&2
        exit 2
      fi
      SLURM_EXTRA_DIRECTIVES+=("$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN_FLAG=1
      shift
      ;;
    --foreground|--tail)
      LOCAL_ONLY_ARGS+=("$1")
      shift
      ;;
    --run-id)
      if [[ $# -lt 2 ]]; then
        echo "--run-id requires a value." >&2
        exit 2
      fi
      LOCAL_ONLY_ARGS+=("$1" "$2")
      shift 2
      ;;
    --)
      shift
      HYDRA_OVERRIDES=("$@")
      break
      ;;
    *.yaml|*.yml|*::*)
      if [[ "${CONFIG_ARG_SEEN}" == "1" ]]; then
        echo "Only one config path is allowed." >&2
        exit 2
      fi
      CONFIG_REFERENCE="$1"
      CONFIG_ARG_SEEN=1
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      echo "Put Hydra overrides after '--'." >&2
      exit 2
      ;;
    *)
      echo "Unexpected argument: $1" >&2
      echo "Only one config is allowed; put Hydra overrides after '--'." >&2
      exit 2
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

if [[ "${LAUNCH_MODE}" == "auto" ]]; then
  if command -v sbatch >/dev/null 2>&1 && [[ -z "${SLURM_JOB_ID:-}" ]]; then
    LAUNCH_MODE="slurm"
  else
    LAUNCH_MODE="local"
  fi
fi

if [[ "${LAUNCH_MODE}" == "slurm" && "${#LOCAL_ONLY_ARGS[@]}" -gt 0 ]]; then
  echo "${LOCAL_ONLY_ARGS[0]} is local-only; add --local to use it." >&2
  exit 2
fi
if [[ "${LAUNCH_MODE}" == "local" && "${#SLURM_EXTRA_DIRECTIVES[@]}" -gt 0 ]]; then
  echo "--slurm-args requires Slurm mode; add --slurm to use it." >&2
  exit 2
fi

# These variables must be set before Python, Ray, Torch, or vLLM starts.
export MOPD_GLOBAL_SEED="${MOPD_SEED}"
export PYTHONHASHSEED="${MOPD_SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export FLASH_ATTENTION_DETERMINISTIC="${FLASH_ATTENTION_DETERMINISTIC:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
SEED_OVERRIDES=(
  "++data.seed=${MOPD_SEED}"
  "++actor_rollout_ref.rollout.seed=${MOPD_SEED}"
  "++trainer.seed=${MOPD_SEED}"
)

echo "LAUNCH_MODE=${LAUNCH_MODE}"
echo "CONFIG_REFERENCE=${CONFIG_REFERENCE}"

if [[ "${LAUNCH_MODE}" == "slurm" ]]; then
  SLURM_MEM="${SLURM_MEM:-700G}"
  SLURM_TIME="${SLURM_TIME:-72:00:00}"
  RUN_MOPD_ARGS=(
    "${CONFIG_REFERENCE}"
    --slurm-args "--mem=${SLURM_MEM}"
    --slurm-args "--time=${SLURM_TIME}"
  )
  if [[ "${#SLURM_EXTRA_DIRECTIVES[@]}" -gt 0 ]]; then
    for slurm_directive in "${SLURM_EXTRA_DIRECTIVES[@]}"; do
      RUN_MOPD_ARGS+=(--slurm-args "${slurm_directive}")
    done
  fi
  RUN_MOPD_ARGS+=(--slurm)
  if [[ "${DRY_RUN_FLAG}" == "1" ]]; then
    echo "Generating a Slurm script without submitting a job."
    RUN_MOPD_ARGS+=(--dry-run)
  fi
  RUN_MOPD_ARGS+=(--)
  if [[ "${#HYDRA_OVERRIDES[@]}" -gt 0 ]]; then
    RUN_MOPD_ARGS+=("${HYDRA_OVERRIDES[@]}")
  fi
  RUN_MOPD_ARGS+=("${SEED_OVERRIDES[@]}")
  exec bash "${SCRIPT_DIR}/scripts/run_mopd.sh" "${RUN_MOPD_ARGS[@]}"
fi

export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

LOCAL_LAUNCH_ARGS=("${CONFIG_REFERENCE}")
if [[ "${#LOCAL_ONLY_ARGS[@]}" -gt 0 ]]; then
  LOCAL_LAUNCH_ARGS+=("${LOCAL_ONLY_ARGS[@]}")
fi
if [[ "${DRY_RUN_FLAG}" == "1" ]]; then
  LOCAL_LAUNCH_ARGS+=(--dry-run)
fi
LOCAL_LAUNCH_ARGS+=(--)
if [[ "${#HYDRA_OVERRIDES[@]}" -gt 0 ]]; then
  LOCAL_LAUNCH_ARGS+=("${HYDRA_OVERRIDES[@]}")
fi
LOCAL_LAUNCH_ARGS+=(
  "${SEED_OVERRIDES[@]}"
)

exec bash "${SCRIPT_DIR}/scripts/run_local_mopd_training.sh" \
  "${LOCAL_LAUNCH_ARGS[@]}"
