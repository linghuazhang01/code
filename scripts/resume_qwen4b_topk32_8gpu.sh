#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_topk32.yaml"
CHECKPOINT_DIR="${CODE_DIR}/checkpoints/MOPD/qwen4b-from-30b-a3b-instruct-2507-math-code-science-topk32"
CHECKPOINT_TRACKER="${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
MOPD_RUN_ID="${MOPD_RUN_ID:-qwen4b_topk32_resume_$(date +%Y%m%d_%H%M%S)}"
MOPD_SEED="${MOPD_SEED:-42}"

export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export MOPD_GLOBAL_SEED="${MOPD_SEED}"
export PYTHONHASHSEED="${MOPD_SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export FLASH_ATTENTION_DETERMINISTIC="${FLASH_ATTENTION_DETERMINISTIC:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

for argument in "$@"; do
  case "${argument}" in
    --foreground | --tail | --dry-run)
      ;;
    *)
      echo "Unsupported argument: ${argument}" >&2
      echo "Allowed arguments: --foreground, --tail, --dry-run" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${CHECKPOINT_TRACKER}" ]]; then
  echo "Checkpoint tracker not found: ${CHECKPOINT_TRACKER}" >&2
  echo "Refusing to resume W&B run 14r9am6q from scratch." >&2
  exit 2
fi

latest_step="$(<"${CHECKPOINT_TRACKER}")"
if [[ ! "${latest_step}" =~ ^[0-9]+$ ]]; then
  echo "Invalid checkpoint step in ${CHECKPOINT_TRACKER}: ${latest_step}" >&2
  exit 2
fi

latest_checkpoint="${CHECKPOINT_DIR}/global_step_${latest_step}"
if [[ ! -d "${latest_checkpoint}" ]]; then
  echo "Tracked checkpoint not found: ${latest_checkpoint}" >&2
  echo "Refusing to resume W&B run 14r9am6q from scratch." >&2
  exit 2
fi

if [[ ! -d "${latest_checkpoint}/actor" ]]; then
  echo "Actor checkpoint not found: ${latest_checkpoint}/actor" >&2
  exit 2
fi

if [[ ! -f "${latest_checkpoint}/data.pt" ]]; then
  echo "Dataloader checkpoint not found: ${latest_checkpoint}/data.pt" >&2
  exit 2
fi

echo "Resuming checkpoint: ${latest_checkpoint}"
echo "Resuming W&B run: lz101-rice-university/MOPD/14r9am6q"
echo "Visible GPUs: ${GPU_IDS}"

cd "${CODE_DIR}"
exec bash "${CODE_DIR}/scripts/run_local_mopd_training.sh" \
  "${CONFIG_PATH}" \
  --run-id "${MOPD_RUN_ID}" \
  "$@" \
  -- \
  "++data.seed=${MOPD_SEED}" \
  "++actor_rollout_ref.rollout.seed=${MOPD_SEED}" \
  "++trainer.seed=${MOPD_SEED}" \
  "trainer.resume_from_path=${latest_checkpoint}"
