#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_topk32.yaml"
MOPD_SEED="${MOPD_SEED:-42}"

# These variables must be set before Python, Ray, Torch, or vLLM starts.
export MOPD_GLOBAL_SEED="${MOPD_SEED}"
export PYTHONHASHSEED="${MOPD_SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export FLASH_ATTENTION_DETERMINISTIC="${FLASH_ATTENTION_DETERMINISTIC:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
exec bash "${SCRIPT_DIR}/scripts/run_local_mopd_training.sh" \
  "${CONFIG_PATH}" \
  "$@" \
  -- \
  "++data.seed=${MOPD_SEED}" \
  "++actor_rollout_ref.rollout.seed=${MOPD_SEED}" \
  "++trainer.seed=${MOPD_SEED}"
