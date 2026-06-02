#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/opd_mopd}"
ENV_FILE="${REMOTE_ROOT}/env.sh"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

GOPD_DIR="${GOPD_DIR:-${G_OPD_DIR:-${REMOTE_ROOT}/G-OPD}}"
OPD_CODE_DIR="${OPD_CODE_DIR:-${CODE_DIR}}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
ENV_NAME="${ENV_NAME:-mopd-verl}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-${GOPD_DIR}/smoke_data}"
SMOKE_CONFIG="${SMOKE_CONFIG:-${OPD_CODE_DIR}/configs/mopd_audit_smoke.yaml}"
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen3-0.6B}"
LOG_DIR="${REMOTE_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/one_step_smoke_${TIMESTAMP}.log}"
LOCK_FILE="${REMOTE_ROOT}/one_step_smoke.lock"

mkdir -p "${LOG_DIR}" "${SMOKE_DATA_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another one-step smoke run is already active: ${LOCK_FILE}" >&2
  exit 75
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cleanup_ray() {
  ray stop --force >/tmp/ray_stop_after_one_step_smoke.log 2>&1 || true
}
trap cleanup_ray EXIT
ray stop --force >/tmp/ray_stop_before_one_step_smoke.log 2>&1 || true

export PYTHONPATH="${OPD_CODE_DIR}:${GOPD_DIR}/verl:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REMOTE_ROOT}/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export USED_MODEL="${USED_MODEL:-no_api}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export RAY_DISABLE_DOCKER_CPU_WARNING="${RAY_DISABLE_DOCKER_CPU_WARNING:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-120}"

python -m mopd_verl.smoke_data "${SMOKE_DATA_DIR}"
python -m mopd_verl.prepare_data inspect "${SMOKE_DATA_DIR}/train.parquet"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

cd "${GOPD_DIR}/verl"

COMMAND=(
  "${OPD_CODE_DIR}/scripts/run_math_code_mopd.sh"
  --
  "data.train_files=['${SMOKE_DATA_DIR}/train.parquet']"
  "data.val_files=['${SMOKE_DATA_DIR}/val.parquet']"
  "actor_rollout_ref.model.path=${SMOKE_MODEL}"
  "actor_rollout_ref.model.base_model_path=${SMOKE_MODEL}"
  "actor_rollout_ref.ref.model.path=${SMOKE_MODEL}"
  "actor_rollout_ref.ref.model.base_model_path=${SMOKE_MODEL}"
  "trainer.default_local_dir=${REMOTE_ROOT}/checkpoints/smoke"
)

echo "log_file=${LOG_FILE}"
echo "smoke_model=${SMOKE_MODEL}"
echo "smoke_config=${SMOKE_CONFIG}"

MOPD_CONFIG="${SMOKE_CONFIG}" PYTHON_BIN=python "${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"

echo "one_step_smoke_done" | tee -a "${LOG_FILE}"
echo "log_file=${LOG_FILE}" | tee -a "${LOG_FILE}"
