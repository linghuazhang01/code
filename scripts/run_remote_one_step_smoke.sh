#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_remote_one_step_smoke.sh

Run this on the remote host from the synced OPD-code checkout after
scripts/setup_remote_training_env.sh.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/logs}"
ENV_FILE="${ENV_FILE:-${LOG_DIR}/env.sh}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
ENV_NAME="${ENV_NAME:-mopd-verl}"
VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-${CODE_DIR}/third_party/verl}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-${CODE_DIR}/smoke_data}"
SMOKE_CONFIG="${SMOKE_CONFIG:-${CODE_DIR}/configs/mopd_audit_smoke.yaml}"
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen3-0.6B}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/one_step_smoke_${TIMESTAMP}.log}"
LOCK_FILE="${LOG_DIR}/one_step_smoke.lock"

if [[ ! -f "${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" ]]; then
  echo "Vendored verl runtime not found: ${VERL_RUNTIME_DIR}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${SMOKE_DATA_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another one-step smoke run is already active: ${LOCK_FILE}" >&2
  exit 75
fi

# shellcheck disable=SC1090
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cleanup_ray() {
  ray stop --force >/tmp/ray_stop_after_one_step_smoke.log 2>&1 || true
}
trap cleanup_ray EXIT
ray stop --force >/tmp/ray_stop_before_one_step_smoke.log 2>&1 || true

export PYTHONPATH="${CODE_DIR}:${VERL_RUNTIME_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${CODE_DIR}/hf_home}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
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
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

cd "${CODE_DIR}"

COMMAND=(
  "${CODE_DIR}/scripts/run_math_code_mopd.sh"
  --
  "data.train_files=['smoke_data/train.parquet']"
  "data.val_files=['smoke_data/val.parquet']"
  "actor_rollout_ref.model.path=${SMOKE_MODEL}"
  "actor_rollout_ref.model.base_model_path=${SMOKE_MODEL}"
  "actor_rollout_ref.ref.model.path=${SMOKE_MODEL}"
  "actor_rollout_ref.ref.model.base_model_path=${SMOKE_MODEL}"
  "trainer.default_local_dir=checkpoints/smoke"
)

echo "log_file=${LOG_FILE}"
echo "smoke_model=${SMOKE_MODEL}"
echo "smoke_config=${SMOKE_CONFIG}"
echo "code_dir=${CODE_DIR}"
echo "verl_runtime_dir=${VERL_RUNTIME_DIR}"

MOPD_CONFIG="${SMOKE_CONFIG}" PYTHON_BIN=python "${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"

echo "one_step_smoke_done" | tee -a "${LOG_FILE}"
echo "log_file=${LOG_FILE}" | tee -a "${LOG_FILE}"
