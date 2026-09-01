#!/usr/bin/env bash
set -euo pipefail

export LANG=C
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
CODE_DIR="$(cd "${CODE_DIR}" && pwd -P)"
REMOTE_PYTHON_DEFAULT="/home/shuang_qiu/env/miniconda3/envs/mopd-verl/bin/python"

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

validate_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer: ${value}"
}

MODEL_PATH=""
EVAL_MODEL=""
G_OPD_DIR=""
DATASETS=""
OUTPUT_ROOT=""
RUN_TAG=""
GPU_COUNT=""
SHARDS_PER_DATASET=""
MIN_ROWS_PER_SHARD=""
MAX_SAMPLES=""
RESUME=0
INCLUDE_MMLUPRO_500=0
SCORE_CODE=1
STANDARD_PROTOCOL=0
REFERENCE_ANCHOR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="${2:?--model_path requires a value}"; shift 2 ;;
    --eval_model_path) EVAL_MODEL="${2:?--eval_model_path requires a value}"; shift 2 ;;
    --gopd_dir) G_OPD_DIR="${2:?--gopd_dir requires a value}"; shift 2 ;;
    --datasets) DATASETS="${2:?--datasets requires a value}"; shift 2 ;;
    --output_root) OUTPUT_ROOT="${2:?--output_root requires a value}"; shift 2 ;;
    --run_tag) RUN_TAG="${2:?--run_tag requires a value}"; shift 2 ;;
    --gpus) GPU_COUNT="${2:?--gpus requires a value}"; shift 2 ;;
    --shards_per_dataset) SHARDS_PER_DATASET="${2:?--shards_per_dataset requires a value}"; shift 2 ;;
    --min_rows_per_shard) MIN_ROWS_PER_SHARD="${2:?--min_rows_per_shard requires a value}"; shift 2 ;;
    --max_samples) MAX_SAMPLES="${2:?--max_samples requires a value}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --include_mmlupro_500) INCLUDE_MMLUPRO_500=1; shift ;;
    --no_score_code) SCORE_CODE=0; shift ;;
    --standard_protocol) STANDARD_PROTOCOL=1; shift ;;
    --reference_anchor|--reference-anchor) REFERENCE_ANCHOR=1; shift ;;
    *) fail "unknown parallel worker argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "parallel worker must run inside a Slurm allocation"
[[ -n "${MODEL_PATH}" && -n "${DATASETS}" && -n "${OUTPUT_ROOT}" && -n "${RUN_TAG}" ]] \
  || fail "model_path, datasets, output_root, and run_tag are required"
[[ -z "${G_OPD_DIR}" || -d "${G_OPD_DIR}" ]] || fail "G-OPD checkout is missing: ${G_OPD_DIR}"
validate_positive_integer "--gpus" "${GPU_COUNT}"
validate_positive_integer "--shards_per_dataset" "${SHARDS_PER_DATASET}"
validate_positive_integer "--min_rows_per_shard" "${MIN_ROWS_PER_SHARD}"
[[ -z "${MAX_SAMPLES}" ]] || validate_positive_integer "--max_samples" "${MAX_SAMPLES}"

PYTHON_BIN="${SLURM_EVAL_PYTHON:-${REMOTE_PYTHON_DEFAULT}}"
[[ -x "${PYTHON_BIN}" ]] || fail "Python executable is not runnable: ${PYTHON_BIN}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
unset ROCR_VISIBLE_DEVICES

VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a VISIBLE_DEVICE_ARRAY <<<"${VISIBLE_DEVICES}"
[[ -n "${VISIBLE_DEVICES}" && "${#VISIBLE_DEVICE_ARRAY[@]}" == "${GPU_COUNT}" ]] \
  || fail "expected ${GPU_COUNT} Slurm GPUs, got: ${VISIBLE_DEVICES:-none}"
declare -A SEEN_GPUS=()
for device in "${VISIBLE_DEVICE_ARRAY[@]}"; do
  [[ -n "${device}" && -z "${SEEN_GPUS[${device}]:-}" ]] \
    || fail "expected ${GPU_COUNT} unique Slurm GPU IDs"
  SEEN_GPUS["${device}"]=1
done

TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_EVAL_CPUS:-64}}"
validate_positive_integer "SLURM_CPUS_PER_TASK" "${TOTAL_CPUS}"
CPUS_PER_WORKER=$((TOTAL_CPUS / GPU_COUNT))
[[ "${CPUS_PER_WORKER}" -gt 0 ]] || CPUS_PER_WORKER=1

MATH_SAMPLES="${SLURM_EVAL_MATH_SAMPLES:-8}"
CODE_SAMPLES="${SLURM_EVAL_CODE_SAMPLES:-8}"
SCIENCE_SAMPLES="${SLURM_EVAL_SCIENCE_SAMPLES:-8}"
BASE_SEED="${SLURM_EVAL_SEED:-42}"
MAX_NEW_TOKENS="${SLURM_EVAL_MAX_TOKENS:-16384}"
TEMPERATURE="${SLURM_EVAL_TEMPERATURE:-1.0}"
TOP_P="${SLURM_EVAL_TOP_P:-1.0}"
BATCH_SIZE="${SLURM_EVAL_BATCH_SIZE:-24}"
GPU_MEMORY="${SLURM_EVAL_GPU_MEMORY:-0.85}"
MAX_MODEL_LEN="${SLURM_EVAL_MAX_MODEL_LEN:-18432}"
MAX_NUM_BATCHED_TOKENS="${SLURM_EVAL_MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${SLURM_EVAL_MAX_NUM_SEQS:-24}"
for value_name in MATH_SAMPLES CODE_SAMPLES SCIENCE_SAMPLES; do
  validate_positive_integer "${value_name}" "${!value_name}"
done
if [[ "${STANDARD_PROTOCOL}" == "1" ]]; then
  CANONICAL_DATASETS="aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus,lcb_v5,lcb_v6,gpqa_diamond,mmlupro_500_seed42"
  [[ "${GPU_COUNT}" == "2" || "${GPU_COUNT}" == "3" || "${GPU_COUNT}" == "4" ]] \
    || fail "standard protocol requires DP=2, DP=3, or DP=4"
  [[ "${MATH_SAMPLES}" == "8" && "${CODE_SAMPLES}" == "8" && "${SCIENCE_SAMPLES}" == "8" ]] \
    || fail "standard protocol requires K=8 for every domain"
  [[ "${BASE_SEED}" == "42" ]] || fail "standard protocol requires seed 42"
  [[ "${DATASETS}" == "${CANONICAL_DATASETS}" ]] \
    || fail "standard protocol requires the canonical ordered ten-dataset batch"
  [[ "${SHARDS_PER_DATASET}" -ge "${GPU_COUNT}" && "${MIN_ROWS_PER_SHARD}" == "1" ]] \
    || fail "standard protocol requires at least one dynamic micro-shard per GPU"
  [[ "${TEMPERATURE}" == "1.0" && "${TOP_P}" == "1.0" && "${MAX_NEW_TOKENS}" == "16384" ]] \
    || fail "standard protocol sampling configuration differs"
  [[ "${SCORE_CODE}" == "1" && -n "${G_OPD_DIR}" ]] \
    || fail "standard protocol requires Code scoring and a G-OPD checkout"
  if [[ "${REFERENCE_ANCHOR}" == "0" ]]; then
    [[ "$(basename "${MODEL_PATH}")" == "global_step_60" ]] \
      || fail "standard protocol requires global_step_60 or --reference-anchor"
  fi
fi
for value_name in MAX_NEW_TOKENS BATCH_SIZE MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS; do
  validate_positive_integer "${value_name}" "${!value_name}"
done
[[ "${BASE_SEED}" =~ ^[0-9]+$ ]] || fail "SLURM_EVAL_SEED must be non-negative: ${BASE_SEED}"

SUITE_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
MANIFEST_PATH="${SUITE_ROOT}/suite_manifest.json"
mkdir -p "${OUTPUT_ROOT}"
command -v flock >/dev/null 2>&1 || fail "flock is required for suite-level locking"
exec 8>"${SUITE_ROOT}.lock"
flock -n 8 || fail "another job is already evaluating suite: ${SUITE_ROOT}"

CODE_SANDBOX_IMAGE="${MOPD_CODE_SANDBOX_IMAGE:-verlai/verl:vllm023.dev1}"
CODE_SANDBOX_IMAGE_ID="disabled"
if [[ "${SCORE_CODE}" == "1" \
  && ( "${DATASETS}" == *"humaneval_plus"* || "${DATASETS}" == *"mbpp_plus"* ) ]]; then
  command -v docker >/dev/null 2>&1 || fail "Docker is required for isolated Code scoring"
  docker image inspect "${CODE_SANDBOX_IMAGE}" >/dev/null
  CODE_SANDBOX_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${CODE_SANDBOX_IMAGE}")"
  [[ -n "${CODE_SANDBOX_IMAGE_ID}" ]] || fail "Could not resolve Docker image ID: ${CODE_SANDBOX_IMAGE}"
  export MOPD_CODE_SANDBOX=docker
  export MOPD_CODE_SANDBOX_IMAGE="${CODE_SANDBOX_IMAGE}"
fi

if [[ -z "${EVAL_MODEL}" ]]; then
  EVAL_MODEL="$(CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICE_ARRAY[0]}" \
    bash "${CODE_DIR}/scripts/prepare_eval_model.sh" \
      --model-path "${MODEL_PATH}" \
      --python "${PYTHON_BIN}")"
fi
[[ -d "${EVAL_MODEL}" ]] || fail "prepared eval model is missing: ${EVAL_MODEL}"

PLAN_COMMAND=(
  "${PYTHON_BIN}" -m eval.parallel_eval plan
  --code-dir "${CODE_DIR}"
  --suite-root "${SUITE_ROOT}"
  --run-tag "${RUN_TAG}"
  --model-path "${MODEL_PATH}"
  --eval-model-path "${EVAL_MODEL}"
  --datasets "${DATASETS}"
  --shards-per-dataset "${SHARDS_PER_DATASET}"
  --worker-count "${GPU_COUNT}"
  --min-rows-per-shard "${MIN_ROWS_PER_SHARD}"
  --math-samples "${MATH_SAMPLES}"
  --code-samples "${CODE_SAMPLES}"
  --science-samples "${SCIENCE_SAMPLES}"
  --base-seed "${BASE_SEED}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --batch-size "${BATCH_SIZE}"
  --gpu-memory "${GPU_MEMORY}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --code-sandbox-image "${CODE_SANDBOX_IMAGE}"
  --code-sandbox-image-id "${CODE_SANDBOX_IMAGE_ID}"
)
[[ -z "${G_OPD_DIR}" ]] || PLAN_COMMAND+=(--gopd-dir "${G_OPD_DIR}")
[[ -z "${MAX_SAMPLES}" ]] || PLAN_COMMAND+=(--max-samples-per-dataset "${MAX_SAMPLES}")
[[ "${INCLUDE_MMLUPRO_500}" == "0" ]] || PLAN_COMMAND+=(--include-mmlupro-500)
[[ "${RESUME}" == "0" ]] || PLAN_COMMAND+=(--resume)
[[ "${SCORE_CODE}" == "1" ]] || PLAN_COMMAND+=(--no-score-code)

printf '[parallel-eval] job_id=%s host=%s gpus=%s cpus_per_worker=%s shards_per_dataset=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "${GPU_COUNT}" "${CPUS_PER_WORKER}" "${SHARDS_PER_DATASET}"
EXPECTED_GPU_COUNT="${GPU_COUNT}" "${PYTHON_BIN}" - <<'PY'
import os
import torch

assert torch.cuda.is_available()
expected = int(os.environ["EXPECTED_GPU_COUNT"])
assert torch.cuda.device_count() == expected, (torch.cuda.device_count(), expected)
print(
    "[parallel-eval] CUDA_WITNESS",
    torch.cuda.device_count(),
    [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
)
PY
"${PLAN_COMMAND[@]}"
mkdir -p "${SUITE_ROOT}/logs"
export MOPD_ALLOW_SIMPLE_SCORER_FALLBACK="${MOPD_ALLOW_SIMPLE_SCORER_FALLBACK:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

pids=()
for worker_id in "${!VISIBLE_DEVICE_ARRAY[@]}"; do
  WORKER_ARGS=(
    --manifest "${MANIFEST_PATH}"
    --eval-model-path "${EVAL_MODEL}"
    --worker-id "${worker_id}"
  )
  [[ "${RESUME}" == "0" ]] || WORKER_ARGS+=(--resume)
  CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICE_ARRAY[${worker_id}]}" \
  OMP_NUM_THREADS="${CPUS_PER_WORKER}" \
  MKL_NUM_THREADS="${CPUS_PER_WORKER}" \
  OPENBLAS_NUM_THREADS="${CPUS_PER_WORKER}" \
  NUMEXPR_NUM_THREADS="${CPUS_PER_WORKER}" \
    "${PYTHON_BIN}" -m eval.parallel_worker "${WORKER_ARGS[@]}" \
      >"${SUITE_ROOT}/logs/gpu_worker_${worker_id}.log" 2>&1 &
  pids+=("$!")
done

remaining="${#pids[@]}"
status=0
while (( remaining > 0 )); do
  if wait -n; then
    remaining=$((remaining - 1))
  else
    status=1
    break
  fi
done
if [[ "${status}" != "0" ]]; then
  kill "${pids[@]}" 2>/dev/null || true
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  fail "one or more parallel evaluation workers failed"
fi

"${PYTHON_BIN}" -m eval.parallel_eval merge --manifest "${MANIFEST_PATH}"
printf '[parallel-eval] complete output=%s\n' "${SUITE_ROOT}"
