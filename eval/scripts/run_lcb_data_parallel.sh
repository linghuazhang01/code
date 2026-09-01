#!/usr/bin/env bash
set -euo pipefail

export LANG=C
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
REMOTE_ROOT="$(cd "${CODE_DIR}/.." && pwd -P)"

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

MODEL_PATH=""
CHECKPOINT_PATH=""
OUTPUT_ROOT=""
G_OPD_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python}"
RELEASES="v5,v6"
GPU_COUNT=4
SHARDS_PER_DATASET=16
RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="${2:?--model_path requires a value}"; shift 2 ;;
    --checkpoint_path) CHECKPOINT_PATH="${2:?--checkpoint_path requires a value}"; shift 2 ;;
    --output_root) OUTPUT_ROOT="${2:?--output_root requires a value}"; shift 2 ;;
    --gopd_dir) G_OPD_DIR="${2:?--gopd_dir requires a value}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?--python requires a value}"; shift 2 ;;
    --releases) RELEASES="${2:?--releases requires a value}"; shift 2 ;;
    --gpus) GPU_COUNT="${2:?--gpus requires a value}"; shift 2 ;;
    --shards_per_dataset) SHARDS_PER_DATASET="${2:?--shards_per_dataset requires a value}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    *) fail "unknown LCB data-parallel argument: $1" ;;
  esac
done

[[ -d "${MODEL_PATH}" ]] || fail "model path is missing: ${MODEL_PATH}"
[[ -d "${G_OPD_DIR}" ]] || fail "G-OPD checkout is missing: ${G_OPD_DIR}"
[[ -n "${OUTPUT_ROOT}" ]] || fail "--output_root is required"
[[ -x "$(command -v "${PYTHON_BIN}" 2>/dev/null || true)" || -x "${PYTHON_BIN}" ]] \
  || fail "Python is not runnable: ${PYTHON_BIN}"
[[ "${GPU_COUNT}" == "4" ]] || fail "LCB standard data parallelism requires exactly four GPUs"
[[ "${SHARDS_PER_DATASET}" =~ ^[1-9][0-9]*$ ]] || fail "invalid shard count"
export HF_HOME="${HF_HOME:-${REMOTE_ROOT}/hf_home}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

IFS=',' read -r -a VISIBLE_DEVICE_ARRAY <<<"${CUDA_VISIBLE_DEVICES:-}"
[[ "${#VISIBLE_DEVICE_ARRAY[@]}" == "${GPU_COUNT}" ]] \
  || fail "expected four visible GPUs, got: ${CUDA_VISIBLE_DEVICES:-none}"
declare -A SEEN_GPUS=()
for device in "${VISIBLE_DEVICE_ARRAY[@]}"; do
  [[ -n "${device}" && -z "${SEEN_GPUS[${device}]:-}" ]] \
    || fail "four unique visible GPU IDs are required"
  SEEN_GPUS["${device}"]=1
done

DATASETS=""
IFS=',' read -r -a RELEASE_ARRAY <<<"${RELEASES}"
for release in "${RELEASE_ARRAY[@]}"; do
  release="${release//[[:space:]]/}"
  [[ "${release}" == "v5" || "${release}" == "v6" ]] \
    || fail "unsupported LiveCodeBench release: ${release}"
  [[ -z "${DATASETS}" ]] || DATASETS+=","
  DATASETS+="lcb_${release}"
done

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${MODEL_PATH}}"
SUITE_ROOT="${OUTPUT_ROOT}/lcb_parallel"
MANIFEST_PATH="${SUITE_ROOT}/suite_manifest.json"
PLAN_ARGS=(
  --code-dir "${CODE_DIR}"
  --suite-root "${SUITE_ROOT}"
  --run-tag lcb_parallel
  --model-path "${CHECKPOINT_PATH}"
  --eval-model-path "${MODEL_PATH}"
  --gopd-dir "${G_OPD_DIR}"
  --datasets "${DATASETS}"
  --worker-count 4
  --shards-per-dataset "${SHARDS_PER_DATASET}"
  --min-rows-per-shard 1
  --math-samples 8
  --code-samples 8
  --science-samples 8
  --base-seed 42
  --max-new-tokens 16384
  --temperature 1.0
  --top-p 1.0
  --batch-size 24
  --gpu-memory 0.85
  --max-model-len 18432
  --max-num-batched-tokens 32768
  --max-num-seqs 24
  --no-score-code
)
[[ "${RESUME}" == "0" ]] || PLAN_ARGS+=(--resume)

mkdir -p "${OUTPUT_ROOT}"
"${PYTHON_BIN}" -m eval.parallel_eval plan "${PLAN_ARGS[@]}"
mkdir -p "${SUITE_ROOT}/logs"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

pids=()
for worker_id in "${!VISIBLE_DEVICE_ARRAY[@]}"; do
  WORKER_ARGS=(
    --manifest "${MANIFEST_PATH}"
    --eval-model-path "${MODEL_PATH}"
    --worker-id "${worker_id}"
  )
  [[ "${RESUME}" == "0" ]] || WORKER_ARGS+=(--resume)
  CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICE_ARRAY[${worker_id}]}" \
    "${PYTHON_BIN}" -m eval.parallel_worker "${WORKER_ARGS[@]}" \
      >"${SUITE_ROOT}/logs/gpu_worker_${worker_id}.log" 2>&1 &
  pids+=("$!")
done

remaining="${#pids[@]}"
while (( remaining > 0 )); do
  if wait -n; then
    remaining=$((remaining - 1))
  else
    kill "${pids[@]}" 2>/dev/null || true
    for pid in "${pids[@]}"; do
      wait "${pid}" 2>/dev/null || true
    done
    fail "one or more LCB data-parallel workers failed"
  fi
done

"${PYTHON_BIN}" -m eval.parallel_eval merge --manifest "${MANIFEST_PATH}"
printf '[lcb-dp] complete output=%s\n' "${SUITE_ROOT}"
