#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${CODE_DIR}/../models/Qwen3-1.7B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${CODE_DIR}/../models/Nemotron-Research-GooseReason-4B-Instruct}"
GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CODE_DIR}/data/eval_data/results/two_model_full_training/${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SHARD_SIZE="${SHARD_SIZE:-10000}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"
MAX_SAMPLES_PER_DOMAIN="${MAX_SAMPLES_PER_DOMAIN:-}"
# Match the GooseReason training config's validation-inference settings.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-24}"
GPU_MEMORY="${GPU_MEMORY:-0.6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-18432}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
CONFIRM_FULL_TRAINING="${CONFIRM_FULL_TRAINING:-0}"

for flag in RESUME DRY_RUN CONFIRM_FULL_TRAINING; do
  [[ "${!flag}" == "0" || "${!flag}" == "1" ]] || {
    echo "${flag} must be 0 or 1: ${!flag}" >&2
    exit 2
  }
done
[[ "${SHARD_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "SHARD_SIZE must be a positive integer: ${SHARD_SIZE}" >&2
  exit 2
}
[[ "${MIN_FREE_GB}" =~ ^[0-9]+$ ]] || {
  echo "MIN_FREE_GB must be a non-negative integer: ${MIN_FREE_GB}" >&2
  exit 2
}
if [[ -n "${MAX_SAMPLES_PER_DOMAIN}" && ! "${MAX_SAMPLES_PER_DOMAIN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_SAMPLES_PER_DOMAIN must be a positive integer when provided." >&2
  exit 2
fi
[[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "NUM_SAMPLES must be a positive integer: ${NUM_SAMPLES}" >&2
  exit 2
}
"${PYTHON_BIN}" - "${NUM_SAMPLES}" "${TEMPERATURE}" <<'PY'
import math
import sys

num_samples = int(sys.argv[1])
temperature = float(sys.argv[2])
if not math.isfinite(temperature) or temperature < 0:
    raise SystemExit("TEMPERATURE must be a finite non-negative number.")
if num_samples > 1 and temperature <= 0:
    raise SystemExit(
        "NUM_SAMPLES>1 requires TEMPERATURE>0 to avoid duplicate greedy rollouts."
    )
PY
if [[ "${DRY_RUN}" == "0" && "${CONFIRM_FULL_TRAINING}" != "1" ]]; then
  echo "Full-training eval covers about 118,567 prompts per model and is expensive." >&2
  echo "Rerun with CONFIRM_FULL_TRAINING=1 after checking GPU time and disk capacity." >&2
  exit 2
fi

ensure_disk_capacity() {
  [[ "${MIN_FREE_GB}" != "0" ]] || return
  mkdir -p "${OUTPUT_ROOT}"
  local available_kb
  local required_kb
  available_kb="$(df -Pk "${OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
  required_kb=$((MIN_FREE_GB * 1024 * 1024))
  if [[ ! "${available_kb}" =~ ^[0-9]+$ || "${available_kb}" -lt "${required_kb}" ]]; then
    echo "Full-training eval requires at least ${MIN_FREE_GB} GiB free under ${OUTPUT_ROOT}." >&2
    echo "Set MIN_FREE_GB=0 only after independently checking output capacity." >&2
    exit 2
  fi
}

if [[ "${DRY_RUN}" == "0" ]]; then
  ensure_disk_capacity
fi

export IFBENCH_REPO="${IFBENCH_REPO:-${CODE_DIR}/../temp/IFBench}"
export MOPD_ALLOW_SIMPLE_SCORER_FALLBACK=0
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${DRY_RUN}" == "0" ]]; then
  "${CODE_DIR}/scripts/prepare_ifbench_runtime.sh"
fi

DOMAIN_NAMES=(math code if science)
DATASET_KEYS=(training_full_math training_full_code training_full_if training_full_science)
DATA_FILES=(
  "${CODE_DIR}/data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"
  "${CODE_DIR}/data/G-OPD-Training-Data/Eurus/code_train.parquet"
  "${CODE_DIR}/data/G-OPD-Training-Data/IF/train.parquet"
  "${CODE_DIR}/data/G-OPD-Training-Data/Science/train.parquet"
)
MANIFEST_INITIALIZED=0

row_count() {
  local data_file="$1"
  "${PYTHON_BIN}" - "${data_file}" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Missing training parquet: {path}")
print(pq.ParquetFile(path).metadata.num_rows)
PY
}

write_suite_manifest() {
  local suite_status="$1"
  local manifest_command=(
    "${PYTHON_BIN}"
    "${CODE_DIR}/eval/full_training_manifest.py"
    --code-dir "${CODE_DIR}"
    --output-root "${OUTPUT_ROOT}"
    --status "${suite_status}"
    --run-tag "${RUN_TAG}"
    --student-model-path "${STUDENT_MODEL_PATH}"
    --teacher-model-path "${TEACHER_MODEL_PATH}"
    --gpu-id "${GPU_ID}"
    --shard-size "${SHARD_SIZE}"
    --minimum-free-gib "${MIN_FREE_GB}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --num-samples "${NUM_SAMPLES}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --seed "${SEED}"
    --batch-size "${BATCH_SIZE}"
    --gpu-memory "${GPU_MEMORY}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
  )
  [[ -z "${MAX_SAMPLES_PER_DOMAIN}" ]] || {
    manifest_command+=(--max-samples-per-domain "${MAX_SAMPLES_PER_DOMAIN}")
  }
  [[ "${RESUME}" == "0" ]] || manifest_command+=(--resume)
  [[ "${MANIFEST_INITIALIZED}" == "0" ]] || manifest_command+=(--manifest-initialized)
  "${manifest_command[@]}"
  MANIFEST_INITIALIZED=1
}

run_shard() {
  local model_label="$1"
  local model_path="$2"
  local domain="$3"
  local dataset_key="$4"
  local start="$5"
  local count="$6"
  local shard_index="$7"
  local shard_dir="${OUTPUT_ROOT}/${model_label}/${domain}/shard_${shard_index}"
  local run_id="${model_label}_full_training_${domain}_${shard_index}_${RUN_TAG}"
  local extra_args=()

  if [[ "${RESUME}" == "1" && -f "${shard_dir}/SUCCESS" ]]; then
    echo "[two-model-full-training] skip completed shard: ${shard_dir}"
    return
  fi
  if [[ "${RESUME}" == "1" && -s "${shard_dir}/thinking_eval_samples.jsonl" ]]; then
    extra_args+=(--resume)
  fi
  [[ "${DRY_RUN}" == "0" ]] || extra_args+=(--dry-run)

  echo "[two-model-full-training] model=${model_label} domain=${domain} start=${start} count=${count}"
  local command=(
    "${CODE_DIR}/scripts/run_local_eval.sh"
    --model-path "${model_path}"
    --datasets "${dataset_key}"
    --modes non_thinking
    --backend vllm
    --tensor-parallel-size 1
    --batch-size "${BATCH_SIZE}"
    --gpu-memory "${GPU_MEMORY}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --enforce-eager
    --disable-chunked-prefill
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --num-samples "${NUM_SAMPLES}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --seed "${SEED}"
    --sample-offset "${start}"
    --max-samples "${count}"
    --python "${PYTHON_BIN}"
    --run-id "${run_id}"
    --output-dir "${shard_dir}"
    --score-code
    --save-completions
    "${extra_args[@]}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}"
    return
  fi
  mkdir -p "${shard_dir}"
  local tee_args=()
  [[ "${RESUME}" == "0" ]] || tee_args+=(-a)
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}" 2>&1 \
    | tee "${tee_args[@]}" "${shard_dir}/run.log"
  touch "${shard_dir}/SUCCESS"
}

run_model() {
  local model_label="$1"
  local model_path="$2"
  local domain_index

  for domain_index in "${!DOMAIN_NAMES[@]}"; do
    local domain="${DOMAIN_NAMES[${domain_index}]}"
    local dataset_key="${DATASET_KEYS[${domain_index}]}"
    local data_file="${DATA_FILES[${domain_index}]}"
    local total_rows
    total_rows="$(row_count "${data_file}")"
    if [[ -n "${MAX_SAMPLES_PER_DOMAIN}" && "${MAX_SAMPLES_PER_DOMAIN}" -lt "${total_rows}" ]]; then
      total_rows="${MAX_SAMPLES_PER_DOMAIN}"
    fi

    local start=0
    local shard_number=0
    while [[ "${start}" -lt "${total_rows}" ]]; do
      local count="${SHARD_SIZE}"
      if [[ $((start + count)) -gt "${total_rows}" ]]; then
        count=$((total_rows - start))
      fi
      local shard_index
      shard_index="$(printf '%05d_%09d_%09d' "${shard_number}" "${start}" "$((start + count))")"
      run_shard "${model_label}" "${model_path}" "${domain}" "${dataset_key}" \
        "${start}" "${count}" "${shard_index}"
      start=$((start + count))
      shard_number=$((shard_number + 1))
    done
  done
}

if [[ "${DRY_RUN}" == "1" ]]; then
  write_suite_manifest dry_run
else
  write_suite_manifest running
fi
run_model qwen3_1p7b "${STUDENT_MODEL_PATH}"
run_model goosereason_4b "${TEACHER_MODEL_PATH}"
if [[ "${DRY_RUN}" == "0" ]]; then
  write_suite_manifest complete
fi

echo "[two-model-full-training] complete: ${OUTPUT_ROOT}"
