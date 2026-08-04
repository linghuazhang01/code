#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${CODE_DIR}/../models/Qwen3-1.7B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${CODE_DIR}/../models/Nemotron-Research-GooseReason-4B-Instruct}"
GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CODE_DIR}/data/eval_data/results/two_model_ood/${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# LiveCodeBench is opt-in because its official protocol uses sampled K=4
# rollouts (temperature=1, max_tokens=16384), unlike this greedy suite.
OOD_DATASETS="${OOD_DATASETS:-aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus,ifeval,ifbench,gpqa_diamond}"
# Match the GooseReason training config's validation-inference settings.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-24}"
GPU_MEMORY="${GPU_MEMORY:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-18432}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_WANDB_ENABLED="${EVAL_WANDB_ENABLED:-1}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-mopd-eval}"
EVAL_WANDB_ENTITY="${EVAL_WANDB_ENTITY:-${WANDB_ENTITY:-}}"
EVAL_WANDB_MODE="${EVAL_WANDB_MODE:-online}"
EVAL_WANDB_UPLOAD_RAW="${EVAL_WANDB_UPLOAD_RAW:-1}"
EVAL_WANDB_TIMEOUT_SECONDS="${EVAL_WANDB_TIMEOUT_SECONDS:-1800}"
EVAL_WANDB_ENV_FILE="${EVAL_WANDB_ENV_FILE:-${CODE_DIR}/.env.local}"

for flag in RESUME DRY_RUN EVAL_WANDB_ENABLED EVAL_WANDB_UPLOAD_RAW; do
  [[ "${!flag}" == "0" || "${!flag}" == "1" ]] || {
    echo "${flag} must be 0 or 1: ${!flag}" >&2
    exit 2
  }
done
[[ "${EVAL_WANDB_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] || {
  echo "EVAL_WANDB_TIMEOUT_SECONDS must be a non-negative integer." >&2
  exit 2
}

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

export IFBENCH_REPO="${IFBENCH_REPO:-${CODE_DIR}/../temp/IFBench}"
export MOPD_ALLOW_SIMPLE_SCORER_FALLBACK=0
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${DRY_RUN}" == "0" ]]; then
  "${CODE_DIR}/scripts/prepare_ifbench_runtime.sh"
fi

run_model() {
  local model_label="$1"
  local model_path="$2"
  local output_dir="${OUTPUT_ROOT}/${model_label}"
  local run_id="${model_label}_ood_${RUN_TAG}"
  local extra_args=()

  [[ -z "${MAX_SAMPLES}" ]] || extra_args+=(--max-samples "${MAX_SAMPLES}")
  [[ "${RESUME}" == "0" ]] || extra_args+=(--resume)
  [[ "${DRY_RUN}" == "0" ]] || extra_args+=(--dry-run)
  if [[ "${EVAL_WANDB_ENABLED}" == "1" ]]; then
    extra_args+=(
      --wandb-project "${EVAL_WANDB_PROJECT}"
      --wandb-group "two_model_ood_${RUN_TAG}"
      --wandb-mode "${EVAL_WANDB_MODE}"
      --wandb-timeout-seconds "${EVAL_WANDB_TIMEOUT_SECONDS}"
      --wandb-env-file "${EVAL_WANDB_ENV_FILE}"
    )
    [[ -z "${EVAL_WANDB_ENTITY}" ]] || extra_args+=(--wandb-entity "${EVAL_WANDB_ENTITY}")
    [[ "${EVAL_WANDB_UPLOAD_RAW}" == "0" ]] || extra_args+=(--wandb-upload-raw)
  fi

  echo "[two-model-ood] model=${model_label} path=${model_path} output=${output_dir}"
  local command=(
    "${CODE_DIR}/scripts/run_local_eval.sh"
      --model-path "${model_path}" \
      --datasets "${OOD_DATASETS}" \
      --modes non_thinking \
      --backend vllm \
      --tensor-parallel-size 1 \
      --batch-size "${BATCH_SIZE}" \
      --gpu-memory "${GPU_MEMORY}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --enforce-eager \
      --disable-chunked-prefill \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --num-samples "${NUM_SAMPLES}" \
      --temperature "${TEMPERATURE}" \
      --top-p "${TOP_P}" \
      --seed "${SEED}" \
      --python "${PYTHON_BIN}" \
      --run-id "${run_id}" \
      --output-dir "${output_dir}" \
      --score-code \
      --save-completions \
      "${extra_args[@]}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}"
    return
  fi
  if [[ "${RESUME}" == "0" && -d "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty output directory: ${output_dir}" >&2
    echo "Use a new RUN_TAG, or set RESUME=1 with identical settings." >&2
    exit 2
  fi
  mkdir -p "${output_dir}"
  local tee_args=()
  [[ "${RESUME}" == "0" ]] || tee_args+=(-a)
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}" 2>&1 \
    | tee "${tee_args[@]}" "${output_dir}/run.log"
}

run_model qwen3_1p7b "${STUDENT_MODEL_PATH}"
run_model goosereason_4b "${TEACHER_MODEL_PATH}"

echo "[two-model-ood] complete: ${OUTPUT_ROOT}"
