#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="$(cd "${EVAL_DIR}/.." && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-$(cd "${CODE_DIR}/.." && pwd)}"

if [[ -f "${CODE_DIR}/logs/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CODE_DIR}/logs/env.sh"
fi

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
ENV_NAME="${ENV_NAME:-mopd-verl}"
MODEL_PATH="${MODEL_PATH:-${REMOTE_ROOT}/models/Qwen3-4B}"
RUN_ID="${RUN_ID:-qwen3_4b_thinking_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_DIR}/results/${RUN_ID}}"
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-}"
MAX_NEW_TOKENS_THINKING="${MAX_NEW_TOKENS_THINKING:-65536}"
MAX_NEW_TOKENS_NON_THINKING="${MAX_NEW_TOKENS_NON_THINKING:-16384}"
MAX_NEW_TOKENS_THINKING_MATH="${MAX_NEW_TOKENS_THINKING_MATH:-}"
MAX_NEW_TOKENS_THINKING_CODE="${MAX_NEW_TOKENS_THINKING_CODE:-}"
MAX_NEW_TOKENS_NON_THINKING_MATH="${MAX_NEW_TOKENS_NON_THINKING_MATH:-}"
MAX_NEW_TOKENS_NON_THINKING_CODE="${MAX_NEW_TOKENS_NON_THINKING_CODE:-}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
BACKEND="${BACKEND:-vllm}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-${REMOTE_ROOT}/hf_home}"
SCORE_CODE="${SCORE_CODE:-1}"
SAVE_COMPLETIONS="${SAVE_COMPLETIONS:-1}"
INCLUDE_SEARCH="${INCLUDE_SEARCH:-1}"
SEARCH_DATA_FILES="${SEARCH_DATA_FILES:-${CODE_DIR}/data/SearchQA/test.parquet}"
INCLUDE_GREASONER="${INCLUDE_GREASONER:-1}"
GREASONER_DATA_FILES="${GREASONER_DATA_FILES:-${CODE_DIR}/eval/domains/greasoner/data/WebInstructVerified/test.parquet}"
INCLUDE_TOOLRL="${INCLUDE_TOOLRL:-1}"
TOOLRL_DATA_FILES="${TOOLRL_DATA_FILES:-${CODE_DIR}/eval/domains/toolrl/data/BFCL/test.parquet ${CODE_DIR}/eval/domains/toolrl/data/API-Bank/test.parquet ${CODE_DIR}/eval/domains/toolrl/data/Bamboogle/test.parquet}"

export CUDA_VISIBLE_DEVICES
export HF_ENDPOINT
export HF_HOME
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
DATA_FILES=(
  "${CODE_DIR}/eval/domains/math/data/AIME24/test.parquet"
  "${CODE_DIR}/eval/domains/math/data/AIME25/test.parquet"
  "${CODE_DIR}/eval/domains/math/data/HMMT25Feb/test.parquet"
  "${CODE_DIR}/eval/domains/math/data/HMMT25Nov/test.parquet"
  "${CODE_DIR}/eval/domains/code/data/HumanEvalPlus/test.parquet"
  "${CODE_DIR}/eval/domains/code/data/MBPPPlus/test.parquet"
)
if [[ "${INCLUDE_GREASONER}" == "1" ]]; then
  for greasoner_data_file in ${GREASONER_DATA_FILES}; do
    if [[ -f "${greasoner_data_file}" ]]; then
      DATA_FILES+=("${greasoner_data_file}")
    else
      echo "[thinking-eval] warning: missing GReasoner eval data, skipping: ${greasoner_data_file}" >&2
    fi
  done
fi
if [[ "${INCLUDE_TOOLRL}" == "1" ]]; then
  for toolrl_data_file in ${TOOLRL_DATA_FILES}; do
    if [[ -f "${toolrl_data_file}" ]]; then
      DATA_FILES+=("${toolrl_data_file}")
    else
      echo "[thinking-eval] warning: missing ToolRL eval data, skipping: ${toolrl_data_file}" >&2
    fi
  done
fi
if [[ "${INCLUDE_SEARCH}" == "1" ]]; then
  for search_data_file in ${SEARCH_DATA_FILES}; do
    if [[ -f "${search_data_file}" ]]; then
      DATA_FILES+=("${search_data_file}")
    else
      echo "[thinking-eval] warning: missing search eval data, skipping: ${search_data_file}" >&2
    fi
  done
fi

ARGS=(
  -m eval.runner
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --data-files "${DATA_FILES[@]}"
  --modes non_thinking thinking
  --max-new-tokens-thinking "${MAX_NEW_TOKENS_THINKING}"
  --max-new-tokens-non-thinking "${MAX_NEW_TOKENS_NON_THINKING}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --backend "${BACKEND}"
  --batch-size "${BATCH_SIZE}"
  --torch-dtype "${TORCH_DTYPE}"
  --device-map "${DEVICE_MAP}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --skip-missing-data-files
)

if [[ -n "${MAX_SAMPLES_PER_DATASET}" ]]; then
  ARGS+=(--max-samples-per-dataset "${MAX_SAMPLES_PER_DATASET}")
fi
if [[ -n "${MAX_NEW_TOKENS_THINKING_MATH}" ]]; then
  ARGS+=(--max-new-tokens-thinking-math "${MAX_NEW_TOKENS_THINKING_MATH}")
fi
if [[ -n "${MAX_NEW_TOKENS_THINKING_CODE}" ]]; then
  ARGS+=(--max-new-tokens-thinking-code "${MAX_NEW_TOKENS_THINKING_CODE}")
fi
if [[ -n "${MAX_NEW_TOKENS_NON_THINKING_MATH}" ]]; then
  ARGS+=(--max-new-tokens-non-thinking-math "${MAX_NEW_TOKENS_NON_THINKING_MATH}")
fi
if [[ -n "${MAX_NEW_TOKENS_NON_THINKING_CODE}" ]]; then
  ARGS+=(--max-new-tokens-non-thinking-code "${MAX_NEW_TOKENS_NON_THINKING_CODE}")
fi
if [[ "${SCORE_CODE}" == "1" ]]; then
  ARGS+=(--score-code)
fi
if [[ "${SAVE_COMPLETIONS}" == "1" ]]; then
  ARGS+=(--save-completions)
fi

cd "${CODE_DIR}"
"${PYTHON_BIN}" "${ARGS[@]}"
"${PYTHON_BIN}" -m eval.report \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  --model-path "${MODEL_PATH}" \
  --status final

echo "[thinking-eval] outputs: ${OUTPUT_DIR}"
echo "[thinking-eval] summary: ${OUTPUT_DIR}/thinking_eval_summary.csv"
echo "[thinking-eval] report: ${OUTPUT_DIR}/README.md"
