#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
API_FILE="${API_FILE:-${CODE_DIR}/api.sh}"

if [[ -f "${API_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${API_FILE}"
fi

# api.sh compatibility:
#   dashscope_ak      -> OPENAI_API_KEY
#   dashscope_baseurl -> OPENAI_BASE_URL
#   model             -> JUDGE_MODEL
if [[ -z "${OPENAI_API_KEY:-}" && -n "${dashscope_ak:-}" ]]; then
  export OPENAI_API_KEY="${dashscope_ak}"
fi
if [[ -z "${OPENAI_BASE_URL:-}" && -n "${dashscope_baseurl:-}" ]]; then
  export OPENAI_BASE_URL="${dashscope_baseurl}"
fi
if [[ -z "${JUDGE_MODEL:-}" && -n "${model:-}" ]]; then
  export JUDGE_MODEL="${model}"
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
RUN_ID="${RUN_ID:-qwen3_4b_nonthinking_greasoner_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/eval/results/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/run.log}"

DOMAINS="${DOMAINS:-greasoner}"
DATASETS="${DATASETS:-mmlupro gpqa_d supergpqa theoremqa bbeh}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
API_BANK_DIR="${API_BANK_DIR:-${CODE_DIR}/../temp/grpo_sources/ToolRL/benchmarks/API-Bank}"
API_BANK_LEVELS="${API_BANK_LEVELS:-1 2 3}"

JUDGE_BASE_URL="${JUDGE_BASE_URL:-${OPENAI_BASE_URL:-}}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"

if [[ " ${DATASETS} " == *" theoremqa "* ]]; then
  if [[ -z "${JUDGE_API_KEY}" || -z "${JUDGE_BASE_URL}" || -z "${JUDGE_MODEL}" ]]; then
    cat >&2 <<'MSG'
[qwen3-4b-nonthinking-eval] theoremqa requires judge API settings.
Set them in api.sh or env:
  OPENAI_API_KEY / OPENAI_BASE_URL / JUDGE_MODEL
or:
  dashscope_ak / dashscope_baseurl / model
MSG
    exit 2
  fi
fi

ARGS=(
  --domains ${DOMAINS}
  --datasets ${DATASETS}
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --enable-thinking false
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --api-bank-dir "${API_BANK_DIR}"
  --api-bank-levels ${API_BANK_LEVELS}
  --judge-base-url "${JUDGE_BASE_URL}"
  --judge-api-key "${JUDGE_API_KEY}"
  --judge-model "${JUDGE_MODEL}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi

echo "[qwen3-4b-nonthinking-eval] model: ${MODEL_PATH}"
echo "[qwen3-4b-nonthinking-eval] domains: ${DOMAINS}"
echo "[qwen3-4b-nonthinking-eval] datasets: ${DATASETS}"
echo "[qwen3-4b-nonthinking-eval] output: ${OUTPUT_DIR}"
echo "[qwen3-4b-nonthinking-eval] log: ${LOG_FILE}"
echo "[qwen3-4b-nonthinking-eval] judge model: ${JUDGE_MODEL}"

cd "${CODE_DIR}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[qwen3-4b-nonthinking-eval] dry run only; command is ready."
  echo "[qwen3-4b-nonthinking-eval] set DRY_RUN=0 or omit it to execute."
  exit 0
fi

mkdir -p "$(dirname "${LOG_FILE}")"
"${CODE_DIR}/eval/scripts/run_official_eval.sh" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
