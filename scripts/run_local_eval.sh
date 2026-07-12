#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
Run the parquet-based OPD evaluation with the local Transformers runtime.

Usage:
  scripts/run_local_eval.sh --model-path PATH [options]

Options:
  --model-path PATH       Local model directory or Hugging Face model id.
  --datasets NAMES        Comma-separated datasets (default: aime24,aime25,
                          hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus).
                          Also supports ifeval, ifbench, and gpqa_diamond.
  --modes NAMES           Comma-separated modes: non_thinking,thinking
                          (default: non_thinking).
  --max-samples N         Maximum examples per dataset (default: all).
  --max-new-tokens N      Generation limit for every selected mode (default: 8192).
  --num-samples N         Rollouts per prompt (default: 1; GRPO AIME paper eval: 32).
  --temperature FLOAT     Sampling temperature (default: 0; GRPO AIME paper eval: 1).
  --top-p FLOAT           Nucleus sampling threshold (default: 1.0).
  --seed N                Base generation seed (default: 42).
  --output-dir PATH       Result directory (default: data/eval_data/results/<run-id>).
  --run-id ID             Run identifier used in the report.
  --python PATH           Python executable (default: $PYTHON or python3).
  --score-code            Execute generated code for Code scoring; use only in
                          an isolated environment.
  --save-completions      Save full model completions in JSONL output.
  --dry-run               Validate inputs and print the command only.
  -h, --help              Show this help.

Examples:
  scripts/run_local_eval.sh --model-path ../models/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
    --datasets aime24 --max-samples 2 --save-completions

  scripts/run_local_eval.sh --model-path Qwen/Qwen3-4B \
    --datasets aime24,humaneval_plus --modes non_thinking,thinking --score-code
USAGE
}

MODEL_PATH="${MODEL_PATH:-}"
DATASETS="${DATASETS:-aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus}"
MODES="${MODES:-non_thinking}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-local_eval_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PYTHON_BIN="${PYTHON:-python3}"
SCORE_CODE=0
SAVE_COMPLETIONS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="${2:?--model-path requires a value}"; shift 2 ;;
    --datasets) DATASETS="${2:?--datasets requires a value}"; shift 2 ;;
    --modes) MODES="${2:?--modes requires a value}"; shift 2 ;;
    --max-samples) MAX_SAMPLES="${2:?--max-samples requires a value}"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="${2:?--max-new-tokens requires a value}"; shift 2 ;;
    --num-samples) NUM_SAMPLES="${2:?--num-samples requires a value}"; shift 2 ;;
    --temperature) TEMPERATURE="${2:?--temperature requires a value}"; shift 2 ;;
    --top-p) TOP_P="${2:?--top-p requires a value}"; shift 2 ;;
    --seed) SEED="${2:?--seed requires a value}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
    --run-id) RUN_ID="${2:?--run-id requires a value}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?--python requires a value}"; shift 2 ;;
    --score-code) SCORE_CODE=1; shift ;;
    --save-completions) SAVE_COMPLETIONS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${MODEL_PATH}" ]] || { echo "--model-path is required" >&2; exit 2; }
[[ "${MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max-new-tokens must be a positive integer" >&2
  exit 2
}
[[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || { echo "--num-samples must be a positive integer" >&2; exit 2; }
[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "--seed must be a non-negative integer" >&2; exit 2; }
if [[ -n "${MAX_SAMPLES}" && ! "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-samples must be a positive integer" >&2
  exit 2
fi

IFS=',' read -r -a DATASET_NAMES <<< "${DATASETS}"
DATA_FILES=()
for name in "${DATASET_NAMES[@]}"; do
  name="${name//[[:space:]]/}"
  case "${name}" in
    aime24) relative_path="data/eval_data/math/AIME24/test.parquet" ;;
    aime25) relative_path="data/eval_data/math/AIME25/test.parquet" ;;
    hmmt25feb) relative_path="data/eval_data/math/HMMT25Feb/test.parquet" ;;
    hmmt25nov) relative_path="data/eval_data/math/HMMT25Nov/test.parquet" ;;
    humaneval_plus) relative_path="data/eval_data/code/HumanEvalPlus/test.parquet" ;;
    mbpp_plus) relative_path="data/eval_data/code/MBPPPlus/test.parquet" ;;
    livecodebench) relative_path="data/eval_data/code/LiveCodeBench/test.parquet" ;;
    ifeval) relative_path="data/eval_data/ifbench/IFEval.parquet" ;;
    ifbench) relative_path="data/eval_data/ifbench/IFBench_test.parquet" ;;
    gpqa_diamond) relative_path="data/eval_data/science/gpqa.parquet" ;;
    *)
      echo "Unknown dataset '${name}'." >&2
      echo "Valid names: aime24 aime25 hmmt25feb hmmt25nov humaneval_plus mbpp_plus livecodebench ifeval ifbench gpqa_diamond" >&2
      exit 2
      ;;
  esac
  data_file="${CODE_DIR}/${relative_path}"
  [[ -f "${data_file}" ]] || {
    echo "Missing eval data: ${data_file}" >&2
    echo "Run eval/scripts/prepare_paper_eval_data.sh first." >&2
    exit 2
  }
  DATA_FILES+=("${data_file}")
done

IFS=',' read -r -a MODE_NAMES <<< "${MODES}"
for index in "${!MODE_NAMES[@]}"; do
  mode="${MODE_NAMES[${index}]//[[:space:]]/}"
  MODE_NAMES[${index}]="${mode}"
  [[ "${mode}" == "thinking" || "${mode}" == "non_thinking" ]] || {
    echo "Unknown mode '${mode}'; use thinking or non_thinking." >&2
    exit 2
  }
done

OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/data/eval_data/results/${RUN_ID}}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MOPD_ALLOW_SIMPLE_SCORER_FALLBACK="${MOPD_ALLOW_SIMPLE_SCORER_FALLBACK:-1}"

CMD=(
  "${PYTHON_BIN}" -m eval.runner
  --model-path "${MODEL_PATH}"
  --data-files "${DATA_FILES[@]}"
  --output-dir "${OUTPUT_DIR}"
  --modes "${MODE_NAMES[@]}"
  --backend transformers
  --device-map auto
  --torch-dtype auto
  --max-new-tokens-thinking "${MAX_NEW_TOKENS}"
  --max-new-tokens-non-thinking "${MAX_NEW_TOKENS}"
  --num-samples "${NUM_SAMPLES}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --seed "${SEED}"
)
[[ -z "${MAX_SAMPLES}" ]] || CMD+=(--max-samples-per-dataset "${MAX_SAMPLES}")
[[ "${SCORE_CODE}" == "0" ]] || CMD+=(--score-code)
[[ "${SAVE_COMPLETIONS}" == "0" ]] || CMD+=(--save-completions)

printf '[local-eval] model: %s\n' "${MODEL_PATH}"
printf '[local-eval] datasets: %s\n' "${DATASETS}"
printf '[local-eval] modes: %s\n' "${MODES}"
printf '[local-eval] output: %s\n' "${OUTPUT_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[local-eval] command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
}
"${PYTHON_BIN}" -c 'import accelerate, pandas, pyarrow, torch, transformers' || {
  echo "Missing local eval dependencies; install transformers, accelerate, torch, pandas, and pyarrow." >&2
  exit 2
}

mkdir -p "${OUTPUT_DIR}"
cd "${CODE_DIR}"
"${CMD[@]}"
"${PYTHON_BIN}" -m eval.report \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  --model-path "${MODEL_PATH}" \
  --status final

printf '[local-eval] report: %s/README.md\n' "${OUTPUT_DIR}"
