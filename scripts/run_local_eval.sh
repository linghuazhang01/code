#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
Run the parquet-based OPD evaluation with Transformers or vLLM.

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
  --backend NAME          transformers or vllm (default: transformers).
  --tensor-parallel-size N
                          vLLM tensor-parallel GPU count (default: 1).
  --batch-size N          vLLM generation batch size (default: 8).
  --gpu-memory FLOAT      vLLM GPU memory utilization (default: 0.9).
  --torch-dtype NAME      Model dtype (default: auto).
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

  CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/run_local_eval.sh \
    --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 --backend vllm \
    --tensor-parallel-size 4 --datasets aime24,gpqa_diamond
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
BACKEND="${BACKEND:-transformers}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
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
    --backend) BACKEND="${2:?--backend requires a value}"; shift 2 ;;
    --tensor-parallel-size) TENSOR_PARALLEL_SIZE="${2:?--tensor-parallel-size requires a value}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:?--batch-size requires a value}"; shift 2 ;;
    --gpu-memory) GPU_MEMORY_UTILIZATION="${2:?--gpu-memory requires a value}"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="${2:?--torch-dtype requires a value}"; shift 2 ;;
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
[[ "${BACKEND}" == "transformers" || "${BACKEND}" == "vllm" ]] || {
  echo "--backend must be transformers or vllm" >&2
  exit 2
}
[[ "${TENSOR_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--tensor-parallel-size must be a positive integer" >&2
  exit 2
}
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "--batch-size must be a positive integer" >&2; exit 2; }
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
  --backend "${BACKEND}"
  --torch-dtype "${TORCH_DTYPE}"
  --max-new-tokens-thinking "${MAX_NEW_TOKENS}"
  --max-new-tokens-non-thinking "${MAX_NEW_TOKENS}"
  --num-samples "${NUM_SAMPLES}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --seed "${SEED}"
)
if [[ "${BACKEND}" == "vllm" ]]; then
  CMD+=(
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --batch-size "${BATCH_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  )
else
  CMD+=(--device-map auto)
fi
[[ -z "${MAX_SAMPLES}" ]] || CMD+=(--max-samples-per-dataset "${MAX_SAMPLES}")
[[ "${SCORE_CODE}" == "0" ]] || CMD+=(--score-code)
[[ "${SAVE_COMPLETIONS}" == "0" ]] || CMD+=(--save-completions)

printf '[local-eval] model: %s\n' "${MODEL_PATH}"
printf '[local-eval] datasets: %s\n' "${DATASETS}"
printf '[local-eval] modes: %s\n' "${MODES}"
printf '[local-eval] backend: %s\n' "${BACKEND}"
if [[ "${BACKEND}" == "vllm" ]]; then
  printf '[local-eval] tensor parallel size: %s\n' "${TENSOR_PARALLEL_SIZE}"
fi
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
REQUIRED_MODULES='import pandas, pyarrow, torch, transformers'
[[ "${BACKEND}" != "transformers" ]] || REQUIRED_MODULES+=', accelerate'
[[ "${BACKEND}" != "vllm" ]] || REQUIRED_MODULES+=', vllm'
"${PYTHON_BIN}" -c "${REQUIRED_MODULES}" || {
  echo "Missing dependencies for eval backend '${BACKEND}'." >&2
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
