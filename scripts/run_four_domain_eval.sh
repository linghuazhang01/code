#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
Run the supported four-domain OPD evaluation and generate Markdown/JSON/CSV reports.

Usage:
  scripts/run_four_domain_eval.sh --model-path PATH [run_local_eval options]

Default benchmark set:
  Math:    AIME24, AIME25, HMMT25Feb, HMMT25Nov
  Code:    HumanEvalPlus, MBPPPlus
  IF:      IFEval, IFBench
  Science: GPQA-Diamond

HLE is intentionally excluded because the official judge is not implemented.
LiveCodeBench is opt-in because the local 1,055-row file is not pinned to the
M2RL paper's separate v5/v6 releases.

Environment overrides:
  MODEL_PATH          Model directory or Hugging Face model id.
  EVAL_DATASETS       Comma-separated dataset keys.
  EVAL_MODES          non_thinking, thinking, or both (default: non_thinking).
  EVAL_BACKEND        transformers or vllm (default: transformers).
  EVAL_TP_SIZE        vLLM tensor-parallel GPU count (default: 1).
  EVAL_BATCH_SIZE     vLLM request batch size (default: 8).
  EVAL_GPU_MEMORY     vLLM GPU memory utilization (default: 0.9).
  EVAL_TORCH_DTYPE    Model dtype (default: auto).
  EVAL_MAX_TOKENS     Generation limit (default: 8192).
  EVAL_NUM_SAMPLES    Rollouts per prompt (default: 1).
  EVAL_TEMPERATURE    Sampling temperature (default: 0).
  EVAL_TOP_P          Nucleus sampling threshold (default: 1.0).
  EVAL_SEED           Base seed (default: 42).

Example smoke run:
  scripts/run_four_domain_eval.sh --model-path /path/to/model --max-samples 2

Example 8-GPU vLLM run for selected datasets:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  MODEL_PATH=/path/to/model EVAL_BACKEND=vllm EVAL_TP_SIZE=8 \
  EVAL_DATASETS=aime24,aime25,gpqa_diamond \
    scripts/run_four_domain_eval.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATASETS="${EVAL_DATASETS:-aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus,ifeval,ifbench,gpqa_diamond}"
MODES="${EVAL_MODES:-non_thinking}"
MAX_TOKENS="${EVAL_MAX_TOKENS:-8192}"
NUM_SAMPLES="${EVAL_NUM_SAMPLES:-1}"
TEMPERATURE="${EVAL_TEMPERATURE:-0}"
TOP_P="${EVAL_TOP_P:-1.0}"
SEED="${EVAL_SEED:-42}"
BACKEND="${EVAL_BACKEND:-transformers}"
TP_SIZE="${EVAL_TP_SIZE:-1}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GPU_MEMORY="${EVAL_GPU_MEMORY:-0.9}"
TORCH_DTYPE="${EVAL_TORCH_DTYPE:-auto}"
export IFBENCH_REPO="${IFBENCH_REPO:-${CODE_DIR}/../temp/IFBench}"
DRY_RUN=0
ARGS=("$@")
for ((index = 0; index < ${#ARGS[@]}; index++)); do
  case "${ARGS[${index}]}" in
    --datasets) DATASETS="${ARGS[$((index + 1))]:?--datasets requires a value}" ;;
    --modes) MODES="${ARGS[$((index + 1))]:?--modes requires a value}" ;;
    --backend) BACKEND="${ARGS[$((index + 1))]:?--backend requires a value}" ;;
    --tensor-parallel-size) TP_SIZE="${ARGS[$((index + 1))]:?--tensor-parallel-size requires a value}" ;;
    --batch-size) BATCH_SIZE="${ARGS[$((index + 1))]:?--batch-size requires a value}" ;;
    --gpu-memory) GPU_MEMORY="${ARGS[$((index + 1))]:?--gpu-memory requires a value}" ;;
    --torch-dtype) TORCH_DTYPE="${ARGS[$((index + 1))]:?--torch-dtype requires a value}" ;;
    --dry-run) DRY_RUN=1 ;;
  esac
done

cat <<EOF
[four-domain-eval] HLE excluded: official judge is not implemented.
[four-domain-eval] Code scoring executes generated code in child processes.
[four-domain-eval] datasets: ${DATASETS}
[four-domain-eval] backend: ${BACKEND}
EOF

if [[ "${DRY_RUN}" == "0" && ("${DATASETS}" == *"ifeval"* || "${DATASETS}" == *"ifbench"*) ]]; then
  "${CODE_DIR}/scripts/prepare_ifbench_runtime.sh"
fi

exec "${CODE_DIR}/scripts/run_local_eval.sh" \
  --datasets "${DATASETS}" \
  --modes "${MODES}" \
  --max-new-tokens "${MAX_TOKENS}" \
  --num-samples "${NUM_SAMPLES}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --seed "${SEED}" \
  --backend "${BACKEND}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --gpu-memory "${GPU_MEMORY}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --score-code \
  --save-completions \
  "$@"
