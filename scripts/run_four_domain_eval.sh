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
  EVAL_DATASETS       Comma-separated dataset keys.
  EVAL_MODES          non_thinking, thinking, or both (default: non_thinking).
  EVAL_MAX_TOKENS     Generation limit (default: 8192).
  EVAL_NUM_SAMPLES    Rollouts per prompt (default: 1).
  EVAL_TEMPERATURE    Sampling temperature (default: 0).
  EVAL_TOP_P          Nucleus sampling threshold (default: 1.0).
  EVAL_SEED           Base seed (default: 42).

Example smoke run:
  scripts/run_four_domain_eval.sh --model-path /path/to/model --max-samples 2
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
export IFBENCH_REPO="${IFBENCH_REPO:-${CODE_DIR}/../temp/IFBench}"
DRY_RUN=0
for arg in "$@"; do
  [[ "${arg}" != "--dry-run" ]] || DRY_RUN=1
done

cat <<EOF
[four-domain-eval] HLE excluded: official judge is not implemented.
[four-domain-eval] Code scoring executes generated code in child processes.
[four-domain-eval] datasets: ${DATASETS}
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
  --score-code \
  --save-completions \
  "$@"
