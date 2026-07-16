#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${1:-}}"
RUN_ID="${RUN_ID:-four_domain_3gpu_smoke_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/data/eval_data/results/${RUN_ID}}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
IFBENCH_REPO="${IFBENCH_REPO:-${CODE_DIR}/../IFBench}"
PYTHON_BIN="${PYTHON_BIN:-python}"

[[ -n "${MODEL_PATH}" ]] || { echo "MODEL_PATH or first argument is required." >&2; exit 2; }
[[ -f "${IFBENCH_REPO}/evaluation_lib.py" ]] || {
  echo "IFBench evaluator not found: ${IFBENCH_REPO}" >&2
  exit 2
}

export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"
export IFBENCH_REPO
export TOKENIZERS_PARALLELISM=false
export MOPD_ALLOW_SIMPLE_SCORER_FALLBACK=0

GROUP_0=(
  data/eval_data/math/AIME24/test.parquet
  data/eval_data/math/AIME25/test.parquet
  data/eval_data/math/HMMT25Feb/test.parquet
  data/eval_data/math/HMMT25Nov/test.parquet
)
GROUP_1=(
  data/eval_data/code/HumanEvalPlus/test.parquet
  data/eval_data/code/MBPPPlus/test.parquet
  data/eval_data/science/GPQA/test.parquet
)
GROUP_2=(
  data/eval_data/if/IFEval/test.parquet
  data/eval_data/if/IFBench/test.parquet
)

mkdir -p "${OUTPUT_DIR}"
cd "${CODE_DIR}"

run_group() {
  local gpu="$1"
  local shard_dir="${OUTPUT_DIR}/gpu${gpu}"
  shift
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m eval.runner \
    --model-path "${MODEL_PATH}" \
    --data-files "$@" \
    --output-dir "${shard_dir}" \
    --modes non_thinking \
    --backend vllm \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --max-samples-per-dataset "${MAX_SAMPLES}" \
    --max-new-tokens-non-thinking "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --top-p 1 \
    --num-samples 1 \
    --seed 42 \
    --score-code \
    --save-completions \
    >"${shard_dir}.log" 2>&1
}

run_group 0 "${GROUP_0[@]}" & pid0=$!
run_group 1 "${GROUP_1[@]}" & pid1=$!
run_group 2 "${GROUP_2[@]}" & pid2=$!

status=0
wait "${pid0}" || status=1
wait "${pid1}" || status=1
wait "${pid2}" || status=1
if [[ "${status}" != "0" ]]; then
  echo "One or more eval shards failed. Inspect ${OUTPUT_DIR}/gpu*.log" >&2
  exit 1
fi

cat "${OUTPUT_DIR}"/gpu*/thinking_eval_samples.jsonl > "${OUTPUT_DIR}/thinking_eval_samples.jsonl"
"${PYTHON_BIN}" -m eval.report \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  --model-path "${MODEL_PATH}" \
  --status final \
  --expected-total 45 \
  --notes "3-GPU data-parallel smoke; 5 examples per dataset; HLE excluded."

echo "[3gpu-eval] report: ${OUTPUT_DIR}/README.md"
