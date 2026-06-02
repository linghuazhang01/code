#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/opd_mopd}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
ENV_NAME="${ENV_NAME:-mopd-verl}"
G_OPD_DIR="${G_OPD_DIR:-${REMOTE_ROOT}/G-OPD}"
OPD_CODE_DIR="${OPD_CODE_DIR:-${REMOTE_ROOT}/OPD-code}"

if [[ -f "${REMOTE_ROOT}/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${REMOTE_ROOT}/env.sh"
fi

MODEL_PATH="${MODEL_PATH:-${1:-}}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required." >&2
  exit 2
fi

MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
SAFE_MODEL_NAME="${MODEL_NAME//[^A-Za-z0-9_.-]/_}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${REMOTE_ROOT}/eval_outputs/paper_suite/step_${MOPD_GLOBAL_STEP:-manual}}"
PAPER_EVAL_DATASETS="${PAPER_EVAL_DATASETS-aime24,aime25,hmmt25_feb,hmmt25_nov,humaneval_plus,mbpp_plus,lcb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MATH_N="${MATH_N:-32}"
MATH_MAX_TOKENS="${MATH_MAX_TOKENS:-16384}"
MATH_MAX_NUM_SEQS="${MATH_MAX_NUM_SEQS:-32}"
EVALPLUS_GREEDY="${EVALPLUS_GREEDY:-0}"
EVALPLUS_TEMPERATURE="${EVALPLUS_TEMPERATURE:-1.0}"
EVALPLUS_TOP_P="${EVALPLUS_TOP_P:-1.0}"
EVALPLUS_N="${EVALPLUS_N:-4}"
LCB_RELEASE_VERSION="${LCB_RELEASE_VERSION:-release_v6}"
LCB_N="${LCB_N:-4}"
LCB_TEMPERATURE="${LCB_TEMPERATURE:-1.0}"
LCB_TOP_P="${LCB_TOP_P:-1.0}"
LCB_MAX_TOKENS="${LCB_MAX_TOKENS:-16384}"

export PATH="${CONDA_ROOT}/bin:${PATH}"
export PYTHONPATH="${OPD_CODE_DIR}:${G_OPD_DIR}/verl:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${REMOTE_ROOT}/hf_home}"
export CUDA_VISIBLE_DEVICES
export EVAL_OUTPUT_DIR
export MODEL_PATH
export PAPER_EVAL_DATASETS

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

mkdir -p "${EVAL_OUTPUT_DIR}"
cd "${G_OPD_DIR}"

contains_dataset() {
  local target="$1"
  [[ ",${PAPER_EVAL_DATASETS}," == *",${target},"* ]]
}

run_math_eval() {
  local dataset_key="$1"
  local input_file="$2"
  local output_file="${EVAL_OUTPUT_DIR}/${dataset_key}_${SAFE_MODEL_NAME}.jsonl"
  echo "[paper-eval] math ${dataset_key}: ${input_file}"
  python math_eval/eval_math.py \
    --input_file "${input_file}" \
    --model_path "${MODEL_PATH}" \
    --output_file "${output_file}" \
    --max_tokens "${MATH_MAX_TOKENS}" \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs "${MATH_MAX_NUM_SEQS}" \
    --n "${MATH_N}" \
    2>&1 | tee "${EVAL_OUTPUT_DIR}/${dataset_key}.log"
}

run_evalplus() {
  local dataset="$1"
  echo "[paper-eval] evalplus ${dataset}"
  bash code_eval/scripts/run_evalplus.sh \
    "${dataset}" \
    "${MODEL_PATH}" \
    "${EVALPLUS_GREEDY}" \
    "${EVALPLUS_TEMPERATURE}" \
    "${EVALPLUS_TOP_P}" \
    "${EVALPLUS_N}" \
    2>&1 | tee "${EVAL_OUTPUT_DIR}/evalplus_${dataset}.log"
}

run_lcb() {
  echo "[paper-eval] LiveCodeBench ${LCB_RELEASE_VERSION}"
  cd "${G_OPD_DIR}/code_eval/coding/LiveCodeBench"
  python -m lcb_runner.runner.main \
    --model "${SAFE_MODEL_NAME}" \
    --local_model_path "${MODEL_PATH}" \
    --trust_remote_code \
    --scenario codegeneration \
    --release_version "${LCB_RELEASE_VERSION}" \
    --tensor_parallel_size "$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)" \
    --use_cache \
    --n "${LCB_N}" \
    --temperature "${LCB_TEMPERATURE}" \
    --max_tokens "${LCB_MAX_TOKENS}" \
    --custom_output_save_name "${SAFE_MODEL_NAME}" \
    --top_p "${LCB_TOP_P}" \
    --timeout 60 \
    --evaluate --continue_existing --continue_existing_with_eval \
    2>&1 | tee "${EVAL_OUTPUT_DIR}/lcb.log"
  cd "${G_OPD_DIR}"
}

if contains_dataset "aime24"; then
  run_math_eval "aime24" "data/aime24/test.jsonl"
fi
if contains_dataset "aime25"; then
  run_math_eval "aime25" "data/aime25/test.jsonl"
fi
if contains_dataset "hmmt25_feb"; then
  run_math_eval "hmmt25_feb" "data/hmmt25_feb/test.jsonl"
fi
if contains_dataset "hmmt25_nov"; then
  run_math_eval "hmmt25_nov" "data/hmmt25_nov/test.jsonl"
fi
if contains_dataset "humaneval_plus"; then
  run_evalplus "humaneval"
fi
if contains_dataset "mbpp_plus"; then
  run_evalplus "mbpp"
fi
if contains_dataset "lcb"; then
  run_lcb
fi

python - <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ["EVAL_OUTPUT_DIR"])
summary = {
    "datasets": [item for item in os.environ["PAPER_EVAL_DATASETS"].split(",") if item],
    "model_path": os.environ["MODEL_PATH"],
    "output_dir": str(output_dir),
    "logs": sorted(path.name for path in output_dir.glob("*.log")),
    "jsonl_outputs": sorted(path.name for path in output_dir.glob("*.jsonl")),
}
(output_dir / "paper_eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
