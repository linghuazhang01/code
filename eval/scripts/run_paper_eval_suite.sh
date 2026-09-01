#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="$(cd "${EVAL_DIR}/.." && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-$(cd "${CODE_DIR}/.." && pwd)}"
DEFAULT_G_OPD_DIR="${REMOTE_ROOT}/G-OPD"
if [[ ! -d "${DEFAULT_G_OPD_DIR}" && -d "${REMOTE_ROOT}/../G-OPD" ]]; then
  DEFAULT_G_OPD_DIR="$(cd "${REMOTE_ROOT}/../G-OPD" && pwd)"
fi
G_OPD_DIR="${G_OPD_DIR:-${DEFAULT_G_OPD_DIR}}"
OPD_CODE_DIR="${OPD_CODE_DIR:-${CODE_DIR}}"

if [[ -f "${OPD_CODE_DIR}/logs/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${OPD_CODE_DIR}/logs/env.sh"
elif [[ -f "${REMOTE_ROOT}/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${REMOTE_ROOT}/env.sh"
fi

MODEL_PATH="${MODEL_PATH:-${1:-}}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required." >&2
  exit 2
fi

MODEL_BASENAME="$(basename "${MODEL_PATH}")"
MODEL_PARENT_NAME="$(basename "$(dirname "${MODEL_PATH}")")"
MODEL_NAME="${MODEL_NAME:-${MODEL_PARENT_NAME}__${MODEL_BASENAME}}"
SAFE_MODEL_NAME="${MODEL_NAME//[^A-Za-z0-9_.-]/_}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${REMOTE_ROOT}/eval_outputs/paper_suite/${SAFE_MODEL_NAME}}"
PAPER_EVAL_DATASETS="${PAPER_EVAL_DATASETS-aime24,aime25,hmmt25_feb,hmmt25_nov,humaneval_plus,mbpp_plus,lcb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MATH_N="${MATH_N:-8}"
MATH_MAX_TOKENS="${MATH_MAX_TOKENS:-16384}"
MATH_MAX_NUM_SEQS="${MATH_MAX_NUM_SEQS:-32}"
EVALPLUS_GREEDY="${EVALPLUS_GREEDY:-0}"
EVALPLUS_TEMPERATURE="${EVALPLUS_TEMPERATURE:-1.0}"
EVALPLUS_TOP_P="${EVALPLUS_TOP_P:-1.0}"
EVALPLUS_N="${EVALPLUS_N:-8}"
LCB_RELEASE_VERSIONS="${LCB_RELEASE_VERSIONS:-v5,v6}"
LCB_MODEL_STYLE_NAME="${LCB_MODEL_STYLE_NAME:-Qwen3-4B-NonThinking}"
LCB_CHAT_TEMPLATE_TOKENIZER="Qwen/Qwen3-4B"
LCB_N="${LCB_N:-8}"
LCB_TEMPERATURE="${LCB_TEMPERATURE:-1.0}"
LCB_TOP_P="${LCB_TOP_P:-1.0}"
LCB_MAX_TOKENS="${LCB_MAX_TOKENS:-16384}"

LCB_RUNNER_DIR="${G_OPD_DIR}/code_eval/coding/LiveCodeBench"
export PYTHONPATH="${OPD_CODE_DIR}:${G_OPD_DIR}/verl:${LCB_RUNNER_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REMOTE_ROOT}/hf_home}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES
export EVAL_OUTPUT_DIR
export MODEL_PATH
export PAPER_EVAL_DATASETS
export MATH_N
export EVALPLUS_N
export LCB_N
export LCB_RELEASE_VERSIONS
export LCB_MODEL_STYLE_NAME
export LCB_CHAT_TEMPLATE_TOKENIZER

mkdir -p "${EVAL_OUTPUT_DIR}"
PREPARED_MODEL_PATH="$(
  bash "${OPD_CODE_DIR}/scripts/prepare_eval_model.sh" \
    --model-path "${MODEL_PATH}" \
    --python "$(command -v python)"
)"
MODEL_RUNTIME_PATH="${PREPARED_MODEL_PATH}"
if [[ -d "${PREPARED_MODEL_PATH}" ]]; then
  MODEL_ALIAS_ROOT="${MOPD_MODEL_ALIAS_ROOT:-${REMOTE_ROOT}/eval_model_aliases}"
  MODEL_ALIAS_PATH="${MODEL_ALIAS_ROOT}/${SAFE_MODEL_NAME}"
  mkdir -p "${MODEL_ALIAS_ROOT}"
  if [[ -L "${MODEL_ALIAS_PATH}" ]]; then
    if [[ "$(readlink "${MODEL_ALIAS_PATH}")" != "${PREPARED_MODEL_PATH}" ]]; then
      echo "Model alias points to a different checkpoint: ${MODEL_ALIAS_PATH}" >&2
      exit 2
    fi
  elif [[ -e "${MODEL_ALIAS_PATH}" ]]; then
    echo "Refusing to replace non-symlink model alias: ${MODEL_ALIAS_PATH}" >&2
    exit 2
  else
    ln -s "${PREPARED_MODEL_PATH}" "${MODEL_ALIAS_PATH}"
  fi
  MODEL_RUNTIME_PATH="${MODEL_ALIAS_PATH}"
fi
export MODEL_RUNTIME_PATH
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
    --model_path "${MODEL_RUNTIME_PATH}" \
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
  local artifact_dir="${EVAL_OUTPUT_DIR}/evalplus/${dataset}"
  echo "[paper-eval] evalplus ${dataset}"
  bash code_eval/scripts/run_evalplus.sh \
    "${dataset}" \
    "${MODEL_RUNTIME_PATH}" \
    "${EVALPLUS_GREEDY}" \
    "${EVALPLUS_TEMPERATURE}" \
    "${EVALPLUS_TOP_P}" \
    "${EVALPLUS_N}" \
    2>&1 | tee "${EVAL_OUTPUT_DIR}/evalplus_${dataset}.log"
  mkdir -p "${artifact_dir}"
  find "evalplus_results/${dataset}" -maxdepth 1 -type f \
    -name "*${SAFE_MODEL_NAME}*" -exec cp -p {} "${artifact_dir}/" \;
}

run_lcb_data_parallel() {
  [[ "${LCB_N}" == "8" ]] || { echo "LCB_N must be 8" >&2; return 2; }
  [[ "${LCB_TEMPERATURE}" == "1.0" && "${LCB_TOP_P}" == "1.0" ]] \
    || { echo "LCB sampling must use temperature=1.0 and top_p=1.0" >&2; return 2; }
  [[ "${LCB_MAX_TOKENS}" == "16384" ]] \
    || { echo "LCB_MAX_TOKENS must be 16384" >&2; return 2; }
  echo "[paper-eval] LiveCodeBench DP=4 releases=${LCB_RELEASE_VERSIONS}"
  PYTHON_BIN="$(command -v python)" \
    bash "${OPD_CODE_DIR}/eval/scripts/run_lcb_data_parallel.sh" \
      --model_path "${MODEL_RUNTIME_PATH}" \
      --checkpoint_path "${MODEL_PATH}" \
      --output_root "${EVAL_OUTPUT_DIR}" \
      --gopd_dir "${G_OPD_DIR}" \
      --releases "${LCB_RELEASE_VERSIONS}" \
      --gpus 4 \
      --shards_per_dataset 16
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
  run_lcb_data_parallel
fi

python - <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ["EVAL_OUTPUT_DIR"])
summary = {
    "datasets": [item for item in os.environ["PAPER_EVAL_DATASETS"].split(",") if item],
    "livecodebench_releases": [
        item for item in os.environ["LCB_RELEASE_VERSIONS"].split(",") if item
    ],
    "livecodebench_model_style": os.environ["LCB_MODEL_STYLE_NAME"],
    "livecodebench_chat_template_enable_thinking": False,
    "livecodebench_chat_template_tokenizer": os.environ[
        "LCB_CHAT_TEMPLATE_TOKENIZER"
    ],
    "livecodebench_runtime_dirs": {
        release: str(output_dir / "lcb_parallel")
        for release in os.environ["LCB_RELEASE_VERSIONS"].split(",")
        if release
    },
    "model_path": os.environ["MODEL_PATH"],
    "model_runtime_path": os.environ["MODEL_RUNTIME_PATH"],
    "output_dir": str(output_dir),
    "rollouts": {
        "math": int(os.environ["MATH_N"]),
        "evalplus": int(os.environ["EVALPLUS_N"]),
        "livecodebench": int(os.environ["LCB_N"]),
    },
    "logs": sorted(path.name for path in output_dir.glob("*.log")),
    "jsonl_outputs": sorted(path.name for path in output_dir.glob("*.jsonl")),
    "evalplus_artifacts": sorted(
        str(path.relative_to(output_dir))
        for path in (output_dir / "evalplus").glob("*/*")
        if path.is_file()
    ),
}
(output_dir / "paper_eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
