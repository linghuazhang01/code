#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="$(cd "${EVAL_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${1:-}}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required." >&2
  exit 2
fi

MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
SAFE_MODEL_NAME="${MODEL_NAME//[^A-Za-z0-9_.-]/_}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/data/eval_data/results/standard_ood/${SAFE_MODEL_NAME}_${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
GENERATION_SEED="${GENERATION_SEED:-42}"

DATA_FILE="${CODE_DIR}/data/eval_data/science/MMLU-Pro/subsets/openprm_style_500_seed42/test.parquet"
MANIFEST_FILE="${CODE_DIR}/data/eval_data/science/MMLU-Pro/subsets/openprm_style_500_seed42/manifest.json"

for flag in DRY_RUN RESUME; do
  [[ "${!flag}" == "0" || "${!flag}" == "1" ]] || {
    echo "${flag} must be 0 or 1: ${!flag}" >&2
    exit 2
  }
done
[[ "${SAMPLE_OFFSET}" =~ ^[0-9]+$ ]] || {
  echo "SAMPLE_OFFSET must be a non-negative integer: ${SAMPLE_OFFSET}" >&2
  exit 2
}
if [[ -n "${MAX_SAMPLES}" && ! "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_SAMPLES must be a positive integer when provided: ${MAX_SAMPLES}" >&2
  exit 2
fi
[[ "${GENERATION_SEED}" =~ ^[0-9]+$ ]] || {
  echo "GENERATION_SEED must be a non-negative integer: ${GENERATION_SEED}" >&2
  exit 2
}

PINNED_VALIDATION_JSON="$(
  PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    -m eval.domains.science.pinned_mmlupro \
    --data-file "${DATA_FILE}" \
    --manifest-file "${MANIFEST_FILE}"
)"

COMMAND=(
  "${CODE_DIR}/eval/scripts/run_official_eval.sh"
  --domains science
  --datasets mmlupro_500_seed42
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --tensor-parallel-size 1
  --gpu-memory-utilization 0.85
  --max-model-len 18432
  --max-tokens 16384
  --temperature 1.0
  --top-p 1.0
  --num-samples 4
  --seed "${GENERATION_SEED}"
  --enable-thinking false
)
[[ "${SAMPLE_OFFSET}" == "0" ]] || COMMAND+=(--sample-offset "${SAMPLE_OFFSET}")
[[ -z "${MAX_SAMPLES}" ]] || COMMAND+=(--max-samples "${MAX_SAMPLES}")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[standard-ood] CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ "${RESUME}" == "1" && -f "${OUTPUT_DIR}/SUCCESS" ]]; then
  echo "[standard-ood] already complete, skipping: ${OUTPUT_DIR}"
  exit 0
fi
if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty output directory: ${OUTPUT_DIR}" >&2
  echo "Use a new OUTPUT_DIR; incomplete MMLU-Pro runs are not resumable." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" - \
  "${OUTPUT_DIR}/standard_ood_manifest.json" \
  "${MODEL_PATH}" \
  "${DATA_FILE}" \
  "${PINNED_VALIDATION_JSON}" \
  "${SAMPLE_OFFSET}" \
  "${MAX_SAMPLES}" \
  "${GENERATION_SEED}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


validation = json.loads(sys.argv[4])
payload = {
    "benchmark": "MMLU-Pro-500",
    "dataset_key": "mmlupro_500_seed42",
    "model_path": sys.argv[2],
    "data_file": sys.argv[3],
    "data_sha256": validation["data_sha256"],
    "selected_ids_sha256": validation["selected_ids_sha256"],
    "protocol": {
        "questions": 500,
        "rollouts_per_question": 4,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": int(sys.argv[7]),
        "max_tokens": 16384,
        "max_model_len": 18432,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "enable_thinking": False,
        "sample_offset": int(sys.argv[5]),
        "max_samples": int(sys.argv[6]) if sys.argv[6] else None,
    },
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"

"${COMMAND[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
touch "${OUTPUT_DIR}/SUCCESS"
echo "[standard-ood] complete: ${OUTPUT_DIR}"
