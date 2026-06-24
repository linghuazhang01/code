#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_OUTPUT_DIR="${GENERAL_REASONER_TRAIN_OUTPUT_DIR:-${CODE_DIR}/data/GeneralReasoner/WebInstructVerified}"
EVAL_OUTPUT_DIR="${GENERAL_REASONER_EVAL_OUTPUT_DIR:-${CODE_DIR}/eval/domains/greasoner/data/WebInstructVerified}"

export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"

mkdir -p "${TRAIN_OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

if [[ "${GENERAL_REASONER_FROM_HF:-1}" == "1" ]]; then
  DATASET_NAME="${GENERAL_REASONER_DATASET_NAME:-TIGER-Lab/WebInstruct-verified}"
  TEST_MAX_SAMPLES="${GENERAL_REASONER_TEST_MAX_SAMPLES:-100}"
  python -m mopd_verl.prepare_data prepare-general-reasoner-hf \
    --dataset-name "${DATASET_NAME}" \
    --output-dir "${TRAIN_OUTPUT_DIR}" \
    --test-max-samples "${TEST_MAX_SAMPLES}"
  if [[ -f "${TRAIN_OUTPUT_DIR}/test.parquet" ]]; then
    cp -f "${TRAIN_OUTPUT_DIR}/test.parquet" "${EVAL_OUTPUT_DIR}/test.parquet"
  fi
else
  INPUT_DIR="${GENERAL_REASONER_INPUT_DIR:-${CODE_DIR}/data/GeneralReasoner/WebInstructVerified/raw}"
  TRAIN_INPUT="${GENERAL_REASONER_TRAIN_INPUT:-${INPUT_DIR}/train.parquet}"
  TEST_INPUT="${GENERAL_REASONER_TEST_INPUT:-${INPUT_DIR}/test.parquet}"
  python -m mopd_verl.prepare_data prepare-general-reasoner \
    --input "${TRAIN_INPUT}" \
    --output "${TRAIN_OUTPUT_DIR}/train.parquet" \
    --split train
  python -m mopd_verl.prepare_data prepare-general-reasoner \
    --input "${TEST_INPUT}" \
    --output "${EVAL_OUTPUT_DIR}/test.parquet" \
    --split test \
    --max-samples "${GENERAL_REASONER_TEST_MAX_SAMPLES:-100}"
fi

echo "[general-reasoner-data] train: ${TRAIN_OUTPUT_DIR}/train.parquet"
echo "[general-reasoner-data] eval: ${EVAL_OUTPUT_DIR}/test.parquet"
