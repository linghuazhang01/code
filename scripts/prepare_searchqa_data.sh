#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUT_DIR="${SEARCHQA_INPUT_DIR:-${CODE_DIR}/data/SearchQA/raw}"
OUTPUT_DIR="${SEARCHQA_OUTPUT_DIR:-${CODE_DIR}/data/SearchQA}"
TRAIN_INPUT="${SEARCHQA_TRAIN_INPUT:-${INPUT_DIR}/train.parquet}"
TEST_INPUT="${SEARCHQA_TEST_INPUT:-${INPUT_DIR}/test.parquet}"

export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

python -m mopd_verl.prepare_data prepare-searchqa \
  --input "${TRAIN_INPUT}" \
  --output "${OUTPUT_DIR}/train.parquet" \
  --split train

python -m mopd_verl.prepare_data prepare-searchqa \
  --input "${TEST_INPUT}" \
  --output "${OUTPUT_DIR}/test.parquet" \
  --split test

echo "[searchqa-data] wrote ${OUTPUT_DIR}/train.parquet and ${OUTPUT_DIR}/test.parquet"
