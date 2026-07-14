#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/download_training_assets.sh

Download and validate the data/model assets for the current Qwen3-4B student
and Qwen3-30B-A3B-Instruct-2507 teacher math/code/IF/science profiles.

Default assets:
  data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet
  data/G-OPD-Training-Data/Eurus/code_train.parquet
  data/G-OPD-Training-Data/IF/train.parquet
  data/G-OPD-Training-Data/Science/train.parquet
  ../models/Qwen3-4B
  ../models/Qwen3-30B-A3B-Instruct-2507

Environment knobs:
  DATA_DIR=$CODE_DIR/data/G-OPD-Training-Data
  DATASET_ID=icemoon28/MOPD-Training-Data
  DATASET_REVISION=main
  EVAL_DATA_DIR=$CODE_DIR/data/eval_data
  MODEL_ROOT=<parent of OPD-code>/models
  PYTHON_BIN=<auto-detected python or python3>
  MODEL_BACKEND=huggingface
  DOWNLOAD_DATA=1
  DOWNLOAD_MODELS=1
  REQUIRE_MATH_CODE_TRAIN_DATA=1
  REQUIRE_4DOMAIN_TRAIN_DATA=1
  REQUIRE_MODELS=1
  DOWNLOAD_BASE_4B=$DOWNLOAD_MODELS
  DOWNLOAD_QWEN30B=$DOWNLOAD_MODELS
  DOWNLOAD_LCB=0
  MIN_FREE_GB=0
  PREPARE_M2RL_EVAL_DATA=0
  CHECK_M2RL_EVAL_DATA=0
  REQUIRE_M2RL_EVAL_DATA=0
  IF_VAL_SOURCE=/path/to/raw_if_val.parquet
  SCIENCE_VAL_SOURCE=/path/to/raw_science_val.parquet
  NEMOTRON_RL_SOURCE=/path/to/instruction_following.jsonl

Use DOWNLOAD_MODELS=0 REQUIRE_MODELS=1 to verify already-downloaded models.
Set REQUIRE_M2RL_EVAL_DATA=1 when IF/science validation parquet files should
be mandatory for the selected config.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${CODE_DIR}/data/G-OPD-Training-Data}"
DATASET_ID="${DATASET_ID:-icemoon28/MOPD-Training-Data}"
DATASET_REVISION="${DATASET_REVISION:-main}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${CODE_DIR}/data/eval_data}"
MODEL_ROOT="${MODEL_ROOT:-$(cd "${CODE_DIR}/.." && pwd)/models}"
PYTHON_BIN="${PYTHON_BIN:-}"
MODEL_BACKEND="${MODEL_BACKEND:-huggingface}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
REQUIRE_MATH_CODE_TRAIN_DATA="${REQUIRE_MATH_CODE_TRAIN_DATA:-1}"
REQUIRE_4DOMAIN_TRAIN_DATA="${REQUIRE_4DOMAIN_TRAIN_DATA:-1}"
REQUIRE_MODELS="${REQUIRE_MODELS:-1}"
DOWNLOAD_BASE_4B="${DOWNLOAD_BASE_4B:-${DOWNLOAD_MODELS}}"
DOWNLOAD_QWEN30B="${DOWNLOAD_QWEN30B:-${DOWNLOAD_MODELS}}"
REQUIRE_BASE_4B="${REQUIRE_BASE_4B:-${REQUIRE_MODELS}}"
REQUIRE_QWEN30B="${REQUIRE_QWEN30B:-${REQUIRE_MODELS}}"
DOWNLOAD_LCB="${DOWNLOAD_LCB:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-0}"
PREPARE_M2RL_EVAL_DATA="${PREPARE_M2RL_EVAL_DATA:-0}"
CHECK_M2RL_EVAL_DATA="${CHECK_M2RL_EVAL_DATA:-0}"
REQUIRE_M2RL_EVAL_DATA="${REQUIRE_M2RL_EVAL_DATA:-0}"
IF_VAL_SOURCE="${IF_VAL_SOURCE:-}"
SCIENCE_VAL_SOURCE="${SCIENCE_VAL_SOURCE:-}"
NEMOTRON_RL_SOURCE="${NEMOTRON_RL_SOURCE:-}"
M2RL_EVAL_MAX_SAMPLES="${M2RL_EVAL_MAX_SAMPLES:-}"
IF_VAL_MAX_SAMPLES="${IF_VAL_MAX_SAMPLES:-${M2RL_EVAL_MAX_SAMPLES}}"
SCIENCE_VAL_MAX_SAMPLES="${SCIENCE_VAL_MAX_SAMPLES:-${M2RL_EVAL_MAX_SAMPLES}}"
IF_EVAL_OUTPUT="${IF_EVAL_OUTPUT:-${EVAL_DATA_DIR}/ifbench/IFBench_test.parquet}"
SCIENCE_EVAL_OUTPUT="${SCIENCE_EVAL_OUTPUT:-${EVAL_DATA_DIR}/science/gpqa.parquet}"
BASE_4B_MODEL_ID="${BASE_4B_MODEL_ID:-Qwen/Qwen3-4B}"
BASE_4B_DIR_NAME="${BASE_4B_DIR_NAME:-Qwen3-4B}"
QWEN30B_MODEL_ID="${QWEN30B_MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
QWEN30B_DIR_NAME="${QWEN30B_DIR_NAME:-Qwen3-30B-A3B-Instruct-2507}"

export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "python or python3 is required." >&2
    exit 1
  fi
fi

ensure_parquet_support() {
  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1 || "${PYTHON_BIN}" -m pip install "pandas>=2.0" "pyarrow>=19.0.0"
import pandas
import pyarrow
PY
}

validate_training_data() {
  ensure_parquet_support
  DATA_DIR="${DATA_DIR}" \
  REQUIRE_4DOMAIN_TRAIN_DATA="${REQUIRE_4DOMAIN_TRAIN_DATA}" \
    "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

import pyarrow.parquet as pq

data_dir = Path(os.environ["DATA_DIR"])
required_files = [
    "DeepMath-103K/train_filtered_level6.parquet",
    "Eurus/code_train.parquet",
]
if os.environ["REQUIRE_4DOMAIN_TRAIN_DATA"] == "1":
    required_files.extend(["IF/train.parquet", "Science/train.parquet"])
missing = []
invalid = []
for rel_path in required_files:
    path = data_dir / rel_path
    if not path.exists():
        missing.append(str(path))
        continue
    with path.open("rb") as handle:
        header = handle.read(96)
    if header.startswith(b"version https://git-lfs.github.com/spec"):
        invalid.append(f"{path} is still a Git LFS pointer")
        continue
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:  # noqa: BLE001 - report all data readiness failures.
        invalid.append(f"{path} is not readable as parquet: {exc}")
        continue
    print(f"training_data {path} rows={parquet_file.metadata.num_rows}")

if missing or invalid:
    for item in missing:
        print(f"missing train data: {item}")
    for item in invalid:
        print(f"invalid train data: {item}")
    raise SystemExit(1)
PY
}

prepare_or_check_m2rl_eval() {
  local should_run=0
  if [[ "${PREPARE_M2RL_EVAL_DATA}" == "1" ]]; then
    should_run=1
  fi
  if [[ -n "${IF_VAL_SOURCE}" || -n "${SCIENCE_VAL_SOURCE}" || -n "${NEMOTRON_RL_SOURCE}" ]]; then
    should_run=1
  fi
  if [[ "${CHECK_M2RL_EVAL_DATA}" == "1" || "${REQUIRE_M2RL_EVAL_DATA}" == "1" ]]; then
    should_run=1
  fi

  if [[ "${should_run}" != "1" ]]; then
    echo "IF/science eval preparation skipped."
    echo "Set PREPARE_M2RL_EVAL_DATA=1 or REQUIRE_M2RL_EVAL_DATA=1 when needed."
    return
  fi

  IF_VAL_SOURCE="${IF_VAL_SOURCE}" \
  SCIENCE_VAL_SOURCE="${SCIENCE_VAL_SOURCE}" \
  NEMOTRON_RL_SOURCE="${NEMOTRON_RL_SOURCE}" \
  M2RL_EVAL_MAX_SAMPLES="${M2RL_EVAL_MAX_SAMPLES}" \
  IF_VAL_MAX_SAMPLES="${IF_VAL_MAX_SAMPLES}" \
  SCIENCE_VAL_MAX_SAMPLES="${SCIENCE_VAL_MAX_SAMPLES}" \
  IF_EVAL_OUTPUT="${IF_EVAL_OUTPUT}" \
  SCIENCE_EVAL_OUTPUT="${SCIENCE_EVAL_OUTPUT}" \
  REQUIRE_M2RL_EVAL_DATA="${REQUIRE_M2RL_EVAL_DATA}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/prepare_m2rl_eval_data.sh"
}

prepare_models() {
  if [[ "${DOWNLOAD_MODELS}" != "1" && "${REQUIRE_MODELS}" != "1" ]]; then
    echo "Model download and validation skipped."
    return
  fi

  MODEL_ROOT="${MODEL_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  MODEL_BACKEND="${MODEL_BACKEND}" \
  DOWNLOAD_STUDENT=0 \
  REQUIRE_STUDENT=0 \
  DOWNLOAD_BASE_4B="${DOWNLOAD_BASE_4B}" \
  REQUIRE_BASE_4B="${REQUIRE_BASE_4B}" \
  BASE_4B_MODEL_ID="${BASE_4B_MODEL_ID}" \
  BASE_4B_DIR_NAME="${BASE_4B_DIR_NAME}" \
  DOWNLOAD_REASONING_TEACHER=0 \
  REQUIRE_REASONING_TEACHER=0 \
    bash "${SCRIPT_DIR}/download_mopd_models.sh"

  MODEL_ROOT="${MODEL_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  MODEL_BACKEND="${MODEL_BACKEND}" \
  QWEN30B_MODEL_ID="${QWEN30B_MODEL_ID}" \
  QWEN30B_DIR_NAME="${QWEN30B_DIR_NAME}" \
  DOWNLOAD_QWEN30B="${DOWNLOAD_QWEN30B}" \
  REQUIRE_QWEN30B="${REQUIRE_QWEN30B}" \
  MIN_FREE_GB="${MIN_FREE_GB}" \
    bash "${SCRIPT_DIR}/download_qwen30b_teacher.sh"
}

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
  DATA_DIR="${DATA_DIR}" \
  DATASET_ID="${DATASET_ID}" \
  DATASET_REVISION="${DATASET_REVISION}" \
  EVAL_DATA_DIR="${EVAL_DATA_DIR}" \
  DOWNLOAD_LCB="${DOWNLOAD_LCB}" \
  REQUIRE_4DOMAIN_TRAIN_DATA="${REQUIRE_4DOMAIN_TRAIN_DATA}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/download_mopd_data.sh"
else
  echo "Training data download skipped: ${DATA_DIR}"
fi

if [[ "${REQUIRE_MATH_CODE_TRAIN_DATA}" == "1" || "${REQUIRE_4DOMAIN_TRAIN_DATA}" == "1" ]]; then
  validate_training_data
fi

prepare_or_check_m2rl_eval
prepare_models

cat <<EOF
Qwen3-30B-A3B-Instruct-2507 training assets ready.
  Data: ${DATA_DIR}
  Math train: ${DATA_DIR}/DeepMath-103K/train_filtered_level6.parquet
  Code train: ${DATA_DIR}/Eurus/code_train.parquet
  IF train: ${DATA_DIR}/IF/train.parquet
  Science train: ${DATA_DIR}/Science/train.parquet
  Student: ${MODEL_ROOT}/${BASE_4B_DIR_NAME}
  Teacher math/code/IF/science: ${MODEL_ROOT}/${QWEN30B_DIR_NAME}
EOF
