#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/setup_remote_training_env.sh

Run this on the remote host from an already-activated conda environment.
The script installs Python packages needed for MOPD training.

Environment knobs:
  VERL_RUNTIME_DIR=$CODE_DIR/third_party/verl
  HF_HOME=$CODE_DIR/hf_home
  HF_XET_HIGH_PERFORMANCE=1
  INSTALL_VERL_DEPS=1
  INSTALL_M2RL_IF_DEPS=1
  CHECK_MOPD_DATA=1
  PREPARE_M2RL_EVAL_DATA=0
  CHECK_M2RL_EVAL_DATA=0
  REQUIRE_M2RL_EVAL_DATA=0
  IF_VAL_SOURCE=/path/to/raw_if_val.parquet
  SCIENCE_VAL_SOURCE=/path/to/raw_science_val.parquet
  NEMOTRON_RL_SOURCE=/path/to/instruction_following.jsonl
  PULL_GIT_LFS_DATA=1
  DATA_DIR=$CODE_DIR/data/G-OPD-Training-Data
  FORCE_REINSTALL=0
  INSTALL_SGLANG=0
  USE_MEGATRON=0
  REQUIREMENT_FILE=$CODE_DIR/requirement.txt

The script does not clone G-OPD. Training imports verl from third_party/verl.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-${CODE_DIR}/third_party/verl}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-${CODE_DIR}/smoke_data}"
DATA_DIR="${DATA_DIR:-${CODE_DIR}/data/G-OPD-Training-Data}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/logs}"
HF_HOME="${HF_HOME:-${CODE_DIR}/hf_home}"
INSTALL_VERL_DEPS="${INSTALL_VERL_DEPS:-1}"
INSTALL_M2RL_IF_DEPS="${INSTALL_M2RL_IF_DEPS:-1}"
CHECK_MOPD_DATA="${CHECK_MOPD_DATA:-1}"
PREPARE_M2RL_EVAL_DATA="${PREPARE_M2RL_EVAL_DATA:-0}"
CHECK_M2RL_EVAL_DATA="${CHECK_M2RL_EVAL_DATA:-0}"
REQUIRE_M2RL_EVAL_DATA="${REQUIRE_M2RL_EVAL_DATA:-0}"
IF_VAL_SOURCE="${IF_VAL_SOURCE:-}"
SCIENCE_VAL_SOURCE="${SCIENCE_VAL_SOURCE:-}"
NEMOTRON_RL_SOURCE="${NEMOTRON_RL_SOURCE:-}"
M2RL_EVAL_MAX_SAMPLES="${M2RL_EVAL_MAX_SAMPLES:-}"
IF_VAL_MAX_SAMPLES="${IF_VAL_MAX_SAMPLES:-${M2RL_EVAL_MAX_SAMPLES}}"
SCIENCE_VAL_MAX_SAMPLES="${SCIENCE_VAL_MAX_SAMPLES:-${M2RL_EVAL_MAX_SAMPLES}}"
IF_EVAL_OUTPUT="${IF_EVAL_OUTPUT:-${CODE_DIR}/eval/domains/ifbench/data/IFBench_test.parquet}"
SCIENCE_EVAL_OUTPUT="${SCIENCE_EVAL_OUTPUT:-${CODE_DIR}/eval/domains/science/data/gpqa.parquet}"
PULL_GIT_LFS_DATA="${PULL_GIT_LFS_DATA:-1}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
INSTALL_SGLANG="${INSTALL_SGLANG:-0}"
USE_MEGATRON="${USE_MEGATRON:-0}"
REQUIREMENT_FILE="${REQUIREMENT_FILE:-${CODE_DIR}/requirement.txt}"
INSTALL_STAMP="${LOG_DIR}/.mopd_vendored_verl_install_done"
MOPD_REQUIRED_DATA_FILES=(
  "DeepMath-103K/train_filtered_level6.parquet"
  "Eurus/code_train.parquet"
  "IF/train.parquet"
  "Science/train.parquet"
)

if [[ ! -f "${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" ]]; then
  echo "Vendored verl runtime not found: ${VERL_RUNTIME_DIR}" >&2
  echo "Expected ${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${HF_HOME}" "${SMOKE_DATA_DIR}"

if [[ "${PULL_GIT_LFS_DATA}" == "1" ]]; then
  if git -C "${CODE_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git -C "${CODE_DIR}" lfs version >/dev/null 2>&1; then
      lfs_include=""
      for rel_path in "${MOPD_REQUIRED_DATA_FILES[@]}"; do
        data_path="data/G-OPD-Training-Data/${rel_path}"
        if [[ -z "${lfs_include}" ]]; then
          lfs_include="${data_path}"
        else
          lfs_include="${lfs_include},${data_path}"
        fi
      done
      git -C "${CODE_DIR}" lfs pull --include "${lfs_include}"
    else
      echo "git-lfs is unavailable; skipping automatic MOPD data pull." >&2
    fi
  fi
fi

python -m pip install --upgrade pip setuptools wheel

export PYTHONPATH="${CODE_DIR}:${VERL_RUNTIME_DIR}:${PYTHONPATH:-}"
export HF_HOME
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

if [[ "${INSTALL_VERL_DEPS}" == "1" ]]; then
  if [[ "${FORCE_REINSTALL}" == "1" || ! -f "${INSTALL_STAMP}" ]]; then
    cd "${VERL_RUNTIME_DIR}"
    USE_MEGATRON="${USE_MEGATRON}" USE_SGLANG="${INSTALL_SGLANG}" bash scripts/install_vllm_sglang_mcore.sh
    touch "${INSTALL_STAMP}"
    cd "${CODE_DIR}"
  else
    echo "Using existing verl dependency install stamp: ${INSTALL_STAMP}"
  fi
fi

[[ -f "${REQUIREMENT_FILE}" ]] || {
  echo "Missing requirement file: ${REQUIREMENT_FILE}" >&2
  exit 2
}
python -m pip install --upgrade -r "${REQUIREMENT_FILE}"

if [[ "${INSTALL_M2RL_IF_DEPS}" == "1" ]]; then
  if ! python - <<'PY' >/dev/null 2>&1; then
import importlib

for package in ("langdetect", "nltk", "immutabledict", "emoji", "syllapy", "unicodedata2"):
    importlib.import_module(package)
from verifiable_instructions import instructions_registry

required = {
    "length_constraints:number_words",
    "length_constraints:nth_paragraph_first_word",
    "last_word:last_word_answer",
}
missing = required.difference(instructions_registry.INSTRUCTION_DICT)
if missing:
    raise RuntimeError(f"verifiable_instructions missing required ids: {sorted(missing)}")
PY
    echo "IF/science dependencies are missing after installing ${REQUIREMENT_FILE}." >&2
    exit 2
  fi
fi

if [[ "${PREPARE_M2RL_EVAL_DATA}" == "1" || -n "${IF_VAL_SOURCE}" || -n "${SCIENCE_VAL_SOURCE}" || -n "${NEMOTRON_RL_SOURCE}" ]]; then
  PYTHON_BIN=python \
  IF_VAL_SOURCE="${IF_VAL_SOURCE}" \
  SCIENCE_VAL_SOURCE="${SCIENCE_VAL_SOURCE}" \
  NEMOTRON_RL_SOURCE="${NEMOTRON_RL_SOURCE}" \
  M2RL_EVAL_MAX_SAMPLES="${M2RL_EVAL_MAX_SAMPLES}" \
  IF_VAL_MAX_SAMPLES="${IF_VAL_MAX_SAMPLES}" \
  SCIENCE_VAL_MAX_SAMPLES="${SCIENCE_VAL_MAX_SAMPLES}" \
  IF_EVAL_OUTPUT="${IF_EVAL_OUTPUT}" \
  SCIENCE_EVAL_OUTPUT="${SCIENCE_EVAL_OUTPUT}" \
  REQUIRE_M2RL_EVAL_DATA="${REQUIRE_M2RL_EVAL_DATA}" \
    bash "${CODE_DIR}/scripts/prepare_m2rl_eval_data.sh"
elif [[ "${CHECK_M2RL_EVAL_DATA}" == "1" ]]; then
  PYTHON_BIN=python \
  IF_EVAL_OUTPUT="${IF_EVAL_OUTPUT}" \
  SCIENCE_EVAL_OUTPUT="${SCIENCE_EVAL_OUTPUT}" \
  REQUIRE_M2RL_EVAL_DATA=1 \
    bash "${CODE_DIR}/scripts/prepare_m2rl_eval_data.sh"
fi

python -m mopd_verl.smoke_data "${SMOKE_DATA_DIR}"
python -m mopd_verl.prepare_data inspect "${SMOKE_DATA_DIR}/train.parquet"

if [[ "${CHECK_MOPD_DATA}" == "1" ]]; then
  DATA_DIR="${DATA_DIR}" python - <<'PY'
from pathlib import Path
import os

import pyarrow.parquet as pq

data_dir = Path(os.environ["DATA_DIR"])
required_files = [
    "DeepMath-103K/train_filtered_level6.parquet",
    "Eurus/code_train.parquet",
    "IF/train.parquet",
    "Science/train.parquet",
]
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
    print(f"mopd_data {path} rows={parquet_file.metadata.num_rows}")

if missing or invalid:
    for item in missing:
        print(f"missing MOPD data: {item}")
    for item in invalid:
        print(f"invalid MOPD data: {item}")
    raise SystemExit(1)
PY
fi

cat > "${LOG_DIR}/env.sh" <<EOF
export CODE_DIR="${CODE_DIR}"
export VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR}"
export SMOKE_DATA_DIR="${SMOKE_DATA_DIR}"
export DATA_DIR="${DATA_DIR}"
export LOG_DIR="${LOG_DIR}"
export HF_HOME="${HF_HOME}"
export HF_XET_HIGH_PERFORMANCE="\${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_DATASETS_CACHE="\${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export WANDB_MODE="\${WANDB_MODE:-disabled}"
export INSTALL_M2RL_IF_DEPS="${INSTALL_M2RL_IF_DEPS}"
export CHECK_MOPD_DATA="${CHECK_MOPD_DATA}"
export CHECK_M2RL_EVAL_DATA="${CHECK_M2RL_EVAL_DATA}"
export IF_EVAL_OUTPUT="${IF_EVAL_OUTPUT}"
export SCIENCE_EVAL_OUTPUT="${SCIENCE_EVAL_OUTPUT}"
export PULL_GIT_LFS_DATA="${PULL_GIT_LFS_DATA}"
export PYTHONPATH="${CODE_DIR}:${VERL_RUNTIME_DIR}:\${PYTHONPATH:-}"
EOF

INSTALL_VERL_DEPS="${INSTALL_VERL_DEPS}" python - <<'PY'
import importlib
import os
import sys

packages = ["yaml", "transformers", "click", "pandas", "pyarrow"]
if os.environ["INSTALL_VERL_DEPS"] == "1":
    packages.extend(["torch", "vllm", "ray", "verl"])

print("python", sys.version.split()[0], sys.executable)
for package in packages:
    module = importlib.import_module(package)
    print(package, getattr(module, "__version__", "unknown"))

try:
    registry = importlib.import_module("verifiable_instructions.instructions_registry")
except ImportError:
    print("verifiable_instructions unavailable")
else:
    required = [
        "length_constraints:number_words",
        "length_constraints:nth_paragraph_first_word",
        "last_word:last_word_answer",
    ]
    missing = [item for item in required if item not in registry.INSTRUCTION_DICT]
    if missing:
        raise RuntimeError(f"verifiable_instructions missing required ids: {missing}")
    print("verifiable_instructions", "ok")
PY

echo "Environment ready."
echo "CODE_DIR=${CODE_DIR}"
echo "VERL_RUNTIME_DIR=${VERL_RUNTIME_DIR}"
echo "ENV_FILE=${LOG_DIR}/env.sh"
