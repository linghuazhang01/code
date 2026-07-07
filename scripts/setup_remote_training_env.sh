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
  PULL_GIT_LFS_DATA=1
  DATA_DIR=$CODE_DIR/data/G-OPD-Training-Data
  FORCE_REINSTALL=0
  INSTALL_SGLANG=0
  USE_MEGATRON=0

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
PULL_GIT_LFS_DATA="${PULL_GIT_LFS_DATA:-1}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
INSTALL_SGLANG="${INSTALL_SGLANG:-0}"
USE_MEGATRON="${USE_MEGATRON:-0}"
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

python -m pip install --upgrade \
  "transformers[hf_xet]==4.51.3" \
  "tokenizers>=0.21.1,<0.22" \
  "huggingface_hub>=0.30.0,<1.0" \
  pyyaml \
  pandas \
  pyarrow \
  "tensorboard==2.20.0" \
  "protobuf<5.0,>=3.20.3" \
  "opentelemetry-exporter-prometheus==0.47b0" \
  hf_xet \
  modelscope

if [[ "${INSTALL_M2RL_IF_DEPS}" == "1" ]]; then
  if ! python - <<'PY' >/dev/null 2>&1; then
import importlib

for package in ("langdetect", "nltk", "immutabledict", "emoji", "syllapy", "unicodedata2"):
    importlib.import_module(package)
PY
    python -m pip install --no-cache-dir \
      langdetect \
      nltk \
      immutabledict \
      emoji \
      syllapy \
      unicodedata2
  fi

  if ! python - <<'PY' >/dev/null 2>&1; then
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
    python -m pip install --no-cache-dir \
      git+https://github.com/abukharin-nv/verifiable-instructions.git
  fi
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
export PULL_GIT_LFS_DATA="${PULL_GIT_LFS_DATA}"
export PYTHONPATH="${CODE_DIR}:${VERL_RUNTIME_DIR}:\${PYTHONPATH:-}"
EOF

python - <<'PY'
import importlib
import sys

packages = ["yaml", "torch", "transformers", "vllm", "ray", "click", "pandas", "pyarrow", "verl"]
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
