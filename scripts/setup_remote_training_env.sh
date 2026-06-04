#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/setup_remote_training_env.sh

Run this on the remote host from the synced OPD-code checkout.

Environment knobs:
  CONDA_ROOT=<auto-detected, usually $HOME/miniconda3>
  ENV_NAME=mopd-verl
  VERL_RUNTIME_DIR=$CODE_DIR/third_party/verl
  HF_HOME=$CODE_DIR/hf_home
  HF_XET_HIGH_PERFORMANCE=1
  INSTALL_VERL_DEPS=1
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

if [[ -z "${CONDA_ROOT:-}" ]]; then
  for candidate in "${HOME}/miniconda3" "/root/miniconda3" "/opt/conda"; do
    if [[ -x "${candidate}/bin/conda" ]]; then
      CONDA_ROOT="${candidate}"
      break
    fi
  done
fi

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
ENV_NAME="${ENV_NAME:-mopd-verl}"
VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-${CODE_DIR}/third_party/verl}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-${CODE_DIR}/smoke_data}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/logs}"
HF_HOME="${HF_HOME:-${CODE_DIR}/hf_home}"
INSTALL_VERL_DEPS="${INSTALL_VERL_DEPS:-1}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
INSTALL_SGLANG="${INSTALL_SGLANG:-0}"
USE_MEGATRON="${USE_MEGATRON:-0}"
INSTALL_STAMP="${LOG_DIR}/.mopd_vendored_verl_install_done"

if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
  echo "Missing conda at ${CONDA_ROOT}/bin/conda" >&2
  echo "Set CONDA_ROOT to the conda installation directory." >&2
  exit 1
fi

if [[ ! -f "${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" ]]; then
  echo "Vendored verl runtime not found: ${VERL_RUNTIME_DIR}" >&2
  echo "Expected ${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${HF_HOME}" "${SMOKE_DATA_DIR}"
export PATH="${CONDA_ROOT}/bin:${PATH}"
# shellcheck disable=SC1090
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10 pip
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel

export PYTHONPATH="${CODE_DIR}:${VERL_RUNTIME_DIR}:${PYTHONPATH:-}"
export HF_HOME
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
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
  tensorboard \
  hf_xet \
  modelscope

python -m mopd_verl.smoke_data "${SMOKE_DATA_DIR}"
python -m mopd_verl.prepare_data inspect "${SMOKE_DATA_DIR}/train.parquet"

cat > "${LOG_DIR}/env.sh" <<EOF
export CODE_DIR="${CODE_DIR}"
export CONDA_ROOT="${CONDA_ROOT}"
export ENV_NAME="${ENV_NAME}"
export VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR}"
export SMOKE_DATA_DIR="${SMOKE_DATA_DIR}"
export LOG_DIR="${LOG_DIR}"
export HF_HOME="${HF_HOME}"
export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_XET_HIGH_PERFORMANCE="\${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_DATASETS_CACHE="\${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export WANDB_MODE="\${WANDB_MODE:-disabled}"
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
PY

echo "Environment ready."
echo "CODE_DIR=${CODE_DIR}"
echo "CONDA_ENV=${ENV_NAME}"
echo "VERL_RUNTIME_DIR=${VERL_RUNTIME_DIR}"
echo "ENV_FILE=${LOG_DIR}/env.sh"
