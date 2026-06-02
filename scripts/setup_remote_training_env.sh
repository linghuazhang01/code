#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/opd_mopd}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
ENV_NAME="${ENV_NAME:-mopd-verl}"
G_OPD_REPO="${G_OPD_REPO:-https://github.com/RUCBM/G-OPD.git}"
G_OPD_DIR="${G_OPD_DIR:-${REMOTE_ROOT}/G-OPD}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-${REMOTE_ROOT}/smoke_data}"
LOG_DIR="${LOG_DIR:-${REMOTE_ROOT}/logs}"
HF_HOME="${HF_HOME:-${REMOTE_ROOT}/hf_home}"
INSTALL_SGLANG="${INSTALL_SGLANG:-0}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
UPDATE_GOPD="${UPDATE_GOPD:-0}"
SKIP_FLASHINFER="${SKIP_FLASHINFER:-1}"

if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
  echo "Missing conda at ${CONDA_ROOT}/bin/conda" >&2
  exit 1
fi

mkdir -p "${REMOTE_ROOT}" "${LOG_DIR}" "${HF_HOME}"
export PATH="${CONDA_ROOT}/bin:${PATH}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10 pip
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel

if [[ ! -d "${G_OPD_DIR}/verl" ]]; then
  git clone --depth 1 "${G_OPD_REPO}" "${G_OPD_DIR}"
elif [[ "${UPDATE_GOPD}" == "1" && -d "${G_OPD_DIR}/.git" ]]; then
  git -C "${G_OPD_DIR}" pull --ff-only
else
  echo "Using existing G-OPD checkout at ${G_OPD_DIR}"
fi

export PYTHONPATH="${CODE_DIR}:${G_OPD_DIR}/verl:${PYTHONPATH:-}"
python "${CODE_DIR}/scripts/apply_gopd_audit_patch.py" "${G_OPD_DIR}"

INSTALL_SCRIPT="${G_OPD_DIR}/verl/scripts/install_vllm_sglang_mcore.sh"
if [[ "${SKIP_FLASHINFER}" == "1" && -f "${INSTALL_SCRIPT}" ]]; then
  python - "${INSTALL_SCRIPT}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = """# Install flashinfer-0.2.2.post1+cu124 (cxx11abi=False)
# vllm-0.8.3 does not support flashinfer>=0.2.3
# see https://github.com/vllm-project/vllm/pull/15777
wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl && \\
    pip install --no-cache-dir flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl"""
new = """# FlashInfer is optional for the single-step smoke path. GitHub wheel
# downloads are often slow on rented instances, so skip it by default.
echo "Skipping FlashInfer wheel download; set SKIP_FLASHINFER=0 and rerun if production rollout requires it" """
if old in text and "Skipping FlashInfer wheel download" not in text:
    path.write_text(text.replace(old, new))
PY
fi

INSTALL_STAMP="${REMOTE_ROOT}/.gopd_verl_install_done"
if [[ "${FORCE_REINSTALL}" == "1" || ! -f "${INSTALL_STAMP}" ]]; then
  cd "${G_OPD_DIR}/verl"
  USE_MEGATRON=0 USE_SGLANG="${INSTALL_SGLANG}" bash scripts/install_vllm_sglang_mcore.sh
  touch "${INSTALL_STAMP}"
fi

python -m pip install --upgrade \
  "transformers[hf_xet]==4.51.3" \
  "tokenizers>=0.21.1,<0.22" \
  "huggingface-hub>=0.30.0,<1.0" \
  "ray[default]==2.46.0" \
  "numpy<2.0.0" \
  "opentelemetry-api==1.26.0" \
  "opentelemetry-sdk==1.26.0" \
  "opentelemetry-exporter-otlp==1.26.0" \
  "opentelemetry-exporter-otlp-proto-grpc==1.26.0" \
  "opentelemetry-exporter-otlp-proto-http==1.26.0" \
  "click<8.2" \
  math-verify \
  pyyaml \
  pandas \
  pyarrow \
  tensorboard \
  huggingface_hub \
  hf-transfer

python -m mopd_verl.smoke_data "${SMOKE_DATA_DIR}"
python -m mopd_verl.prepare_data inspect "${SMOKE_DATA_DIR}/train.parquet"

cat > "${REMOTE_ROOT}/env.sh" <<EOF
export REMOTE_ROOT="${REMOTE_ROOT}"
export CONDA_ROOT="${CONDA_ROOT}"
export ENV_NAME="${ENV_NAME}"
export G_OPD_DIR="${G_OPD_DIR}"
export OPD_CODE_DIR="${CODE_DIR}"
export SMOKE_DATA_DIR="${SMOKE_DATA_DIR}"
export LOG_DIR="${LOG_DIR}"
export HF_HOME="${HF_HOME}"
export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export WANDB_MODE="\${WANDB_MODE:-disabled}"
export PYTHONPATH="${CODE_DIR}:${G_OPD_DIR}/verl:\${PYTHONPATH:-}"
EOF

python - <<'PY'
import importlib
import sys

packages = ["torch", "transformers", "vllm", "ray", "click", "pandas", "pyarrow"]
print("python", sys.version.split()[0], sys.executable)
for package in packages:
    module = importlib.import_module(package)
    print(package, getattr(module, "__version__", "unknown"))
PY

echo "Environment ready: ${REMOTE_ROOT}"
