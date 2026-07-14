#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/setup_training_env.sh

Create or update the MOPD Conda environment exclusively from ENV_FILE,
verify the training runtime, and optionally download training assets.

Common flow:
  bash scripts/setup_training_env.sh
  source logs/activate_training_env.sh

Blackwell sm_120 flow:
  ENV_NAME=mopd-verl-blackwell \
  ENV_FILE="$(pwd)/environment.blackwell.yml" \
    bash scripts/setup_training_env.sh

Environment knobs:
  CONDA_ROOT=$HOME/miniconda3
  ENV_NAME=mopd-verl
  ENV_FILE="$(pwd)/environment.yml"
  INSTALL_MINICONDA=1
  UPDATE_ENV=1
  INSTALL_GIT_LFS=1
  PULL_GIT_LFS_DATA=0
  REGISTER_KERNEL=0
  KERNEL_NAME=$ENV_NAME
  KERNEL_DISPLAY_NAME=MOPD ($ENV_NAME)
  DOWNLOAD_ASSETS=0

Dependency versions and package sources must be declared only in ENV_FILE.
The default profile targets Linux x86_64 with CUDA 12-compatible NVIDIA
drivers. Use environment.blackwell.yml for CUDA 12.8 / PyTorch 2.8 Blackwell
GPUs. Asset variables are forwarded to scripts/download_training_assets.sh
when DOWNLOAD_ASSETS=1.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ENV_NAME:-mopd-verl}"
ENV_FILE="${ENV_FILE:-${CODE_DIR}/environment.yml}"
if [[ "${ENV_FILE}" != /* ]]; then
  ENV_FILE="$(cd "$(dirname "${ENV_FILE}")" && pwd)/$(basename "${ENV_FILE}")"
fi
INSTALL_MINICONDA="${INSTALL_MINICONDA:-1}"
UPDATE_ENV="${UPDATE_ENV:-1}"
INSTALL_GIT_LFS="${INSTALL_GIT_LFS:-1}"
PULL_GIT_LFS_DATA="${PULL_GIT_LFS_DATA:-0}"
REGISTER_KERNEL="${REGISTER_KERNEL:-0}"
KERNEL_NAME="${KERNEL_NAME:-${ENV_NAME}}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-MOPD (${ENV_NAME})}"
DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-0}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/logs}"
HF_HOME="${HF_HOME:-${CODE_DIR}/hf_home}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-${CODE_DIR}/smoke_data}"

MINICONDA_INSTALLER=""

cleanup() {
  if [[ -n "${MINICONDA_INSTALLER}" ]]; then
    rm -f "${MINICONDA_INSTALLER}"
  fi
}

trap cleanup EXIT

download_file() {
  local url="$1"
  local output="$2"

  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "${output}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --output-document="${output}" "${url}"
  else
    echo "curl or wget is required to install Miniconda." >&2
    return 1
  fi
}

ensure_supported_platform() {
  local system
  local machine
  system="$(uname -s)"
  machine="$(uname -m)"
  if [[ "${system}" != "Linux" || "${machine}" != "x86_64" ]]; then
    echo "${ENV_FILE} targets Linux x86_64; detected ${system} ${machine}." >&2
    return 2
  fi
}

ensure_git_lfs() {
  if [[ "${PULL_GIT_LFS_DATA}" != "1" && "${INSTALL_GIT_LFS}" != "1" ]]; then
    return
  fi
  if git lfs version >/dev/null 2>&1; then
    git lfs install --local >/dev/null 2>&1 || true
    return
  fi
  if [[ "${INSTALL_GIT_LFS}" != "1" ]]; then
    echo "git-lfs is unavailable and INSTALL_GIT_LFS=${INSTALL_GIT_LFS}." >&2
    return
  fi
  if [[ "$(id -u)" == "0" && -x "$(command -v apt-get)" ]]; then
    apt-get update
    apt-get install -y git-lfs
    git lfs install --system || git lfs install --local || true
    return
  fi
  echo "git-lfs is unavailable; install it manually if repository LFS data is required." >&2
}

find_conda_root() {
  local candidate

  if command -v conda >/dev/null 2>&1; then
    conda info --base
    return 0
  fi
  if [[ -n "${CONDA_ROOT:-}" ]]; then
    [[ -x "${CONDA_ROOT}/bin/conda" ]] || return 1
    printf '%s\n' "${CONDA_ROOT}"
    return 0
  fi
  for candidate in "${HOME}/miniconda3" "/root/miniconda3" "/opt/conda" "/opt/anaconda3"; do
    if [[ -x "${candidate}/bin/conda" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

install_miniconda() {
  local target_root="$1"
  local installer_url

  if [[ "${INSTALL_MINICONDA}" != "1" ]]; then
    echo "Conda was not found and INSTALL_MINICONDA=${INSTALL_MINICONDA}." >&2
    return 1
  fi
  installer_url="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
  MINICONDA_INSTALLER="$(mktemp /tmp/mopd-miniconda.XXXXXX.sh)"
  if [[ -e "${target_root}" && ! -x "${target_root}/bin/conda" ]]; then
    local backup_root="${target_root}.incomplete.$(date +%Y%m%d_%H%M%S)"
    echo "Moving incomplete Conda directory to ${backup_root}"
    mv "${target_root}" "${backup_root}"
  fi
  download_file "${installer_url}" "${MINICONDA_INSTALLER}"
  test -s "${MINICONDA_INSTALLER}"
  bash "${MINICONDA_INSTALLER}" -b -p "${target_root}"
  rm -f "${MINICONDA_INSTALLER}"
  MINICONDA_INSTALLER=""
}

prepare_conda() {
  if detected_conda_root="$(find_conda_root)"; then
    CONDA_ROOT="${detected_conda_root}"
  else
    CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
    install_miniconda "${CONDA_ROOT}"
  fi
  export CONDA_ROOT
  export PATH="${CONDA_ROOT}/bin:${PATH}"
  # shellcheck disable=SC1090
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda --version
}

environment_exists() {
  conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
}

sync_environment() {
  [[ -f "${ENV_FILE}" ]] || {
    echo "Missing environment definition: ${ENV_FILE}" >&2
    return 2
  }

  cd "${CODE_DIR}"
  if environment_exists; then
    if [[ "${UPDATE_ENV}" == "1" ]]; then
      echo "Updating ${ENV_NAME} exclusively from ${ENV_FILE}."
      conda env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune
    else
      echo "Using existing Conda environment: ${ENV_NAME}"
    fi
  else
    echo "Creating ${ENV_NAME} exclusively from ${ENV_FILE}."
    conda env create --name "${ENV_NAME}" --file "${ENV_FILE}"
  fi
}

run_in_training_env() {
  conda run --no-capture-output -n "${ENV_NAME}" "$@"
}

verify_environment() {
  export CODE_DIR ENV_FILE HF_HOME SMOKE_DATA_DIR
  run_in_training_env python -m pip check
  run_in_training_env env \
    PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}" \
    ENV_FILE="${ENV_FILE}" \
    HF_HOME="${HF_HOME}" \
    SMOKE_DATA_DIR="${SMOKE_DATA_DIR}" \
    python - <<'PY'
import importlib
import importlib.metadata
import os
import re
import sys
from pathlib import Path

import yaml

packages = [
    "torch",
    "vllm",
    "ray",
    "verl",
    "transformers",
    "pandas",
    "pyarrow",
    "yaml",
]
environment_data = yaml.safe_load(Path(os.environ["ENV_FILE"]).read_text(encoding="utf-8"))
pip_dependencies = []
for dependency in environment_data.get("dependencies", []):
    if isinstance(dependency, dict):
        pip_dependencies.extend(dependency.get("pip", []))

normalized_dependencies = [str(dependency).lower() for dependency in pip_dependencies]
if any("flash_attn" in dependency or "flash-attn" in dependency for dependency in normalized_dependencies):
    packages.append("flash_attn")
else:
    print("flash_attn skipped: not declared by the selected environment file")

exact_pins = {}
for dependency in pip_dependencies:
    match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^;\s]+)$", str(dependency))
    if match:
        exact_pins[match.group(1).lower().replace("_", "-")] = match.group(2)

for distribution in (
    "torch",
    "torchvision",
    "torchaudio",
    "torchdata",
    "vllm",
    "xformers",
    "transformers",
    "tensordict",
    "ray",
    "accelerate",
    "datasets",
    "peft",
    "liger-kernel",
    "opentelemetry-exporter-prometheus",
    "tensorboard",
):
    expected = exact_pins.get(distribution)
    if expected is None:
        continue
    actual = importlib.metadata.version(distribution)
    if actual != expected:
        raise RuntimeError(f"{distribution} version mismatch: expected {expected}, got {actual}")
    print(f"verified_pin {distribution}=={actual}")
print("python", sys.version.split()[0], sys.executable)
for package in packages:
    module = importlib.import_module(package)
    print(package, getattr(module, "__version__", "unknown"))

torch = importlib.import_module("torch")
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())

registry = importlib.import_module("verifiable_instructions.instructions_registry")
required = {
    "length_constraints:number_words",
    "length_constraints:nth_paragraph_first_word",
    "last_word:last_word_answer",
}
missing = required.difference(registry.INSTRUCTION_DICT)
if missing:
    raise RuntimeError(f"verifiable_instructions missing required ids: {sorted(missing)}")
PY

  run_in_training_env env \
    PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}" \
    python -m mopd_verl.smoke_data "${SMOKE_DATA_DIR}"
  run_in_training_env env \
    PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}" \
    python -m mopd_verl.prepare_data inspect "${SMOKE_DATA_DIR}/train.parquet"
}

write_activation_file() {
  cat > "${LOG_DIR}/activate_training_env.sh" <<EOF
#!/usr/bin/env bash
export CODE_DIR="${CODE_DIR}"
export HF_HOME="${HF_HOME}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:\${PYTHONPATH:-}"
export CONDA_ROOT="${CONDA_ROOT}"
export PATH="\${CONDA_ROOT}/bin:\${PATH}"
source "\${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
EOF
  chmod +x "${LOG_DIR}/activate_training_env.sh"
}

ensure_supported_platform
mkdir -p "${LOG_DIR}" "${HF_HOME}" "${SMOKE_DATA_DIR}"
ensure_git_lfs
prepare_conda
sync_environment
verify_environment

if [[ "${DOWNLOAD_ASSETS}" == "1" ]]; then
  run_in_training_env env \
    PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}" \
    bash "${SCRIPT_DIR}/download_training_assets.sh"
fi

if [[ "${REGISTER_KERNEL}" == "1" ]]; then
  run_in_training_env python -m ipykernel install \
    --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"
fi

write_activation_file

echo "Training environment ready."
echo "CODE_DIR=${CODE_DIR}"
echo "ENV_NAME=${ENV_NAME}"
echo "ENV_FILE=${ENV_FILE}"
echo "ACTIVATE_FILE=${LOG_DIR}/activate_training_env.sh"
