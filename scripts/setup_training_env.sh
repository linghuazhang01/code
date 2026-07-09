#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/setup_training_env.sh

Create or reuse a conda environment, install the vendored verl training stack,
and optionally download the assets for the Qwen30B four-domain MOPD run.

Common fresh-remote flow:
  bash scripts/setup_training_env.sh
  source logs/activate_training_env.sh

  DOWNLOAD_ASSETS=1 \
  MODEL_ROOT=/root/autodl-tmp/opd_mopd/models \
  DATA_DIR=/root/autodl-tmp/opd_mopd/OPD-code/data/G-OPD-Training-Data \
    bash scripts/setup_training_env.sh

Environment knobs:
  CONDA_ROOT=$HOME/miniconda3
  ENV_NAME=mopd-verl
  PYTHON_VERSION=3.10
  CONDA_CHANNEL=conda-forge
  INSTALL_MINICONDA=1
  USE_CURRENT_ENV=0
  INSTALL_VERL_DEPS=1
  INSTALL_M2RL_IF_DEPS=1
  CHECK_MOPD_DATA=0
  PULL_GIT_LFS_DATA=0
  INSTALL_GIT_LFS=1
  FORCE_REINSTALL=0
  INSTALL_SGLANG=0
  USE_MEGATRON=0
  REGISTER_KERNEL=0
  KERNEL_NAME=$ENV_NAME
  KERNEL_DISPLAY_NAME=MOPD ($ENV_NAME)
  DOWNLOAD_ASSETS=0
  REQUIREMENT_FILE=$CODE_DIR/requirement.txt

Asset variables are forwarded to scripts/download_training_assets.sh when
DOWNLOAD_ASSETS=1. Set USE_CURRENT_ENV=1 to install into the active Python
environment instead of creating or using conda.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ENV_NAME:-mopd-verl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_CHANNEL="${CONDA_CHANNEL:-conda-forge}"
INSTALL_MINICONDA="${INSTALL_MINICONDA:-1}"
USE_CURRENT_ENV="${USE_CURRENT_ENV:-0}"
INSTALL_VERL_DEPS="${INSTALL_VERL_DEPS:-1}"
INSTALL_M2RL_IF_DEPS="${INSTALL_M2RL_IF_DEPS:-1}"
CHECK_MOPD_DATA="${CHECK_MOPD_DATA:-0}"
PULL_GIT_LFS_DATA="${PULL_GIT_LFS_DATA:-0}"
INSTALL_GIT_LFS="${INSTALL_GIT_LFS:-1}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
INSTALL_SGLANG="${INSTALL_SGLANG:-0}"
USE_MEGATRON="${USE_MEGATRON:-0}"
REGISTER_KERNEL="${REGISTER_KERNEL:-0}"
KERNEL_NAME="${KERNEL_NAME:-${ENV_NAME}}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-MOPD (${ENV_NAME})}"
DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-0}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/logs}"
HF_HOME="${HF_HOME:-${CODE_DIR}/hf_home}"
REQUIREMENT_FILE="${REQUIREMENT_FILE:-${CODE_DIR}/requirement.txt}"

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
    echo "Installing git-lfs with apt-get."
    apt-get update
    apt-get install -y git-lfs
    git lfs install --system || git lfs install --local || true
    return
  fi
  echo "git-lfs is unavailable; install it manually if repo LFS data is required." >&2
}

find_conda_root() {
  local candidate

  if [[ -n "${CONDA_ROOT:-}" ]]; then
    if [[ -x "${CONDA_ROOT}/bin/conda" ]]; then
      printf '%s\n' "${CONDA_ROOT}"
      return 0
    fi
    return 1
  fi

  for candidate in "${HOME}/miniconda3" "/root/miniconda3" "/opt/conda"; do
    if [[ -x "${candidate}/bin/conda" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

install_miniconda() {
  local target_root="$1"
  local machine
  local installer_arch
  local installer_url

  if [[ "${INSTALL_MINICONDA}" != "1" ]]; then
    echo "Conda was not found and INSTALL_MINICONDA=${INSTALL_MINICONDA}." >&2
    return 1
  fi

  machine="$(uname -m)"
  case "${machine}" in
    x86_64|amd64)
      installer_arch="x86_64"
      ;;
    aarch64|arm64)
      installer_arch="aarch64"
      ;;
    *)
      echo "Unsupported architecture for automatic Miniconda install: ${machine}" >&2
      return 1
      ;;
  esac

  installer_url="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${installer_arch}.sh}"
  MINICONDA_INSTALLER="$(mktemp /tmp/mopd-miniconda.XXXXXX.sh)"

  if [[ -e "${target_root}" && ! -x "${target_root}/bin/conda" ]]; then
    local backup_root="${target_root}.incomplete.$(date +%Y%m%d_%H%M%S)"
    echo "Moving incomplete conda directory to ${backup_root}"
    mv "${target_root}" "${backup_root}"
  fi

  echo "Downloading Miniconda from ${installer_url}"
  download_file "${installer_url}" "${MINICONDA_INSTALLER}"
  test -s "${MINICONDA_INSTALLER}"
  bash "${MINICONDA_INSTALLER}" -b -p "${target_root}"
  rm -f "${MINICONDA_INSTALLER}"
  MINICONDA_INSTALLER=""
}

prepare_conda_env() {
  if detected_conda_root="$(find_conda_root)"; then
    CONDA_ROOT="${detected_conda_root}"
  else
    CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
    install_miniconda "${CONDA_ROOT}"
  fi

  if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
    echo "Conda is unavailable at ${CONDA_ROOT}." >&2
    return 1
  fi

  export CONDA_ROOT
  export PATH="${CONDA_ROOT}/bin:${PATH}"
  # shellcheck disable=SC1090
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"

  echo "Using conda: ${CONDA_ROOT}"
  conda --version

  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Creating ${ENV_NAME} with Python ${PYTHON_VERSION} from ${CONDA_CHANNEL}."
    conda create -y \
      --override-channels \
      --channel "${CONDA_CHANNEL}" \
      --name "${ENV_NAME}" \
      "python=${PYTHON_VERSION}" \
      pip
  else
    echo "Using existing conda environment: ${ENV_NAME}"
  fi
}

run_in_training_env() {
  if [[ "${USE_CURRENT_ENV}" == "1" ]]; then
    "$@"
  else
    conda run --no-capture-output -n "${ENV_NAME}" "$@"
  fi
}

mkdir -p "${LOG_DIR}" "${HF_HOME}"
ensure_git_lfs

if [[ "${USE_CURRENT_ENV}" != "1" ]]; then
  prepare_conda_env
else
  current_python_bin="${PYTHON_BIN:-python}"
  if ! command -v "${current_python_bin}" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
    current_python_bin="python3"
  fi
  echo "Using current Python environment: $("${current_python_bin}" -c 'import sys; print(sys.executable)')"
fi

run_in_training_env env \
  INSTALL_VERL_DEPS="${INSTALL_VERL_DEPS}" \
  INSTALL_M2RL_IF_DEPS="${INSTALL_M2RL_IF_DEPS}" \
  CHECK_MOPD_DATA="${CHECK_MOPD_DATA}" \
  PULL_GIT_LFS_DATA="${PULL_GIT_LFS_DATA}" \
  FORCE_REINSTALL="${FORCE_REINSTALL}" \
  INSTALL_SGLANG="${INSTALL_SGLANG}" \
  USE_MEGATRON="${USE_MEGATRON}" \
  LOG_DIR="${LOG_DIR}" \
  HF_HOME="${HF_HOME}" \
  REQUIREMENT_FILE="${REQUIREMENT_FILE}" \
  bash "${SCRIPT_DIR}/setup_remote_training_env.sh"

if [[ "${DOWNLOAD_ASSETS}" == "1" ]]; then
  run_in_training_env bash "${SCRIPT_DIR}/download_training_assets.sh"
fi

if [[ "${REGISTER_KERNEL}" == "1" ]]; then
  run_in_training_env python -m pip install --upgrade ipykernel
  run_in_training_env python -m ipykernel install \
    --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"
fi

cat > "${LOG_DIR}/activate_training_env.sh" <<EOF
#!/usr/bin/env bash
export CODE_DIR="${CODE_DIR}"
export HF_HOME="${HF_HOME}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:\${PYTHONPATH:-}"
EOF

if [[ "${USE_CURRENT_ENV}" != "1" ]]; then
  cat >> "${LOG_DIR}/activate_training_env.sh" <<EOF
export CONDA_ROOT="${CONDA_ROOT}"
export PATH="\${CONDA_ROOT}/bin:\${PATH}"
source "\${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
EOF
fi

chmod +x "${LOG_DIR}/activate_training_env.sh"

echo "Training environment ready."
echo "CODE_DIR=${CODE_DIR}"
echo "ENV_NAME=${ENV_NAME}"
echo "ACTIVATE_FILE=${LOG_DIR}/activate_training_env.sh"
