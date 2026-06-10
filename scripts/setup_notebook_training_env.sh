#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/setup_notebook_training_env.sh

Run this from a Jupyter/Notebook cell:
  !bash scripts/setup_notebook_training_env.sh

The script:
  1. Finds or installs Miniconda.
  2. Creates the environment from conda-forge without Anaconda ToS prompts.
  3. Runs the regular OPD/verl environment setup.
  4. Registers a Jupyter kernel for the new environment.

Environment knobs:
  CONDA_ROOT=$HOME/miniconda3
  ENV_NAME=mopd-verl
  CONDA_CHANNEL=conda-forge
  INSTALL_MINICONDA=1
  INSTALL_VERL_DEPS=1
  FORCE_REINSTALL=0
  INSTALL_SGLANG=0
  USE_MEGATRON=0
  REGISTER_KERNEL=1
  KERNEL_NAME=mopd-verl
  KERNEL_DISPLAY_NAME=MOPD (mopd-verl)
  MINICONDA_URL=<auto-detected Linux installer URL>
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

on_error() {
  local exit_code=$?
  echo >&2
  echo "Notebook environment setup failed at line ${BASH_LINENO[0]} (exit ${exit_code})." >&2
  echo "Re-run this script in a Notebook cell to retain the full command output." >&2
  exit "${exit_code}"
}

MINICONDA_INSTALLER=""

cleanup() {
  if [[ -n "${MINICONDA_INSTALLER}" ]]; then
    rm -f "${MINICONDA_INSTALLER}"
  fi
}

trap on_error ERR
trap cleanup EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ENV_NAME:-mopd-verl}"
CONDA_CHANNEL="${CONDA_CHANNEL:-conda-forge}"
INSTALL_MINICONDA="${INSTALL_MINICONDA:-1}"
INSTALL_VERL_DEPS="${INSTALL_VERL_DEPS:-1}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
INSTALL_SGLANG="${INSTALL_SGLANG:-0}"
USE_MEGATRON="${USE_MEGATRON:-0}"
REGISTER_KERNEL="${REGISTER_KERNEL:-1}"
KERNEL_NAME="${KERNEL_NAME:-${ENV_NAME}}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-MOPD (${ENV_NAME})}"

find_conda_root() {
  local candidate

  if [[ -n "${CONDA_ROOT:-}" && -x "${CONDA_ROOT}/bin/conda" ]]; then
    printf '%s\n' "${CONDA_ROOT}"
    return 0
  fi

  for candidate in "${HOME}/miniconda3" "/root/miniconda3" "/opt/conda"; do
    if [[ -x "${candidate}/bin/conda" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

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

install_miniconda() {
  local target_root="$1"
  local installer
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
  installer="$(mktemp /tmp/mopd-miniconda.XXXXXX.sh)"
  MINICONDA_INSTALLER="${installer}"

  if [[ -e "${target_root}" && ! -x "${target_root}/bin/conda" ]]; then
    local backup_root="${target_root}.incomplete.$(date +%Y%m%d_%H%M%S)"
    echo "Moving incomplete conda directory to ${backup_root}"
    mv "${target_root}" "${backup_root}"
  fi

  echo "Downloading Miniconda from ${installer_url}"
  download_file "${installer_url}" "${installer}"
  test -s "${installer}"
  bash "${installer}" -b -p "${target_root}"
  rm -f "${installer}"
  MINICONDA_INSTALLER=""
}

if detected_conda_root="$(find_conda_root)"; then
  CONDA_ROOT="${detected_conda_root}"
else
  CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
  install_miniconda "${CONDA_ROOT}"
fi

if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
  echo "Conda installation is unavailable at ${CONDA_ROOT}." >&2
  exit 1
fi

export CONDA_ROOT
export PATH="${CONDA_ROOT}/bin:${PATH}"
# shellcheck disable=SC1090
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

echo "Using conda: ${CONDA_ROOT}"
conda --version

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Creating ${ENV_NAME} from ${CONDA_CHANNEL} without default Anaconda channels."
  conda create -y \
    --override-channels \
    --channel "${CONDA_CHANNEL}" \
    --name "${ENV_NAME}" \
    python=3.10 \
    pip
else
  echo "Using existing conda environment: ${ENV_NAME}"
fi

CONDA_ROOT="${CONDA_ROOT}" \
ENV_NAME="${ENV_NAME}" \
INSTALL_VERL_DEPS="${INSTALL_VERL_DEPS}" \
FORCE_REINSTALL="${FORCE_REINSTALL}" \
INSTALL_SGLANG="${INSTALL_SGLANG}" \
USE_MEGATRON="${USE_MEGATRON}" \
bash "${SCRIPT_DIR}/setup_remote_training_env.sh"

if [[ "${REGISTER_KERNEL}" == "1" ]]; then
  conda run --no-capture-output -n "${ENV_NAME}" \
    python -m pip install --upgrade ipykernel
  conda run --no-capture-output -n "${ENV_NAME}" \
    python -m ipykernel install \
      --user \
      --name "${KERNEL_NAME}" \
      --display-name "${KERNEL_DISPLAY_NAME}"
fi

cat <<EOF

Notebook environment ready.
  Code directory: ${CODE_DIR}
  Conda root: ${CONDA_ROOT}
  Conda environment: ${ENV_NAME}
  Jupyter kernel: ${KERNEL_DISPLAY_NAME}

Select "${KERNEL_DISPLAY_NAME}" as the Notebook kernel, or restart the current
kernel before importing packages installed by this script.
EOF
