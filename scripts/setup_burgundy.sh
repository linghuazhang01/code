#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/setup_burgundy.sh [--phases PHASES] [--dry-run]
Bootstrap MOPD on the CityUHK Burgundy cluster. The default flow is:
  repository -> Conda environment -> training data -> models -> verification
Run the environment and asset phases inside a Slurm allocation, for example:
  srun --partition=gpu_a100 --nodes=1 --ntasks=1 --gres=gpu:a100:1 \
    --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
  bash scripts/setup_burgundy.sh
Options:
  --phases LIST  Comma-separated subset of repo,env,data,models,verify or all.
  --dry-run      Validate and print the resolved configuration without writes.
  -h, --help     Show this help.
Authentication is read only from the standard tools (for example HF_TOKEN or
an existing Hugging Face login). Do not place credentials in this script.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISCOVERED_CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-${HOME}/scratch/opd}"

if [[ -f "${DISCOVERED_CODE_DIR}/environment.yml" ]]; then
  CODE_DIR="${CODE_DIR:-${DISCOVERED_CODE_DIR}}"
else
  CODE_DIR="${CODE_DIR:-${REMOTE_ROOT}/mopd_code}"
fi

REPO_URL="${REPO_URL:-https://github.com/linghuazhang01/code.git}"
REPO_REF="${REPO_REF:-main}"
CONDA_ROOT="${CONDA_ROOT:-${REMOTE_ROOT}/miniforge3}"
MINIFORGE_VERSION="${MINIFORGE_VERSION:-26.5.3-0}"
MINIFORGE_URL="${MINIFORGE_URL:-https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh}"
MINIFORGE_SHA256="${MINIFORGE_SHA256:-14db468222ad564658656f769506056209b6dc375f5e7dfd31eb5ebbf08fa529}"
ENV_NAME="${ENV_NAME:-mopd-verl-a100}"
ENV_FILE="${ENV_FILE:-${CODE_DIR}/environment.yml}"
MODEL_ROOT="${MODEL_ROOT:-${REMOTE_ROOT}/models}"
DATA_DIR="${DATA_DIR:-${CODE_DIR}/data/G-OPD-Training-Data}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${CODE_DIR}/data/eval_data}"
HF_HOME="${HF_HOME:-${REMOTE_ROOT}/hf_home}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${REMOTE_ROOT}/pip_cache}"
LOG_DIR="${LOG_DIR:-${REMOTE_ROOT}/logs}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-${REMOTE_ROOT}/smoke_data}"
MODEL_BACKEND="${MODEL_BACKEND:-huggingface}"
DATASET_ID="${DATASET_ID:-icemoon28/MOPD-Training-Data}"
DATASET_REVISION="${DATASET_REVISION:-main}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
UPDATE_REPO="${UPDATE_REPO:-0}"
UPDATE_ENV="${UPDATE_ENV:-1}"
ALLOW_LOGIN_NODE="${ALLOW_LOGIN_NODE:-0}"
ALLOW_NON_SCRATCH_ROOT="${ALLOW_NON_SCRATCH_ROOT:-0}"
DOWNLOAD_LCB="${DOWNLOAD_LCB:-0}"
PHASES="${PHASES:-all}"
DRY_RUN=0

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

phase_enabled() {
  local phase="$1"
  case ",${PHASES}," in
    *,all,*|*,"${phase}",*) return 0 ;;
    *) return 1 ;;
  esac
}

validate_boolean() {
  local name="$1"
  local value="$2"
  [[ "${value}" == "0" || "${value}" == "1" ]] || {
    die "${name} must be 0 or 1, got ${value}"
  }
}

validate_phases() {
  local phase
  local -a selected_phases
  IFS=',' read -r -a selected_phases <<<"${PHASES}"
  ((${#selected_phases[@]} > 0)) || die "PHASES cannot be empty"
  for phase in "${selected_phases[@]}"; do
    case "${phase}" in
      all|repo|env|data|models|verify) ;;
      *) die "unsupported phase: ${phase}" ;;
    esac
  done
}

validate_configuration() {
  local canonical_path
  local canonical_root
  local canonical_scratch
  local path_name
  validate_phases
  validate_boolean UPDATE_REPO "${UPDATE_REPO}"
  validate_boolean UPDATE_ENV "${UPDATE_ENV}"
  validate_boolean ALLOW_LOGIN_NODE "${ALLOW_LOGIN_NODE}"
  validate_boolean ALLOW_NON_SCRATCH_ROOT "${ALLOW_NON_SCRATCH_ROOT}"
  validate_boolean DOWNLOAD_LCB "${DOWNLOAD_LCB}"

  [[ "${ENV_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid ENV_NAME"
  [[ "${REPO_REF}" =~ ^[A-Za-z0-9._/-]+$ ]] || die "invalid REPO_REF"
  [[ "${MIN_FREE_GB}" =~ ^[0-9]+$ ]] || die "MIN_FREE_GB must be an integer"

  if [[ "${DRY_RUN}" != "1" && "${ALLOW_NON_SCRATCH_ROOT}" != "1" ]]; then
    command -v python3 >/dev/null 2>&1 || die "python3 is required for path safety"
    canonical_scratch="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${HOME}/scratch")"
    canonical_root="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${REMOTE_ROOT}")"
    case "${canonical_root}" in
      "${canonical_scratch}"|"${canonical_scratch}/"*) ;;
      *) die "REMOTE_ROOT must resolve under ${HOME}/scratch" ;;
    esac
    for path_name in CODE_DIR CONDA_ROOT MODEL_ROOT DATA_DIR EVAL_DATA_DIR HF_HOME PIP_CACHE_DIR LOG_DIR SMOKE_DATA_DIR; do
      [[ "${!path_name}" == /* ]] || die "${path_name} must be absolute"
      canonical_path="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${!path_name}")"
      case "${canonical_path}" in
        "${canonical_root}"|"${canonical_root}/"*) ;;
        *) die "${path_name} must resolve under REMOTE_ROOT" ;;
      esac
    done
  fi
}

require_compute_allocation() {
  local gpu_info
  local gpu_memory
  local needs_compute=0
  local partition
  local phase
  for phase in env data models verify; do
    if phase_enabled "${phase}"; then
      needs_compute=1
    fi
  done
  if [[ "${needs_compute}" != "1" || "${ALLOW_LOGIN_NODE}" == "1" ]]; then
    return
  fi
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "env/data/models/verify must run inside Slurm"
  partition="${SLURM_JOB_PARTITION:-}"
  if [[ -z "${partition}" ]] && command -v squeue >/dev/null 2>&1; then
    partition="$(squeue -h -j "${SLURM_JOB_ID}" -o '%P')"
  fi
  [[ "${partition}" == "gpu_a100" ]] || die "expected gpu_a100 partition, got ${partition:-unknown}"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
  gpu_info="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | sed -n '1p')"
  [[ "${gpu_info}" == *A100* ]] || die "expected an A100 GPU, got ${gpu_info:-none}"
  gpu_memory="$(printf '%s' "${gpu_info##*,}" | tr -d '[:space:]')"
  [[ "${gpu_memory}" =~ ^[0-9]+$ ]] || die "cannot parse A100 memory: ${gpu_info}"
  ((gpu_memory >= 80000)) || die "expected A100 80GB, got ${gpu_memory} MiB"
}

report_configuration() {
  cat <<EOF
Burgundy bootstrap configuration
  phases=${PHASES}
  dry_run=${DRY_RUN}
  remote_root=${REMOTE_ROOT}
  code_dir=${CODE_DIR}
  repo_ref=${REPO_REF}
  conda_root=${CONDA_ROOT}
  env_name=${ENV_NAME}
  env_file=${ENV_FILE}
  data_dir=${DATA_DIR}
  model_root=${MODEL_ROOT}
  hf_home=${HF_HOME}
  slurm_job_id=${SLURM_JOB_ID:-none}
EOF
}

validate_repo_tree() {
  local required_path
  for required_path in \
    "${CODE_DIR}/environment.yml" \
    "${CODE_DIR}/scripts/setup_training_env.sh" \
    "${CODE_DIR}/scripts/download_training_assets.sh"; do
    [[ -f "${required_path}" ]] || die "missing repository file: ${required_path}"
  done
}

prepare_repository() {
  local clone_dir
  mkdir -p "${REMOTE_ROOT}"
  if [[ -f "${CODE_DIR}/environment.yml" ]]; then
    printf 'Using existing source tree: %s\n' "${CODE_DIR}"
  elif [[ -e "${CODE_DIR}" ]]; then
    die "CODE_DIR exists but is not a complete MOPD checkout: ${CODE_DIR}"
  else
    clone_dir="$(mktemp -d "$(dirname "${CODE_DIR}")/.mopd-clone.XXXXXX")"
    printf 'Cloning repository ref %s into %s\n' "${REPO_REF}" "${CODE_DIR}"
    if GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout \
      "${REPO_URL}" "${clone_dir}" && \
      git -C "${clone_dir}" fetch --depth 1 origin "${REPO_REF}" && \
      git -C "${clone_dir}" checkout --detach FETCH_HEAD; then
      mv "${clone_dir}" "${CODE_DIR}"
    else
      printf 'Incomplete clone retained for diagnosis: %s\n' "${clone_dir}" >&2
      return 1
    fi
  fi

  if [[ "${UPDATE_REPO}" == "1" ]]; then
    [[ -d "${CODE_DIR}/.git" ]] || die "UPDATE_REPO=1 requires a Git checkout"
    [[ -z "$(git -C "${CODE_DIR}" status --porcelain)" ]] || {
      die "refusing to update a dirty remote checkout"
    }
    git -C "${CODE_DIR}" fetch --depth 1 origin "${REPO_REF}"
    git -C "${CODE_DIR}" checkout --detach FETCH_HEAD
  fi
  validate_repo_tree
}

prepare_environment() {
  mkdir -p "${LOG_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${SMOKE_DATA_DIR}"
  CONDA_ROOT="${CONDA_ROOT}" \
  MINICONDA_URL="${MINIFORGE_URL}" \
  MINICONDA_SHA256="${MINIFORGE_SHA256}" \
  PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
  ENV_NAME="${ENV_NAME}" \
  ENV_FILE="${ENV_FILE}" \
  INSTALL_MINICONDA=1 \
  UPDATE_ENV="${UPDATE_ENV}" \
  INSTALL_GIT_LFS=0 \
  PULL_GIT_LFS_DATA=0 \
  REGISTER_KERNEL=0 \
  DOWNLOAD_ASSETS=0 \
  LOG_DIR="${LOG_DIR}" \
  HF_HOME="${HF_HOME}" \
  SMOKE_DATA_DIR="${SMOKE_DATA_DIR}" \
    bash "${CODE_DIR}/scripts/setup_training_env.sh"
}

training_python() {
  local python_bin="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"
  [[ -x "${python_bin}" ]] || die "training Python is missing: ${python_bin}"
  printf '%s\n' "${python_bin}"
}

prepare_data() {
  local python_bin
  python_bin="$(training_python)"
  DATA_DIR="${DATA_DIR}" \
  DATASET_ID="${DATASET_ID}" \
  DATASET_REVISION="${DATASET_REVISION}" \
  EVAL_DATA_DIR="${EVAL_DATA_DIR}" \
  MODEL_ROOT="${MODEL_ROOT}" \
  HF_HOME="${HF_HOME}" \
  PYTHON_BIN="${python_bin}" \
  DOWNLOAD_DATA=1 \
  DOWNLOAD_MODELS=0 \
  REQUIRE_MATH_CODE_TRAIN_DATA=1 \
  REQUIRE_4DOMAIN_TRAIN_DATA=1 \
  REQUIRE_MODELS=0 \
  DOWNLOAD_LCB="${DOWNLOAD_LCB}" \
    bash "${CODE_DIR}/scripts/download_training_assets.sh"
}

prepare_models() {
  local python_bin
  python_bin="$(training_python)"
  DATA_DIR="${DATA_DIR}" \
  EVAL_DATA_DIR="${EVAL_DATA_DIR}" \
  MODEL_ROOT="${MODEL_ROOT}" \
  HF_HOME="${HF_HOME}" \
  PYTHON_BIN="${python_bin}" \
  MODEL_BACKEND="${MODEL_BACKEND}" \
  DOWNLOAD_DATA=0 \
  DOWNLOAD_MODELS=1 \
  REQUIRE_MATH_CODE_TRAIN_DATA=0 \
  REQUIRE_4DOMAIN_TRAIN_DATA=0 \
  REQUIRE_MODELS=1 \
  MIN_FREE_GB="${MIN_FREE_GB}" \
    bash "${CODE_DIR}/scripts/download_training_assets.sh"
}

verify_installation() {
  local python_bin
  local setup_id
  local record_dir
  python_bin="$(training_python)"
  setup_id="$(date +%Y%m%d_%H%M%S)"
  record_dir="${LOG_DIR}/burgundy-setup-${setup_id}"
  mkdir -p "${record_dir}"

  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export HF_HOME
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
  export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
  export PYTHONHASHSEED=0
  export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"

  "${python_bin}" - <<'PY' | tee "${record_dir}/cuda-witness.txt"
import json
import random

import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the Burgundy environment")

left = torch.randn(64, 64, device="cuda")
right = torch.randn(64, 64, device="cuda")
result = left @ right
if result.shape != (64, 64) or not torch.isfinite(result).all():
    raise RuntimeError("seeded CUDA matrix witness failed")

payload = {
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "shape": list(result.shape),
    "torch": torch.__version__,
}
print("CUDA_WITNESS", json.dumps(payload, sort_keys=True))
PY

  {
    printf 'timestamp=%s\n' "$(date -Is)"
    printf 'host=%s\n' "$(hostname)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-none}"
    printf 'code_dir=%s\n' "${CODE_DIR}"
    printf 'git_commit=%s\n' "$(git -C "${CODE_DIR}" rev-parse HEAD 2>/dev/null || echo synced-tree)"
    printf 'env_name=%s\n' "${ENV_NAME}"
    printf 'env_file=%s\n' "${ENV_FILE}"
    printf 'dataset_id=%s\n' "${DATASET_ID}"
    printf 'dataset_revision=%s\n' "${DATASET_REVISION}"
    printf 'data_dir=%s\n' "${DATA_DIR}"
    printf 'model_root=%s\n' "${MODEL_ROOT}"
  } | tee "${record_dir}/manifest.txt"

  "${python_bin}" --version 2>&1 | tee "${record_dir}/python-version.txt"
  "${python_bin}" -m pip freeze >"${record_dir}/pip-freeze.txt"
  nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version \
    --format=csv,noheader >"${record_dir}/gpu.txt"
  printf 'Verification record: %s\n' "${record_dir}"
}

while (($# > 0)); do
  case "$1" in
    --phases)
      (($# >= 2)) || die "--phases requires a value"
      PHASES="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

validate_configuration
report_configuration
if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

require_compute_allocation
if phase_enabled repo; then
  prepare_repository
else
  validate_repo_tree
fi
if phase_enabled env; then
  prepare_environment
fi
if phase_enabled data; then
  prepare_data
fi
if phase_enabled models; then
  prepare_models
fi
if phase_enabled verify; then
  verify_installation
fi

cat <<EOF
Burgundy bootstrap complete.
  Activate: source ${LOG_DIR}/activate_training_env.sh
  Code:     ${CODE_DIR}
  Data:     ${DATA_DIR}
  Models:   ${MODEL_ROOT}
EOF
