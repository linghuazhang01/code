#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash bootstrap_qwen30b_mopd_training.sh [-- <extra hydra overrides...>]

Clone or update OPD-code, optionally overlay a data-only zip, install the
training environment, download Qwen30B models, verify four-domain assets, and
launch the split-teacher MOPD run.

Defaults target the fsdp compatibility profile derived from:
  configs/mopd_qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_6gpu_fsdp.yaml

Environment knobs:
  REPO_URL=http://github.com/linghuazhang01/code.git
  REPO_REF=bowen
  CHECKOUT_DIR=/root/autodl-tmp/opd_mopd/OPD-code
  BUNDLE_ZIP=                      # data-only zip created by package_qwen30b_mopd_bundle.sh
  BUNDLE_EXTRACT_DIR=/root/autodl-tmp/opd_mopd/bundle_extract
  BUNDLE_REPLACE_EXISTING=1       # overwrite existing data files when overlaying
  UPDATE_EXISTING=1
  GIT_TIMEOUT_SECONDS=300
  GIT_HTTP_VERSION=HTTP/1.1
  GIT_CLONE_DEPTH=1              # set to 0 for a full clone

  GPU_PROFILE=8gpu                 # 8gpu or 4gpu
  GPU_IDS=<auto: 0..7 or 0..3>
  RUN_ID=<auto>
  FOREGROUND=0
  TAIL_LOG=0
  DRY_RUN=0

  INSTALL_ENV=1
  PREPARE_ASSETS=1
  MODEL_ROOT=/root/autodl-tmp/opd_mopd/models
  DATA_DIR=$CHECKOUT_DIR/data/G-OPD-Training-Data
  DOWNLOAD_DATA=<auto: 0 for BUNDLE_ZIP, otherwise 1>
  DOWNLOAD_MODELS=1
  REQUIRE_4DOMAIN_TRAIN_DATA=1
  REQUIRE_MODELS=1
  MIN_FREE_GB=100
  MODEL_BACKEND=modelscope

  CONDA_ROOT=/root/autodl-tmp/opd_mopd/miniconda3
  ENV_NAME=mopd-verl
  WANDB_MODE=disabled

Examples:
  GPU_PROFILE=8gpu bash bootstrap_qwen30b_mopd_training.sh

  GPU_PROFILE=4gpu GPU_IDS=0,1,2,3 DRY_RUN=1 \
    bash bootstrap_qwen30b_mopd_training.sh -- trainer.total_training_steps=2

  WANDB_MODE=online WANDB_API_KEY=... GPU_PROFILE=8gpu \
    bash bootstrap_qwen30b_mopd_training.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

EXTRA_HYDRA_OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      EXTRA_HYDRA_OVERRIDES=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Put Hydra overrides after '--'." >&2
      exit 2
      ;;
  esac
done

REPO_URL="${REPO_URL:-http://github.com/linghuazhang01/code.git}"
REPO_REF="${REPO_REF:-bowen}"
CHECKOUT_DIR="${CHECKOUT_DIR:-/root/autodl-tmp/opd_mopd/OPD-code}"
BUNDLE_ZIP="${BUNDLE_ZIP:-}"
BUNDLE_EXTRACT_DIR="${BUNDLE_EXTRACT_DIR:-/root/autodl-tmp/opd_mopd/bundle_extract}"
BUNDLE_REPLACE_EXISTING="${BUNDLE_REPLACE_EXISTING:-1}"
UPDATE_EXISTING="${UPDATE_EXISTING:-1}"
GIT_TIMEOUT_SECONDS="${GIT_TIMEOUT_SECONDS:-300}"
GIT_HTTP_VERSION="${GIT_HTTP_VERSION:-HTTP/1.1}"
GIT_CLONE_DEPTH="${GIT_CLONE_DEPTH:-1}"
GPU_PROFILE="${GPU_PROFILE:-8gpu}"
INSTALL_ENV="${INSTALL_ENV:-1}"
PREPARE_ASSETS="${PREPARE_ASSETS:-1}"
MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/opd_mopd/models}"
DATA_DIR="${DATA_DIR:-${CHECKOUT_DIR}/data/G-OPD-Training-Data}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
REQUIRE_4DOMAIN_TRAIN_DATA="${REQUIRE_4DOMAIN_TRAIN_DATA:-1}"
REQUIRE_MODELS="${REQUIRE_MODELS:-1}"
if [[ -z "${DOWNLOAD_DATA:-}" ]]; then
  if [[ -n "${BUNDLE_ZIP}" ]]; then
    DOWNLOAD_DATA=0
  else
    DOWNLOAD_DATA=1
  fi
fi
MIN_FREE_GB="${MIN_FREE_GB:-100}"
MODEL_BACKEND="${MODEL_BACKEND:-modelscope}"
CONDA_ROOT="${CONDA_ROOT:-/root/autodl-tmp/opd_mopd/miniconda3}"
ENV_NAME="${ENV_NAME:-mopd-verl}"
WANDB_MODE="${WANDB_MODE:-disabled}"
FOREGROUND="${FOREGROUND:-0}"
TAIL_LOG="${TAIL_LOG:-0}"
DRY_RUN="${DRY_RUN:-0}"

log_step() {
  local message="$1"
  printf '\n[%s] >>> %s\n' "$(date '+%F %T')" "${message}"
}

log_done() {
  local message="$1"
  printf '[%s] <<< %s\n' "$(date '+%F %T')" "${message}"
}

run_git() {
  local status
  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout "${GIT_TIMEOUT_SECONDS}" env GIT_HTTP_VERSION="${GIT_HTTP_VERSION}" git "$@"
    status=$?
  else
    env GIT_HTTP_VERSION="${GIT_HTTP_VERSION}" git "$@"
    status=$?
  fi
  set -e
  if [[ "${status}" == "124" ]]; then
    echo "Git command timed out after ${GIT_TIMEOUT_SECONDS}s: git $*" >&2
  fi
  return "${status}"
}

validate_data_bundle_entries() {
  local entry
  while IFS= read -r entry; do
    [[ -n "${entry}" ]] || continue
    case "${entry}" in
      /*|../*|*/../*)
        echo "Unsafe bundle entry: ${entry}" >&2
        exit 2
        ;;
      data/G-OPD-Training-Data/*|eval/domains/*/data/*)
        ;;
      *)
        echo "Non-data bundle entry is not allowed: ${entry}" >&2
        exit 2
        ;;
    esac
  done < <(unzip -Z1 "${BUNDLE_ZIP}")
}

unpack_data_bundle() {
  log_step "STEP 1/4 data bundle overlay: ${BUNDLE_ZIP} -> ${CHECKOUT_DIR}"
  [[ -f "${BUNDLE_ZIP}" ]] || {
    echo "BUNDLE_ZIP does not exist: ${BUNDLE_ZIP}" >&2
    exit 2
  }
  if ! command -v unzip >/dev/null 2>&1; then
    echo "unzip is required for BUNDLE_ZIP mode." >&2
    exit 2
  fi

  validate_data_bundle_entries
  mkdir -p "${BUNDLE_EXTRACT_DIR}" "${CHECKOUT_DIR}"
  echo "Data bundle extract root: ${CHECKOUT_DIR}"
  if [[ "${BUNDLE_REPLACE_EXISTING}" == "1" ]]; then
    unzip -oq "${BUNDLE_ZIP}" -d "${CHECKOUT_DIR}"
  else
    unzip -nq "${BUNDLE_ZIP}" -d "${CHECKOUT_DIR}"
  fi
  log_done "STEP 1/4 data bundle overlay ready"
}

clone_or_update_repo() {
  log_step "STEP 1/4 git clone/update: ${REPO_URL} -> ${CHECKOUT_DIR}"
  echo "Git timeout: ${GIT_TIMEOUT_SECONDS}s"
  echo "Git HTTP version: ${GIT_HTTP_VERSION}"
  echo "Git clone depth: ${GIT_CLONE_DEPTH}"
  if [[ -d "${CHECKOUT_DIR}/.git" ]]; then
    echo "Using existing checkout: ${CHECKOUT_DIR}"
    if [[ "${UPDATE_EXISTING}" == "1" ]]; then
      echo "Fetching origin..."
      run_git -C "${CHECKOUT_DIR}" fetch origin
      if [[ -n "${REPO_REF}" ]]; then
        echo "Checking out ${REPO_REF}..."
        run_git -C "${CHECKOUT_DIR}" checkout "${REPO_REF}"
        echo "Fast-forward pulling ${REPO_REF}..."
        run_git -C "${CHECKOUT_DIR}" pull --ff-only origin "${REPO_REF}" || run_git -C "${CHECKOUT_DIR}" pull --ff-only
      else
        echo "Fast-forward pulling current branch..."
        run_git -C "${CHECKOUT_DIR}" pull --ff-only
      fi
    else
      echo "UPDATE_EXISTING=0, skipping git fetch/pull."
    fi
    log_done "STEP 1/4 git checkout ready"
    return
  fi

  if [[ -e "${CHECKOUT_DIR}" ]]; then
    echo "CHECKOUT_DIR exists but is not a git checkout: ${CHECKOUT_DIR}" >&2
    exit 2
  fi

  mkdir -p "$(dirname "${CHECKOUT_DIR}")"
  echo "Cloning repository..."
  if ! [[ "${GIT_CLONE_DEPTH}" =~ ^[0-9]+$ ]]; then
    echo "GIT_CLONE_DEPTH must be a non-negative integer: ${GIT_CLONE_DEPTH}" >&2
    exit 2
  fi
  clone_args=(clone)
  if [[ "${GIT_CLONE_DEPTH}" != "0" ]]; then
    clone_args+=(--depth "${GIT_CLONE_DEPTH}" --single-branch)
  fi
  if [[ -n "${REPO_REF}" ]]; then
    clone_args+=(--branch "${REPO_REF}")
  fi
  clone_args+=("${REPO_URL}" "${CHECKOUT_DIR}")
  run_git "${clone_args[@]}"
  if [[ -n "${REPO_REF}" ]]; then
    echo "Checking out ${REPO_REF}..."
    run_git -C "${CHECKOUT_DIR}" checkout "${REPO_REF}"
  fi
  log_done "STEP 1/4 git checkout ready"
}

prepare_code() {
  clone_or_update_repo
  if [[ -n "${BUNDLE_ZIP}" ]]; then
    unpack_data_bundle
  fi
}

profile_overrides() {
  local profile="$1"
  case "${profile}" in
    4|4gpu)
      GPU_PROFILE="4gpu"
      GPU_IDS="${GPU_IDS:-0,1,2,3}"
      PROFILE_TRAIN_BATCH_SIZE=384
      PROFILE_ACTOR_GPUS=3
      PROFILE_REF_GPUS=1
      PROFILE_ROLLOUT_TP=1
      PROFILE_RAY_CPUS=16
      PROFILE_NAME="qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_4gpu_4domain_fsdp"
      ;;
    8|8gpu)
      GPU_PROFILE="8gpu"
      GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
      PROFILE_TRAIN_BATCH_SIZE=768
      PROFILE_ACTOR_GPUS=6
      PROFILE_REF_GPUS=2
      PROFILE_ROLLOUT_TP=1
      PROFILE_RAY_CPUS=32
      PROFILE_NAME="qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_8gpu_4domain_fsdp"
      ;;
    *)
      echo "Unsupported GPU_PROFILE=${profile}; expected 4gpu or 8gpu." >&2
      exit 2
      ;;
  esac
}

profile_overrides "${GPU_PROFILE}"
prepare_code

cd "${CHECKOUT_DIR}"

if [[ ! -f scripts/setup_training_env.sh ]]; then
  echo "Missing scripts/setup_training_env.sh in ${CHECKOUT_DIR}" >&2
  exit 2
fi
if [[ ! -f scripts/download_training_assets.sh ]]; then
  echo "Missing scripts/download_training_assets.sh in ${CHECKOUT_DIR}" >&2
  exit 2
fi
if [[ ! -f scripts/start_remote_mopd_training.sh ]]; then
  echo "Missing scripts/start_remote_mopd_training.sh in ${CHECKOUT_DIR}" >&2
  exit 2
fi

BASE_CONFIG="configs/mopd_qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_6gpu_fsdp.yaml"
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing base config: ${BASE_CONFIG}" >&2
  exit 2
fi

if [[ "${INSTALL_ENV}" == "1" ]]; then
  log_step "STEP 2/4 environment install: conda env ${ENV_NAME}"
  CONDA_ROOT="${CONDA_ROOT}" \
  ENV_NAME="${ENV_NAME}" \
  DOWNLOAD_ASSETS=0 \
    bash scripts/setup_training_env.sh
  log_done "STEP 2/4 environment install finished"
else
  log_step "STEP 2/4 environment install skipped: INSTALL_ENV=0"
  log_done "STEP 2/4 environment install skipped"
fi

if [[ -f logs/activate_training_env.sh ]]; then
  echo "Activating training environment from logs/activate_training_env.sh"
  # shellcheck disable=SC1091
  source logs/activate_training_env.sh
elif [[ -d "${CONDA_ROOT}/envs/${ENV_NAME}/bin" ]]; then
  echo "Using existing conda environment on PATH: ${CONDA_ROOT}/envs/${ENV_NAME}"
  export PATH="${CONDA_ROOT}/envs/${ENV_NAME}/bin:${CONDA_ROOT}/bin:${PATH}"
fi

if [[ "${PREPARE_ASSETS}" == "1" ]]; then
  log_step "STEP 3/4 asset preparation: data verify/download + Qwen3-4B + Qwen3-30B-A3B"
  MODEL_ROOT="${MODEL_ROOT}" \
  DATA_DIR="${DATA_DIR}" \
  PYTHON_BIN="${PYTHON_BIN:-python}" \
  MODEL_BACKEND="${MODEL_BACKEND}" \
  DOWNLOAD_DATA="${DOWNLOAD_DATA}" \
  DOWNLOAD_MODELS="${DOWNLOAD_MODELS}" \
  REQUIRE_4DOMAIN_TRAIN_DATA="${REQUIRE_4DOMAIN_TRAIN_DATA}" \
  REQUIRE_MODELS="${REQUIRE_MODELS}" \
  MIN_FREE_GB="${MIN_FREE_GB}" \
    scripts/download_training_assets.sh
  log_done "STEP 3/4 asset preparation finished"
else
  log_step "STEP 3/4 data/model download skipped: PREPARE_ASSETS=0"
  log_done "STEP 3/4 data/model download skipped"
fi

RUN_ID="${RUN_ID:-${PROFILE_NAME}_$(date +%Y%m%d_%H%M%S)}"
export WANDB_MODE

LAUNCH_ARGS=()
if [[ "${FOREGROUND}" == "1" ]]; then
  LAUNCH_ARGS+=(--foreground)
fi
if [[ "${TAIL_LOG}" == "1" ]]; then
  LAUNCH_ARGS+=(--tail)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  LAUNCH_ARGS+=(--dry-run --foreground)
fi

PROFILE_OVERRIDES=(
  "data.train_batch_size=${PROFILE_TRAIN_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PROFILE_TRAIN_BATCH_SIZE}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${PROFILE_ROLLOUT_TP}"
  "actor_rollout_ref.worker_placement.actor_rollout.n_gpus_per_node=${PROFILE_ACTOR_GPUS}"
  "actor_rollout_ref.worker_placement.ref_policy.n_gpus_per_node=${PROFILE_REF_GPUS}"
  "trainer.n_gpus_per_node=${PROFILE_ACTOR_GPUS}"
  "ray_kwargs.ray_init.num_cpus=${PROFILE_RAY_CPUS}"
  "mopd_audit.output_dir=audit/${RUN_ID}"
  "trainer.default_local_dir=checkpoints/MOPD/${RUN_ID}"
  "trainer.experiment_name=${RUN_ID}"
)

cat <<EOF
[$(date '+%F %T')] >>> STEP 4/4 launch training
== Qwen30B MOPD bootstrap ==
CHECKOUT_DIR=${CHECKOUT_DIR}
REPO_URL=${REPO_URL}
REPO_REF=${REPO_REF}
BUNDLE_ZIP=${BUNDLE_ZIP}
GPU_PROFILE=${GPU_PROFILE}
GPU_IDS=${GPU_IDS}
BASE_CONFIG=${BASE_CONFIG}
RUN_ID=${RUN_ID}
MODEL_ROOT=${MODEL_ROOT}
DATA_DIR=${DATA_DIR}
DOWNLOAD_DATA=${DOWNLOAD_DATA}
DOWNLOAD_MODELS=${DOWNLOAD_MODELS}
MODEL_BACKEND=${MODEL_BACKEND}
WANDB_MODE=${WANDB_MODE}
EOF

GPU_IDS="${GPU_IDS}" \
MOPD_REMOTE_CONDA_ENV="${CONDA_ROOT}/envs/${ENV_NAME}" \
MOPD_REMOTE_CONDA_ROOT="${CONDA_ROOT}" \
  scripts/start_remote_mopd_training.sh \
    "${BASE_CONFIG}" \
    --run-id "${RUN_ID}" \
    "${LAUNCH_ARGS[@]}" \
    -- \
    "${PROFILE_OVERRIDES[@]}" \
    "${EXTRA_HYDRA_OVERRIDES[@]}"

log_done "STEP 4/4 launch training command finished"
