#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  REMOTE=root@host REMOTE_PORT=22 scripts/deploy_qwen30b_mopd_zip_remote.sh [-- <extra hydra overrides...>]

Build a local code+data zip bundle, upload it to the remote data disk, and run
the remote bootstrap. Models and the conda environment are installed on the
remote host.

Required:
  REMOTE=root@connect.westb.seetacloud.com

Common knobs:
  REMOTE_PORT=22
  REMOTE_WORKDIR=/root/autodl-tmp/opd_mopd
  BUNDLE_ZIP=<auto-created by package_qwen30b_mopd_bundle.sh>
  GPU_PROFILE=8gpu                 # 8gpu or 4gpu
  DRY_RUN=0
  INSTALL_ENV=1
  PREPARE_ASSETS=1
  MODEL_BACKEND=modelscope
  DOWNLOAD_DATA=0                  # data comes from the uploaded zip
  DOWNLOAD_MODELS=1
  INSTALL_VERL_DEPS=1
  INSTALL_M2RL_IF_DEPS=1
  MIN_FREE_GB=100

Optional password automation:
  export SSHPASS=...
  # sshpass is used only when SSHPASS is set and sshpass is installed.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE="${REMOTE:-}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/root/autodl-tmp/opd_mopd}"
BUNDLE_ZIP="${BUNDLE_ZIP:-}"
GPU_PROFILE="${GPU_PROFILE:-8gpu}"
DRY_RUN="${DRY_RUN:-0}"
INSTALL_ENV="${INSTALL_ENV:-1}"
PREPARE_ASSETS="${PREPARE_ASSETS:-1}"
MODEL_BACKEND="${MODEL_BACKEND:-modelscope}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
REQUIRE_4DOMAIN_TRAIN_DATA="${REQUIRE_4DOMAIN_TRAIN_DATA:-1}"
REQUIRE_MODELS="${REQUIRE_MODELS:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
BUNDLE_REPLACE_EXISTING="${BUNDLE_REPLACE_EXISTING:-1}"
ENV_NAME="${ENV_NAME:-mopd-verl}"
INSTALL_VERL_DEPS="${INSTALL_VERL_DEPS:-1}"
INSTALL_M2RL_IF_DEPS="${INSTALL_M2RL_IF_DEPS:-1}"
CHECK_MOPD_DATA="${CHECK_MOPD_DATA:-0}"
PULL_GIT_LFS_DATA="${PULL_GIT_LFS_DATA:-0}"
INSTALL_GIT_LFS="${INSTALL_GIT_LFS:-1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
FOREGROUND="${FOREGROUND:-0}"
TAIL_LOG="${TAIL_LOG:-0}"

if [[ -z "${REMOTE}" ]]; then
  usage >&2
  exit 2
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

log_step() {
  printf '\n[%s] >>> %s\n' "$(date '+%F %T')" "$1"
}

log_done() {
  printf '[%s] <<< %s\n' "$(date '+%F %T')" "$1"
}

ssh_base=(ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no)
scp_base=(scp -P "${REMOTE_PORT}" -o StrictHostKeyChecking=no)
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  ssh_base=(sshpass -e "${ssh_base[@]}")
  scp_base=(sshpass -e "${scp_base[@]}")
fi

if [[ -z "${BUNDLE_ZIP}" ]]; then
  log_step "STEP 1/5 local package: code + four-domain data zip"
  "${SCRIPT_DIR}/package_qwen30b_mopd_bundle.sh"
  BUNDLE_ZIP="$(ls -t "${CODE_DIR}"/temp/opd_qwen30b_mopd_bundle_*.zip | head -n 1)"
  log_done "STEP 1/5 local package finished"
else
  log_step "STEP 1/5 local package skipped: BUNDLE_ZIP=${BUNDLE_ZIP}"
  [[ -f "${BUNDLE_ZIP}" ]] || {
    echo "BUNDLE_ZIP does not exist: ${BUNDLE_ZIP}" >&2
    exit 2
  }
  log_done "STEP 1/5 local package ready"
fi

remote_bundle="${REMOTE_WORKDIR}/bundles/$(basename "${BUNDLE_ZIP}")"
remote_bootstrap="${REMOTE_WORKDIR}/bootstrap_qwen30b_mopd_training.sh"

log_step "STEP 2/5 remote prepare directories: ${REMOTE}:${REMOTE_WORKDIR}"
"${ssh_base[@]}" "${REMOTE}" "mkdir -p '${REMOTE_WORKDIR}/bundles'"
log_done "STEP 2/5 remote directories ready"

log_step "STEP 3/5 upload bundle and bootstrap"
"${scp_base[@]}" "${BUNDLE_ZIP}" "${REMOTE}:${remote_bundle}"
"${scp_base[@]}" "${SCRIPT_DIR}/bootstrap_qwen30b_mopd_training.sh" "${REMOTE}:${remote_bootstrap}"
"${ssh_base[@]}" "${REMOTE}" "chmod +x '${remote_bootstrap}'"
log_done "STEP 3/5 upload finished"

log_step "STEP 4/5 remote bootstrap: env/model install + dry-run/train launch"
remote_command=(
  "BUNDLE_ZIP='${remote_bundle}'"
  "CHECKOUT_DIR='${REMOTE_WORKDIR}/OPD-code'"
  "BUNDLE_EXTRACT_DIR='${REMOTE_WORKDIR}/bundle_extract'"
  "BUNDLE_REPLACE_EXISTING='${BUNDLE_REPLACE_EXISTING}'"
  "MODEL_ROOT='${REMOTE_WORKDIR}/models'"
  "CONDA_ROOT='${REMOTE_WORKDIR}/miniconda3'"
  "ENV_NAME='${ENV_NAME}'"
  "GPU_PROFILE='${GPU_PROFILE}'"
  "DRY_RUN='${DRY_RUN}'"
  "INSTALL_ENV='${INSTALL_ENV}'"
  "PREPARE_ASSETS='${PREPARE_ASSETS}'"
  "INSTALL_VERL_DEPS='${INSTALL_VERL_DEPS}'"
  "INSTALL_M2RL_IF_DEPS='${INSTALL_M2RL_IF_DEPS}'"
  "CHECK_MOPD_DATA='${CHECK_MOPD_DATA}'"
  "PULL_GIT_LFS_DATA='${PULL_GIT_LFS_DATA}'"
  "INSTALL_GIT_LFS='${INSTALL_GIT_LFS}'"
  "MODEL_BACKEND='${MODEL_BACKEND}'"
  "DOWNLOAD_DATA='${DOWNLOAD_DATA}'"
  "DOWNLOAD_MODELS='${DOWNLOAD_MODELS}'"
  "REQUIRE_4DOMAIN_TRAIN_DATA='${REQUIRE_4DOMAIN_TRAIN_DATA}'"
  "REQUIRE_MODELS='${REQUIRE_MODELS}'"
  "MIN_FREE_GB='${MIN_FREE_GB}'"
  "WANDB_MODE='${WANDB_MODE}'"
  "FOREGROUND='${FOREGROUND}'"
  "TAIL_LOG='${TAIL_LOG}'"
  "bash '${remote_bootstrap}'"
)
if [[ "${#EXTRA_HYDRA_OVERRIDES[@]}" -gt 0 ]]; then
  quoted_overrides=()
  for override in "${EXTRA_HYDRA_OVERRIDES[@]}"; do
    quoted_overrides+=("$(printf "%q" "${override}")")
  done
  remote_command+=("--" "${quoted_overrides[@]}")
fi
"${ssh_base[@]}" "${REMOTE}" "${remote_command[*]}"
log_done "STEP 4/5 remote bootstrap finished"

log_step "STEP 5/5 remote status"
"${ssh_base[@]}" "${REMOTE}" "df -h '${REMOTE_WORKDIR}' || true; screen -ls || true"
log_done "STEP 5/5 remote status finished"
