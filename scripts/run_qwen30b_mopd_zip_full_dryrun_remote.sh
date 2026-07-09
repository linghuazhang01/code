#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  REMOTE=root@host REMOTE_PORT=22 scripts/run_qwen30b_mopd_zip_full_dryrun_remote.sh [-- <extra hydra overrides...>]

Reset a remote data-disk workdir and run the complete zip-based dry-run flow:
  1. package local code + four-domain data into a zip
  2. upload the zip to the remote workdir
  3. unpack the zip on the remote host
  4. install the conda training environment on the remote data disk
  5. download/validate Qwen3-4B and Qwen3-30B-A3B models on the remote host
  6. run the training command in DRY_RUN mode

Required:
  REMOTE=root@connect.westb.seetacloud.com

Defaults:
  REMOTE_PORT=22
  REMOTE_WORKDIR=/root/autodl-tmp/opd_mopd_full_dryrun
  RESET_REMOTE=1
  GPU_PROFILE=4gpu
  MODEL_BACKEND=modelscope
  MIN_FREE_GB=100
  ENV_NAME=mopd-verl

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

REMOTE="${REMOTE:-}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/root/autodl-tmp/opd_mopd_full_dryrun}"
RESET_REMOTE="${RESET_REMOTE:-1}"
GPU_PROFILE="${GPU_PROFILE:-4gpu}"
MODEL_BACKEND="${MODEL_BACKEND:-modelscope}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
ENV_NAME="${ENV_NAME:-mopd-verl}"
BUNDLE_ZIP="${BUNDLE_ZIP:-}"
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

quote() {
  printf "%q" "$1"
}

ssh_base=(ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no)
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  ssh_base=(sshpass -e "${ssh_base[@]}")
fi

assert_safe_remote_workdir() {
  case "${REMOTE_WORKDIR}" in
    /root/autodl-tmp/opd_*|/tmp/opd_*|/workspace/opd_*)
      ;;
    *)
      cat >&2 <<EOF
Refusing to reset unsafe REMOTE_WORKDIR=${REMOTE_WORKDIR}
Use a dedicated path like /root/autodl-tmp/opd_mopd_full_dryrun.
EOF
      exit 2
      ;;
  esac
}

reset_remote_state() {
  if [[ "${RESET_REMOTE}" != "1" ]]; then
    log_step "STEP 0/6 remote reset skipped: RESET_REMOTE=${RESET_REMOTE}"
    log_done "STEP 0/6 remote reset skipped"
    return
  fi

  assert_safe_remote_workdir
  log_step "STEP 0/6 remote reset: ${REMOTE}:${REMOTE_WORKDIR}"
  local remote_workdir_q
  remote_workdir_q="$(quote "${REMOTE_WORKDIR}")"
  "${ssh_base[@]}" "${REMOTE}" "set -euo pipefail
    mkdir -p ${remote_workdir_q}
    rm -rf \
      ${remote_workdir_q}/OPD-code \
      ${remote_workdir_q}/models \
      ${remote_workdir_q}/miniconda3 \
      ${remote_workdir_q}/bundle_extract \
      ${remote_workdir_q}/bundles \
      ${remote_workdir_q}/bootstrap_qwen30b_mopd_training.sh
    mkdir -p ${remote_workdir_q}/bundles
    ps -ef | grep -E 'main_ppo|run_mopd|git-lfs|git lfs' | grep -v grep || true
    screen -ls || true
    df -h ${remote_workdir_q}
  "
  log_done "STEP 0/6 remote reset finished"
}

reset_remote_state

log_step "STEP 1/6 run zip deploy full dry-run flow"
deploy_args=()
if [[ "${#EXTRA_HYDRA_OVERRIDES[@]}" -gt 0 ]]; then
  deploy_args+=(--)
  deploy_args+=("${EXTRA_HYDRA_OVERRIDES[@]}")
fi

REMOTE="${REMOTE}" \
REMOTE_PORT="${REMOTE_PORT}" \
REMOTE_WORKDIR="${REMOTE_WORKDIR}" \
BUNDLE_ZIP="${BUNDLE_ZIP}" \
GPU_PROFILE="${GPU_PROFILE}" \
DRY_RUN=1 \
INSTALL_ENV=1 \
PREPARE_ASSETS=1 \
MODEL_BACKEND="${MODEL_BACKEND}" \
DOWNLOAD_DATA=0 \
DOWNLOAD_MODELS=1 \
REQUIRE_4DOMAIN_TRAIN_DATA=1 \
REQUIRE_MODELS=1 \
MIN_FREE_GB="${MIN_FREE_GB}" \
BUNDLE_REPLACE_EXISTING=1 \
ENV_NAME="${ENV_NAME}" \
INSTALL_VERL_DEPS=1 \
INSTALL_M2RL_IF_DEPS=1 \
CHECK_MOPD_DATA=0 \
PULL_GIT_LFS_DATA=0 \
INSTALL_GIT_LFS=1 \
WANDB_MODE="${WANDB_MODE}" \
FOREGROUND="${FOREGROUND}" \
TAIL_LOG="${TAIL_LOG}" \
  "${SCRIPT_DIR}/deploy_qwen30b_mopd_zip_remote.sh" "${deploy_args[@]}"
log_done "STEP 1/6 full dry-run flow finished"
