#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_qwen30b_mopd_zip_train.sh [-- <extra hydra overrides...>]

This script is self-contained when placed next to split parts:
  opd_qwen30b_mopd_code_data.zip.part-00
  opd_qwen30b_mopd_code_data.zip.part-01

It uploads the code+data zip parts to the remote data disk, reconstructs the
zip remotely, extracts the bootstrap script locally from the parts, uploads it,
installs/validates the environment and models remotely, then starts real
Qwen30B MOPD training.

Common environment knobs:
  BUNDLE_ZIP=./opd_qwen30b_mopd_code_data.zip
  REMOTE=root@connect.westb.seetacloud.com
  REMOTE_PORT=16968
  REMOTE_WORKDIR=/root/autodl-tmp/opd_mopd_zip_train
  GPU_PROFILE=4gpu
  MODEL_BACKEND=modelscope
  INSTALL_VERL_DEPS=1
  INSTALL_M2RL_IF_DEPS=1
  MIN_FREE_GB=100
  WANDB_MODE=disabled
  FOREGROUND=0
  TAIL_LOG=0
  RESET_REMOTE=0

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

BUNDLE_ZIP="${BUNDLE_ZIP:-${SCRIPT_DIR}/opd_qwen30b_mopd_code_data.zip}"
REMOTE="${REMOTE:-root@connect.westb.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-16968}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/root/autodl-tmp/opd_mopd_zip_train}"
GPU_PROFILE="${GPU_PROFILE:-4gpu}"
MODEL_BACKEND="${MODEL_BACKEND:-modelscope}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
ENV_NAME="${ENV_NAME:-mopd-verl}"
INSTALL_VERL_DEPS="${INSTALL_VERL_DEPS:-1}"
INSTALL_M2RL_IF_DEPS="${INSTALL_M2RL_IF_DEPS:-1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
FOREGROUND="${FOREGROUND:-0}"
TAIL_LOG="${TAIL_LOG:-0}"
RESET_REMOTE="${RESET_REMOTE:-0}"

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

assert_safe_remote_workdir() {
  case "${REMOTE_WORKDIR}" in
    /root/autodl-tmp/opd_*|/tmp/opd_*|/workspace/opd_*)
      ;;
    *)
      cat >&2 <<EOF
Refusing unsafe REMOTE_WORKDIR=${REMOTE_WORKDIR}
Use a dedicated data-disk path like /root/autodl-tmp/opd_mopd_zip_train.
EOF
      exit 2
      ;;
  esac
}

command -v unzip >/dev/null 2>&1 || {
  echo "unzip is required to extract the remote bootstrap from the zip." >&2
  exit 2
}
command -v ssh >/dev/null 2>&1 || {
  echo "ssh is required." >&2
  exit 2
}
command -v scp >/dev/null 2>&1 || {
  echo "scp is required." >&2
  exit 2
}

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

BUNDLE_PARTS=()
LOCAL_BUNDLE_FOR_READ="${BUNDLE_ZIP}"
if [[ ! -f "${BUNDLE_ZIP}" ]]; then
  shopt -s nullglob
  BUNDLE_PARTS=("${BUNDLE_ZIP}".part-*)
  shopt -u nullglob
  if [[ "${#BUNDLE_PARTS[@]}" -eq 0 ]]; then
    echo "Missing bundle zip or split parts: ${BUNDLE_ZIP}(.part-*)" >&2
    exit 2
  fi
  LOCAL_BUNDLE_FOR_READ="${TMP_DIR}/$(basename "${BUNDLE_ZIP}")"
  cat "${BUNDLE_PARTS[@]}" > "${LOCAL_BUNDLE_FOR_READ}"
fi

BOOTSTRAP_LOCAL="${TMP_DIR}/bootstrap_qwen30b_mopd_training.sh"
unzip -p "${LOCAL_BUNDLE_FOR_READ}" OPD-code/scripts/bootstrap_qwen30b_mopd_training.sh > "${BOOTSTRAP_LOCAL}" || {
  echo "Failed to extract OPD-code/scripts/bootstrap_qwen30b_mopd_training.sh from ${BUNDLE_ZIP}" >&2
  exit 2
}
chmod +x "${BOOTSTRAP_LOCAL}"

ssh_base=(ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=no)
scp_base=(scp -P "${REMOTE_PORT}" -o StrictHostKeyChecking=no)
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  ssh_base=(sshpass -e "${ssh_base[@]}")
  scp_base=(sshpass -e "${scp_base[@]}")
fi

REMOTE_BUNDLE="${REMOTE_WORKDIR}/bundles/$(basename "${BUNDLE_ZIP}")"
REMOTE_BOOTSTRAP="${REMOTE_WORKDIR}/bootstrap_qwen30b_mopd_training.sh"
REMOTE_PART_DIR="${REMOTE_WORKDIR}/bundles/$(basename "${BUNDLE_ZIP}").parts"
remote_workdir_q="$(quote "${REMOTE_WORKDIR}")"
remote_bundle_q="$(quote "${REMOTE_BUNDLE}")"
remote_bootstrap_q="$(quote "${REMOTE_BOOTSTRAP}")"
remote_part_dir_q="$(quote "${REMOTE_PART_DIR}")"

cat <<EOF
== Qwen30B MOPD zip train ==
BUNDLE_ZIP=${BUNDLE_ZIP}
REMOTE=${REMOTE}
REMOTE_PORT=${REMOTE_PORT}
REMOTE_WORKDIR=${REMOTE_WORKDIR}
GPU_PROFILE=${GPU_PROFILE}
MODEL_BACKEND=${MODEL_BACKEND}
INSTALL_VERL_DEPS=${INSTALL_VERL_DEPS}
INSTALL_M2RL_IF_DEPS=${INSTALL_M2RL_IF_DEPS}
MIN_FREE_GB=${MIN_FREE_GB}
WANDB_MODE=${WANDB_MODE}
FOREGROUND=${FOREGROUND}
TAIL_LOG=${TAIL_LOG}
RESET_REMOTE=${RESET_REMOTE}
EOF

if [[ "${RESET_REMOTE}" == "1" ]]; then
  assert_safe_remote_workdir
  log_step "STEP 0/5 remote reset: ${REMOTE}:${REMOTE_WORKDIR}"
  "${ssh_base[@]}" "${REMOTE}" "set -euo pipefail
    rm -rf ${remote_workdir_q}
    mkdir -p ${remote_workdir_q}/bundles
    df -h ${remote_workdir_q}
  "
  log_done "STEP 0/5 remote reset finished"
fi

log_step "STEP 1/5 remote prepare directories"
"${ssh_base[@]}" "${REMOTE}" "mkdir -p ${remote_workdir_q}/bundles"
log_done "STEP 1/5 remote directories ready"

log_step "STEP 2/5 upload code+data zip parts and bootstrap"
if [[ "${#BUNDLE_PARTS[@]}" -gt 0 ]]; then
  "${ssh_base[@]}" "${REMOTE}" "rm -rf ${remote_part_dir_q}; mkdir -p ${remote_part_dir_q}"
  "${scp_base[@]}" "${BUNDLE_PARTS[@]}" "${REMOTE}:${REMOTE_PART_DIR}/"
  "${ssh_base[@]}" "${REMOTE}" "cat ${remote_part_dir_q}/$(basename "${BUNDLE_ZIP}").part-* > ${remote_bundle_q}"
else
  "${scp_base[@]}" "${BUNDLE_ZIP}" "${REMOTE}:${REMOTE_BUNDLE}"
fi
"${scp_base[@]}" "${BOOTSTRAP_LOCAL}" "${REMOTE}:${REMOTE_BOOTSTRAP}"
"${ssh_base[@]}" "${REMOTE}" "chmod +x ${remote_bootstrap_q}"
log_done "STEP 2/5 upload finished"

log_step "STEP 3/5 remote environment/model preparation and real training launch"
remote_command=(
  "BUNDLE_ZIP=${remote_bundle_q}"
  "CHECKOUT_DIR=${remote_workdir_q}/OPD-code"
  "BUNDLE_EXTRACT_DIR=${remote_workdir_q}/bundle_extract"
  "BUNDLE_REPLACE_EXISTING=1"
  "MODEL_ROOT=${remote_workdir_q}/models"
  "CONDA_ROOT=${remote_workdir_q}/miniconda3"
  "ENV_NAME='${ENV_NAME}'"
  "GPU_PROFILE='${GPU_PROFILE}'"
  "DRY_RUN=0"
  "INSTALL_ENV=1"
  "INSTALL_VERL_DEPS='${INSTALL_VERL_DEPS}'"
  "INSTALL_M2RL_IF_DEPS='${INSTALL_M2RL_IF_DEPS}'"
  "PREPARE_ASSETS=1"
  "MODEL_BACKEND='${MODEL_BACKEND}'"
  "DOWNLOAD_DATA=0"
  "DOWNLOAD_MODELS=1"
  "REQUIRE_4DOMAIN_TRAIN_DATA=1"
  "REQUIRE_MODELS=1"
  "MIN_FREE_GB='${MIN_FREE_GB}'"
  "WANDB_MODE='${WANDB_MODE}'"
  "FOREGROUND='${FOREGROUND}'"
  "TAIL_LOG='${TAIL_LOG}'"
  "bash ${remote_bootstrap_q}"
)
if [[ "${#EXTRA_HYDRA_OVERRIDES[@]}" -gt 0 ]]; then
  quoted_overrides=()
  for override in "${EXTRA_HYDRA_OVERRIDES[@]}"; do
    quoted_overrides+=("$(quote "${override}")")
  done
  remote_command+=("--" "${quoted_overrides[@]}")
fi
"${ssh_base[@]}" "${REMOTE}" "${remote_command[*]}"
log_done "STEP 3/5 remote launch command finished"

log_step "STEP 4/5 remote training status"
"${ssh_base[@]}" "${REMOTE}" "df -h ${remote_workdir_q} || true; screen -ls || true; ls -t ${remote_workdir_q}/OPD-code/logs/*.log 2>/dev/null | head -n 5 || true"
log_done "STEP 4/5 remote status finished"

log_step "STEP 5/5 done"
log_done "Real training flow submitted"
