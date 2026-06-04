#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/sync_and_start_remote_mopd.sh <config> [--run-id RUN_ID] [-- <hydra overrides...>]
  scripts/sync_and_start_remote_mopd.sh [<config>] --sync-only

Examples:
  scripts/sync_and_start_remote_mopd.sh --sync-only

  scripts/sync_and_start_remote_mopd.sh configs/mopd_formal_single_a800.yaml --run-id mopd_manual_test

  scripts/sync_and_start_remote_mopd.sh configs/mopd_formal_single_a800.yaml \
    --run-id mopd_bsz128 \
    -- data.train_batch_size=128 data.val_batch_size=128 trainer.val_before_train=false

Notes:
  - <config> is required for launch. It is optional for --sync-only.
  - Relative config paths are resolved in this local repo, then run from the synced remote repo.
  - Training imports verl from this repo's third_party/verl directory after sync.
  - SSH password auth is automatic when local ssh.sh exists; set USE_SSHPASS=0 to disable it.
  - Rsync progress is shown by default; set RSYNC_PROGRESS=0 to disable progress output.
  - Extra Hydra overrides are passed exactly after '--'; the script does not add hidden training overrides.
USAGE
}

quote() {
  printf "%q" "$1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-root@connect.nma1.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-51568}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/opd_mopd}"
REMOTE_CODE_DIR="${REMOTE_CODE_DIR:-${REMOTE_ROOT}/OPD-code}"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-mopd-verl}"
PYTHON_BIN="${PYTHON_BIN:-}"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="${LOG_DIR:-${REMOTE_CODE_DIR}/logs}"
STOP_STALE_RAY="${STOP_STALE_RAY:-1}"
ASSUME_YES="${ASSUME_YES:-0}"
USE_SSHPASS="${USE_SSHPASS:-auto}"
SSH_PASSWORD_FILE="${SSH_PASSWORD_FILE:-${LOCAL_CODE_DIR}/ssh.sh}"
SSH_PASSWORD_LINE="${SSH_PASSWORD_LINE:-2}"
SSH_STRICT_HOST_KEY_CHECKING="${SSH_STRICT_HOST_KEY_CHECKING:-no}"
RSYNC_PROGRESS="${RSYNC_PROGRESS:-1}"
RSYNC_STATS="${RSYNC_STATS:-1}"

SSH_AUTH_PREFIX=()
SSHPASS_TEMP_FILE=""

cleanup_ssh_auth() {
  if [[ -n "${SSHPASS_TEMP_FILE}" && -f "${SSHPASS_TEMP_FILE}" ]]; then
    rm -f "${SSHPASS_TEMP_FILE}"
  fi
}
trap cleanup_ssh_auth EXIT

setup_ssh_auth() {
  case "${USE_SSHPASS}" in
    0|false|False|no|No)
      return
      ;;
  esac

  if [[ ! -f "${SSH_PASSWORD_FILE}" ]]; then
    if [[ "${USE_SSHPASS}" == "1" || "${USE_SSHPASS}" == "true" || "${USE_SSHPASS}" == "yes" ]]; then
      echo "SSH password file not found: ${SSH_PASSWORD_FILE}" >&2
      exit 2
    fi
    return
  fi

  if ! command -v sshpass >/dev/null 2>&1; then
    if [[ "${USE_SSHPASS}" == "1" || "${USE_SSHPASS}" == "true" || "${USE_SSHPASS}" == "yes" ]]; then
      echo "sshpass is required for passwordless launch but was not found." >&2
      exit 2
    fi
    echo "sshpass not found; SSH may prompt for a password." >&2
    return
  fi

  local password
  password="$(sed -n "${SSH_PASSWORD_LINE}p" "${SSH_PASSWORD_FILE}" | tr -d '\r')"
  if [[ -z "${password}" ]]; then
    echo "No password found at ${SSH_PASSWORD_FILE}:${SSH_PASSWORD_LINE}" >&2
    exit 2
  fi

  SSHPASS_TEMP_FILE="$(mktemp "${TMPDIR:-/tmp}/opd_sshpass.XXXXXX")"
  chmod 600 "${SSHPASS_TEMP_FILE}"
  printf "%s\n" "${password}" > "${SSHPASS_TEMP_FILE}"
  SSH_AUTH_PREFIX=(sshpass -f "${SSHPASS_TEMP_FILE}")
}

SSH_BASE_OPTS=(-o "StrictHostKeyChecking=${SSH_STRICT_HOST_KEY_CHECKING}" -p "${REMOTE_PORT}")
RSYNC_RSH="ssh -o StrictHostKeyChecking=${SSH_STRICT_HOST_KEY_CHECKING} -p ${REMOTE_PORT}"
RSYNC_PROGRESS_ARGS=()
if [[ "${RSYNC_PROGRESS}" != "0" && "${RSYNC_PROGRESS}" != "false" && "${RSYNC_PROGRESS}" != "False" ]]; then
  if rsync --help 2>/dev/null | grep -q -- "--info="; then
    RSYNC_PROGRESS_ARGS=(--info=progress2 --human-readable)
  else
    RSYNC_PROGRESS_ARGS=(--progress)
  fi
fi
RSYNC_STATS_ARGS=()
if [[ "${RSYNC_STATS}" != "0" && "${RSYNC_STATS}" != "false" && "${RSYNC_STATS}" != "False" ]]; then
  RSYNC_STATS_ARGS=(--stats)
fi

ssh_remote() {
  "${SSH_AUTH_PREFIX[@]}" ssh "${SSH_BASE_OPTS[@]}" "${REMOTE_HOST}" "$@"
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CONFIG_ARG=""
RUN_ID=""
SYNC_ONLY=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --run-id)
      if [[ $# -lt 2 ]]; then
        echo "--run-id requires a value" >&2
        exit 2
      fi
      RUN_ID="$2"
      shift 2
      ;;
    --sync-only)
      SYNC_ONLY=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    -*)
      echo "Unknown script option: $1" >&2
      echo "Put Hydra overrides after '--'." >&2
      exit 2
      ;;
    *)
      if [[ -n "${CONFIG_ARG}" ]]; then
        echo "Only one config path is allowed. Extra Hydra overrides must go after '--'." >&2
        exit 2
      fi
      CONFIG_ARG="$1"
      shift
      ;;
  esac
done

if [[ -z "${CONFIG_ARG}" && "${SYNC_ONLY}" != "1" ]]; then
  echo "<config> is required unless --sync-only is set." >&2
  usage >&2
  exit 2
fi

if [[ -z "${CONFIG_ARG}" && "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  echo "Hydra overrides are only valid when launching with an explicit config." >&2
  exit 2
fi

REMOTE_CONFIG=""
CONFIG_LABEL="sync_only"
if [[ -n "${CONFIG_ARG}" ]]; then
  if [[ "${CONFIG_ARG}" == /* ]]; then
    REMOTE_CONFIG="${CONFIG_ARG}"
    CONFIG_LABEL="$(basename "${CONFIG_ARG}")"
  else
    if [[ ! -f "${LOCAL_CODE_DIR}/${CONFIG_ARG}" ]]; then
      echo "Local config not found: ${LOCAL_CODE_DIR}/${CONFIG_ARG}" >&2
      exit 2
    fi
    REMOTE_CONFIG="${REMOTE_CODE_DIR}/${CONFIG_ARG}"
    CONFIG_LABEL="${CONFIG_ARG##*/}"
  fi
fi

if [[ "${SYNC_ONLY}" == "1" && -n "${RUN_ID}" ]]; then
  echo "--run-id has no effect with --sync-only." >&2
  exit 2
fi

if [[ "${SYNC_ONLY}" != "1" ]]; then
  if [[ -z "${RUN_ID}" ]]; then
    config_stem="${CONFIG_LABEL%.*}"
    RUN_ID="${config_stem}_$(date +%Y%m%d_%H%M%S)"
  fi
  LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
  GPU_CSV="${LOG_DIR}/${RUN_ID}_gpu.csv"
else
  LOG_FILE=""
  GPU_CSV=""
fi

REMOTE_START_CONFIG_ARG="${REMOTE_CONFIG}"
if [[ -n "${CONFIG_ARG}" && "${CONFIG_ARG}" != /* ]]; then
  REMOTE_START_CONFIG_ARG="${CONFIG_ARG}"
fi

if [[ "${SYNC_ONLY}" != "1" ]]; then
  EXTRA_ARGS_Q=""
  for arg in "${EXTRA_ARGS[@]}"; do
    EXTRA_ARGS_Q+=" $(quote "${arg}")"
  done
else
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    echo "Hydra overrides are only valid when launching, not with --sync-only." >&2
    exit 2
  fi
  EXTRA_ARGS_Q=""
fi

echo "== Local code dir =="
echo "${LOCAL_CODE_DIR}"
echo
echo "== Remote target =="
echo "${REMOTE_HOST}:${REMOTE_CODE_DIR}"
echo
echo "== SSH auth =="
if [[ "${USE_SSHPASS}" == "0" || "${USE_SSHPASS}" == "false" || "${USE_SSHPASS}" == "False" ]]; then
  echo "sshpass disabled; SSH may use keys or prompt interactively"
elif [[ -f "${SSH_PASSWORD_FILE}" && "$(command -v sshpass || true)" != "" ]]; then
  echo "sshpass enabled via ${SSH_PASSWORD_FILE}:${SSH_PASSWORD_LINE}"
else
  echo "sshpass unavailable; SSH may use keys or prompt interactively"
fi
echo
echo "== Rsync progress =="
if [[ "${#RSYNC_PROGRESS_ARGS[@]}" -gt 0 ]]; then
  printf "%s\n" "enabled: ${RSYNC_PROGRESS_ARGS[*]}"
else
  echo "disabled"
fi
echo
if [[ "${SYNC_ONLY}" == "1" ]]; then
  echo "== Mode =="
  echo "sync only; no training will be launched"
  if [[ -n "${REMOTE_CONFIG}" ]]; then
    echo
    echo "== Config path checked locally =="
    echo "${REMOTE_CONFIG}"
  fi
else
  echo "== Explicit config =="
  echo "${REMOTE_CONFIG}"
  echo
  echo "== Run id / log =="
  echo "${RUN_ID}"
  echo "${LOG_FILE}"
fi
echo

if [[ "${ASSUME_YES}" != "1" ]]; then
  if [[ "${SYNC_ONLY}" == "1" ]]; then
    read -r -p "Type SYNC to rsync local code only: " confirm
    expected_confirm="SYNC"
  else
    read -r -p "Type START to rsync local code and launch this config: " confirm
    expected_confirm="START"
  fi
  if [[ "${confirm}" != "${expected_confirm}" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

echo "== Syncing local code to remote =="
setup_ssh_auth
ssh_remote "mkdir -p $(quote "${REMOTE_CODE_DIR}")"
"${SSH_AUTH_PREFIX[@]}" rsync -az --delete --partial \
  "${RSYNC_PROGRESS_ARGS[@]}" \
  "${RSYNC_STATS_ARGS[@]}" \
  --exclude ".git/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".DS_Store" \
  --exclude ".env" \
  --exclude "ssh.sh" \
  --exclude "temp/" \
  --exclude "logs/" \
  --exclude "hf_home/" \
  --exclude "smoke_data/" \
  --exclude "tensorboard_log/" \
  --exclude "checkpoints/" \
  --exclude "audit/" \
  --exclude "eval_outputs/" \
  -e "${RSYNC_RSH}" \
  "${LOCAL_CODE_DIR}/" \
  "${REMOTE_HOST}:${REMOTE_CODE_DIR}/"

if [[ "${SYNC_ONLY}" == "1" ]]; then
  echo "Sync complete. --sync-only was set, so no training was launched."
  exit 0
fi

echo "== Launching remote screen =="
REMOTE_CMD=$(cat <<REMOTE
set -euo pipefail
cd $(quote "${REMOTE_CODE_DIR}")
export CONDA_SH=$(quote "${CONDA_SH}")
export CONDA_ENV=$(quote "${CONDA_ENV}")
export PYTHON_BIN=$(quote "${PYTHON_BIN}")
export GPU_ID=$(quote "${GPU_ID}")
export LOG_DIR=$(quote "${LOG_DIR}")
export STOP_STALE_RAY=$(quote "${STOP_STALE_RAY}")
bash scripts/start_remote_mopd_training.sh $(quote "${REMOTE_START_CONFIG_ARG}") --run-id $(quote "${RUN_ID}")${EXTRA_ARGS_Q:+ --${EXTRA_ARGS_Q}}
REMOTE
)

ssh_remote "bash -lc $(quote "${REMOTE_CMD}")"

echo
echo "== Follow logs =="
echo "ssh -p ${REMOTE_PORT} ${REMOTE_HOST} 'tail -f ${LOG_FILE}'"
