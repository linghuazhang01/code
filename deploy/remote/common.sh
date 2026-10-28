#!/usr/bin/env bash
# Shared layout and network settings for the remote half of code/deploy. Sourced by the other remote scripts.
# Everything lives under REMOTE_ROOT (deploy/target.env); network routes come from state/net.resolved.env (net_check.sh).

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$(cd "${DEPLOY_DIR}/.." && pwd)"
# shellcheck source=/dev/null
[[ -f "${DEPLOY_DIR}/target.env" ]] && source "${DEPLOY_DIR}/target.env"
# shellcheck source=/dev/null
[[ -f "${DEPLOY_DIR}/network.env" ]] && source "${DEPLOY_DIR}/network.env"
# Run on the machine itself without target.env: everything goes next to the code checkout
REMOTE_ROOT="${REMOTE_ROOT:-$(dirname "${CODE_DIR}")}"

# The one training + evaluation stack (author decision 2026-10-28); env and verify phases refuse anything else.
REQ_TORCH=2.6.0
REQ_VLLM=0.8.5.post1
REQ_TRANSFORMERS=4.51.3
CONDA_ROOT="${REMOTE_ROOT}/miniforge3"
ENV_PREFIX="${CONDA_ROOT}/envs/mopd-verl"
TOOLS_PREFIX="${CONDA_ROOT}/envs/opd-tools"   # small env for downloads, so data/models do not wait for the big env
PYTHON="${ENV_PREFIX}/bin/python"
TOOLS_PYTHON="${TOOLS_PREFIX}/bin/python"
DATA_DIR="${CODE_DIR}/data/G-OPD-Training-Data"
EVAL_DATA_DIR="${CODE_DIR}/data/eval_data"
GOPD_DIR="${DATA_DIR}/.eval-source/G-OPD"
GOPD_COMMIT=37371a4c31ad7947746200d234161769191f4748
MODEL_ROOT="${REMOTE_ROOT}/models"          # = <parent of code>/models, the default of scripts/download_*.sh
CKPT_ROOT="${REMOTE_ROOT}/eval_models"
EVAL_OUT="${REMOTE_ROOT}/eval_outputs"
LCB_EXTRA="${REMOTE_ROOT}/lcb_extra"
STATE_DIR="${REMOTE_ROOT}/state"
LOG_DIR="${REMOTE_ROOT}/logs"
CKPT_REPO=icemoon28/opd-checkpoints
MINIFORGE_VERSION=26.5.3-0
MINIFORGE_SHA256=14db468222ad564658656f769506056209b6dc375f5e7dfd31eb5ebbf08fa529
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"

GPU_IDS="${GPU_IDS:-0}"
MIN_FREE_GIB="${MIN_FREE_GIB:-100}"
export HF_HOME="${REMOTE_ROOT}/hf_home"
export PIP_CACHE_DIR="${REMOTE_ROOT}/pip_cache"
export TMPDIR="${REMOTE_ROOT}/tmp"
mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
die() { log "ERROR: $*" >&2; exit 2; }

free_gib() { df -BG --output=avail "$1" 2>/dev/null | tail -n 1 | tr -dc '0-9'; }
disk_ok() {
  local free; free="$(free_gib "${REMOTE_ROOT}")"
  [[ -n "${free}" && "${free}" -ge "${MIN_FREE_GIB}" ]] || { log "STOP: ${REMOTE_ROOT} has ${free:-?} GiB free < ${MIN_FREE_GIB} GiB"; return 1; }
}

# Prefix a github.com URL with GITHUB_PREFIX (prefix-style proxy) when the resolved route needs it.
gh_url() { if [[ -n "${GITHUB_PREFIX:-}" ]]; then printf '%s/%s\n' "${GITHUB_PREFIX%/}" "$1"; else printf '%s\n' "$1"; fi; }

# Export the routes chosen by net_check.sh. Uploads must not use HF_ENDPOINT (mirrors are read-only); upload code
# passes endpoint=https://huggingface.co explicitly.
apply_network() {
  # shellcheck source=/dev/null
  [[ -f "${STATE_DIR}/net.resolved.env" ]] && source "${STATE_DIR}/net.resolved.env"
  if [[ -n "${NET_PROXY:-}" ]]; then
    export http_proxy="${NET_PROXY}" https_proxy="${NET_PROXY}" HTTP_PROXY="${NET_PROXY}" HTTPS_PROXY="${NET_PROXY}"
    export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
  fi
  [[ -n "${HF_ENDPOINT_RESOLVED:-}" ]] && export HF_ENDPOINT="${HF_ENDPOINT_RESOLVED}"
  [[ -n "${PIP_INDEX_RESOLVED:-}" ]] && export PIP_INDEX_URL="${PIP_INDEX_RESOLVED}"
  if [[ -n "${GITHUB_PREFIX:-}" ]]; then  # git clone / pip git+https through the prefix proxy, without touching ~/.gitconfig
    export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="url.${GITHUB_PREFIX%/}/https://github.com/.insteadOf" GIT_CONFIG_VALUE_0="https://github.com/"
  fi
  {
    printf 'channels:\n  - conda-forge\nchannel_priority: strict\nshow_channel_urls: true\n'
    [[ -n "${CONDA_MIRROR_RESOLVED:-}" ]] && printf 'custom_channels:\n  conda-forge: %s\n' "${CONDA_MIRROR_RESOLVED%/}"
  } > "${STATE_DIR}/condarc"
  export CONDARC="${STATE_DIR}/condarc"
  export HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120
}
apply_network
