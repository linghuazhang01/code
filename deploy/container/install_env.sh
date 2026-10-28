#!/usr/bin/env bash
# Container setup, step 1 of 2: install only the OPD environment, no data and no models (step 2 is
# deploy/container/download_data_models.sh). Works in a `docker build` RUN line (no GPU needed) and in a container's
# start-up script. Run it from the code checkout.
# The environment is the one training + evaluation stack: torch 2.6.0, vLLM 0.8.5.post1, transformers 4.51.3
# (environment.yml declares the packages, deploy/env.lock.txt pins every pip package). It goes into its own Miniforge
# under --prefix, so the Python and CUDA packages that come with the image are left alone.
#
#   bash deploy/container/install_env.sh                                  # Miniforge in /opt/miniforge3
#   bash deploy/container/install_env.sh --prefix /mnt/disk/miniforge3    # persistent disk, mounted at the same path at
#                                                                         # every start: later starts skip the install
#   bash deploy/container/install_env.sh --clean                          # docker build: drop conda and pip caches
#   bash deploy/container/install_env.sh --verify 0,1,2,3                 # on the GPUs: CUDA check + NCCL check
#   source /opt/miniforge3/opd_env.sh                                     # afterwards: PATH, PYTHON, PYTHONPATH
#
# Options:  --prefix DIR (default /opt/miniforge3; the env is DIR/envs/mopd-verl)  --clean  --verify GPU_IDS
#           --force (rebuild the env)
# Network:  HTTPS_PROXY / HTTP_PROXY / NO_PROXY work as usual for curl, conda, pip and git. Without a proxy:
#           GITHUB_PROXY=https://ghfast.top   prefix for github.com (Miniforge installer, flash-attn wheel, one git package)
#           PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
#           CONDA_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud   (serves conda-forge)
# Needs:    Linux x86_64, bash, curl, git; at run time an NVIDIA driver with CUDA 12.4 support and no Blackwell GPU
#           (torch 2.6 has no kernels for compute capability >= 10).
set -euo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Same pins as deploy/remote/common.sh
REQ_TORCH=2.6.0
REQ_VLLM=0.8.5.post1
REQ_TRANSFORMERS=4.51.3
MINIFORGE_VERSION=26.5.3-0
MINIFORGE_SHA256=14db468222ad564658656f769506056209b6dc375f5e7dfd31eb5ebbf08fa529
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"

PREFIX=/opt/miniforge3; CLEAN=0; VERIFY=""; FORCE=0
while (( $# )); do
  case "$1" in
    --prefix) PREFIX="${2:?--prefix needs a directory}"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    --verify) VERIFY="${2:?--verify needs GPU ids, e.g. 0,1,2,3}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[[ "${PREFIX}" == /* ]] || { echo "--prefix must be an absolute path" >&2; exit 2; }
PREFIX="${PREFIX%/}"
ENV_PREFIX="${PREFIX}/envs/mopd-verl"
PY="${ENV_PREFIX}/bin/python"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
die() { log "ERROR: $*" >&2; exit 2; }
gh_url() { if [[ -n "${GITHUB_PROXY:-}" ]]; then printf '%s/%s\n' "${GITHUB_PROXY%/}" "$1"; else printf '%s\n' "$1"; fi; }

gpu_check() {  # no GPU (docker build) is fine; a Blackwell GPU is not
  local cap
  if ! cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')" || [[ -z "${cap}" ]]; then
    log "no GPU visible (normal during docker build)"
    return 0
  fi
  nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv,noheader
  [[ "${cap%%.*}" -lt 10 ]] || die "GPU compute capability ${cap} (Blackwell): torch ${REQ_TORCH} cannot run on it"
}

check_stack() {  # the env must be exactly the unified stack
  "${PY}" - "${REQ_TORCH}" "${REQ_VLLM}" "${REQ_TRANSFORMERS}" <<'PY'
import sys
import importlib.metadata as md
want = dict(zip(("torch", "vllm", "transformers"), sys.argv[1:4]))
have = {k: md.version(k) for k in want}
have["flash_attn"] = md.version("flash_attn")
print("stack:", " ".join(f"{k}={v}" for k, v in have.items()))
bad = {k: (have[k], v) for k, v in want.items() if have[k].split("+")[0] != v}
if bad:
    sys.exit(f"stack differs from the unified one: {bad}")
PY
}

log "OPD environment: prefix ${PREFIX}, code ${CODE_DIR}"
[[ "$(uname -s)-$(uname -m)" == Linux-x86_64 ]] || die "environment.yml targets Linux x86_64, this is $(uname -s)-$(uname -m)"
for c in curl git sha256sum; do
  command -v "${c}" >/dev/null || die "${c} is missing (Debian/Ubuntu image: apt-get install -y curl git)"
done
[[ -f "${CODE_DIR}/environment.yml" && -f "${CODE_DIR}/deploy/env.lock.txt" ]] \
  || die "no environment.yml / deploy/env.lock.txt under ${CODE_DIR}; run this script from the OPD code checkout"
gpu_check

# Network settings apply to this script only: no ~/.gitconfig, ~/.condarc or pip.conf is written
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
if [[ -n "${PIP_MIRROR:-}" ]]; then export PIP_INDEX_URL="${PIP_MIRROR}"; fi
if [[ -n "${GITHUB_PROXY:-}" ]]; then  # pip's git+https install goes through the prefix proxy too
  export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="url.${GITHUB_PROXY%/}/https://github.com/.insteadOf" GIT_CONFIG_VALUE_0="https://github.com/"
fi
{
  printf 'channels:\n  - conda-forge\nchannel_priority: strict\n'
  if [[ -n "${CONDA_MIRROR:-}" ]]; then printf 'custom_channels:\n  conda-forge: %s\n' "${CONDA_MIRROR%/}"; fi
} > "${WORK}/condarc"
export CONDARC="${WORK}/condarc" PIP_CONSTRAINT="${CODE_DIR}/deploy/env.lock.txt" PIP_ROOT_USER_ACTION=ignore
if (( CLEAN )); then export PIP_NO_CACHE_DIR=1; fi

# 1. Miniforge
if [[ -x "${PREFIX}/bin/conda" ]]; then
  log "Miniforge already at ${PREFIX}"
else
  [[ ! -e "${PREFIX}" || -z "$(ls -A "${PREFIX}")" ]] || die "${PREFIX} exists, is not empty and has no conda; pick another --prefix"
  rmdir "${PREFIX}" 2>/dev/null || true
  mkdir -p "$(dirname "${PREFIX}")"
  log "Miniforge ${MINIFORGE_VERSION} -> ${PREFIX}"
  curl -fL --retry 5 -o "${WORK}/miniforge.sh" "$(gh_url "${MINIFORGE_URL}")"
  echo "${MINIFORGE_SHA256}  ${WORK}/miniforge.sh" | sha256sum -c -
  bash "${WORK}/miniforge.sh" -b -p "${PREFIX}"
fi
CONDA="${PREFIX}/bin/conda"

# 2. the mopd-verl env
if [[ -x "${PY}" ]] && (( ! FORCE )) && check_stack >/dev/null 2>&1; then
  log "mopd-verl already installed at ${ENV_PREFIX}; skipping (--force rebuilds it)"
else
  if (( FORCE )) || [[ -d "${ENV_PREFIX}" && ! -x "${PY}" ]]; then  # --force, or what a failed create left behind
    rm -rf "${ENV_PREFIX}"
  fi
  # only github.com URLs change, and only when a prefix proxy is in use
  if [[ -n "${GITHUB_PROXY:-}" ]]; then
    sed "s#https://github.com/#${GITHUB_PROXY%/}/https://github.com/#g" "${CODE_DIR}/environment.yml" > "${WORK}/environment.yml"
  else
    cp "${CODE_DIR}/environment.yml" "${WORK}/environment.yml"
  fi
  if [[ -x "${PY}" ]]; then
    log "updating ${ENV_PREFIX}: its versions differ from the pinned stack"
    "${CONDA}" env update -p "${ENV_PREFIX}" -f "${WORK}/environment.yml" --prune
  else
    log "creating ${ENV_PREFIX} from environment.yml, pip constraints deploy/env.lock.txt"
    "${CONDA}" env create -p "${ENV_PREFIX}" -f "${WORK}/environment.yml"
  fi
fi
check_stack || die "${ENV_PREFIX} is not torch ${REQ_TORCH} / vLLM ${REQ_VLLM} / transformers ${REQ_TRANSFORMERS}"
if (( CLEAN )); then "${CONDA}" clean -afy >/dev/null; log "conda package caches removed"; fi

# 3. activation file
cat > "${PREFIX}/opd_env.sh" <<EOF
# source ${PREFIX}/opd_env.sh   (written by deploy/container/install_env.sh, $(date '+%F %T'))
# Puts the mopd-verl env first on PATH and sets PYTHON / PYTHONPATH for start.sh and the scripts in the checkout.
# When the checkout is somewhere else at run time, set OPD_CODE_DIR before sourcing.
export OPD_CODE_DIR="\${OPD_CODE_DIR:-${CODE_DIR}}"
export PATH="${ENV_PREFIX}/bin:\${PATH}"
export PYTHON="${PY}"
export PYTHONPATH="\${OPD_CODE_DIR}:\${OPD_CODE_DIR}/third_party/verl\${PYTHONPATH:+:\${PYTHONPATH}}"
EOF

# 4. optional GPU check (needs the GPUs, so not during docker build)
if [[ -n "${VERIFY}" ]]; then
  log "verify on GPUs ${VERIFY}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${VERIFY%%,*}" "${PY}" - <<'PY' || die "CUDA check failed"
import torch, vllm, flash_attn  # noqa: F401  (imports are part of the check)
assert torch.cuda.is_available(), "CUDA is unavailable: no GPU in the container, or a driver without CUDA 12.4 support"
a = torch.randn(64, 64, device="cuda"); b = torch.randn(64, 64, device="cuda")
assert torch.isfinite(a @ b).all()
print("CUDA ok:", torch.cuda.get_device_name(0))
PY
  if [[ "${VERIFY}" == *,* ]]; then
    PYTHON="${PY}" bash "${CODE_DIR}/scripts/gpu_connectivity_check.sh" "${VERIFY}" || die "multi-GPU check failed on ${VERIFY}"
  fi
fi

log "done: ${ENV_PREFIX}"
log "use it with: source ${PREFIX}/opd_env.sh"
