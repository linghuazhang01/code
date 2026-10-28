#!/usr/bin/env bash
# Probe Hugging Face, GitHub, PyPI, conda-forge and W&B from the target and pick one route per service
# (direct, proxy or mirror) according to PROXY_MODE. Writes state/net.resolved.env, which common.sh applies.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rm -f "${STATE_DIR}/net.resolved.env"
unset HF_ENDPOINT PIP_INDEX_URL GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0 http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

command -v curl >/dev/null 2>&1 || die "curl is required on the target"
PROXY_MODE="${PROXY_MODE:-auto}"
case "${PROXY_MODE}" in
  http) PROXY_CANDIDATE="${HTTP_PROXY_URL:?PROXY_MODE=http needs HTTP_PROXY_URL}" ;;
  ssh-reverse) PROXY_CANDIDATE="http://127.0.0.1:${REMOTE_PROXY_PORT:-17890}" ;;
  auto) PROXY_CANDIDATE="${HTTP_PROXY_URL:-}"
        [[ -z "${PROXY_CANDIDATE}" ]] && curl -s -m 3 -o /dev/null "http://127.0.0.1:${REMOTE_PROXY_PORT:-17890}" \
          && PROXY_CANDIDATE="http://127.0.0.1:${REMOTE_PROXY_PORT:-17890}" ;;
  direct|mirror) PROXY_CANDIDATE="" ;;
  *) die "unknown PROXY_MODE ${PROXY_MODE}" ;;
esac

# probe URL [proxy] -> prints the HTTP code (000 = unreachable); any code 200-399 counts as reachable
probe() {
  local args=(-s -L -o /dev/null -m 12 -r 0-1023 -w '%{http_code}')
  [[ -n "${2:-}" ]] && args+=(-x "$2") || args+=(--noproxy '*')
  curl "${args[@]}" "$1" 2>/dev/null || true
}
ok() { [[ "$1" =~ ^[23][0-9][0-9]$ ]]; }

HF_URL="https://huggingface.co/api/models/Qwen/Qwen3-1.7B"
GH_URL="${MINIFORGE_URL}"
PIP_URL="https://pypi.org/simple/pip/"
CONDA_URL="https://conda.anaconda.org/conda-forge/noarch/current_repodata.json"
WANDB_URL="https://api.wandb.ai/"

declare -A DIRECT VIA
for s in hf gh pip conda wandb; do
  case $s in hf) u=$HF_URL ;; gh) u=$GH_URL ;; pip) u=$PIP_URL ;; conda) u=$CONDA_URL ;; wandb) u=$WANDB_URL ;; esac
  DIRECT[$s]=$(probe "$u"); VIA[$s]=000
  [[ -n "${PROXY_CANDIDATE}" ]] && VIA[$s]=$(probe "$u" "${PROXY_CANDIDATE}")
done
M_HF=000; M_GH=000; M_PIP=000; M_CONDA=000
[[ -n "${HF_MIRROR:-}" ]] && M_HF=$(probe "${HF_MIRROR%/}/api/models/Qwen/Qwen3-1.7B")
[[ -n "${GITHUB_PROXY:-}" ]] && M_GH=$(probe "${GITHUB_PROXY%/}/${GH_URL}")
[[ -n "${PIP_MIRROR:-}" ]] && M_PIP=$(probe "${PIP_MIRROR%/}/pip/")
[[ -n "${CONDA_MIRROR:-}" ]] && M_CONDA=$(probe "${CONDA_MIRROR%/}/conda-forge/noarch/current_repodata.json")

NET_PROXY=""; HF_EP=""; GH_PREFIX=""; PIP_IDX=""; CONDA_M=""
proxy_works=0
if [[ -n "${PROXY_CANDIDATE}" ]] && { ok "${VIA[hf]}" || ok "${VIA[gh]}" || ok "${VIA[pip]}"; }; then proxy_works=1; fi
case "${PROXY_MODE}" in
  http|ssh-reverse) (( proxy_works )) || die "proxy ${PROXY_CANDIDATE} does not reach huggingface.co, github.com or pypi.org"
                    NET_PROXY="${PROXY_CANDIDATE}" ;;
  mirror) HF_EP="${HF_MIRROR:-}"; GH_PREFIX="${GITHUB_PROXY:-}"; PIP_IDX="${PIP_MIRROR:-}"; CONDA_M="${CONDA_MIRROR:-}" ;;
  auto)
    all_direct=1; for s in hf gh pip conda; do ok "${DIRECT[$s]}" || all_direct=0; done
    if (( ! all_direct )) && (( proxy_works )); then
      NET_PROXY="${PROXY_CANDIDATE}"
    elif (( ! all_direct )); then  # per-service mirrors where direct fails
      ok "${DIRECT[hf]}" || { ok "${M_HF}" && HF_EP="${HF_MIRROR}"; }
      ok "${DIRECT[gh]}" || { ok "${M_GH}" && GH_PREFIX="${GITHUB_PROXY}"; }
      ok "${DIRECT[pip]}" || { ok "${M_PIP}" && PIP_IDX="${PIP_MIRROR}"; }
      ok "${DIRECT[conda]}" || { ok "${M_CONDA}" && CONDA_M="${CONDA_MIRROR}"; }
    fi ;;
esac

# Can the target talk to huggingface.co itself (needed for uploads; mirrors cannot take uploads)?
UPLOAD_ROUTE=relay
if ok "${DIRECT[hf]}" || { [[ -n "${NET_PROXY}" ]] && ok "${VIA[hf]}"; }; then UPLOAD_ROUTE=direct; fi
case "${UPLOAD_MODE:-auto}" in direct) UPLOAD_ROUTE=direct ;; relay) UPLOAD_ROUTE=relay ;; esac

route() {  # service direct_code via_code mirror_code chosen_mirror
  if [[ -n "${NET_PROXY}" ]]; then ok "$3" && echo "proxy" || echo "UNREACHABLE"
  elif [[ -n "$5" ]]; then echo "mirror"
  elif ok "$2"; then echo "direct"
  else echo "UNREACHABLE"; fi
}
{
  printf '%-12s %-7s %-7s %-7s %s\n' service direct proxy mirror chosen
  printf '%-12s %-7s %-7s %-7s %s\n' huggingface "${DIRECT[hf]}" "${VIA[hf]}" "${M_HF}" "$(route hf "${DIRECT[hf]}" "${VIA[hf]}" "${M_HF}" "${HF_EP}")"
  printf '%-12s %-7s %-7s %-7s %s\n' github "${DIRECT[gh]}" "${VIA[gh]}" "${M_GH}" "$(route gh "${DIRECT[gh]}" "${VIA[gh]}" "${M_GH}" "${GH_PREFIX}")"
  printf '%-12s %-7s %-7s %-7s %s\n' pypi "${DIRECT[pip]}" "${VIA[pip]}" "${M_PIP}" "$(route pip "${DIRECT[pip]}" "${VIA[pip]}" "${M_PIP}" "${PIP_IDX}")"
  printf '%-12s %-7s %-7s %-7s %s\n' conda-forge "${DIRECT[conda]}" "${VIA[conda]}" "${M_CONDA}" "$(route conda "${DIRECT[conda]}" "${VIA[conda]}" "${M_CONDA}" "${CONDA_M}")"
  printf '%-12s %-7s %-7s %-7s %s\n' wandb "${DIRECT[wandb]}" "${VIA[wandb]}" "-" "$( [[ -n "${NET_PROXY}" ]] && echo proxy || echo direct)"
  printf 'mode=%s proxy=%s upload_route=%s\n' "${PROXY_MODE}" "${NET_PROXY:-none}" "${UPLOAD_ROUTE}"
} | tee "${STATE_DIR}/net.table.txt"

cat > "${STATE_DIR}/net.resolved.env" <<EOF
# written by net_check.sh at $(date -Is)
NET_PROXY="${NET_PROXY}"
HF_ENDPOINT_RESOLVED="${HF_EP}"
GITHUB_PREFIX="${GH_PREFIX}"
PIP_INDEX_RESOLVED="${PIP_IDX}"
CONDA_MIRROR_RESOLVED="${CONDA_M}"
UPLOAD_ROUTE="${UPLOAD_ROUTE}"
EOF
grep -q UNREACHABLE "${STATE_DIR}/net.table.txt" && { log "WARNING: some services are unreachable; set a proxy or mirrors in deploy/network.env"; exit 1; }
exit 0
