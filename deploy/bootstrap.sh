#!/usr/bin/env bash
# Run on your own computer. Puts OPD on a target machine over SSH and drives it there.
#   bash deploy/bootstrap.sh init                 # create deploy/target.env and deploy/network.env, then edit them
#   bash deploy/bootstrap.sh check                # SSH test, copy deploy/, probe HF / GitHub / PyPI / conda routes on the target
#   bash deploy/bootstrap.sh setup [PHASES]       # copy code, install env + data + models + eval runtime (background on target)
#   bash deploy/bootstrap.sh status | logs [phase]
#   bash deploy/bootstrap.sh run <plan.tsv>       # training / evaluation / upload plan (deploy/plans/*.tsv), background on target
#   bash deploy/bootstrap.sh fetch [local_dir]    # copy finished eval suites back (no shards, no EvalPlus cache)
#   bash deploy/bootstrap.sh relay-upload <local_dir> [glob]   # upload fetched suites from this computer (its proxy)
#   bash deploy/bootstrap.sh secrets              # copy ~/.config/opd-private/opd.env (HF_TOKEN, WANDB_API_KEY) to the target
#   bash deploy/bootstrap.sh tunnel start|stop    # PROXY_MODE=ssh-reverse: keep this computer's proxy forwarded to the target
#   bash deploy/bootstrap.sh shell                # interactive SSH session
set -euo pipefail
DEPLOY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "${DEPLOY}/.." && pwd)"
CMD="${1:-help}"; shift || true

if [[ "${CMD}" == init ]]; then
  for f in target network; do
    if [[ -f "${DEPLOY}/${f}.env" ]]; then echo "keep existing deploy/${f}.env"; else cp "${DEPLOY}/${f}.env.example" "${DEPLOY}/${f}.env"; echo "created deploy/${f}.env"; fi
  done
  echo "Edit the SSH block in deploy/target.env (and deploy/network.env if the target needs a proxy), then run: bash deploy/bootstrap.sh check"
  exit 0
fi
[[ "${CMD}" == help || "${CMD}" == -h ]] && { sed -n '2,13p' "$0"; exit 0; }
[[ -f "${DEPLOY}/target.env" ]] || { echo "deploy/target.env is missing: bash deploy/bootstrap.sh init" >&2; exit 2; }
# shellcheck source=/dev/null
source "${DEPLOY}/target.env"
# shellcheck source=/dev/null
[[ -f "${DEPLOY}/network.env" ]] && source "${DEPLOY}/network.env"
SSH_KEY="${SSH_KEY/#\~/${HOME}}"
CTL="${HOME}/.ssh/cm-opd-deploy-%r@%h:%p"
# shellcheck disable=SC2206
SSH=(ssh -p "${SSH_PORT:-22}" -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ControlMaster=auto -o "ControlPath=${CTL}" -o ControlPersist=10m ${SSH_EXTRA_OPTS:-})
[[ -n "${SSH_KEY:-}" ]] && SSH+=(-i "${SSH_KEY}" -o IdentitiesOnly=yes)
TARGET="${SSH_USER}@${SSH_HOST}"
RR="${REMOTE_ROOT:?REMOTE_ROOT is not set in deploy/target.env}"
remote() { "${SSH[@]}" "${TARGET}" "$@"; }
rsync_to() { RSYNC_RSH="${SSH[*]}" rsync -az --delete "$@"; }   # RSYNC_RSH: macOS rsync 2.6.9 mis-parses -e with --exclude
q() { printf '%q' "$1"; }

sync_deploy() {
  # REMOTE_ROOT must be a directory these scripts own: refuse a non-empty directory without our marker,
  # because code sync uses rsync --delete inside it
  remote "if [ -d $(q "${RR}") ] && [ ! -e $(q "${RR}")/.opd_deploy_root ] && [ -n \"\$(ls -A $(q "${RR}") 2>/dev/null)\" ]; then echo 'REFUSE: REMOTE_ROOT ${RR} exists, is not empty and was not created by deploy/'; exit 3; fi; mkdir -p $(q "${RR}")/code && touch $(q "${RR}")/.opd_deploy_root" \
    || { echo "pick an empty or new REMOTE_ROOT in deploy/target.env" >&2; exit 3; }
  rsync_to --exclude '.tunnel.*' "${DEPLOY}/" "${TARGET}:${RR}/code/deploy/"
}
sync_code() {
  if [[ "${CODE_SOURCE:-rsync}" == git ]]; then
    remote "set -e; if [ ! -d $(q "${RR}")/code/.git ]; then git clone --filter=blob:none $(q "${REPO_URL}") $(q "${RR}")/code.tmp && rm -rf $(q "${RR}")/code.tmp/deploy && mv $(q "${RR}")/code.tmp/* $(q "${RR}")/code.tmp/.git $(q "${RR}")/code/ && rmdir $(q "${RR}")/code.tmp; fi; git -C $(q "${RR}")/code fetch --depth 1 origin $(q "${REPO_REF}") && git -C $(q "${RR}")/code checkout --detach FETCH_HEAD"
  else
    # this checkout as it is (uncommitted changes included); data, logs, outputs and secrets stay behind
    rsync_to --exclude '/.git/' --exclude '/data/' --exclude '/hf_home/' --exclude '/logs/' --exclude '/wandb/' \
      --exclude '/eval_outputs/' --exclude '/outputs/' --exclude '/checkpoints/' --exclude '/profile_output/' --exclude '/plan/' \
      --exclude '/temp/' --exclude '/audit/' --exclude '/mopd_audit/' --exclude '/experiment_records/' --exclude '/analysis-output' \
      --exclude '/refine-logs/' --exclude '/reports/' --exclude '/.runtime/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
      --exclude '/ssh*.sh' --exclude '/api.sh' --exclude '.env' --exclude '.env.*' --exclude '/deploy/' \
      "${CODE}/" "${TARGET}:${RR}/code/"
    { git -C "${CODE}" rev-parse HEAD 2>/dev/null || echo no-git; git -C "${CODE}" status --porcelain 2>/dev/null | wc -l | xargs echo dirty_files; date '+%Y-%m-%dT%H:%M:%S%z'; } \
      | remote "cat > $(q "${RR}")/code/DEPLOY_SOURCE.txt"
  fi
  sync_deploy
}
launch() {  # launch NAME COMMAND...: detached on the target, survives SSH drops
  # the cd must happen inside the detached bash: `cd X && setsid ... &` backgrounds the whole list, which keeps the
  # SSH channel open until the job ends
  local name="$1"; shift
  remote "mkdir -p $(q "${RR}")/logs; setsid nohup bash -c $(q "cd ${RR}/code && $*") > $(q "${RR}")/logs/${name}.out 2>&1 < /dev/null & echo started ${name}: log ${RR}/logs/${name}.out"
}
tunnel_start() {
  [[ "${PROXY_MODE:-auto}" == ssh-reverse ]] || return 0
  if [[ -f "${DEPLOY}/.tunnel.pid" ]] && kill -0 "$(cat "${DEPLOY}/.tunnel.pid")" 2>/dev/null; then return 0; fi
  nc -z 127.0.0.1 "${LOCAL_PROXY_PORT:-7890}" || { echo "no proxy listening on 127.0.0.1:${LOCAL_PROXY_PORT:-7890} here" >&2; exit 2; }
  ( while true; do
      ssh -p "${SSH_PORT:-22}" ${SSH_KEY:+-i "${SSH_KEY}"} -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
        -N -R "${REMOTE_PROXY_PORT:-17890}:127.0.0.1:${LOCAL_PROXY_PORT:-7890}" ${SSH_EXTRA_OPTS:-} "${TARGET}"
      echo "$(date '+%F %T') tunnel dropped, reconnecting in 10 s"; sleep 10
    done ) >> "${DEPLOY}/.tunnel.log" 2>&1 &
  echo $! > "${DEPLOY}/.tunnel.pid"; sleep 3
  echo "reverse tunnel: target 127.0.0.1:${REMOTE_PROXY_PORT:-17890} -> here 127.0.0.1:${LOCAL_PROXY_PORT:-7890} (pid $(cat "${DEPLOY}/.tunnel.pid"), log deploy/.tunnel.log)"
}
tunnel_stop() {
  [[ -f "${DEPLOY}/.tunnel.pid" ]] || { echo "no tunnel"; return 0; }
  pkill -P "$(cat "${DEPLOY}/.tunnel.pid")" 2>/dev/null || true; kill "$(cat "${DEPLOY}/.tunnel.pid")" 2>/dev/null || true
  rm -f "${DEPLOY}/.tunnel.pid"; echo "tunnel stopped"
}

case "${CMD}" in
  check)
    remote "echo ssh ok: \$(whoami)@\$(hostname) \$(uname -m); nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo no-nvidia-smi; df -h $(q "$(dirname "${RR}")") | tail -n 1"
    tunnel_start; sync_deploy
    remote "bash $(q "${RR}")/code/deploy/remote/net_check.sh" || echo "some services are unreachable: set PROXY_MODE / mirrors in deploy/network.env and rerun check"
    ;;
  setup)
    what="${1:-all}"   # env | data | all | comma-separated phases
    case "${what}" in env|data|all) sel="${what}" ;; *) sel="--phases ${what}" ;; esac
    tunnel_start; sync_code
    remote "bash $(q "${RR}")/code/deploy/remote/setup.sh ${sel} --dry-run"
    launch setup "bash deploy/remote/setup.sh ${sel}"
    echo "follow: bash deploy/bootstrap.sh logs     (reruns skip finished phases; FORCE=1 on the target redoes them)"
    ;;
  status)
    remote "cd $(q "${RR}") && for f in state/*.done; do [ -e \"\$f\" ] && echo \"\$(basename \"\$f\" .done): done \$(cat \"\$f\")\"; done; cat state/net.table.txt 2>/dev/null; tail -n 5 logs/setup.out logs/pipeline.out 2>/dev/null; ls eval_outputs 2>/dev/null | head -50"
    ;;
  logs)
    if [[ -n "${1:-}" ]]; then remote "tail -n 50 -f $(q "${RR}")/logs/setup_${1}.log"; else remote "tail -n 50 -f $(q "${RR}")/logs/setup.out"; fi
    ;;
  run)
    plan="${1:?usage: bootstrap.sh run <plan.tsv>}"
    tunnel_start; sync_deploy
    rel="deploy/plans/$(basename "${plan}")"
    remote "mkdir -p $(q "${RR}")/code/deploy/plans"
    rsync_to "${plan}" "${TARGET}:${RR}/code/${rel}"
    launch pipeline "bash deploy/remote/run_pipeline.sh ${rel}"
    ;;
  fetch)
    dest="${1:-${CODE}/../analysis/deploy-$(date +%Y%m%d)-${SSH_HOST//[^A-Za-z0-9]/_}}"; mkdir -p "${dest}"
    RSYNC_RSH="${SSH[*]}" rsync -az --exclude 'shards/' --exclude 'cache/' --exclude '.locks/' "${TARGET}:${RR}/eval_outputs/" "${dest}/"
    echo "fetched into ${dest}"
    ;;
  relay-upload)
    src="${1:?usage: bootstrap.sh relay-upload <local_dir> [glob]}"; glob="${2:-*}"
    : "${RESULTS_REPO:?set RESULTS_REPO in deploy/target.env}"
    # shellcheck source=/dev/null
    [[ -f "${HOME}/.config/opd-private/opd.env" ]] && { set -a; source "${HOME}/.config/opd-private/opd.env"; set +a; }
    if [[ "${PROXY_MODE:-auto}" == ssh-reverse || -n "${LOCAL_PROXY_PORT:-}" ]] && nc -z 127.0.0.1 "${LOCAL_PROXY_PORT:-7890}" 2>/dev/null; then
      export https_proxy="http://127.0.0.1:${LOCAL_PROXY_PORT:-7890}" http_proxy="http://127.0.0.1:${LOCAL_PROXY_PORT:-7890}"
    fi
    # shellcheck disable=SC2086
    python3 "${DEPLOY}/remote/upload_results.py" --repo "${RESULTS_REPO}" --repo-type "${RESULTS_REPO_TYPE:-dataset}" \
      --prefix "${RESULTS_PREFIX:-eval_results}" ${src}/${glob}
    ;;
  secrets)
    f="${HOME}/.config/opd-private/opd.env"
    [[ -f "${f}" ]] || { echo "no ${f} (put HF_TOKEN=... and WANDB_API_KEY=... there)" >&2; exit 2; }
    read -r -p "copy ${f} to ${TARGET}:~/.config/opd-private/opd.env (mode 600)? [y/N] " ans
    [[ "${ans}" == y ]] || exit 0
    remote "umask 077; mkdir -p ~/.config/opd-private && cat > ~/.config/opd-private/opd.env" < "${f}"
    echo "copied"
    ;;
  tunnel) case "${1:-start}" in start) PROXY_MODE=ssh-reverse tunnel_start ;; stop) tunnel_stop ;; *) echo "tunnel start|stop" ;; esac ;;
  shell) "${SSH[@]}" -t "${TARGET}" "cd $(q "${RR}") 2>/dev/null; exec \$SHELL -l" ;;
  *) echo "unknown command ${CMD}"; sed -n '2,13p' "$0"; exit 2 ;;
esac
