#!/usr/bin/env bash
# One script, run on your own computer: install OPD on a machine over SSH, step by step, then report what finished
# and where it failed. The machine gets the one training + evaluation stack (torch 2.6.0, vLLM 0.8.5.post1,
# transformers 4.51.3), training/eval data, models and the official-scoring runtime under --root.
#
#   bash deploy/opd_deploy.sh --ssh "ssh -p 30022 root@1.2.3.4" --root /data/opd
#   bash deploy/opd_deploy.sh --host 1.2.3.4 --user root --port 30022 --key ~/.ssh/id_ed25519 --root /data/opd --gpus 0,1,2,3
#   bash deploy/opd_deploy.sh --ssh "ssh root@1.2.3.4" --root /data/opd --status     # progress of the run on that machine
#
# SSH:      --ssh "ssh ..."  (as you would type it: user@host, -p, -i, -J, -o)  or --host --user --port --key --jump
#           A password, if the machine asks for one, is typed by you once at the start (the connection is reused).
# Target:   --root DIR (required; new or empty)  --gpus 0,1,...  --min-free-gib N (default 100; CityU H200: 500)
# Content:  --students "Qwen/Qwen3-1.7B Qwen/Qwen3-4B"  --teacher Qwen/Qwen3-30B-A3B-Instruct-2507 ("" = none)
#           --checkpoints "name ..." (deploy/eval_models.tsv)  --no-train-data  --no-eval
# Network:  --proxy auto|direct|http|ssh-reverse|mirror  --http-proxy URL  --local-proxy-port 7890
#           --hf-mirror URL  --github-proxy URL  --pip-mirror URL  --conda-mirror URL  --upload auto|direct|relay
# Steps:    connect, sync, network, base, env, data, models, ckpts, evalrt, verify[, plan]
#           --steps a,b,...  --sequential (env and the downloads one after another; default: in parallel)
#           --force (redo finished steps)  --plan deploy/plans/x.tsv (run it after the install and follow it)
#           --status (no changes, report only)  --no-follow (start the install and return)
set -uo pipefail
if (( BASH_VERSINFO[0] < 4 )); then  # macOS /bin/bash is 3.2; this script needs associative arrays
  for b in /opt/homebrew/bin/bash /usr/local/bin/bash; do [[ -x "$b" ]] && exec "$b" "$0" "$@"; done
  echo "needs bash >= 4 (macOS: brew install bash)" >&2; exit 2
fi
DEPLOY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "${DEPLOY}/.." && pwd)"

# ---------- arguments ----------
HOST=""; USER_=""; PORT=""; KEY=""; JUMP=""; SSH_O=(); RR=""; GPUS=0; MIN_FREE=100
STUDENTS="Qwen/Qwen3-1.7B Qwen/Qwen3-4B"; TEACHER="Qwen/Qwen3-30B-A3B-Instruct-2507"; CKPTS=""; TRAIN_DATA=1; EVAL=1
PROXY=auto; HTTP_PROXY_URL=""; LOCAL_PROXY_PORT=7890; HF_MIRROR=https://hf-mirror.com; GITHUB_PROXY=""; PIP_MIRROR=""; CONDA_MIRROR=""; UPLOAD=auto
STEPS=all; SEQ=0; FORCE=0; PLAN=""; STATUS_ONLY=0; FOLLOW=1
usage() { sed -n '2,23p' "$0"; exit "${1:-0}"; }
parse_ssh() {  # "ssh -p 22 -i key -J jump -o X=Y user@host"
  local -a w; read -ra w <<< "$1"; local i=0
  [[ "${w[0]:-}" == ssh ]] && i=1
  while (( i < ${#w[@]} )); do
    case "${w[$i]}" in
      -p) PORT="${w[$((i+1))]}"; i=$((i+2)) ;;
      -i) KEY="${w[$((i+1))]}"; i=$((i+2)) ;;
      -J) JUMP="${w[$((i+1))]}"; i=$((i+2)) ;;
      -o) SSH_O+=(-o "${w[$((i+1))]}"); i=$((i+2)) ;;
      -l) USER_="${w[$((i+1))]}"; i=$((i+2)) ;;
      -*) i=$((i+1)) ;;
      *@*) USER_="${w[$i]%@*}"; HOST="${w[$i]#*@}"; i=$((i+1)) ;;
      *) HOST="${w[$i]}"; i=$((i+1)) ;;
    esac
  done
}
while (( $# )); do
  case "$1" in
    --ssh) parse_ssh "$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;      --user) USER_="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;      --key) KEY="$2"; shift 2 ;;       --jump) JUMP="$2"; shift 2 ;;
    --root) RR="$2"; shift 2 ;;        --gpus) GPUS="$2"; shift 2 ;;     --min-free-gib) MIN_FREE="$2"; shift 2 ;;
    --students) STUDENTS="$2"; shift 2 ;; --teacher) TEACHER="$2"; shift 2 ;; --checkpoints) CKPTS="$2"; shift 2 ;;
    --no-train-data) TRAIN_DATA=0; shift ;; --no-eval) EVAL=0; shift ;;
    --proxy) PROXY="$2"; shift 2 ;;    --http-proxy) HTTP_PROXY_URL="$2"; shift 2 ;; --local-proxy-port) LOCAL_PROXY_PORT="$2"; shift 2 ;;
    --hf-mirror) HF_MIRROR="$2"; shift 2 ;; --github-proxy) GITHUB_PROXY="$2"; shift 2 ;;
    --pip-mirror) PIP_MIRROR="$2"; shift 2 ;; --conda-mirror) CONDA_MIRROR="$2"; shift 2 ;; --upload) UPLOAD="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;    --sequential) SEQ=1; shift ;;     --force) FORCE=1; shift ;;
    --plan) PLAN="$2"; shift 2 ;;      --status) STATUS_ONLY=1; shift ;; --no-follow) FOLLOW=0; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 2 ;;
  esac
done
[[ -n "${HOST}" && -n "${RR}" ]] || { echo "need --ssh (or --host) and --root" >&2; usage 2; }
[[ "${RR}" == /* ]] || { echo "--root must be an absolute path" >&2; exit 2; }
KEY="${KEY/#\~/${HOME}}"
TARGET="${USER_:+${USER_}@}${HOST}"
SSH=(ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ControlMaster=auto -o "ControlPath=${HOME}/.ssh/cm-opd-deploy-%C" -o ControlPersist=2h "${SSH_O[@]}")
[[ -n "${PORT}" ]] && SSH+=(-p "${PORT}")
[[ -n "${KEY}" ]] && SSH+=(-i "${KEY}" -o IdentitiesOnly=yes)
[[ -n "${JUMP}" ]] && SSH+=(-J "${JUMP}")
remote() { "${SSH[@]}" "${TARGET}" "$@"; }
q() { printf '%q' "$1"; }
want() { [[ ",${STEPS}," == *",all,"* || ",${STEPS}," == *",$1,"* ]]; }

# ---------- progress bookkeeping ----------
ALL_STEPS=(connect sync network base env data models ckpts evalrt verify)
[[ -n "${PLAN}" ]] && ALL_STEPS+=(plan)
declare -A ST TM NOTE
for s in "${ALL_STEPS[@]}"; do ST[$s]="-"; TM[$s]=""; NOTE[$s]=""; done
now() { date '+%H:%M:%S'; }
say() { printf '%s %s\n' "$(now)" "$*"; }
mark() { ST[$1]="$2"; [[ -n "${3:-}" ]] && NOTE[$1]="$3"; }
fmt_secs() { local s="${1:-}"; [[ -z "$s" ]] && return; printf '%dm%02ds' $((s / 60)) $((s % 60)); }
summary() {
  local s first_fail=""
  echo
  echo "================ OPD deploy: ${TARGET}:${RR} ================"
  printf '%-9s %-9s %-8s %s\n' step status time note
  for s in "${ALL_STEPS[@]}"; do
    printf '%-9s %-9s %-8s %s\n' "$s" "${ST[$s]}" "$(fmt_secs "${TM[$s]}")" "${NOTE[$s]}"
    [[ -z "${first_fail}" && "${ST[$s]}" == FAILED ]] && first_fail="$s"
  done
  if [[ -n "${first_fail}" ]]; then
    echo; echo "FIRST FAILURE: ${first_fail} — ${NOTE[${first_fail}]}"
    if [[ " base env data models ckpts evalrt verify " == *" ${first_fail} "* ]]; then
      echo "--- last lines of ${RR}/logs/setup_${first_fail}.log ---"
      remote "tail -n 25 $(q "${RR}")/logs/setup_${first_fail}.log" 2>/dev/null
    fi
    echo "Fix the cause and run the same command again: finished steps are skipped."
  elif [[ "${ST[verify]}" == ok ]]; then
    echo; echo "READY. On the machine: source ${RR}/activate.sh   (training: bash start.sh -c configs/<config>.yaml --local)"
  fi
}
trap 'echo; say "interrupted here; anything already started keeps running on the machine (check with --status)"; summary; exit 130' INT

# ---------- remote install progress ----------
read_remote_state() {  # one line per phase: name status secs note
  remote "cd $(q "${RR}")/state 2>/dev/null || exit 0; for p in base env data models ckpts evalrt verify; do
    if [ -f \$p.failed ]; then echo \"\$p FAILED \$(cat \$p.secs 2>/dev/null) \$(cat \$p.failed)\";
    elif [ -f \$p.running ]; then echo \"\$p running \$(( \$(date +%s) - \$(cat \$p.running) ))\";
    elif [ -f \$p.done ]; then echo \"\$p ok \$(cat \$p.secs 2>/dev/null) \$(cat \$p.done)\"; fi; done;
    [ -f setup.exit ] && echo \"EXIT \$(cat setup.exit)\"; [ -f net.table.txt ] && echo \"NET \$(tail -n 1 net.table.txt)\"; true" 2>/dev/null
}
apply_remote_state() {  # returns 0 when the remote setup has finished
  local line name status secs rest finished=1
  while read -r name status secs rest; do
    case "${name}" in
      EXIT) finished=0 ;;
      NET) NOTE[network]="${status} ${secs} ${rest}" ;;
      base|env|data|models|ckpts|evalrt|verify)
        [[ "${secs}" =~ ^[0-9]+$ ]] && TM[$name]="${secs}"
        case "${status}" in
          ok) ST[$name]=ok; NOTE[$name]="" ;;
          FAILED) ST[$name]=FAILED; NOTE[$name]="${rest}" ;;
          running) ST[$name]=running; NOTE[$name]="" ;;
        esac ;;
    esac
  done <<< "$1"
  return "${finished}"
}
follow_setup() {
  local state prev="" lost=0 heartbeat=0 line
  while true; do
    if ! state="$(read_remote_state)"; then
      lost=$((lost + 1)); say "connection lost (${lost}); retrying in 30 s — the install keeps running on the machine"
      (( lost >= 60 )) && { say "giving up following; check later with --status"; return 1; }
      sleep 30; continue
    fi
    lost=0
    apply_remote_state "${state}"; local fin=$?
    line="$(for s in base env data models ckpts evalrt verify; do printf '%s:%s ' "$s" "${ST[$s]}"; done)"
    if [[ "${line}" != "${prev}" ]] || (( heartbeat >= 6 )); then
      local run=""; for s in base env data models ckpts evalrt verify; do [[ "${ST[$s]}" == running ]] && run+=" ${s}($(fmt_secs "${TM[$s]}"))"; done
      say "${line}${run:+ | running:${run}}"; prev="${line}"; heartbeat=0
    fi
    (( fin == 0 )) && return 0
    heartbeat=$((heartbeat + 1)); sleep 20
  done
}

# ---------- status only ----------
if (( STATUS_ONLY )); then
  ST[connect]=ok
  state="$(read_remote_state)" || { echo "cannot reach ${TARGET}"; exit 2; }
  apply_remote_state "${state}"
  remote "tail -n 3 $(q "${RR}")/logs/setup.out 2>/dev/null" | sed 's/^/  setup.out: /'
  summary; exit 0
fi

say "OPD deploy to ${TARGET} (root ${RR}, GPUs ${GPUS}); steps: ${STEPS}"

# 1. connect
t0=$(date +%s)
info="$(remote "echo \"\$(whoami)@\$(hostname) \$(uname -m)\"; command -v rsync >/dev/null && echo rsync=yes || echo rsync=no; command -v curl >/dev/null && echo curl=yes || echo curl=no; command -v git >/dev/null && echo git=yes || echo git=no; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | sort | uniq -c | sed 's/^ */gpu: /'; df -BG --output=avail \$(dirname $(q "${RR}")) 2>/dev/null | tail -n 1 | tr -dc 0-9 | sed 's/^/free_gib=/'" 2>&1)" \
  || { mark connect FAILED "ssh failed: ${info##*$'\n'}"; summary; exit 1; }
TM[connect]=$(( $(date +%s) - t0 ))
echo "${info}" | sed 's/^/  /'
grep -q "curl=no\|git=no" <<< "${info}" && { mark connect FAILED "the machine needs curl and git"; summary; exit 1; }
mark connect ok "$(head -n 1 <<< "${info}")"
REMOTE_RSYNC=$(grep -q rsync=yes <<< "${info}" && echo 1 || echo 0)

# optional reverse tunnel for --proxy ssh-reverse (lives as long as this script's SSH master)
if [[ "${PROXY}" == ssh-reverse ]]; then
  nc -z 127.0.0.1 "${LOCAL_PROXY_PORT}" || { mark network FAILED "no proxy on this computer at 127.0.0.1:${LOCAL_PROXY_PORT}"; summary; exit 1; }
  "${SSH[@]}" -O forward -R "17890:127.0.0.1:${LOCAL_PROXY_PORT}" "${TARGET}" 2>/dev/null \
    && say "reverse proxy: machine 127.0.0.1:17890 -> this computer 127.0.0.1:${LOCAL_PROXY_PORT}" \
    || { mark network FAILED "SSH port forwarding refused by the machine (AllowTcpForwarding?)"; summary; exit 1; }
fi

# 2. sync code + config
if want sync; then
  t0=$(date +%s); say "[sync] code -> ${TARGET}:${RR}/code"
  remote "if [ -d $(q "${RR}") ] && [ ! -e $(q "${RR}")/.opd_deploy_root ] && [ -n \"\$(ls -A $(q "${RR}") 2>/dev/null)\" ]; then exit 3; fi; mkdir -p $(q "${RR}")/code && touch $(q "${RR}")/.opd_deploy_root" \
    || { mark sync FAILED "${RR} exists, is not empty and was not created by this script; use a new --root"; summary; exit 1; }
  EXCL=(/.git/ /data/ /hf_home/ /logs/ /wandb/ /eval_outputs/ /outputs/ /checkpoints/ /profile_output/ /plan/ /temp/ /audit/ /mopd_audit/
        /experiment_records/ /analysis-output /refine-logs/ /reports/ /.runtime/ __pycache__/ .pytest_cache/ '/ssh*.sh' /api.sh .env '.env.*'
        /deploy/target.env /deploy/network.env '/deploy/.tunnel.*')
  if (( REMOTE_RSYNC )); then
    args=(); for e in "${EXCL[@]}"; do args+=(--exclude "$e"); done
    RSYNC_RSH="${SSH[*]}" rsync -az --delete "${args[@]}" "${CODE}/" "${TARGET}:${RR}/code/" || { mark sync FAILED "rsync failed"; summary; exit 1; }
  else
    args=(); for e in "${EXCL[@]}"; do args+=(--exclude "./${e#/}"); done
    COPYFILE_DISABLE=1 tar -C "${CODE}" "${args[@]}" -czf - . | remote "tar -C $(q "${RR}")/code -xzf -" || { mark sync FAILED "tar copy failed"; summary; exit 1; }
  fi
  remote "cat > $(q "${RR}")/code/deploy/target.env" <<EOF
# written by deploy/opd_deploy.sh from $(hostname) at $(date '+%Y-%m-%dT%H:%M:%S%z')
REMOTE_ROOT=${RR}
GPU_IDS=${GPUS}
MIN_FREE_GIB=${MIN_FREE}
STUDENT_MODELS="${STUDENTS}"
TEACHER_MODEL="${TEACHER}"
EVAL_CHECKPOINTS="${CKPTS}"
INSTALL_TRAIN_DATA=${TRAIN_DATA}
INSTALL_EVAL=${EVAL}
EOF
  remote "cat > $(q "${RR}")/code/deploy/network.env" <<EOF
PROXY_MODE=${PROXY}
HTTP_PROXY_URL="${HTTP_PROXY_URL}"
REMOTE_PROXY_PORT=17890
HF_MIRROR="${HF_MIRROR}"
GITHUB_PROXY="${GITHUB_PROXY}"
PIP_MIRROR="${PIP_MIRROR}"
CONDA_MIRROR="${CONDA_MIRROR}"
UPLOAD_MODE=${UPLOAD}
EOF
  { git -C "${CODE}" rev-parse HEAD 2>/dev/null || echo no-git; git -C "${CODE}" status --porcelain 2>/dev/null | wc -l | xargs echo dirty_files; } \
    | remote "cat > $(q "${RR}")/code/DEPLOY_SOURCE.txt"
  TM[sync]=$(( $(date +%s) - t0 )); mark sync ok "commit $(git -C "${CODE}" rev-parse --short HEAD 2>/dev/null) + $(git -C "${CODE}" status --porcelain 2>/dev/null | wc -l | tr -d ' ') local changes"
fi

# 3. network routes
if want network; then
  t0=$(date +%s); say "[network] probing Hugging Face / GitHub / PyPI / conda-forge from the machine"
  out="$(remote "bash $(q "${RR}")/code/deploy/remote/net_check.sh" 2>&1)"; rc=$?
  echo "${out}" | sed 's/^/  /'
  TM[network]=$(( $(date +%s) - t0 ))
  if (( rc == 0 )); then mark network ok "$(tail -n 1 <<< "${out}")"
  else mark network FAILED "unreachable services (see table); set --proxy / mirrors"; summary; exit 1; fi
fi

# 4.-10. install phases (detached on the machine; this script follows the state files)
phases=(); for p in base env data models ckpts evalrt verify; do want "$p" && phases+=("$p"); done
if (( ${#phases[@]} )); then
  sel="$(IFS=,; echo "${phases[*]}")"
  extra=""; (( SEQ )) && extra="--sequential"
  say "[install] starting on the machine: ${sel} ($( (( SEQ )) && echo one after another || echo env in parallel with downloads)) — logs in ${RR}/logs/setup_<step>.log"
  remote "rm -f $(q "${RR}")/state/setup.exit; setsid nohup bash -c $(q "cd ${RR}/code && FORCE=${FORCE} bash deploy/remote/setup.sh --phases ${sel} ${extra}") > $(q "${RR}")/logs/setup.out 2>&1 < /dev/null & echo started" \
    || { mark base FAILED "could not start setup on the machine"; summary; exit 1; }
  if (( ! FOLLOW )); then say "started; follow later with: bash deploy/opd_deploy.sh ... --status"; exit 0; fi
  sleep 5; follow_setup || { summary; exit 1; }
  for p in base env data models ckpts evalrt verify; do want "$p" && [[ "${ST[$p]}" == "-" ]] && mark "$p" skipped "not run (an earlier step failed or it waits for env + data)"; done
fi

# 11. plan (training / evaluation / upload) on the machine
if [[ -n "${PLAN}" && ! " ${ST[*]} " =~ " FAILED " ]]; then
  t0=$(date +%s); rel="deploy/plans/$(basename "${PLAN}")"
  "${SSH[@]}" "${TARGET}" "mkdir -p $(q "${RR}")/code/deploy/plans && cat > $(q "${RR}")/code/${rel}" < "${PLAN}"
  remote "setsid nohup bash -c $(q "cd ${RR}/code && bash deploy/remote/run_pipeline.sh ${rel}") > $(q "${RR}")/logs/pipeline.out 2>&1 < /dev/null & echo started"
  say "[plan] ${rel} started; following ${RR}/logs/pipeline.out"
  seen=0
  while true; do
    out="$(remote "cat $(q "${RR}")/logs/pipeline.out" 2>/dev/null)" || { sleep 30; continue; }
    total=$(wc -l <<< "${out}"); (( total > seen )) && { tail -n +$((seen + 1)) <<< "${out}" | sed 's/^/  /'; seen=${total}; }
    if grep -q "pipeline finished with failures\|ERROR\|failed (see" <<< "${out}"; then mark plan FAILED "$(grep -m1 "failed\|ERROR" <<< "${out}")"; break; fi
    grep -q "pipeline finished$" <<< "${out}" && { mark plan ok; break; }
    sleep 30
  done
  TM[plan]=$(( $(date +%s) - t0 ))
fi

summary
[[ " ${ST[*]} " =~ " FAILED " ]] && exit 1
exit 0
