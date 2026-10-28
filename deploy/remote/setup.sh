#!/usr/bin/env bash
# Install everything OPD needs, under REMOTE_ROOT. One stack for training and evaluation:
# torch 2.6.0, vLLM 0.8.5.post1, transformers 4.51.3 (environment.yml + deploy/env.lock.txt).
#   base   : disk/GPU checks, network routes (net_check.sh), Miniforge, a small download env (opd-tools)
#   env    : the training + evaluation conda env (mopd-verl), versions checked          } run in parallel
#   data   : training parquet, G-OPD eval prompts at the pinned commit, LiveCodeBench  } after base
#   models : STUDENT_MODELS and TEACHER_MODEL from Hugging Face, sha256-checked         }
#   ckpts  : EVAL_CHECKPOINTS (deploy/eval_models.tsv) from icemoon28/opd-checkpoints   }
#   evalrt : official-scoring runtime (EvalPlus, LiveCodeBench sources); runs as soon as env and data are both done
#   verify : versions, CUDA, NCCL check on GPU_IDS
# Finished phases write state/<phase>.done and are skipped on reruns (FORCE=1 redoes them). At the end
# REMOTE_ROOT/activate.sh is (re)written: `source` it to get the env, PYTHONPATH, HF cache and network settings.
#   bash deploy/remote/setup.sh env|data|all        # env = base,env,verify   data = base,data,models,ckpts
#   bash deploy/remote/setup.sh --phases base,env,... [--dry-run]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PHASES=all; DRY=0; SEQUENTIAL=0
while (( $# )); do
  case "$1" in
    env) PHASES=base,env,verify; shift ;;
    data) PHASES=base,data,models,ckpts; shift ;;
    all) PHASES=all; shift ;;
    --phases) PHASES="${2:?}"; shift 2 ;;
    --sequential) SEQUENTIAL=1; shift ;;   # env, data, models, ckpts one after another instead of in parallel
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) die "unknown argument $1" ;;
  esac
done
want() { [[ ",${PHASES}," == *",all,"* || ",${PHASES}," == *",$1,"* ]]; }

check_stack() {  # the env must be exactly the unified stack
  "${PYTHON}" - "${REQ_TORCH}" "${REQ_VLLM}" "${REQ_TRANSFORMERS}" <<'PY'
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

phase_base() {
  local cap
  [[ "$(uname -m)" == x86_64 ]] || die "environment.yml targets Linux x86_64, got $(uname -m)"
  disk_ok || return 1
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv,noheader || log "WARNING: nvidia-smi failed (CPU-only machine?)"
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')"
  if [[ -n "${cap}" && "${cap%%.*}" -ge 10 ]]; then  # torch 2.6 cu124 wheels stop at sm_90
    log "STOP: GPU compute capability ${cap} (Blackwell) is not supported by torch ${REQ_TORCH}; the unified stack cannot run here"
    return 1
  fi
  bash "${DEPLOY_DIR}/remote/net_check.sh"; local rc=$?
  apply_network
  (( rc == 0 )) || log "WARNING: net_check reported unreachable services; continuing with what works"
  if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
    local inst="${TMPDIR}/miniforge.sh"
    curl -fL --retry 5 -o "${inst}" "$(gh_url "${MINIFORGE_URL}")" || return 1
    echo "${MINIFORGE_SHA256}  ${inst}" | sha256sum -c - || return 1
    bash "${inst}" -b -p "${CONDA_ROOT}" || return 1
  fi
  if [[ ! -x "${TOOLS_PYTHON}" ]]; then
    "${CONDA_ROOT}/bin/conda" create -y -p "${TOOLS_PREFIX}" python=3.10 pip || return 1
  fi
  "${TOOLS_PYTHON}" -m pip install -q "huggingface_hub>=0.30,<1.0" hf_transfer hf_xet "pandas>=2.0" "pyarrow>=19.0.0"
}

phase_env() {
  local src="${CODE_DIR}/environment.yml" dst="${STATE_DIR}/environment.yml"
  # environment.yml declares the packages; env.lock.txt pins every pip package to the CityU H200 training env.
  # Only github.com URLs are rewritten when a prefix proxy is in use.
  if [[ -n "${GITHUB_PREFIX:-}" ]]; then
    sed "s#https://github.com/#${GITHUB_PREFIX%/}/https://github.com/#g" "${src}" > "${dst}"
  else
    cp "${src}" "${dst}"
  fi
  export PIP_CONSTRAINT="${DEPLOY_DIR}/env.lock.txt"
  log "env file ${src}, pip constraints ${PIP_CONSTRAINT}"
  if [[ -x "${PYTHON}" ]]; then
    "${CONDA_ROOT}/bin/conda" env update -p "${ENV_PREFIX}" -f "${dst}" --prune || return 1
  else
    "${CONDA_ROOT}/bin/conda" env create -p "${ENV_PREFIX}" -f "${dst}" || return 1
  fi
  check_stack
}

phase_data() {
  DATA_DIR="${DATA_DIR}" EVAL_DATA_DIR="${EVAL_DATA_DIR}" MODEL_ROOT="${MODEL_ROOT}" HF_HOME="${HF_HOME}" \
  PYTHON_BIN="${TOOLS_PYTHON}" DOWNLOAD_DATA=1 DOWNLOAD_MODELS=0 REQUIRE_MODELS=0 \
  REQUIRE_MATH_CODE_TRAIN_DATA="${INSTALL_TRAIN_DATA:-1}" REQUIRE_4DOMAIN_TRAIN_DATA=0 \
  DOWNLOAD_LCB="${INSTALL_EVAL:-1}" PULL_REPO_LFS_FALLBACK=0 \
    bash "${CODE_DIR}/scripts/download_training_assets.sh" || return 1
  if [[ "${INSTALL_EVAL:-1}" == 1 ]]; then
    [[ "$(git -C "${GOPD_DIR}" rev-parse HEAD 2>/dev/null)" == "${GOPD_COMMIT}" ]] || { log "G-OPD checkout is not at ${GOPD_COMMIT}"; return 1; }
  fi
}

phase_models() {
  local m rc=0
  for m in ${STUDENT_MODELS:-} ${TEACHER_MODEL:-}; do
    disk_ok || return 1
    log "model ${m}"
    "${TOOLS_PYTHON}" "${DEPLOY_DIR}/remote/fetch_hf.py" model "${m}" "${MODEL_ROOT}/${m##*/}" || rc=1
  done
  return "${rc}"
}

phase_ckpts() {
  local name path rc=0
  for name in ${EVAL_CHECKPOINTS:-}; do
    path="$(awk -F'\t' -v n="${name}" '$1 == n {print $2}' "${DEPLOY_DIR}/eval_models.tsv")"
    [[ -n "${path}" ]] || { log "unknown checkpoint name ${name}"; rc=1; continue; }
    [[ "${path}" == /* ]] && { log "${name} is a local path, nothing to download"; continue; }
    disk_ok || return 1
    "${TOOLS_PYTHON}" "${DEPLOY_DIR}/remote/fetch_hf.py" prefix "${CKPT_REPO}" "${path}" "${CKPT_ROOT}" || rc=1
  done
  return "${rc}"
}

phase_evalrt() {
  # download_mopd_data.sh checks out only the eval prompts; official scoring also needs code_eval/coding
  # (pinned EvalPlus source) and the LiveCodeBench sources, which G-OPD does not track: the data phase built them
  # from livecodebench/code_generation_lite at the pinned revision; rows and SHA-256 are checked by eval/lcb_official.py
  git -C "${GOPD_DIR}" sparse-checkout add code_eval/coding || return 1
  git -C "${GOPD_DIR}" checkout --detach "${GOPD_COMMIT}" || return 1
  local lcb_dst="${GOPD_DIR}/code_eval/coding/LiveCodeBench/code_generation_lite" f
  mkdir -p "${lcb_dst}"
  for f in test5.jsonl test6.jsonl; do
    [[ -f "${lcb_dst}/${f}" ]] || cp "${DATA_DIR}/.eval-source/LiveCodeBench/${f}" "${lcb_dst}/${f}" || return 1
  done
  (cd "${CODE_DIR}" && "${PYTHON}" -c 'from pathlib import Path; import sys
from eval.lcb_official import validate_lcb_source
for r in ("v5", "v6"): validate_lcb_source(Path(sys.argv[1]), r); print(r, "ok")' "${GOPD_DIR}") || return 1
  PYTHON="${PYTHON}" GOPD_REPO="${GOPD_DIR}" GOPD_COMMIT="${GOPD_COMMIT}" \
    bash "${CODE_DIR}/scripts/prepare_evalplus_rescore_runtime.sh" "${CODE_DIR}" || return 1
  # chat-template tokenizer used by the eval formatter (local_files_only in the standard suite)
  "${TOOLS_PYTHON}" -c 'from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-4B", allow_patterns=["*.json", "merges.txt", "vocab.json"])' || return 1
  # LiveCodeBench prompt modules import `anthropic`, which environment.yml lacks; kept outside the env (last on PYTHONPATH)
  "${PYTHON}" -m pip install -q --target "${LCB_EXTRA}" anthropic
}

phase_verify() {
  local n
  check_stack || return 1
  "${PYTHON}" - <<'PY' || return 1
import torch, vllm, flash_attn
assert torch.cuda.is_available(), "CUDA is unavailable"
a = torch.randn(64, 64, device="cuda"); b = torch.randn(64, 64, device="cuda")
assert torch.isfinite(a @ b).all()
print("CUDA_WITNESS", torch.cuda.get_device_name(0))
PY
  n=$(awk -F, '{print NF}' <<< "${GPU_IDS}")
  if (( n > 1 )); then PYTHON="${PYTHON}" bash "${CODE_DIR}/scripts/gpu_connectivity_check.sh" "${GPU_IDS}" || return 1; fi
  log "verify ok"
}

run_phase() {  # run_phase NAME : one phase, own log; state/NAME.{running,done,failed,secs} for progress reports
  local p="$1" t0 rc
  if [[ -f "${STATE_DIR}/${p}.done" && "${FORCE:-0}" != 1 ]]; then log "skip ${p} (done $(cat "${STATE_DIR}/${p}.done"))"; return 0; fi
  rm -f "${STATE_DIR}/${p}.failed" "${STATE_DIR}/${p}.done"; t0=$(date +%s); echo "${t0}" > "${STATE_DIR}/${p}.running"
  log "start ${p} (log ${LOG_DIR}/setup_${p}.log)"
  "phase_${p}" > "${LOG_DIR}/setup_${p}.log" 2>&1; rc=$?
  rm -f "${STATE_DIR}/${p}.running"; echo $(( $(date +%s) - t0 )) > "${STATE_DIR}/${p}.secs"
  if (( rc == 0 )); then
    date -Is > "${STATE_DIR}/${p}.done"; log "done ${p}"
  else
    echo "rc=${rc} log=${LOG_DIR}/setup_${p}.log" > "${STATE_DIR}/${p}.failed"
    log "FAIL ${p}; last lines of ${LOG_DIR}/setup_${p}.log:"; tail -n 15 "${LOG_DIR}/setup_${p}.log"; return 1
  fi
}

write_activate() {  # REMOTE_ROOT/activate.sh: the env and settings every training / evaluation command should use
  cat > "${REMOTE_ROOT}/activate.sh" <<EOF
# source ${REMOTE_ROOT}/activate.sh   (written by deploy/remote/setup.sh $(date -Is))
export PATH="${ENV_PREFIX}/bin:\${PATH}"
export PYTHON="${PYTHON}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${LCB_EXTRA}\${PYTHONPATH:+:\${PYTHONPATH}}"
export HF_HOME="${HF_HOME}" PIP_CACHE_DIR="${PIP_CACHE_DIR}"
source "${DEPLOY_DIR}/remote/common.sh" >/dev/null   # proxy / mirror routes from state/net.resolved.env
[ -f "\${HOME}/.config/opd-private/opd.env" ] && { set -a; . "\${HOME}/.config/opd-private/opd.env"; set +a; }
cd "${CODE_DIR}"
EOF
}

log "setup on $(hostname): REMOTE_ROOT=${REMOTE_ROOT} code=${CODE_DIR} phases=${PHASES} stack=torch ${REQ_TORCH} / vLLM ${REQ_VLLM} / transformers ${REQ_TRANSFORMERS}"
[[ -f "${CODE_DIR}/environment.yml" ]] || die "no code at ${CODE_DIR}"
if (( DRY )); then
  for p in base env data models ckpts evalrt verify; do printf '  %-7s %-5s %s\n' "$p" "$(want "$p" && echo run || echo -)" "$([[ -f ${STATE_DIR}/$p.done ]] && echo done || echo todo)"; done
  printf '  students: %s\n  teacher: %s\n  eval checkpoints: %s\n' "${STUDENT_MODELS:-}" "${TEACHER_MODEL:-}" "${EVAL_CHECKPOINTS:-}"
  exit 0
fi

rm -f "${STATE_DIR}/setup.exit"; echo $$ > "${STATE_DIR}/setup.pid"
finish() { echo "$1 $(date -Is)" > "${STATE_DIR}/setup.exit"; exit "$1"; }
failed=0
want base && { run_phase base || finish 1; }
pids=(); names=()
for p in env data models ckpts; do
  want "$p" || continue
  if (( SEQUENTIAL )); then run_phase "$p" || { failed=1; log "phase ${p} failed"; }
  else run_phase "$p" & pids+=("$!"); names+=("$p"); fi
done
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { failed=1; log "phase ${names[$i]} failed"; }; done
# evaluation runtime: whenever env and data are both done (whichever script finished second runs it)
if [[ "${INSTALL_EVAL:-1}" == 1 ]]; then
  if [[ -f "${STATE_DIR}/env.done" && -f "${STATE_DIR}/data.done" ]]; then
    if want evalrt || [[ ! -f "${STATE_DIR}/evalrt.done" ]]; then run_phase evalrt || failed=1; fi
  elif want evalrt; then log "evalrt waits until both env and data are done"; fi
fi
if want verify; then
  if [[ -f "${STATE_DIR}/env.done" ]]; then run_phase verify || failed=1; else log "skip verify: env not done"; failed=1; fi
fi
[[ -f "${STATE_DIR}/env.done" ]] && write_activate
(( failed )) && { log "setup finished with failures; fix and rerun (finished phases are skipped)"; finish 1; }
for p in env data models ckpts evalrt verify; do printf '  %-7s %s\n' "$p" "$([[ -f ${STATE_DIR}/$p.done ]] && echo done || echo -)"; done
[[ -f "${REMOTE_ROOT}/activate.sh" ]] && log "ready: source ${REMOTE_ROOT}/activate.sh"
log "setup complete"
finish 0
