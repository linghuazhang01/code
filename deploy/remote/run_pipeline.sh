#!/usr/bin/env bash
# Run a plan file step by step on the target: training, evaluation queues, uploads.
#   bash deploy/remote/run_pipeline.sh deploy/plans/<plan>.tsv
# Plan lines (whitespace-separated, # comments):
#   train      <config, relative to code/>  <gpu ids>  [hydra overrides...]   # start.sh --local --foreground
#   eval       <type:model:seed,...>        <gpu ids>                         # one eval_queue.sh per GPU, in the background
#   wait-eval                                                                 # wait for the eval queues started so far
#   upload     <suite name glob>                                              # finished suites -> RESULTS_REPO
# Training uploads its own step-60 checkpoint when the config enables huggingface_checkpoint; that needs HF_TOKEN
# (~/.config/opd-private/opd.env on the target) and a route to huggingface.co (net_check.sh: upload_route=direct).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PLAN="${1:?usage: run_pipeline.sh <plan.tsv>}"
[[ -f "${PLAN}" ]] || die "no plan file ${PLAN}"
# shellcheck source=/dev/null
[[ -f "${HOME}/.config/opd-private/opd.env" ]] && { set -a; source "${HOME}/.config/opd-private/opd.env"; set +a; }
[[ "${UPLOAD_ROUTE:-relay}" == direct ]] || log "NOTE: upload_route=${UPLOAD_ROUTE:-unknown}: the target cannot reach huggingface.co; disable huggingface_checkpoint uploads in configs and use bootstrap.sh relay-upload"

EVAL_PIDS=()
wait_eval() { local p rc=0; for p in "${EVAL_PIDS[@]}"; do wait "$p" || rc=1; done; EVAL_PIDS=(); return "$rc"; }
gpu_check() {  # author rule: NCCL connectivity check before every multi-GPU launch
  local ids="$1"
  (( $(awk -F, '{print NF}' <<< "${ids}") > 1 )) || return 0
  PYTHON="${PYTHON}" bash "${CODE_DIR}/scripts/gpu_connectivity_check.sh" "${ids}" | tee -a "${LOG_DIR}/pipeline.log"
  return "${PIPESTATUS[0]}"
}

step=0; failed=0
while read -r kind a1 a2 rest; do
  [[ -z "${kind}" || "${kind}" == \#* ]] && continue
  step=$((step + 1))
  case "${kind}" in
    train)
      log "step ${step}: train ${a1} on GPUs ${a2}"
      disk_ok || exit 1
      gpu_check "${a2}" || { log "GPU check failed on ${a2}; not launching"; exit 1; }
      (cd "${CODE_DIR}" && GPU_IDS="${a2}" PYTHON="${PYTHON}" bash start.sh -c "${a1}" --local --foreground ${rest:+-- ${rest}}) \
        >> "${LOG_DIR}/train_step${step}.log" 2>&1 || { log "training step ${step} failed (see ${LOG_DIR}/train_step${step}.log)"; exit 1; }
      ;;
    eval)
      log "step ${step}: eval on GPUs ${a2}: ${a1}"
      IFS=, read -ra gpus <<< "${a2}"
      for g in "${gpus[@]}"; do
        bash "${DEPLOY_DIR}/remote/eval_queue.sh" --gpu "${g}" --jobs "${a1}" >> "${LOG_DIR}/eval_step${step}_gpu${g}.log" 2>&1 &
        EVAL_PIDS+=("$!")
        sleep 30   # stagger vLLM start-up
      done
      ;;
    wait-eval)
      log "step ${step}: waiting for ${#EVAL_PIDS[@]} eval queue(s)"
      wait_eval || failed=1
      ;;
    upload)
      log "step ${step}: upload ${a1}"
      if [[ -z "${RESULTS_REPO:-}" ]]; then log "RESULTS_REPO is empty; skip upload"; continue; fi
      if [[ "${UPLOAD_ROUTE:-relay}" != direct ]]; then log "upload_route=relay; run bootstrap.sh relay-upload from your computer"; continue; fi
      mapfile -t suites < <(compgen -G "${EVAL_OUT}/${a1}" || true)
      (( ${#suites[@]} )) || { log "no suites match ${a1}"; continue; }
      "${TOOLS_PYTHON}" "${DEPLOY_DIR}/remote/upload_results.py" --repo "${RESULTS_REPO}" \
        --repo-type "${RESULTS_REPO_TYPE:-dataset}" --prefix "${RESULTS_PREFIX:-eval_results}" "${suites[@]}" || failed=1
      ;;
    *) die "unknown plan step '${kind}' (line ${step})" ;;
  esac
done < "${PLAN}"
wait_eval || failed=1
(( failed )) && { log "pipeline finished with failures"; exit 1; }
log "pipeline finished"
