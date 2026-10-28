#!/usr/bin/env bash
# One evaluation queue on one GPU (DP1), same protocol as the Burgundy A100 tables: K=8, T=1.0, top-p 1.0,
# 16,384 new tokens, CUDA graphs, vLLM from environment.yml. A job is <type>:<model>:<seed>, type is
#   math4 = AIME24/25 + HMMT25 Feb/Nov,  code4 = HumanEval+, MBPP+, LiveCodeBench v5/v6,  mc8 = both,
# and <model> is a name from deploy/eval_models.tsv. Generation runs on the GPU; merge + official LiveCodeBench +
# official EvalPlus run in the background while the GPU starts the next suite.
# Several queues (one per GPU, any host sharing EVAL_OUT) can take the same job list: the first to create
# eval_outputs/.locks/<tag> runs it. Locks are never released automatically; delete one by hand to rerun a suite.
#   bash deploy/remote/eval_queue.sh --gpu 0 --jobs math4:q1p7b_math_termlook:42,code4:q1p7b_code_termlook:42 [--dry-run|--smoke]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

GPU=""; JOBS=""; MODE=run
while (( $# )); do
  case "$1" in
    --gpu) GPU="${2:?}"; shift 2 ;;
    --jobs) JOBS="${2:?}"; shift 2 ;;
    --dry-run) MODE=dry; shift ;;
    --smoke) MODE=smoke; shift ;;
    *) die "unknown argument $1" ;;
  esac
done
[[ "${GPU}" =~ ^[0-9]+$ ]] || die "--gpu <physical index> is required"
[[ -n "${JOBS}" ]] || die "--jobs is required"
IFS=, read -ra RUN <<< "${JOBS}"

declare -A DSETS=(
  [math4]="aime24,aime25,hmmt25feb,hmmt25nov"
  [code4]="humaneval_plus,mbpp_plus,lcb_v5,lcb_v6"
  [mc8]="aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus,lcb_v5,lcb_v6"
)
declare -A MODELS
while IFS=$'\t' read -r n p; do [[ -z "${n}" || "${n}" == \#* ]] || MODELS[$n]="$p"; done < "${DEPLOY_DIR}/eval_models.tsv"

HW="${EVAL_HW_TAG:-auto}"
if [[ "${HW}" == auto ]]; then
  HW="$(nvidia-smi -i "${GPU}" --query-gpu=name --format=csv,noheader | sed -E 's/^NVIDIA //; s/[- ].*//' | tr 'A-Z' 'a-z')"
fi
DATE_TAG="${EVAL_DATE_TAG:-20261023}"
QTAG="$(hostname -s)_gpu${GPU}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export VLLM_CACHE_ROOT="${TMPDIR}/vllm_${QTAG}" TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/inductor_${QTAG}" TRITON_CACHE_DIR="${TMPDIR}/triton_${QTAG}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}:${LCB_EXTRA}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export LANG=C LC_ALL=C PYTHONINTMAXSTRDIGITS=0 TOKENIZERS_PARALLELISM=false
export MOPD_ALLOW_SIMPLE_SCORER_FALLBACK=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
# Official LiveCodeBench test processes (6 s per-test timeout). The A100 tables used 5 on an 8-CPU job; too many
# processes for the CPUs turn slow tests into spurious timeouts. Default: CPUs minus 3 (generation of the next
# suite runs at the same time), at least 2, at most 64.
NCPU="$(nproc)"; LCB_DEFAULT=$(( NCPU - 3 )); (( LCB_DEFAULT < 2 )) && LCB_DEFAULT=2; (( LCB_DEFAULT > 64 )) && LCB_DEFAULT=64
export MOPD_LCB_PROCS="${MOPD_LCB_PROCS:-${LCB_DEFAULT}}"
export HUMANEVAL_SOURCE="${GOPD_DIR}/code_eval/data/HumanEvalPlus.jsonl" MBPP_SOURCE="${GOPD_DIR}/code_eval/data/MbppPlus.jsonl"
mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${CKPT_ROOT}" "${EVAL_OUT}/.locks"
QLOG="${LOG_DIR}/eval_queue_${QTAG}.status.log"
status() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${QLOG}"; }

model_dir() {  # name -> local directory holding the weights
  local p="${MODELS[$1]}"
  [[ "${p}" == /* ]] && echo "${p}" || echo "${CKPT_ROOT}/${p}"
}
ensure_weights() {  # serialised by flock so two queues never download the same checkpoint at once
  local name="$1" p="${MODELS[$1]}" dir; dir="$(model_dir "$1")"
  ls "${dir}"/model*.safetensors >/dev/null 2>&1 && return 0
  [[ "${p}" == /* ]] && { status "FAIL ${name}: no weights in local path ${dir}"; return 1; }
  [[ "${MODE}" == dry ]] && { status "would download ${name}: ${CKPT_REPO}/${p}"; return 0; }
  status "downloading ${name}: ${CKPT_REPO}/${p}"
  ( flock 9; "${TOOLS_PYTHON}" "${DEPLOY_DIR}/remote/fetch_hf.py" prefix "${CKPT_REPO}" "${p}" "${CKPT_ROOT}" ) \
    9> "${CKPT_ROOT}/.weights.lock" >> "${LOG_DIR}/eval_downloads.log" 2>&1 || { status "FAIL download ${name}"; return 1; }
}

finalize_suite() {  # background: merge (+ official LiveCodeBench), official EvalPlus, completion marker
  local suite="$1" type="$2" tag; tag="$(basename "${suite}")"
  if [[ "${type}" == math4 ]]; then
    "${PYTHON}" -m eval.parallel_eval merge --manifest "${suite}/suite_manifest.json" > "${suite}/logs/merge.log" 2>&1 \
      || { status "FAIL merge ${tag}"; return 1; }
  else
    "${PYTHON}" -m eval.parallel_eval merge --manifest "${suite}/suite_manifest.json" --defer-completion > "${suite}/logs/merge.log" 2>&1 \
      || { status "FAIL merge ${tag}"; return 1; }
    PYTHON="${PYTHON}" bash "${CODE_DIR}/scripts/finalize_local_standard_eval.sh" "${suite}" \
      || { status "FAIL official scoring ${tag} (see ${suite}/logs/official_evalplus.log)"; return 1; }
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "${suite}/COMPLETED_AT_UTC"
  status "eval done ${tag}"
}

status "start mode=${MODE} gpu=${GPU} hw=${HW} host=$(hostname) jobs=${JOBS}"
[[ -x "${CODE_DIR}/.runtime/evalplus-gopd-${GOPD_COMMIT}/venv/bin/python" ]] || { status "FAIL EvalPlus runtime missing (run setup.sh --phases evalrt)"; exit 1; }
[[ -f "${GOPD_DIR}/code_eval/coding/LiveCodeBench/code_generation_lite/test6.jsonl" ]] || { status "FAIL LiveCodeBench sources missing in ${GOPD_DIR}"; exit 1; }
for job in "${RUN[@]}"; do
  IFS=: read -r type name seed <<< "${job}"
  [[ -n "${DSETS[${type}]:-}" ]] || { status "FAIL unknown suite type in ${job}"; exit 2; }
  [[ -n "${MODELS[${name}]:-}" ]] || { status "FAIL unknown model in ${job} (add it to deploy/eval_models.tsv)"; exit 2; }
  [[ "${seed}" =~ ^[0-9]+$ ]] || { status "FAIL bad seed in ${job}"; exit 2; }
  [[ "${MODE}" == dry ]] && ensure_weights "${name}"
done
[[ "${MODE}" == dry ]] && { status "dry: stop before evaluation"; exit 0; }

failed=0; bg=()
for job in "${RUN[@]}"; do
  IFS=: read -r type name seed <<< "${job}"
  tag="${type}_k8_s60_${name}_${HW}_cg_s${seed}_${DATE_TAG}"
  extra=(); [[ "${MODE}" == smoke ]] && { tag="smoke_${tag}"; extra=(--max-samples-per-dataset 2); }
  suite="${EVAL_OUT}/${tag}"
  [[ -f "${suite}/COMPLETED_AT_UTC" ]] && { status "skip ${tag}: complete"; continue; }
  disk_ok || exit 1
  if [[ "${MODE}" != smoke ]] && ! mkdir "${EVAL_OUT}/.locks/${tag}" 2>/dev/null; then status "skip ${tag}: taken by another queue"; continue; fi
  echo "queue=${QTAG} at=$(date -u +%FT%TZ)" > "${EVAL_OUT}/.locks/${tag}/owner" 2>/dev/null
  ensure_weights "${name}" || { rm -rf "${EVAL_OUT}/.locks/${tag}"; failed=$((failed + 1)); continue; }
  model="$(model_dir "${name}")"
  resume=(); [[ -e "${suite}" ]] && resume=(--resume)
  status "eval start ${tag} model=${model} ${resume[*]}"
  cd "${CODE_DIR}" || exit 1
  if [[ ! -f "${suite}/GENERATION_DONE" ]]; then
    "${PYTHON}" -m eval.parallel_eval plan \
      --code-dir "${CODE_DIR}" --suite-root "${suite}" --run-tag "${tag}" \
      --model-path "${model}" --eval-model-path "${model}" --gopd-dir "${GOPD_DIR}" \
      --datasets "${DSETS[${type}]}" --shards-per-dataset 16 --worker-count 1 --min-rows-per-shard 1 \
      --math-samples 8 --code-samples 8 --science-samples 8 --base-seed "${seed}" --seed-sequence-offset 0 \
      --max-new-tokens 16384 --temperature 1.0 --top-p 1.0 --batch-size 24 \
      --gpu-memory "${EVAL_GPU_MEMORY:-0.90}" --max-model-len 18432 --max-num-batched-tokens 32768 --max-num-seqs 24 \
      --no-score-code --code-scoring-workers 0 --code-scoring-pending-shards 2 --cuda-graphs \
      "${extra[@]}" "${resume[@]}" > "${EVAL_OUT}/${tag}.plan.log" 2>&1 \
      || { status "FAIL plan ${tag} (see ${EVAL_OUT}/${tag}.plan.log)"; failed=$((failed + 1)); continue; }
    mkdir -p "${suite}/logs"; mv "${EVAL_OUT}/${tag}.plan.log" "${suite}/logs/plan.log"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      "${PYTHON}" -m eval.parallel_worker --manifest "${suite}/suite_manifest.json" \
        --eval-model-path "${model}" --worker-id 0 "${resume[@]}" > "${suite}/logs/gpu_worker_0.log" 2>&1
    rc=$?; (( rc == 0 )) || { status "FAIL generation ${tag} rc=${rc}"; failed=$((failed + 1)); continue; }
    date -u +%Y-%m-%dT%H:%M:%SZ > "${suite}/GENERATION_DONE"
  fi
  status "generation done ${tag}; scoring in the background"
  finalize_suite "${suite}" "${type}" & bg+=("$!")
  sleep 20
done
for pid in "${bg[@]}"; do wait "${pid}" || failed=$((failed + 1)); done
(( failed )) && { status "finished with ${failed} failed step(s)"; exit 1; }
status "finished all suites"
