#!/usr/bin/env bash
# v12 = v11 with a 32768-token cap for the Math4 re-evaluation of 2026-11-01 (tags math4_32k_*); vLLM needs max_num_batched_tokens >= max_model_len.
# Generic single-A100 eval queue (Burgundy hold-job allocation, DP1) for Math4 / Code4 suites, K=8, step 60, CUDA graphs.
#   srun --jobid=<J> --overlap --ntasks=1 --cpus-per-task=8 --gres=gpu:a100:1 \
#        bash run_eval_jobs_a100.sh --jobs math4:q4b_opdnative:42,code4:q1p7b_code_opdnative:42 [--prefetch|--dry-run]
# A job is <type>:<model>:<seed>. math4 = AIME24/25 + HMMT25Feb/Nov; code4 = HumanEval+, MBPP+, LiveCodeBench v5/v6; mc8 = both.
# Same pipeline as run_mc8_q4b_a100.sh: generation without the in-loop Docker scorer, then merge / official LiveCodeBench /
# official EvalPlus in a background subshell while the GPU starts the next suite. Stack: repo environment.yml env
# (torch 2.6.0 / vLLM 0.8.5.post1), tags end in _a100_cg_s<seed>_20261022. v7 = v6 + 12-run ignore/only (Code/Math) and Math+Code v3-pool ignore/only 1.7B; v6 = v5 + Code-only 1.7B OPD naive entropy top20 / tail80 (q1p7b_code_enttop20 / q1p7b_code_enttail80), DATE_TAG 20261023.
set -uo pipefail

BASE="${OPD_EVAL_BASE:-${HOME}/scratch/opd}"
W="${EVAL_HOME:-${BASE}}"   # v9: writable root (scratch is over its soft quota: data writes fail with Errno 122)
ROOT="${BASE}/mopd_code_a100mc"
PYTHON="${BASE}/miniforge3/envs/mopd-verl-a100/bin/python"
GOPD_COMMIT=37371a4c31ad7947746200d234161769191f4748
GOPD_DIR="${BASE}/a100eval_data/G-OPD"
HF_REPO=icemoon28/opd-checkpoints
HF_ROOT="${W}/a100eval_models"
OUT="${W}/a100eval_outputs_mc"
LOG_ROOT="${W}/a100eval_logs"
GPU_MEMORY="${GPU_MEMORY:-0.90}"
MIN_FREE_GIB=500
DATE_TAG=20261101

declare -A PREFIX
PREFIX[q4b_opdnative]="checkpoints/qwen4b/math/q4b-math-opdnative-toklog-3g3c-b255-s42/global_step_60"
PREFIX[q1p7b_code_opdnative]="checkpoints/code/q1p7b-code-opdnative-toklog-3g3c-b255/global_step_60"
PREFIX[q1p7b_math_opd_toklog]="checkpoints/math/q1p7b-math-top32kl-opd-toklog-3g3c-b255/global_step_60"
PREFIX[q4b_math_fireopd]="checkpoints/qwen4b/math/q4b-math-fireopd-4g4c-b256-s42/global_step_60"
PREFIX[q4b_mc_opdnative]="checkpoints/math_code/q4b-mc-opdnative-8g8c-b528-s42-20261018/global_step_60"
PREFIX[q4b_mc_opd]="checkpoints/math_code/q4b-mc-opd-audit-8g8c-20261017-040055/global_step_60"
PREFIX[q4b_mc_tip]="checkpoints/math_code/q4b-mc-tiptopk32-toklog-20261016-143323/global_step_60"
PREFIX[q1p7b_code_opd]="checkpoints/code/q1p7b-code-top32kl-opd-toklog-3g3c-b255/global_step_60"
PREFIX[q4b_code_opd]="checkpoints/qwen4b/code/q4b-code-opd-toklog-3g3c-b255-s42/global_step_60"
PREFIX[q4b_code_opdnative]="checkpoints/qwen4b/code/q4b-code-opdnative-toklog-3g3c-b255-s42/global_step_60"
PREFIX[q1p7b_math_opdnative]="checkpoints/math/q1p7b-math-opdnative-toklog-3g3c-b255/global_step_60"
PREFIX[q1p7b_math_enttop20]="checkpoints/math/q1p7b-math-opdnative-enttop20-toklog-8g8c-b256-s42-20261021/global_step_60"
PREFIX[q1p7b_math_enttail80]="checkpoints/math/q1p7b-math-opdnative-enttail80-toklog-8g8c-b256-s42-20261021/global_step_60"
PREFIX[q4b_mc_eopd]="checkpoints/math_code/q4b-mc-eopd-8g8c-b528-s42-20261018/global_step_60"
PREFIX[q4b_code_tip]="checkpoints/code/q4b-tip-topk32-code-8c-b256-s42-20261019/global_step_60"
PREFIX[q4b_code_eopd]="checkpoints/code/q4b-eopd-code-8c-b256-s42-20261019/global_step_60"
PREFIX[q4b_code_exopd]="checkpoints/code/q4b-exopd-code-8c-b256-s42-20261019/global_step_60"
PREFIX[q4b_code_fireopd]="checkpoints/code/q4b-fire-opd-code-8c-b256-s42-20261019/global_step_60"
PREFIX[q1p7b_math_termlook]="checkpoints/math/q1p7b-math-opdnative-l12only-termlook-toklog-8g8c-b256-s42/global_step_60"
PREFIX[q1p7b_code_termlook]="checkpoints/code/q1p7b-code-opdnative-l12only-termlook-toklog-8g8c-b256-s42/global_step_60"
PREFIX[q1p7b_code_tip]="checkpoints/code/qwen1p7b-30b-code-tip-topk32-rho50-4gpu-4c-b256-s42/global_step_60"
PREFIX[q1p7b_code_eopd]="checkpoints/code/qwen1p7b-30b-code-eopd-native-4gpu-4c-b256-s42/global_step_60"
PREFIX[q1p7b_mc_opdnative]="checkpoints/baselines/math_code/q1p7b-mc-opdnative-toklog-4g4c-b256-s42/global_step_60"
PREFIX[q1p7b_code_enttop20]="checkpoints/code/q1p7b-code-opdnative-enttop20-toklog-8g8c-b256-s42-20261021/global_step_60"
PREFIX[q1p7b_code_enttail80]="checkpoints/code/q1p7b-code-opdnative-enttail80-toklog-8g8c-b256-s42-20261021/global_step_60"
PREFIX[q1p7b_code_l12ign]="checkpoints/code/q1p7b-code-opdnative-l12ign-toklog-8g8c-b256-s42-20261020/global_step_60"
PREFIX[q1p7b_code_l12only]="checkpoints/code/q1p7b-code-opdnative-l12only-toklog-8g8c-b256-s42-20261020/global_step_60"
PREFIX[q1p7b_math_l12ign]="checkpoints/math/q1p7b-math-opdnative-l12ign-toklog-8g8c-b256-s42-20261020/global_step_60"
PREFIX[q1p7b_math_l12only]="checkpoints/math/q1p7b-math-opdnative-l12only-toklog-8g8c-b256-s42-20261020/global_step_60"
PREFIX[q1p7b_mc_v3ign]="checkpoints/math_code/q1p7b-mc-opdv3ign-toklog-20261016-143323/global_step_60"
PREFIX[q1p7b_mc_v3only]="checkpoints/math_code/q1p7b-mc-opdv3only-8c-20261015-123124/global_step_60"
PREFIX[q1p7b_mc_v3ign_early]="checkpoints/mopd/q1p7b-mc-opdv3ign-20261015-021939/global_step_60"
PREFIX[q1p7b_code_exopd]="checkpoints/code/qwen1p7b-30b-code-exopd-native-lambda1p25-4gpu-4c-b256-s42/global_step_60"
PREFIX[q1p7b_code_fireopd]="checkpoints/code/qwen1p7b-30b-code-fire-opd-native-4gpu-4c-b256-s42/global_step_60"
declare -A DSETS
DSETS[math4]="aime24,aime25,hmmt25feb,hmmt25nov"
DSETS[code4]="humaneval_plus,mbpp_plus,lcb_v5,lcb_v6"
DSETS[mc8]="aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus,lcb_v5,lcb_v6"

JOBS=""; MODE=run
while (( $# )); do
  case "$1" in
    --jobs) JOBS="${2:?--jobs name:seed,...}"; shift 2 ;;
    --prefetch) MODE=prefetch; shift ;;
    --dry-run) MODE=dry; shift ;;
    --smoke) MODE=smoke; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${JOBS}" ]] || { echo "--jobs is required" >&2; exit 2; }
IFS=, read -ra RUN <<< "${JOBS}"

HOSTTAG="${SLURM_JOB_ID:-nojob}${SLOT:-}"  # v8: SLOT=a|b lets two queues share one card (separate tmp/cache dirs and status logs)
export TMPDIR="${W}/tmp/a100x_${HOSTTAG}"
export HF_HOME="${W}/hf_home"
export VLLM_CACHE_ROOT="${W}/tmp/vllm_cache_${HOSTTAG}"
export TORCHINDUCTOR_CACHE_DIR="${W}/tmp/inductor_${HOSTTAG}"
export TRITON_CACHE_DIR="${W}/tmp/triton_${HOSTTAG}"
# lcb_extra holds `anthropic` (+ distro, docstring-parser), which the LiveCodeBench prompt modules import
# but the repo environment.yml env lacks; it is last on the path so it can never shadow an installed package.
export PYTHONPATH="${ROOT}:${ROOT}/third_party/verl:${PYTHONPATH:-}:${BASE}/a100eval_env/lcb_extra"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export LANG=C LC_ALL=C PYTHONINTMAXSTRDIGITS=0 TOKENIZERS_PARALLELISM=false
export MOPD_ALLOW_SIMPLE_SCORER_FALLBACK=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
# Official LiveCodeBench scoring defaults to 64 processes (patched in this tree to read MOPD_LCB_PROCS); the job has 8 CPUs
# shared with the generation of the next suite, so use fewer processes to avoid spurious 6 s test timeouts.
export MOPD_LCB_PROCS="${MOPD_LCB_PROCS:-5}"
export HUMANEVAL_SOURCE="${GOPD_DIR}/code_eval/data/HumanEvalPlus.jsonl"
export MBPP_SOURCE="${GOPD_DIR}/code_eval/data/MbppPlus.jsonl"
unset ROCR_VISIBLE_DEVICES
mkdir -p "${TMPDIR}" "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${HF_ROOT}" "${OUT}" "${LOG_ROOT}"
QLOG="${LOG_ROOT}/x_queue_${HOSTTAG}.status.log"
status() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${QLOG}"; }
has_weights() { ls "$1"/model*.safetensors > /dev/null 2>&1; }

fetch_hub() {
  "${PYTHON}" - "${HF_REPO}" "$1" "${HF_ROOT}" >> "${LOG_ROOT}/hf_download.log" 2>&1 <<'PY'
import datetime as dt, hashlib, json, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo, prefix, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
api = HfApi()
revision = api.repo_info(repo).sha
remote = [i for i in api.list_repo_tree(repo, path_in_repo=prefix, recursive=True, expand=True, revision=revision)
          if getattr(i, "size", None) is not None]
assert any(Path(i.path).name.startswith("model") and i.path.endswith(".safetensors") for i in remote), \
    f"no model*.safetensors under {prefix}@{revision}"
snapshot_download(repo, revision=revision, allow_patterns=[f"{prefix}/*"], local_dir=root)
files = []
for i in remote:
    local = root / i.path
    size = local.stat().st_size
    assert size == i.size, f"size mismatch {i.path}: {size} != {i.size}"
    entry = {"path": i.path, "size": size}
    if i.lfs is not None:
        h = hashlib.sha256()
        with local.open("rb") as f:
            for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
                h.update(block)
        assert h.hexdigest() == i.lfs.sha256, f"sha256 mismatch {i.path}"
        entry["sha256"] = h.hexdigest()
    files.append(entry)
target = root / prefix
(target / "HF_PROVENANCE.json").write_text(json.dumps({
    "time": dt.datetime.now(dt.timezone.utc).isoformat(), "repo": repo, "revision": revision,
    "path_in_repo": prefix, "local_path": str(target.resolve()), "files": files}, indent=2) + "\n")
print(target.resolve())
PY
}

ensure_weights() {
  local name="$1" dir="${HF_ROOT}/${PREFIX[$1]}"
  has_weights "${dir}" && return 0
  [[ "${MODE}" == dry ]] && { status "would download ${name}: ${HF_REPO}/${PREFIX[$name]}"; return 0; }
  status "downloading ${name}: ${HF_REPO}/${PREFIX[$name]}"
  fetch_hub "${PREFIX[$name]}" || { status "FAIL download ${name}; see ${LOG_ROOT}/hf_download.log"; return 1; }
  has_weights "${dir}" || { status "FAIL ${name}: no weights in ${dir} after download"; return 1; }
  status "downloaded ${name}: ${dir}"
}

disk_ok() {  # AGENTS.md launch gate on every filesystem this run touches (all on scratch)
  local mount free_gib
  for mount in "${BASE}" "${TMPDIR}"; do
    free_gib=$(df -BG --output=avail "${mount}" | tail -n 1 | tr -dc '0-9')
    if [[ -z "${free_gib}" || "${free_gib}" -lt "${MIN_FREE_GIB}" ]]; then
      status "STOP: ${mount} has ${free_gib:-?} GiB free < ${MIN_FREE_GIB} GiB gate"; return 1
    fi
  done
}

# Background part: merge (+ official LiveCodeBench), official EvalPlus, then mark the suite complete.
finalize_suite() {
  local suite="$1" type="$2" tag; tag="$(basename "${suite}")"
  if [[ "${type}" == math4 ]]; then  # Math only: plain merge, no Code scoring
    if ! "${PYTHON}" -m eval.parallel_eval merge --manifest "${suite}/suite_manifest.json" > "${suite}/logs/merge.log" 2>&1; then
      status "FAIL merge ${tag} (see ${suite}/logs/merge.log)"; return 1
    fi
  else
    if ! "${PYTHON}" -m eval.parallel_eval merge --manifest "${suite}/suite_manifest.json" --defer-completion \
        > "${suite}/logs/merge.log" 2>&1; then
      status "FAIL merge ${tag} (see ${suite}/logs/merge.log)"; return 1
    fi
    if ! PYTHON="${PYTHON}" bash "${ROOT}/scripts/finalize_local_standard_eval.sh" "${suite}"; then
      status "FAIL official scoring ${tag} (see ${suite}/logs/official_evalplus.log)"; return 1
    fi
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "${suite}/COMPLETED_AT_UTC"
  status "eval done ${tag}"
}

status "start mode=${MODE} jobs=${JOBS} host=$(hostname) slurm_job=${HOSTTAG} cpus=$(nproc) gpu=$(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | tr '\n' ';')"
[[ "$(nvidia-smi -L | wc -l)" == 1 ]] || { status "FAIL expected exactly 1 visible GPU"; exit 1; }
[[ -x "${ROOT}/.runtime/evalplus-gopd-${GOPD_COMMIT}/venv/bin/python" ]] || { status "FAIL pinned EvalPlus runtime missing under ${ROOT}/.runtime"; exit 1; }
[[ -f "${GOPD_DIR}/code_eval/coding/LiveCodeBench/code_generation_lite/test6.jsonl" ]] || { status "FAIL G-OPD LiveCodeBench files missing in ${GOPD_DIR}"; exit 1; }

declare -A SEEN=()
for job in "${RUN[@]}"; do
  IFS=: read -r type name seed <<< "${job}"
  [[ "${type}" == math4 ]] || { status "FAIL v12 runs math4 only, got ${job}"; exit 2; }
  [[ -n "${DSETS[${type}]:-}" ]] || { status "FAIL unknown suite type ${type} in ${job}"; exit 2; }
  [[ -n "${PREFIX[${name}]:-}" ]] || { status "FAIL unknown model ${name} in ${job}"; exit 2; }
  [[ "${seed}" =~ ^[0-9]+$ ]] || { status "FAIL bad seed in ${job}"; exit 2; }
  [[ -n "${SEEN[${name}]:-}" ]] && continue; SEEN[${name}]=1
  [[ "${MODE}" == run ]] || ensure_weights "${name}" || exit 1
done
[[ "${MODE}" == prefetch || "${MODE}" == dry ]] && { status "${MODE}: stop before evaluation"; exit 0; }

failed=0; bg_pids=()
for job in "${RUN[@]}"; do
  IFS=: read -r type name seed <<< "${job}"
  model="${HF_ROOT}/${PREFIX[${name}]}"
  tag="${type}_32k_k8_s60_${name}_a100_cg_s${seed}_${DATE_TAG}"
  extra=(); [[ "${MODE}" == smoke ]] && { tag="smoke_${tag}"; extra=(--max-samples-per-dataset 2); }
  suite="${OUT}/${tag}"
  if [[ -f "${suite}/COMPLETED_AT_UTC" ]]; then status "skip ${tag}: already complete"; continue; fi
  disk_ok || exit 1
  # Several queues (one per A100) share one job list: the first queue to mkdir the lock takes the suite.
  # A lock is never released automatically; delete ${OUT}/.locks/<tag> by hand to re-run a cancelled/failed suite.
  mkdir -p "${OUT}/.locks"
  if [[ "${MODE}" != smoke ]] && ! mkdir "${OUT}/.locks/${tag}" 2>/dev/null; then
    status "skip ${tag}: taken by another queue"; continue
  fi
  echo "job=${HOSTTAG} host=$(hostname) at=$(date -u +%FT%TZ)" > "${OUT}/.locks/${tag}/owner" 2>/dev/null
  # v3: download this model now (serialised by flock so two cards never write the same directory)
  ( flock 9; ensure_weights "${name}" ) 9> "${LOG_ROOT}/.weights.lock" \
    || { status "FAIL weights ${name} for ${tag}"; rm -rf "${OUT}/.locks/${tag}"; failed=$((failed + 1)); continue; }
  resume=(); [[ -e "${suite}" ]] && resume=(--resume)
  mkdir -p "$(dirname "${suite}")"
  status "eval start ${tag} model=${model} ${resume[*]}"
  cd "${ROOT}"
  if [[ ! -f "${suite}/GENERATION_DONE" ]]; then
    "${PYTHON}" -m eval.parallel_eval plan \
      --code-dir "${ROOT}" --suite-root "${suite}" --run-tag "${tag}" \
      --model-path "${model}" --eval-model-path "${model}" --gopd-dir "${GOPD_DIR}" \
      --datasets "${DSETS[${type}]}" --shards-per-dataset 16 --worker-count 1 --min-rows-per-shard 1 \
      --math-samples 8 --code-samples 8 --science-samples 8 --base-seed "${seed}" --seed-sequence-offset 0 \
      --max-new-tokens 32768 --temperature 1.0 --top-p 1.0 --batch-size 24 \
      --gpu-memory "${GPU_MEMORY}" --max-model-len 34816 --max-num-batched-tokens 34816 --max-num-seqs 24 \
      --no-score-code --code-scoring-workers 0 --code-scoring-pending-shards 2 --cuda-graphs \
      "${extra[@]}" "${resume[@]}" > "${OUT}/${tag}.plan.log" 2>&1 \
      || { status "FAIL plan ${tag} (see ${OUT}/${tag}.plan.log)"; failed=$((failed + 1)); continue; }
    mkdir -p "${suite}/logs"; mv "${OUT}/${tag}.plan.log" "${suite}/logs/plan.log"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
      "${PYTHON}" -m eval.parallel_worker --manifest "${suite}/suite_manifest.json" \
        --eval-model-path "${model}" --worker-id 0 "${resume[@]}" > "${suite}/logs/gpu_worker_0.log" 2>&1
    rc=$?
    if (( rc != 0 )); then
      status "FAIL generation ${tag} rc=${rc} (see ${suite}/logs/gpu_worker_0.log)"; failed=$((failed + 1)); continue
    fi
    date -u +%Y-%m-%dT%H:%M:%SZ > "${suite}/GENERATION_DONE"
  fi
  status "generation done ${tag}; merging and scoring in the background, GPU moves on to the next suite"
  finalize_suite "${suite}" "${type}" &
  bg_pids+=("$!")
  sleep 20
done
for pid in "${bg_pids[@]}"; do wait "${pid}" || failed=$((failed + 1)); done
if (( failed )); then status "finished with ${failed} failed step(s)"; exit 1; fi
status "finished all suites"
