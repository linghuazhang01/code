#!/usr/bin/env bash
# Eager vs CUDA-graph evaluation of three step-60 baselines at the seeds chosen on 2026-10-11, for any
# machine with this repo. CityU pool scores (CUDA graphs) in brackets:
#   math4:exopd:2042 / 1042 / 42   Math-only ExOPD        Math4  [25.73 / 25.42 / 25.31]
#   mc8:eopd:2042 / 1042           Math+Code EOPD         Macro8 [31.12 / 30.50]
#   mc8:tip:42                     Math+Code TIP-TopK32   Macro8 [31.64; eager 2026-09-26: 30.35]
# math4 = AIME24, AIME25, HMMT25Feb, HMMT25Nov; mc8 = math4 + HumanEval+, MBPP+, LCB v5, LCB v6; K=8.
#
# Each job runs twice: "cg" (vLLM CUDA graphs, enforce_eager=False) then "eager" (enforce_eager=True).
# Nothing else differs between the two. Both use:
#   - the job's base seed with the CityU 16-shards-per-dataset seed map, checked against reference/
#     before any GPU work
#   - T=1.0, top_p=1.0, max_new_tokens 16384, batch 24
#   - gpu_memory GPU_MEMORY and max_num_seqs MAX_NUM_SEQS
#   - one TP=1 vLLM worker per GPU in GPU_IDS
# Math4 jobs run first. Once a suite's GPU work is done, its merge and CPU-only scoring (LiveCodeBench,
# official EvalPlus) run in the background while the next suite generates. At the end, compare.md puts
# each job's eager and cg scores next to the CityU references, with paired per-problem bootstrap CIs.
#
# Usage:
#   bash run_eager_vs_cg.sh --dry-run        # every check + model download + plan check, no GPU work
#   bash run_eager_vs_cg.sh                  # full run; rerun the same command to resume
#   bash run_eager_vs_cg.sh --jobs mc8:tip:42 --modes eager
#   ... --setup-evalplus (build the pinned EvalPlus runtime once) | --download-only | --compare-only
# Environment:
#   CODE_DIR     this repo (default: three levels up)
#   PYTHON       mopd-verl python (default: python3)
#   GPU_IDS      default: all GPUs
#   GPU_MEMORY   default 0.75
#   MAX_NUM_SEQS default 192
#   RUN_ID       default 20261011
#   OUTPUT_ROOT  default CODE_DIR/experiments_records/eval
#   MODEL_ROOT   download dir, default CODE_DIR/checkpoints/hf_eval_evmode
#   GOPD_DIR     G-OPD checkout, needed for mc8 jobs
#   EXOPD_MATH_DIR, TIP_MC_DIR, EOPD_MC_DIR  optional local copies; the Hub copy is downloaded otherwise
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="${CODE_DIR:-$(cd "${HERE}/../../.." && pwd -P)}"
PYTHON="${PYTHON:-python3}"
GPU_MEMORY="${GPU_MEMORY:-0.75}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-192}"
RUN_ID="${RUN_ID:-20261011}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CODE_DIR}/experiments_records/eval}"
MODEL_ROOT="${MODEL_ROOT:-${CODE_DIR}/checkpoints/hf_eval_evmode}"
GOPD_DIR="${GOPD_DIR:-}"
GOPD_COMMIT="37371a4c31ad7947746200d234161769191f4748"
SANDBOX_IMAGE="${MOPD_CODE_SANDBOX_IMAGE:-verlai/verl:vllm023.dev1}"
HF_REPO="icemoon28/opd-checkpoints"
HF_REVISION="e5903df7d42e244bf10e790a1213e3a44ed2a6ce"   # 2026-10-11, after the EOPD upload; all three verified

# model | Hub path_in_repo | sha256 of model.safetensors (= the weights of the CityU
# reference suites) | env var of an optional local copy
MODELS=(
  "exopd_math|checkpoints/math/qwen1p7b-30b-math-exopd-lambda1p25-4gpu-b255/global_step_60/actor/huggingface|1f8cf3726bbbbaade4e8ffcefa24b6d562ebebd6195998de15104f4991b802ce|EXOPD_MATH_DIR"
  "tip_mc|checkpoints/baselines/math_code/tip_topk32_rho50_8gpu_6s2t_b516/global_step_60|f3a0e14642ae31b0507c50365d4d0816d9bdef1b8477898571d29684fa80f10d|TIP_MC_DIR"
  "eopd_mc|checkpoints/baselines/math_code/qwen1p7b-30b-math-code-eopd-native-4gpu-colocate-b516-s42/global_step_60|1f7b3213e9944bfa4b0d308241fd16a4d0e344c2b64ceb2bfc68e74dbb6fd1da|EOPD_MC_DIR"
)
declare -A JOB_MODEL=([math4:exopd]=exopd_math [mc8:tip]=tip_mc [mc8:eopd]=eopd_mc)
declare -A DATASETS=(
  [math4]="aime24,aime25,hmmt25feb,hmmt25nov"
  [mc8]="aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus,lcb_v5,lcb_v6"
)
# file | git-tracked sha256 | sha256 of the CityU copy | row-content sha256 (the CityU copies of the EvalPlus
# parquets differ in bytes but not in rows)
DATA_FILES=(
  "data/eval_data/math/AIME24/test.parquet|627868d30c5892b3e601bae5acd3e1cf461df61f02b7f76756d2d9ea0916ea32|627868d30c5892b3e601bae5acd3e1cf461df61f02b7f76756d2d9ea0916ea32|f75335c6cf50d3e4f5ccfc50c8e329665f5b2180254d48648de8209aa3db7040"
  "data/eval_data/math/AIME25/test.parquet|447effc654af93583ea53f6696a3420b5729f8e4d066b18946367ac267daae66|447effc654af93583ea53f6696a3420b5729f8e4d066b18946367ac267daae66|cc19ddd31a012a6f54bf5b10153dd057107625e84c17bf3b217b826ffb3e8e05"
  "data/eval_data/math/HMMT25Feb/test.parquet|bf6ab1f7c767d0f8b13cfecb1fa0dd801949a430c8baed96e874c3f027a0a106|bf6ab1f7c767d0f8b13cfecb1fa0dd801949a430c8baed96e874c3f027a0a106|e318e309e3e4dc523f5f6191aa27a5989c8d0d4181d9c26936f688e15409ae50"
  "data/eval_data/math/HMMT25Nov/test.parquet|7ecb74118b1c22906640feaa628eac210c989e4add3880bcc778250e9112b2ef|7ecb74118b1c22906640feaa628eac210c989e4add3880bcc778250e9112b2ef|39358e1a1c2eacc9a82642fd72bb7b6fc94f35a25a9197b8988604155c7728fd"
  "data/eval_data/code/HumanEvalPlus/test.parquet|dddc22ee8d1d029c8c42262a73411be6522317d41e5e08c51edc2809f6aae2b6|9794903790fcb617e3f57d9b8c216ec4efd00c495ae8dec3ae2b77557fd4cb4f|68e8752bd0116e672e43990013443d95d272ab8745864061851400063c50e52f"
  "data/eval_data/code/MBPPPlus/test.parquet|eae8157015e48d59f2dc61b02ef54d4afeb60681683cc606caee0dd1f3a2d41e|0dccd39442731a5e81694bf4284d71dc31c1a35b5629722a4bca945128c78749|937d5441f99b4483f10afdb37de6f031076c36c81ba20b20ce1ea2d9696fb439"
)

JOBS="math4:exopd:2042,math4:exopd:1042,math4:exopd:42,mc8:eopd:2042,mc8:eopd:1042,mc8:tip:42"
MODES="cg,eager"
DRY_RUN=0; DOWNLOAD_ONLY=0; SETUP_EVALPLUS=0; COMPARE_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs) JOBS="${2:?--jobs requires a value}"; shift 2 ;;
    --modes) MODES="${2:?--modes requires a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --download-only) DOWNLOAD_ONLY=1; shift ;;
    --setup-evalplus) SETUP_EVALPLUS=1; shift ;;
    --compare-only) COMPARE_ONLY=1; shift ;;
    -h|--help) sed -n 2,37p "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[evmode %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "[evmode] ERROR: $*" >&2; exit 2; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
if [[ "${PYTHON}" != */* ]]; then PYTHON="$(command -v "${PYTHON}")" || die "python not found"; fi
cd "${CODE_DIR}"
[[ -f eval/parallel_eval.py && -f eval/parallel_worker.py ]] || die "CODE_DIR=${CODE_DIR} is not the OPD code checkout"
COMPARE_DIR="${OUTPUT_ROOT}/evmode_compare_${RUN_ID}"
compare() {
  "${PYTHON}" "${HERE}/eval_modes.py" compare --output-root "${OUTPUT_ROOT}" --run-id "${RUN_ID}" --jobs "${JOBS}" \
    --out-dir "${COMPARE_DIR}" > /dev/null && log "comparison: ${COMPARE_DIR}/compare.md"
}
if (( COMPARE_ONLY )); then compare; cat "${COMPARE_DIR}/compare.md"; exit 0; fi

# Jobs and modes.
IFS=',' read -r -a JOB_LIST <<<"${JOBS}"
IFS=',' read -r -a MODE_LIST <<<"${MODES}"
NEED_MC8=0
declare -A NEED_MODEL=()
for job in "${JOB_LIST[@]}"; do
  IFS=':' read -r setting name seed <<<"${job}"
  [[ -n "${JOB_MODEL[${setting}:${name}]:-}" && "${seed}" =~ ^[0-9]+$ ]] || die "unknown job ${job}"
  [[ "${setting}" == "mc8" ]] && NEED_MC8=1
  NEED_MODEL[${JOB_MODEL[${setting}:${name}]}]=1
done
for mode in "${MODE_LIST[@]}"; do [[ "${mode}" == cg || "${mode}" == eager ]] || die "--modes takes cg and/or eager"; done
if [[ -z "${GPU_IDS:-}" ]]; then GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"; fi
IFS=',' read -r -a GPUS <<<"${GPU_IDS}"
N_GPU="${#GPUS[@]}"
(( N_GPU > 0 )) || die "no GPU in GPU_IDS"

# 1. Eval data: prompt order fixes every shard's seed, so the parquet files must match the CityU ones.
for entry in "${DATA_FILES[@]}"; do
  IFS='|' read -r f sha_git sha_ref sha_rows <<<"${entry}"
  [[ "${f}" == *code* && "${NEED_MC8}" == 0 ]] && continue
  [[ -f "${f}" ]] || die "missing eval data: ${CODE_DIR}/${f}"
  got="$(sha256 "${f}")"
  if [[ "${got}" != "${sha_git}" && "${got}" != "${sha_ref}" ]]; then
    rows="$("${PYTHON}" -c 'import hashlib,json,sys,pyarrow.parquet as pq; print(hashlib.sha256(json.dumps(pq.read_table(sys.argv[1]).to_pylist(),sort_keys=True,default=str).encode()).hexdigest())' "${f}")"
    [[ "${rows}" == "${sha_rows}" ]] || die "eval data differs from the CityU reference suites: ${f}"
  fi
done
log "eval data OK"

# 2. Code scoring (mc8 only): G-OPD with the LiveCodeBench sources, pinned EvalPlus runtime, docker sandbox.
EVALPLUS_PY="${CODE_DIR}/.runtime/evalplus-gopd-${GOPD_COMMIT}/venv/bin/python"
SANDBOX_IMAGE_ID=disabled
if (( NEED_MC8 )); then
  [[ -n "${GOPD_DIR}" && -d "${GOPD_DIR}" ]] || die "mc8 jobs need GOPD_DIR: a G-OPD checkout at ${GOPD_COMMIT} (README.md)"
  GOPD_DIR="$(cd "${GOPD_DIR}" && pwd -P)"
  [[ "$(git -C "${GOPD_DIR}" rev-parse HEAD 2>/dev/null)" == "${GOPD_COMMIT}" ]] || die "G-OPD at ${GOPD_DIR} is not at ${GOPD_COMMIT}"
  LCB_DIR="${GOPD_DIR}/code_eval/coding/LiveCodeBench/code_generation_lite"
  for pair in "test5.jsonl|34dc80fac0fb8c3919835079dafa7831fc10056705d9b0d242003ad3ad1e0f5c" "test6.jsonl|bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"; do
    IFS='|' read -r f want <<<"${pair}"
    [[ -f "${LCB_DIR}/${f}" ]] || die "missing ${LCB_DIR}/${f} (README.md)"
    [[ "$(sha256 "${LCB_DIR}/${f}")" == "${want}" ]] || die "${LCB_DIR}/${f} differs from the pinned LiveCodeBench source"
  done
  export HUMANEVAL_SOURCE="${GOPD_DIR}/code_eval/data/HumanEvalPlus.jsonl"
  export MBPP_SOURCE="${GOPD_DIR}/code_eval/data/MbppPlus.jsonl"
  if (( SETUP_EVALPLUS )); then
    log "building the pinned EvalPlus runtime"
    GOPD_REPO="${GOPD_DIR}" GOPD_COMMIT="${GOPD_COMMIT}" PYTHON="${PYTHON}" bash scripts/prepare_evalplus_rescore_runtime.sh "${CODE_DIR}"
  fi
  [[ -x "${EVALPLUS_PY}" ]] || die "missing EvalPlus runtime ${EVALPLUS_PY}; rerun once with --setup-evalplus"
  command -v docker >/dev/null || die "docker is required for the in-loop HumanEval+/MBPP+ sandbox"
  SANDBOX_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${SANDBOX_IMAGE}" 2>/dev/null)" \
    || die "docker image ${SANDBOX_IMAGE} is missing (docker pull ${SANDBOX_IMAGE})"
  log "code scoring OK: G-OPD ${GOPD_COMMIT:0:8}, LCB v5/v6 sources, EvalPlus runtime, ${SANDBOX_IMAGE}"
fi

# 3. Environment versions of the CityU reference suites (vllm 0.23.0, torch 2.11.0+cu130, H200 NVL).
"${PYTHON}" - <<'PY' || true
import importlib, subprocess
for mod, ver in {"vllm": "0.23.0", "torch": "2.11.0+cu130", "transformers": "4.57.6"}.items():
    try:
        got = importlib.import_module(mod).__version__
    except Exception as exc:  # noqa: BLE001
        got = f"import failed: {exc}"
    print(f"[env] {mod:12s} {got:22s} CityU={ver}" + ("" if got == ver else "   <-- differs"))
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.split("\n")[0]
print(f"[env] GPU          {gpu:22s} CityU=NVIDIA H200 NVL")
PY

# 4. Checkpoints: local copy or Hub download, verified by SHA-256 against the weights of the CityU suites.
declare -A MODEL_DIR=()
for spec in "${MODELS[@]}"; do
  IFS='|' read -r key subdir want envvar <<<"${spec}"
  [[ -n "${NEED_MODEL[${key}]:-}" ]] || continue
  dir="${!envvar:-}"
  if [[ -z "${dir}" ]]; then
    [[ -n "${subdir}" ]] || die "${key} is not on the Hub: set ${envvar} to its HF-format directory (README.md)"
    dir="${MODEL_ROOT}/${subdir}"
    if [[ ! -f "${dir}/model.safetensors" ]]; then
      log "downloading ${HF_REPO}@${HF_REVISION:0:8} ${subdir}"
      "${PYTHON}" - "${HF_REPO}" "${HF_REVISION}" "${subdir}" "${MODEL_ROOT}" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, sub, root = sys.argv[1:]
snapshot_download(repo_id=repo, repo_type="model", revision=rev, allow_patterns=[f"{sub}/*"], local_dir=root)
PY
    fi
  fi
  [[ -f "${dir}/model.safetensors" ]] || die "${key}: no model.safetensors in ${dir}"
  got="$(sha256 "${dir}/model.safetensors")"
  [[ "${got}" == "${want}" ]] || die "${key}: ${dir}/model.safetensors sha256 ${got} != ${want}"
  MODEL_DIR[${key}]="$(cd "${dir}" && pwd -P)"
  log "checkpoint ${key} verified: ${MODEL_DIR[${key}]}"
done
(( DOWNLOAD_ONLY )) && { log "download only: done"; exit 0; }

export LANG=C LC_ALL=C
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS=0 TOKENIZERS_PARALLELISM=false
export MOPD_ALLOW_SIMPLE_SCORER_FALLBACK=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
unset ROCR_VISIBLE_DEVICES
if (( NEED_MC8 )); then export MOPD_CODE_SANDBOX=docker MOPD_CODE_SANDBOX_IMAGE="${SANDBOX_IMAGE}"; fi

# Plan one suite into $1 (suite root) and check it against the reference; $2 job, $3 mode, $4 resume flag.
plan_suite() {
  local suite=$1 job=$2 mode=$3 resume=$4 setting name seed key model_dir model_path args
  IFS=':' read -r setting name seed <<<"${job}"
  key="${JOB_MODEL[${setting}:${name}]}"
  model_dir="${MODEL_DIR[${key}]}"
  model_path="${model_dir%/actor/huggingface}"
  args=(
    "${PYTHON}" -m eval.parallel_eval plan
    --code-dir "${CODE_DIR}" --suite-root "${suite}" --run-tag "$(basename "${suite}")"
    --model-path "${model_path}" --eval-model-path "${model_dir}"
    --datasets "${DATASETS[${setting}]}" --shards-per-dataset 16 --worker-count "${N_GPU}" --min-rows-per-shard 1
    --math-samples 8 --code-samples 8 --science-samples 8
    --base-seed "${seed}" --seed-sequence-offset 0
    --max-new-tokens 16384 --temperature 1.0 --top-p 1.0
    --batch-size 24 --gpu-memory "${GPU_MEMORY}" --max-model-len 18432
    --max-num-batched-tokens 32768 --max-num-seqs "${MAX_NUM_SEQS}"
  )
  if [[ "${setting}" == "mc8" ]]; then
    args+=(--gopd-dir "${GOPD_DIR}" --code-sandbox-image "${SANDBOX_IMAGE}" --code-sandbox-image-id "${SANDBOX_IMAGE_ID}"
           --code-scoring-workers 2 --code-scoring-pending-shards 2)
  else
    args+=(--no-score-code --code-scoring-workers 0 --code-scoring-pending-shards 2)
  fi
  [[ "${mode}" == "cg" ]] && args+=(--cuda-graphs)
  (( resume )) && args+=(--resume)
  mkdir -p "$(dirname "${suite}")"
  local plan_log="$(dirname "${suite}")/$(basename "${suite}").plan.log"
  "${args[@]}" > "${plan_log}" 2>&1 || { cat "${plan_log}" >&2; return 1; }
  mkdir -p "${suite}/logs"
  mv "${plan_log}" "${suite}/logs/plan.log"
  "${PYTHON}" "${HERE}/eval_modes.py" check-plan --manifest "${suite}/suite_manifest.json" --job "${job}" --mode "${mode}"
}

# Merge (+ official LiveCodeBench) and, for mc8, official EvalPlus; then mark the suite complete.
finalize_suite() {
  local suite=$1 setting=$2
  if [[ "${setting}" == "mc8" ]]; then
    "${PYTHON}" -m eval.parallel_eval merge --manifest "${suite}/suite_manifest.json" --defer-completion > "${suite}/logs/merge.log" 2>&1
    PYTHON="${PYTHON}" bash "${CODE_DIR}/scripts/finalize_local_standard_eval.sh" "${suite}"
  else
    "${PYTHON}" -m eval.parallel_eval merge --manifest "${suite}/suite_manifest.json" > "${suite}/logs/merge.log" 2>&1
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "${suite}/COMPLETED_AT_UTC"
}

if (( DRY_RUN )); then
  tmp="$(mktemp -d)"
  for job in "${JOB_LIST[@]}"; do
    for mode in "${MODE_LIST[@]}"; do
      tag="$("${PYTHON}" -c 'import sys; sys.path.insert(0, sys.argv[1]); import eval_modes as m; print(m.fresh_tag(*sys.argv[2:]))' "${HERE}" "${job}" "${mode}" "${RUN_ID}")"
      state=todo; [[ -f "${OUTPUT_ROOT}/${tag}/COMPLETED_AT_UTC" ]] && state=complete
      [[ "${state}" == todo && -d "${OUTPUT_ROOT}/${tag}" ]] && state=resume
      log "plan ${job} ${mode}: ${tag} (${state})"
      plan_suite "${tmp}/${tag}" "${job}" "${mode}" 0 || { rm -rf "${tmp}"; die "plan check failed for ${job} ${mode}"; }
    done
  done
  rm -rf "${tmp}"
  log "dry run OK: ${#JOB_LIST[@]} jobs x ${#MODE_LIST[@]} modes on GPUs ${GPU_IDS}; no GPU work started"
  exit 0
fi

# 5. Suites in order: GPU phase in the foreground, merge + scoring in the background.
mkdir -p "${OUTPUT_ROOT}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/[gpu] /' || true
bg_pids=(); bg_tags=(); failed=()
for job in "${JOB_LIST[@]}"; do
  IFS=':' read -r setting name seed <<<"${job}"
  key="${JOB_MODEL[${setting}:${name}]}"
  for mode in "${MODE_LIST[@]}"; do
    tag="$("${PYTHON}" -c 'import sys; sys.path.insert(0, sys.argv[1]); import eval_modes as m; print(m.fresh_tag(*sys.argv[2:]))' "${HERE}" "${job}" "${mode}" "${RUN_ID}")"
    suite="${OUTPUT_ROOT}/${tag}"
    if [[ -f "${suite}/COMPLETED_AT_UTC" ]]; then log "skip ${tag}: complete"; continue; fi
    resume=0; [[ -e "${suite}" ]] && resume=1
    log "start ${tag} (${job}, ${mode}, resume=${resume}, ${N_GPU} GPUs)"
    if ! plan_suite "${suite}" "${job}" "${mode}" "${resume}"; then failed+=("${tag}"); break; fi
    started=$(date +%s)
    worker_extra=(); (( resume )) && worker_extra=(--resume)
    pids=()
    for i in "${!GPUS[@]}"; do
      CUDA_VISIBLE_DEVICES="${GPUS[$i]}" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
        "${PYTHON}" -m eval.parallel_worker --manifest "${suite}/suite_manifest.json" \
          --eval-model-path "${MODEL_DIR[${key}]}" --worker-id "${i}" ${worker_extra[@]+"${worker_extra[@]}"} \
          > "${suite}/logs/gpu_worker_${i}.log" 2>&1 &
      pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
    if (( status != 0 )); then
      log "a worker of ${tag} failed (see ${suite}/logs/gpu_worker_*.log); stopping. Rerun the same command to resume."
      failed+=("${tag}"); break 2
    fi
    log "GPU phase of ${tag} done in $(( ($(date +%s) - started) / 60 )) min; merging and scoring in the background"
    finalize_suite "${suite}" "${setting}" > "${suite}/logs/finalize_wrapper.log" 2>&1 &
    bg_pids+=("$!"); bg_tags+=("${tag}")
  done
done

for i in "${!bg_pids[@]}"; do
  if wait "${bg_pids[$i]}"; then log "complete: ${bg_tags[$i]}"
  else log "scoring failed: ${bg_tags[$i]} (see its logs/finalize_wrapper.log; rerun the same command)"; failed+=("${bg_tags[$i]}"); fi
done
compare || true
if (( ${#failed[@]} )); then log "finished with failures: ${failed[*]}"; exit 1; fi
log "all suites complete"
