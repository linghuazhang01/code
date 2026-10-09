#!/usr/bin/env bash
# mc8 (Math4 + Code4) evaluation of the current best Math+Code checkpoint, MOPD Structure-only C01
# (q1p7b4g-mc-tl-m05c01-cstruct-8g6s2t-b528-s60, step 60): seed 42, eager vLLM, K=8, on 8 GPUs
# (DP=8, one TP=1 vLLM worker per GPU). The shard/seed layout and sampling settings are those of the
# archived eager suite (Macro8 31.66, 2026-09-26, DP2, gpu_memory 0.85); the CUDA-graph re-eval of the
# same checkpoint gave 31.45. mc8_tools.py scores the run and compares it rollout by rollout with both.
#
# Usage (on a machine with this repo checked out):
#   GOPD_DIR=/path/to/G-OPD bash run_best_mc8_eager_8gpu.sh --dry-run   # checks + download + plan, no GPU work
#   GOPD_DIR=/path/to/G-OPD bash run_best_mc8_eager_8gpu.sh             # full run, then score + compare
#   ... --download-only | --setup-evalplus | --resume RUN_TAG
# Prerequisites (checked before anything runs; see README.md):
#   GOPD_DIR   G-OPD checkout at 37371a4c (https://github.com/RUCBM/G-OPD) with LiveCodeBench sources
#              code_eval/coding/LiveCodeBench/code_generation_lite/test5.jsonl and test6.jsonl
#   docker     image verlai/verl:vllm023.dev1 (in-loop HumanEval+/MBPP+ sandbox)
#   EvalPlus   ${CODE_DIR}/.runtime/evalplus-gopd-37371a4c…/venv (built by --setup-evalplus)
# Environment overrides: CODE_DIR, PYTHON, GPU_IDS (0..7), GPU_MEMORY (0.85), MODEL_ROOT, OUTPUT_ROOT, RUN_TAG.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="${CODE_DIR:-$(cd "${HERE}/../../.." && pwd -P)}"
PYTHON="${PYTHON:-python3}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
GPU_MEMORY="${GPU_MEMORY:-0.85}"
SEED=42
GOPD_DIR="${GOPD_DIR:-}"
GOPD_COMMIT="37371a4c31ad7947746200d234161769191f4748"
SANDBOX_IMAGE="${MOPD_CODE_SANDBOX_IMAGE:-verlai/verl:vllm023.dev1}"

HF_REPO="icemoon28/opd-checkpoints"
HF_REVISION="19d7f7ff9239ffcb783e21f6fca347234f65ea1b"   # 2026-09-25 step-60 upload
HF_SUBDIR="checkpoints/mopd/q1p7b4g-mc-tl-m05c01-cstruct-8g6s2t-b528-s60/global_step_60"
MODEL_SHA256="9b19d2346fe240375b3b3518b17b9dba8c638ea8022a4c1356d9d528cd78fa7a"
MODEL_ROOT="${MODEL_ROOT:-${CODE_DIR}/checkpoints/hf_eval_best_mc8}"
MODEL_DIR="${MODEL_ROOT}/${HF_SUBDIR}"

DATASETS="aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus,lcb_v5,lcb_v6"
# file | git-tracked sha256 | sha256 of the copy used by the reference suites | row-content sha256
DATA_FILES=(
  "data/eval_data/math/AIME24/test.parquet|627868d30c5892b3e601bae5acd3e1cf461df61f02b7f76756d2d9ea0916ea32|627868d30c5892b3e601bae5acd3e1cf461df61f02b7f76756d2d9ea0916ea32|f75335c6cf50d3e4f5ccfc50c8e329665f5b2180254d48648de8209aa3db7040"
  "data/eval_data/math/AIME25/test.parquet|447effc654af93583ea53f6696a3420b5729f8e4d066b18946367ac267daae66|447effc654af93583ea53f6696a3420b5729f8e4d066b18946367ac267daae66|cc19ddd31a012a6f54bf5b10153dd057107625e84c17bf3b217b826ffb3e8e05"
  "data/eval_data/math/HMMT25Feb/test.parquet|bf6ab1f7c767d0f8b13cfecb1fa0dd801949a430c8baed96e874c3f027a0a106|bf6ab1f7c767d0f8b13cfecb1fa0dd801949a430c8baed96e874c3f027a0a106|e318e309e3e4dc523f5f6191aa27a5989c8d0d4181d9c26936f688e15409ae50"
  "data/eval_data/math/HMMT25Nov/test.parquet|7ecb74118b1c22906640feaa628eac210c989e4add3880bcc778250e9112b2ef|7ecb74118b1c22906640feaa628eac210c989e4add3880bcc778250e9112b2ef|39358e1a1c2eacc9a82642fd72bb7b6fc94f35a25a9197b8988604155c7728fd"
  "data/eval_data/code/HumanEvalPlus/test.parquet|dddc22ee8d1d029c8c42262a73411be6522317d41e5e08c51edc2809f6aae2b6|9794903790fcb617e3f57d9b8c216ec4efd00c495ae8dec3ae2b77557fd4cb4f|68e8752bd0116e672e43990013443d95d272ab8745864061851400063c50e52f"
  "data/eval_data/code/MBPPPlus/test.parquet|eae8157015e48d59f2dc61b02ef54d4afeb60681683cc606caee0dd1f3a2d41e|0dccd39442731a5e81694bf4284d71dc31c1a35b5629722a4bca945128c78749|937d5441f99b4483f10afdb37de6f031076c36c81ba20b20ce1ea2d9696fb439"
)
OUTPUT_ROOT="${OUTPUT_ROOT:-${CODE_DIR}/experiments_records/eval}"
RUN_TAG="${RUN_TAG:-mc8_k8_s60_best_c01_cstruct_eager_s42_8gpu_$(date +%Y%m%d_%H%M%S)}"

DRY_RUN=0; DOWNLOAD_ONLY=0; RESUME=0; SETUP_EVALPLUS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --download-only) DOWNLOAD_ONLY=1; shift ;;
    --setup-evalplus) SETUP_EVALPLUS=1; shift ;;
    --resume) RESUME=1; RUN_TAG="${2:?--resume requires the RUN_TAG}"; shift 2 ;;
    -h|--help) sed -n 2,22p "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[best-mc8 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "[best-mc8] ERROR: $*" >&2; exit 2; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
if [[ "${PYTHON}" != */* ]]; then PYTHON="$(command -v "${PYTHON}")"; fi

IFS=',' read -r -a GPUS <<<"${GPU_IDS}"
N_GPU="${#GPUS[@]}"
SUITE_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
EVALPLUS_PY="${CODE_DIR}/.runtime/evalplus-gopd-${GOPD_COMMIT}/venv/bin/python"
cd "${CODE_DIR}"
[[ -f eval/parallel_eval.py && -f eval/parallel_worker.py ]] || die "CODE_DIR=${CODE_DIR} is not the OPD code checkout"

# 1. Eval data: the prompts must match the reference suites (prompt order fixes every shard's seed).
for entry in "${DATA_FILES[@]}"; do
  IFS='|' read -r f sha_git sha_ref sha_rows <<<"${entry}"
  [[ -f "${f}" ]] || die "missing eval data: ${CODE_DIR}/${f}"
  got="$(sha256 "${f}")"
  if [[ "${got}" != "${sha_git}" && "${got}" != "${sha_ref}" ]]; then
    rows="$("${PYTHON}" -c 'import hashlib,json,sys,pyarrow.parquet as pq; print(hashlib.sha256(json.dumps(pq.read_table(sys.argv[1]).to_pylist(),sort_keys=True,default=str).encode()).hexdigest())' "${f}")"
    [[ "${rows}" == "${sha_rows}" ]] || die "eval data content differs from the reference suites: ${f}"
  fi
done
log "eval data OK (6 parquet files match the reference suites)"

# 2. G-OPD checkout (LiveCodeBench data + scorer, EvalPlus sources), docker sandbox image.
[[ -n "${GOPD_DIR}" && -d "${GOPD_DIR}" ]] || die "set GOPD_DIR to a G-OPD checkout at ${GOPD_COMMIT} (git clone https://github.com/RUCBM/G-OPD && git checkout ${GOPD_COMMIT})"
GOPD_DIR="$(cd "${GOPD_DIR}" && pwd -P)"
[[ "$(git -C "${GOPD_DIR}" rev-parse HEAD 2>/dev/null)" == "${GOPD_COMMIT}" ]] || die "G-OPD at ${GOPD_DIR} is not at commit ${GOPD_COMMIT}"
LCB_DIR="${GOPD_DIR}/code_eval/coding/LiveCodeBench/code_generation_lite"
for pair in "test5.jsonl|34dc80fac0fb8c3919835079dafa7831fc10056705d9b0d242003ad3ad1e0f5c" "test6.jsonl|bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"; do
  IFS='|' read -r f want <<<"${pair}"
  [[ -f "${LCB_DIR}/${f}" ]] || die "missing ${LCB_DIR}/${f} (see README.md: LiveCodeBench sources)"
  [[ "$(sha256 "${LCB_DIR}/${f}")" == "${want}" ]] || die "${LCB_DIR}/${f} sha256 differs from the pinned source"
done
export HUMANEVAL_SOURCE="${GOPD_DIR}/code_eval/data/HumanEvalPlus.jsonl"
export MBPP_SOURCE="${GOPD_DIR}/code_eval/data/MbppPlus.jsonl"
log "G-OPD OK: ${GOPD_DIR} @ ${GOPD_COMMIT:0:8}, LiveCodeBench v5/v6 sources verified"

if (( SETUP_EVALPLUS )) || [[ ! -x "${EVALPLUS_PY}" ]]; then
  if (( SETUP_EVALPLUS )); then
    log "building the pinned EvalPlus runtime"
    GOPD_REPO="${GOPD_DIR}" GOPD_COMMIT="${GOPD_COMMIT}" PYTHON="${PYTHON}" bash scripts/prepare_evalplus_rescore_runtime.sh "${CODE_DIR}"
  else
    die "missing EvalPlus runtime ${EVALPLUS_PY}; rerun with --setup-evalplus once"
  fi
fi
command -v docker >/dev/null || die "docker is required for the in-loop HumanEval+/MBPP+ sandbox"
SANDBOX_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${SANDBOX_IMAGE}" 2>/dev/null)" \
  || die "docker image ${SANDBOX_IMAGE} is missing (docker pull ${SANDBOX_IMAGE})"
log "EvalPlus runtime and docker image ${SANDBOX_IMAGE} OK"

# 3. Environment: the reference eager suite ran vllm 0.23.0, torch 2.11.0+cu130, transformers 4.57.6 on H200 NVL.
"${PYTHON}" - <<'PY' || true
import importlib, subprocess
want = {"vllm": "0.23.0", "torch": "2.11.0+cu130", "transformers": "4.57.6"}
for mod, ver in want.items():
    try:
        got = importlib.import_module(mod).__version__
    except Exception as exc:  # noqa: BLE001
        got = f"import failed: {exc}"
    print(f"[env] {mod:12s} {got:22s} reference={ver}" + ("" if got == ver else "   <-- differs: bitwise reproduction unlikely"))
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.split("\n")[0]
except FileNotFoundError:
    gpu = "nvidia-smi not found"
print(f"[env] GPU          {gpu:22s} reference=NVIDIA H200 NVL" + ("" if gpu.strip() == "NVIDIA H200 NVL" else "   <-- differs: bitwise reproduction unlikely"))
PY

# 4. Checkpoint: pinned HF revision, verified by SHA-256.
if [[ -f "${MODEL_DIR}/model.safetensors" && "$(sha256 "${MODEL_DIR}/model.safetensors")" == "${MODEL_SHA256}" ]]; then
  log "checkpoint present and verified: ${MODEL_DIR}"
else
  log "downloading ${HF_REPO}@${HF_REVISION:0:8} ${HF_SUBDIR} -> ${MODEL_ROOT}"
  "${PYTHON}" - "${HF_REPO}" "${HF_REVISION}" "${HF_SUBDIR}" "${MODEL_ROOT}" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, sub, root = sys.argv[1:]
snapshot_download(repo_id=repo, repo_type="model", revision=rev, allow_patterns=[f"{sub}/*"], local_dir=root)
PY
  got="$(sha256 "${MODEL_DIR}/model.safetensors")"
  [[ "${got}" == "${MODEL_SHA256}" ]] || { echo "checkpoint SHA mismatch: ${got} != ${MODEL_SHA256}" >&2; exit 1; }
  log "checkpoint verified: sha256 ${MODEL_SHA256:0:12}…"
fi
(( DOWNLOAD_ONLY )) && { log "download only: done"; exit 0; }

# 5. Plan: reference settings except worker_count (8); eager (no --cuda-graphs).
export LANG=C LC_ALL=C
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS=0 TOKENIZERS_PARALLELISM=false
export MOPD_ALLOW_SIMPLE_SCORER_FALLBACK=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export MOPD_CODE_SANDBOX=docker MOPD_CODE_SANDBOX_IMAGE="${SANDBOX_IMAGE}"
unset ROCR_VISIBLE_DEVICES

if (( ! RESUME )) && [[ -e "${SUITE_ROOT}" ]]; then die "suite exists: ${SUITE_ROOT} (use --resume ${RUN_TAG})"; fi
if (( DRY_RUN )); then SUITE_ROOT="$(mktemp -d)/${RUN_TAG}"; fi
mkdir -p "$(dirname "${SUITE_ROOT}")"
PLAN=(
  "${PYTHON}" -m eval.parallel_eval plan
  --code-dir "${CODE_DIR}" --suite-root "${SUITE_ROOT}" --run-tag "${RUN_TAG}"
  --model-path "${MODEL_DIR}" --eval-model-path "${MODEL_DIR}" --gopd-dir "${GOPD_DIR}"
  --datasets "${DATASETS}" --shards-per-dataset 16 --worker-count "${N_GPU}" --min-rows-per-shard 1
  --math-samples 8 --code-samples 8 --science-samples 8
  --base-seed "${SEED}" --seed-sequence-offset 0
  --max-new-tokens 16384 --temperature 1.0 --top-p 1.0
  --batch-size 24 --gpu-memory "${GPU_MEMORY}" --max-model-len 18432
  --max-num-batched-tokens 32768 --max-num-seqs 24
  --code-sandbox-image "${SANDBOX_IMAGE}" --code-sandbox-image-id "${SANDBOX_IMAGE_ID}"
  --code-scoring-workers 2 --code-scoring-pending-shards 2
)
WORKER_EXTRA=()
if (( RESUME )); then PLAN+=(--resume); WORKER_EXTRA+=(--resume); fi
PLAN_LOG="$(dirname "${SUITE_ROOT}")/${RUN_TAG}.plan.log"
"${PLAN[@]}" > "${PLAN_LOG}" 2>&1 || { cat "${PLAN_LOG}" >&2; exit 1; }
mkdir -p "${SUITE_ROOT}/logs"
mv "${PLAN_LOG}" "${SUITE_ROOT}/logs/plan.log"
"${PYTHON}" "${HERE}/mc8_tools.py" check-manifest --manifest "${SUITE_ROOT}/suite_manifest.json"
"${PYTHON}" -c "import json,sys; e=json.load(open(sys.argv[1]))['execution']; assert e['enforce_eager'] is True, e; print('[plan] enforce_eager=True worker_count=%d gpu_memory=%s' % (e['worker_count'], e['gpu_memory']))" "${SUITE_ROOT}/suite_manifest.json"
if (( DRY_RUN )); then rm -rf "$(dirname "${SUITE_ROOT}")"; log "dry run OK (temporary plan removed); no GPU work started"; exit 0; fi

# 6. Workers: one eager vLLM engine per GPU, dynamic shard claiming in strict dataset waves.
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/[gpu] /' || true
log "launching ${N_GPU} workers on GPUs ${GPU_IDS}; suite ${SUITE_ROOT}"
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
    "${PYTHON}" -m eval.parallel_worker --manifest "${SUITE_ROOT}/suite_manifest.json" \
      --eval-model-path "${MODEL_DIR}" --worker-id "${i}" "${WORKER_EXTRA[@]}" \
      > "${SUITE_ROOT}/logs/gpu_worker_${i}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "a worker failed; see ${SUITE_ROOT}/logs/gpu_worker_*.log, then rerun with --resume ${RUN_TAG}" >&2; exit 1; }
log "generation done; merging (includes official LiveCodeBench scoring)"

# 7. Merge + LCB scoring, then official EvalPlus (CPU), as scripts/run_local_math_parallel_eval.sh does.
"${PYTHON}" -m eval.parallel_eval merge --manifest "${SUITE_ROOT}/suite_manifest.json" --defer-completion \
  > "${SUITE_ROOT}/logs/merge.log" 2>&1
PYTHON="${PYTHON}" bash "${CODE_DIR}/scripts/finalize_local_standard_eval.sh" "${SUITE_ROOT}"
date -u +%Y-%m-%dT%H:%M:%SZ > "${SUITE_ROOT}/COMPLETED_AT_UTC"
log "evaluation complete: ${SUITE_ROOT}"

# 8. Score and compare with the 31.66 (eager) and 31.45 (CUDA graph) references.
"${PYTHON}" "${HERE}/mc8_tools.py" compare --suite "${SUITE_ROOT}" | tee "${SUITE_ROOT}/repro_report.txt"
