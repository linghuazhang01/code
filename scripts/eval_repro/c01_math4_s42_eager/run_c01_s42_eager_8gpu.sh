#!/usr/bin/env bash
# Reproduce the archived sapd_c01 Math4 score 26.04 (job 196, 2026-09-02): seed 42, eager vLLM,
# K=8, the same 64-shard / per-shard seed layout, now on 8 GPUs (DP=8, one TP=1 vLLM worker per GPU).
# Worker count does not change any shard's prompts or seed; only GPU model, vLLM/torch versions and
# kernels can make the text differ. compare_with_reference.py reports how close the run gets.
#
# Usage (from anywhere, on a machine with this repo checked out):
#   bash run_c01_s42_eager_8gpu.sh                 # download (if needed) + evaluate + compare
#   bash run_c01_s42_eager_8gpu.sh --dry-run       # checks, download, plan only; no GPU work
#   bash run_c01_s42_eager_8gpu.sh --download-only
#   bash run_c01_s42_eager_8gpu.sh --resume RUN_TAG
# Environment overrides: CODE_DIR, PYTHON, GPU_IDS (default 0..7), GPU_MEMORY (default 0.85, as job 196),
#   MODEL_ROOT, OUTPUT_ROOT, RUN_TAG, HF_ENDPOINT / HF_TOKEN (the repo is public; no token needed).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="${CODE_DIR:-$(cd "${HERE}/../../.." && pwd -P)}"
PYTHON="${PYTHON:-python3}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
GPU_MEMORY="${GPU_MEMORY:-0.85}"
SEED=42

HF_REPO="icemoon28/opd-checkpoints"
HF_REVISION="1fa94a3be5bc05028fa7e7b33a26a90e354405eb"   # 2026-09-02 upload by training job 193 at step 60
HF_SUBDIR="checkpoints/math/q1p7b-math-top32kl-ns-taxonomy-topp0p05-i1w1-5gpu-b256/global_step_60"
MODEL_SHA256="744f2bf39b534054d9379f16dd2d2eef30190e6538590a5944faa1ad6dfebe43"
MODEL_ROOT="${MODEL_ROOT:-${CODE_DIR}/checkpoints/hf_eval_c01_repro}"
MODEL_DIR="${MODEL_ROOT}/${HF_SUBDIR}"

DATASETS="aime24,aime25,hmmt25feb,hmmt25nov"
declare -A DATA_SHA256=(
  [data/eval_data/math/AIME24/test.parquet]=627868d30c5892b3e601bae5acd3e1cf461df61f02b7f76756d2d9ea0916ea32
  [data/eval_data/math/AIME25/test.parquet]=447effc654af93583ea53f6696a3420b5729f8e4d066b18946367ac267daae66
  [data/eval_data/math/HMMT25Feb/test.parquet]=bf6ab1f7c767d0f8b13cfecb1fa0dd801949a430c8baed96e874c3f027a0a106
  [data/eval_data/math/HMMT25Nov/test.parquet]=7ecb74118b1c22906640feaa628eac210c989e4add3880bcc778250e9112b2ef
)
OUTPUT_ROOT="${OUTPUT_ROOT:-${CODE_DIR}/experiments_records/eval}"
RUN_TAG="${RUN_TAG:-math4_k8_s60_sapd_c01_eager_s42_8gpu_$(date +%Y%m%d_%H%M%S)}"

DRY_RUN=0; DOWNLOAD_ONLY=0; RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --download-only) DOWNLOAD_ONLY=1; shift ;;
    --resume) RESUME=1; RUN_TAG="${2:?--resume requires the RUN_TAG}"; shift 2 ;;
    -h|--help) sed -n 2,16p "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[c01-repro %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
if [[ "${PYTHON}" != */* ]]; then PYTHON="$(command -v "${PYTHON}")"; fi

IFS=',' read -r -a GPUS <<<"${GPU_IDS}"
N_GPU="${#GPUS[@]}"
SUITE_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
cd "${CODE_DIR}"
[[ -f eval/parallel_eval.py && -f eval/parallel_worker.py ]] || { echo "CODE_DIR=${CODE_DIR} is not the OPD code checkout" >&2; exit 2; }

# 1. Eval data must be byte-identical to job 196 (prompt order decides each shard's seed).
for f in "${!DATA_SHA256[@]}"; do
  [[ -f "${f}" ]] || { echo "missing eval data: ${CODE_DIR}/${f}" >&2; exit 2; }
  [[ "$(sha256 "${f}")" == "${DATA_SHA256[${f}]}" ]] || { echo "eval data differs from job 196: ${f}" >&2; exit 2; }
done
log "eval data OK (4 parquet files match job 196)"

# 2. Environment: job 196 ran vllm 0.23.0, torch 2.11.0+cu130, transformers 4.57.6 on NVIDIA H200 NVL.
"${PYTHON}" - <<'PY' || true
import importlib, subprocess
want = {"vllm": "0.23.0", "torch": "2.11.0+cu130", "transformers": "4.57.6"}
for mod, ver in want.items():
    try:
        got = importlib.import_module(mod).__version__
    except Exception as exc:  # noqa: BLE001
        got = f"import failed: {exc}"
    print(f"[env] {mod:12s} {got:22s} job196={ver}" + ("" if got == ver else "   <-- differs: bitwise reproduction unlikely"))
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.split("\n")[0]
except FileNotFoundError:
    gpu = "nvidia-smi not found"
print(f"[env] GPU          {gpu:22s} job196=NVIDIA H200 NVL" + ("" if gpu.strip() == "NVIDIA H200 NVL" else "   <-- differs: bitwise reproduction unlikely"))
PY

# 3. Checkpoint: pinned HF revision, verified by SHA-256.
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

# 4. Plan: same settings as job 196 except worker_count (8) and code scoring (Math-only).
export LANG=C LC_ALL=C
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS=0 TOKENIZERS_PARALLELISM=false
export MOPD_ALLOW_SIMPLE_SCORER_FALLBACK=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
unset ROCR_VISIBLE_DEVICES

if (( ! RESUME )) && [[ -e "${SUITE_ROOT}" ]]; then echo "suite exists: ${SUITE_ROOT} (use --resume ${RUN_TAG})" >&2; exit 2; fi
if (( DRY_RUN )); then SUITE_ROOT="$(mktemp -d)/${RUN_TAG}"; fi
mkdir -p "$(dirname "${SUITE_ROOT}")"
PLAN=(
  "${PYTHON}" -m eval.parallel_eval plan
  --code-dir "${CODE_DIR}" --suite-root "${SUITE_ROOT}" --run-tag "${RUN_TAG}"
  --model-path "${MODEL_DIR}" --eval-model-path "${MODEL_DIR}"
  --datasets "${DATASETS}" --shards-per-dataset 16 --worker-count "${N_GPU}" --min-rows-per-shard 1
  --math-samples 8 --code-samples 8 --science-samples 8
  --base-seed "${SEED}" --seed-sequence-offset 0
  --max-new-tokens 16384 --temperature 1.0 --top-p 1.0
  --batch-size 24 --gpu-memory "${GPU_MEMORY}" --max-model-len 18432
  --max-num-batched-tokens 32768 --max-num-seqs 24
  --no-score-code
)   # no --cuda-graphs: eager, as job 196
(( RESUME )) && PLAN+=(--resume)
"${PLAN[@]}" > "$(dirname "${SUITE_ROOT}")/${RUN_TAG}.plan.log" 2>&1 || { cat "$(dirname "${SUITE_ROOT}")/${RUN_TAG}.plan.log" >&2; exit 1; }
mkdir -p "${SUITE_ROOT}/logs"
mv "$(dirname "${SUITE_ROOT}")/${RUN_TAG}.plan.log" "${SUITE_ROOT}/logs/plan.log"
"${PYTHON}" "${HERE}/compare_with_reference.py" --check-manifest "${SUITE_ROOT}/suite_manifest.json"
"${PYTHON}" -c "import json,sys; e=json.load(open(sys.argv[1]))['execution']; assert e['enforce_eager'] is True, e; print('[plan] enforce_eager=True worker_count=%d gpu_memory=%s' % (e['worker_count'], e['gpu_memory']))" "${SUITE_ROOT}/suite_manifest.json"
if (( DRY_RUN )); then log "dry run OK; plan written to ${SUITE_ROOT} (temp) and removed"; rm -rf "$(dirname "${SUITE_ROOT}")"; exit 0; fi

# 5. Workers: one eager vLLM engine per GPU, dynamic shard claiming in strict dataset waves.
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/[gpu] /' || true
log "launching ${N_GPU} workers on GPUs ${GPU_IDS}; suite ${SUITE_ROOT}"
WORKER_EXTRA=()
(( RESUME )) && WORKER_EXTRA+=(--resume)
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

"${PYTHON}" -m eval.parallel_eval merge --manifest "${SUITE_ROOT}/suite_manifest.json" > "${SUITE_ROOT}/logs/merge.log" 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ > "${SUITE_ROOT}/COMPLETED_AT_UTC"
log "evaluation complete: ${SUITE_ROOT}"

# 6. Score and compare with the 26.04 (eager) and 24.06 (CUDA graph) references.
"${PYTHON}" "${HERE}/compare_with_reference.py" --suite "${SUITE_ROOT}" | tee "${SUITE_ROOT}/repro_report.txt"
