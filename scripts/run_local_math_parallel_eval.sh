#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${PYTHON:-python3}"
MODEL_PATH=""
EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-}"
DATASETS="${EVAL_DATASETS:-aime24,aime25,hmmt25feb,hmmt25nov}"
G_OPD_DIR="${GOPD_DIR:-}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${CODE_DIR}/experiments_records/eval"
GPU_ID_LIST="${GPU_IDS:-}"
GPU_COUNT=4
SHARDS_PER_DATASET=16
MIN_ROWS_PER_SHARD=1
SCHEDULING="${MOPD_EVAL_SCHEDULING:-overlap_dataset_waves_dynamic_microshards}"
MAX_NUM_SEQS="${MOPD_EVAL_MAX_NUM_SEQS:-28}"
ENFORCE_EAGER="${MOPD_EVAL_ENFORCE_EAGER:-0}"
ENABLE_CHUNKED_PREFILL="${MOPD_EVAL_ENABLE_CHUNKED_PREFILL:-1}"
MAX_SAMPLES=""
SCORE_CODE=0
STANDARD_PROTOCOL=0
DEFER_OFFICIAL_CODE_SCORING=0
RESUME=0
DRY_RUN=0
POSITIONAL_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_local_math_parallel_eval.sh --model_path PATH [options]

Options:
  --model_path PATH          Step-60 model/checkpoint path (required).
  --eval_model_path PATH     Prepared evaluation model path (default: model path).
  --datasets NAMES            Comma-separated datasets. Default is Math-only;
                              use the canonical ten-dataset list for 3 domains.
  --gopd_dir PATH             G-OPD checkout, required for lcb_v5/lcb_v6.
  --run_tag TAG              Output suite identifier (default: timestamp).
  --output_root PATH         Parent output directory.
  --gpus N                   Number of local GPUs (default: 4; supports 8).
  --gpu_ids LIST             Physical GPU IDs (default: $GPU_IDS or 0..N-1).
  --shards_per_dataset N     Dynamic micro-shards per dataset (default: 16).
  --strict_waves             Wait for every task in a dataset before the next.
  --overlap_waves            Let idle workers start the next dataset (default).
  --max_num_seqs N           vLLM concurrent sequence cap (default: 28).
  --enforce_eager            Disable CUDA Graph capture.
  --cuda_graphs              Enable CUDA Graph capture (default).
  --enable_chunked_prefill   Enable vLLM chunked prefill (default).
  --disable_chunked_prefill  Disable vLLM chunked prefill.
  --max_samples N             Optional prompt cap per dataset for smoke tests.
  --score_code                Enable isolated Code scoring through Docker.
  --standard_protocol         Enforce canonical 10-dataset, K=8, seed-42 protocol.
  --defer_official_code_scoring
                              Generate canonical Code rollouts without Docker;
                              pinned official EvalPlus post-scoring is required.
  --resume                    Resume a compatible existing suite.
  --dry_run                  Print the direct-local plan without launching workers.
  -h, --help                 Show this help.

By default this launcher evaluates Math-only AIME24, AIME25, HMMT25Feb, and
HMMT25Nov. With --standard_protocol it evaluates the canonical 10-dataset
Math/Code/Science suite using DP=4, four persistent TP=1 vLLM workers, and K=8.
Pass --gpus 8 to use eight persistent TP=1 workers without changing the protocol.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path|--model-path) MODEL_PATH="${2:?$1 requires a value}"; shift 2 ;;
    --eval_model_path|--eval-model-path) EVAL_MODEL_PATH="${2:?$1 requires a value}"; shift 2 ;;
    --datasets) DATASETS="${2:?--datasets requires a value}"; shift 2 ;;
    --gopd_dir|--gopd-dir) G_OPD_DIR="${2:?$1 requires a value}"; shift 2 ;;
    --run_tag|--run-tag) RUN_TAG="${2:?$1 requires a value}"; shift 2 ;;
    --output_root|--output-root) OUTPUT_ROOT="${2:?$1 requires a value}"; shift 2 ;;
    --gpus) GPU_COUNT="${2:?--gpus requires a value}"; shift 2 ;;
    --gpu_ids|--gpu-ids) GPU_ID_LIST="${2:?$1 requires a value}"; shift 2 ;;
    --shards_per_dataset|--shards-per-dataset)
      SHARDS_PER_DATASET="${2:?$1 requires a value}"
      shift 2
      ;;
    --strict_waves|--strict-waves)
      SCHEDULING="strict_dataset_wave_dynamic_microshards"
      shift
      ;;
    --overlap_waves|--overlap-waves)
      SCHEDULING="overlap_dataset_waves_dynamic_microshards"
      shift
      ;;
    --max_num_seqs|--max-num-seqs)
      MAX_NUM_SEQS="${2:?$1 requires a value}"
      shift 2
      ;;
    --enforce_eager|--enforce-eager) ENFORCE_EAGER=1; shift ;;
    --cuda_graphs|--cuda-graphs) ENFORCE_EAGER=0; shift ;;
    --enable_chunked_prefill|--enable-chunked-prefill) ENABLE_CHUNKED_PREFILL=1; shift ;;
    --disable_chunked_prefill|--disable-chunked-prefill) ENABLE_CHUNKED_PREFILL=0; shift ;;
    --max_samples|--max-samples) MAX_SAMPLES="${2:?$1 requires a value}"; shift 2 ;;
    --score_code|--score-code) SCORE_CODE=1; shift ;;
    --standard_protocol) STANDARD_PROTOCOL=1; shift ;;
    --defer_official_code_scoring|--defer-official-code-scoring)
      DEFER_OFFICIAL_CODE_SCORING=1
      shift
      ;;
    --resume) RESUME=1; shift ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) POSITIONAL_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${MODEL_PATH}" && "${#POSITIONAL_ARGS[@]}" -ge 1 ]]; then
  MODEL_PATH="${POSITIONAL_ARGS[0]}"
fi
if [[ "${#POSITIONAL_ARGS[@]}" -ge 2 ]]; then
  RUN_TAG="${POSITIONAL_ARGS[1]}"
fi
if [[ "${#POSITIONAL_ARGS[@]}" -ge 3 ]]; then
  OUTPUT_ROOT="${POSITIONAL_ARGS[2]}"
fi
[[ "${#POSITIONAL_ARGS[@]}" -le 3 ]] || {
  echo "Too many positional arguments; use --help for usage." >&2
  exit 2
}
[[ -n "${MODEL_PATH}" ]] || { echo "--model_path is required" >&2; exit 2; }
[[ "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--gpus must be a positive integer" >&2
  exit 2
}
[[ -n "${DATASETS}" ]] || { echo "--datasets cannot be empty" >&2; exit 2; }
[[ "${SHARDS_PER_DATASET}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--shards_per_dataset must be a positive integer" >&2
  exit 2
}
[[ "${MAX_NUM_SEQS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max_num_seqs must be a positive integer" >&2
  exit 2
}
case "${SCHEDULING}" in
  strict_dataset_wave_dynamic_microshards|overlap_dataset_waves_dynamic_microshards) ;;
  *) echo "Unsupported scheduling mode: ${SCHEDULING}" >&2; exit 2 ;;
esac
[[ "${ENFORCE_EAGER}" == "0" || "${ENFORCE_EAGER}" == "1" ]] || {
  echo "MOPD_EVAL_ENFORCE_EAGER must be 0 or 1" >&2
  exit 2
}
[[ "${ENABLE_CHUNKED_PREFILL}" == "0" || "${ENABLE_CHUNKED_PREFILL}" == "1" ]] || {
  echo "MOPD_EVAL_ENABLE_CHUNKED_PREFILL must be 0 or 1" >&2
  exit 2
}
[[ "${MIN_ROWS_PER_SHARD}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--min_rows_per_shard must be a positive integer" >&2
  exit 2
}
[[ -z "${MAX_SAMPLES}" || "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max_samples must be a positive integer" >&2
  exit 2
}
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "run_tag may contain only letters, numbers, '.', '_', and '-'." >&2
  exit 2
}
if [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}" || true)"
fi
[[ -x "${PYTHON_BIN}" ]] || { echo "Python executable is not runnable: ${PYTHON_BIN:-unset}" >&2; exit 2; }
EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-${MODEL_PATH}}"
if [[ -z "${GPU_ID_LIST}" ]]; then
  default_gpu_ids=()
  for ((gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++)); do
    default_gpu_ids+=("${gpu_index}")
  done
  GPU_ID_LIST="$(IFS=,; echo "${default_gpu_ids[*]}")"
fi
IFS=',' read -r -a GPU_IDS <<<"${GPU_ID_LIST}"
SUITE_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
MANIFEST_PATH="${SUITE_ROOT}/suite_manifest.json"
LOG_ROOT="${SUITE_ROOT}/logs"

export LANG=C
export LC_ALL=C
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MOPD_ALLOW_SIMPLE_SCORER_FALLBACK="${MOPD_ALLOW_SIMPLE_SCORER_FALLBACK:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
unset ROCR_VISIBLE_DEVICES

[[ "${#GPU_IDS[@]}" == "${GPU_COUNT}" ]] || {
  echo "Expected ${GPU_COUNT} comma-separated GPU IDs; got ${#GPU_IDS[@]}" >&2
  exit 2
}
SEEN_GPU_IDS=","
for gpu_id in "${GPU_IDS[@]}"; do
  [[ "${gpu_id}" =~ ^[0-9]+$ ]] || {
    echo "GPU IDs must be non-negative integers: ${gpu_id}" >&2
    exit 2
  }
  [[ "${SEEN_GPU_IDS}" != *",${gpu_id},"* ]] || {
    echo "GPU IDs must be unique: ${gpu_id}" >&2
    exit 2
  }
  SEEN_GPU_IDS+="${gpu_id},"
done
[[ -d "${MODEL_PATH}" ]] || { echo "model path does not exist: ${MODEL_PATH}" >&2; exit 2; }
[[ -d "${EVAL_MODEL_PATH}" ]] || { echo "eval model path does not exist: ${EVAL_MODEL_PATH}" >&2; exit 2; }
[[ -z "${G_OPD_DIR}" || -d "${G_OPD_DIR}" ]] || {
  echo "G-OPD checkout is missing: ${G_OPD_DIR}" >&2
  exit 2
}

CANONICAL_DATASETS="aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus,lcb_v5,lcb_v6,gpqa_diamond,mmlupro_500_seed42"
if [[ "${STANDARD_PROTOCOL}" == "1" ]]; then
  DATASETS="${CANONICAL_DATASETS}"
  SCORE_CODE=1
  SCHEDULING="strict_dataset_wave_dynamic_microshards"
  MAX_NUM_SEQS=24
  ENFORCE_EAGER=1
  ENABLE_CHUNKED_PREFILL=0
  [[ "${G_OPD_DIR}" != "" ]] || {
    echo "--standard_protocol requires --gopd_dir" >&2
    exit 2
  }
fi
if [[ "${DEFER_OFFICIAL_CODE_SCORING}" == "1" ]]; then
  [[ "${STANDARD_PROTOCOL}" == "1" ]] || {
    echo "--defer_official_code_scoring requires --standard_protocol" >&2
    exit 2
  }
  SCORE_CODE=0
fi
if [[ "${SCORE_CODE}" == "1" \
  && ( "${DATASETS}" == *"humaneval_plus"* || "${DATASETS}" == *"mbpp_plus"* ) ]]; then
  CODE_SANDBOX_IMAGE="${MOPD_CODE_SANDBOX_IMAGE:-verlai/verl:vllm023.dev1}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    CODE_SANDBOX_IMAGE_ID="dry-run"
  else
    command -v docker >/dev/null 2>&1 || {
      echo "Docker is required for isolated Code scoring" >&2
      exit 2
    }
    CODE_SANDBOX_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${CODE_SANDBOX_IMAGE}")" || {
      echo "Code scorer Docker image is missing: ${CODE_SANDBOX_IMAGE}" >&2
      exit 2
    }
    export MOPD_CODE_SANDBOX=docker
    export MOPD_CODE_SANDBOX_IMAGE="${CODE_SANDBOX_IMAGE}"
  fi
else
  CODE_SANDBOX_IMAGE="verlai/verl:vllm023.dev1"
  CODE_SANDBOX_IMAGE_ID="disabled"
fi
if [[ "${RESUME}" == "0" && -e "${SUITE_ROOT}" ]]; then
  echo "Evaluation suite already exists: ${SUITE_ROOT}" >&2
  exit 2
fi

PLAN_ARGS=(
  "${PYTHON_BIN}" -m eval.parallel_eval plan
  --code-dir "${CODE_DIR}"
  --suite-root "${SUITE_ROOT}"
  --run-tag "${RUN_TAG}"
  --model-path "${MODEL_PATH}"
  --eval-model-path "${EVAL_MODEL_PATH}"
  --datasets "${DATASETS}"
  --shards-per-dataset "${SHARDS_PER_DATASET}"
  --worker-count "${GPU_COUNT}"
  --min-rows-per-shard "${MIN_ROWS_PER_SHARD}"
  --math-samples 8
  --code-samples 8
  --science-samples 8
  --base-seed 42
  --max-new-tokens 16384
  --temperature 1.0
  --top-p 1.0
  --batch-size 24
  --gpu-memory 0.85
  --max-model-len 18432
  --max-num-batched-tokens 32768
  --max-num-seqs "${MAX_NUM_SEQS}"
  --scheduling "${SCHEDULING}"
)
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  PLAN_ARGS+=(--enforce-eager)
else
  PLAN_ARGS+=(--no-enforce-eager)
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  PLAN_ARGS+=(--enable-chunked-prefill)
else
  PLAN_ARGS+=(--no-enable-chunked-prefill)
fi
[[ -z "${G_OPD_DIR}" ]] || PLAN_ARGS+=(--gopd-dir "${G_OPD_DIR}")
[[ -z "${MAX_SAMPLES}" ]] || PLAN_ARGS+=(--max-samples-per-dataset "${MAX_SAMPLES}")
[[ "${SCORE_CODE}" == "1" ]] && PLAN_ARGS+=(
  --code-sandbox-image "${CODE_SANDBOX_IMAGE}"
  --code-sandbox-image-id "${CODE_SANDBOX_IMAGE_ID}"
) || PLAN_ARGS+=(--no-score-code)
[[ "${RESUME}" == "1" ]] && PLAN_ARGS+=(--resume)
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[local-eval] dry run; no workers launched\n'
  printf '%q ' "${PLAN_ARGS[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}"
"${PLAN_ARGS[@]}" > "${OUTPUT_ROOT}/${RUN_TAG}.plan.log" 2>&1

mkdir -p "${LOG_ROOT}"
mv "${OUTPUT_ROOT}/${RUN_TAG}.plan.log" "${LOG_ROOT}/plan.log"
cat > "${SUITE_ROOT}/RUN_MANIFEST.md" <<EOF
# Direct-local Step-60 evaluation

- scheduler: direct-local
- remote host: $(hostname)
- GPU IDs: ${GPU_ID_LIST}
- topology: DP=${GPU_COUNT}, ${GPU_COUNT} persistent TP=1 vLLM workers
- checkpoint: ${MODEL_PATH}
- eval model path: ${EVAL_MODEL_PATH}
- global step: 60
- datasets: ${DATASETS}
- protocol: $(if [[ "${STANDARD_PROTOCOL}" == "1" ]]; then printf 'canonical 3-domain ten-dataset evaluation'; else printf 'custom partial evaluation'; fi); ${SHARDS_PER_DATASET} micro-shards per dataset; scheduling=${SCHEDULING}
- rollouts: K=8 per prompt/domain, temperature=1.0, top_p=1.0
- generation: max_new_tokens=16384, max_model_len=18432
- execution: batch_size=24, max_num_batched_tokens=32768, max_num_seqs=${MAX_NUM_SEQS}, gpu_memory=0.85, enforce_eager=${ENFORCE_EAGER}, chunked_prefill=${ENABLE_CHUNKED_PREFILL}
- code scoring: $(if [[ "${SCORE_CODE}" == "1" ]]; then printf 'enabled via Docker (%s)' "${CODE_SANDBOX_IMAGE}"; elif [[ "${DEFER_OFFICIAL_CODE_SCORING}" == "1" ]]; then printf 'HumanEvalPlus/MBPPPlus deferred to pinned official EvalPlus post-scoring; LCB uses the official G-OPD runner'; else printf 'disabled'; fi)
- python: ${PYTHON_BIN}
- suite root: ${SUITE_ROOT}
- wrapper log: ${SUITE_ROOT}/wrapper.log
- worker logs: ${LOG_ROOT}/gpu_worker_0.log through gpu_worker_$((GPU_COUNT - 1)).log
- started at UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)

$(if [[ "${STANDARD_PROTOCOL}" == "1" ]]; then printf 'This is the canonical 10-dataset 3-domain run.'; else printf 'This is intentionally a custom partial run and must not be treated as the canonical 10-dataset active ranking.'; fi)
EOF

pids=()
for worker_id in "${!GPU_IDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[${worker_id}]}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}" \
  NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}" \
    "${PYTHON_BIN}" -m eval.parallel_worker \
      --manifest "${MANIFEST_PATH}" \
      --eval-model-path "${EVAL_MODEL_PATH}" \
      --worker-id "${worker_id}" \
      > "${LOG_ROOT}/gpu_worker_${worker_id}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" != "0" ]]; then
  echo "One or more parallel evaluation workers failed" >&2
  exit "${status}"
fi

"${PYTHON_BIN}" -m eval.parallel_eval merge --manifest "${MANIFEST_PATH}" \
  > "${LOG_ROOT}/merge.log" 2>&1
if [[ "${DEFER_OFFICIAL_CODE_SCORING}" == "1" ]]; then
  rm -f "${SUITE_ROOT}/SUCCESS"
  date -u +%Y-%m-%dT%H:%M:%SZ \
    > "${SUITE_ROOT}/GENERATION_COMPLETE_PENDING_OFFICIAL_EVALPLUS"
  "${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["status"] = "generation_complete_pending_official_evalplus"
manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY
  echo "[local-eval] generation complete; official EvalPlus scoring pending output=${SUITE_ROOT}"
else
  date -u +%Y-%m-%dT%H:%M:%SZ > "${SUITE_ROOT}/COMPLETED_AT_UTC"
  echo "[local-eval] complete output=${SUITE_ROOT}"
fi
