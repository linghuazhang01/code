#!/usr/bin/env bash
set -euo pipefail

export LANG=C
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

usage() {
  cat <<'USAGE'
Submit a data-parallel OPD evaluation to a dynamic Slurm GPU worker pool.

Usage:
  ./slurm_parallel_eval.sh --model_path PATH [options]

Each dataset is split into disjoint prompt shards. One persistent worker owns
each allocated GPU, loads one vLLM engine, and atomically claims pending shards,
so a GPU that finishes early immediately starts another domain shard without
reloading the model. tensor_parallel_size remains 1: this is Data Parallel.

Options:
  --model_path PATH          Model/checkpoint path.
  --datasets NAMES           Comma-separated standard datasets.
  --output_root PATH         Parent output directory.
  --run_tag TAG              Suite identifier (default: timestamp).
  --gpus N                   Slurm GPUs and worker count (default: 4).
  --shards_per_dataset N     Maximum prompt shards per dataset (default: same as GPUs).
  --min_rows_per_shard N     Avoid undersized vLLM batches (default: 24).
  --max_samples N            Maximum prompts per dataset, useful for smoke tests.
  --include_mmlupro_500      Include the pinned MMLU-Pro-500 protocol.
  --resume                   Resume compatible standard shards and skip completed ones.
  --no_score_code            Generate Code answers without executing the scorer.
  --dry_run                  Print the sbatch command without submitting.
  -h, --help                 Show this help.

Environment overrides:
  SLURM_EVAL_PARTITION       default: compute
  SLURM_EVAL_TIME            default: 48:00:00
  SLURM_EVAL_MEMORY          default: 400G
  SLURM_EVAL_CPUS            default: 64
  SLURM_EVAL_MATH_SAMPLES    default: 16
  SLURM_EVAL_CODE_SAMPLES    default: 4
  SLURM_EVAL_SCIENCE_SAMPLES default: 4
  SLURM_EVAL_SEED            default: 42
USAGE
}

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

validate_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer: ${value}"
}

quote_command() {
  printf '%q ' "$@"
  printf '\n'
}

MODEL_PATH=""
DATASETS="aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus,gpqa_diamond"
OUTPUT_ROOT="${SCRIPT_DIR}/data/eval_data/results/slurm_eval"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
GPU_COUNT=4
SHARDS_PER_DATASET=""
MIN_ROWS_PER_SHARD=24
MAX_SAMPLES=""
RESUME=0
INCLUDE_MMLUPRO_500=0
SCORE_CODE=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path|--model-path) MODEL_PATH="${2:?$1 requires a value}"; shift 2 ;;
    --datasets) DATASETS="${2:?--datasets requires a value}"; shift 2 ;;
    --output_root|--output-root) OUTPUT_ROOT="${2:?$1 requires a value}"; shift 2 ;;
    --run_tag|--run-tag) RUN_TAG="${2:?$1 requires a value}"; shift 2 ;;
    --gpus) GPU_COUNT="${2:?--gpus requires a value}"; shift 2 ;;
    --shards_per_dataset|--shards-per-dataset)
      SHARDS_PER_DATASET="${2:?$1 requires a value}"
      shift 2
      ;;
    --min_rows_per_shard|--min-rows-per-shard)
      MIN_ROWS_PER_SHARD="${2:?$1 requires a value}"
      shift 2
      ;;
    --max_samples|--max-samples) MAX_SAMPLES="${2:?$1 requires a value}"; shift 2 ;;
    --include_mmlupro_500|--include-mmlupro-500) INCLUDE_MMLUPRO_500=1; shift ;;
    --resume) RESUME=1; shift ;;
    --no_score_code|--no-score-code) SCORE_CODE=0; shift ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[[ -n "${MODEL_PATH}" ]] || fail "--model_path is required"
[[ -d "${MODEL_PATH}" ]] || fail "model path does not exist: ${MODEL_PATH}"
[[ -n "${DATASETS}" && -n "${RUN_TAG}" ]] || fail "datasets and run_tag cannot be empty"
[[ -z "${SLURM_JOB_ID:-}" ]] || fail "submit mode must run from a login shell"
validate_positive_integer "--gpus" "${GPU_COUNT}"
SHARDS_PER_DATASET="${SHARDS_PER_DATASET:-${GPU_COUNT}}"
validate_positive_integer "--shards_per_dataset" "${SHARDS_PER_DATASET}"
validate_positive_integer "--min_rows_per_shard" "${MIN_ROWS_PER_SHARD}"
[[ -z "${MAX_SAMPLES}" ]] || validate_positive_integer "--max_samples" "${MAX_SAMPLES}"

PARTITION="${SLURM_EVAL_PARTITION:-compute}"
TIME_LIMIT="${SLURM_EVAL_TIME:-48:00:00}"
MEMORY="${SLURM_EVAL_MEMORY:-400G}"
CPUS="${SLURM_EVAL_CPUS:-64}"
validate_positive_integer "SLURM_EVAL_CPUS" "${CPUS}"

LOG_DIR="${SCRIPT_DIR}/logs/slurm_eval"
JOB_NAME="mopd_parallel_eval_${RUN_TAG}"
JOB_NAME="$(printf '%s' "${JOB_NAME}" | sed -E 's/[^A-Za-z0-9._-]+/_/g' | cut -c1-100)"
mkdir -p "${LOG_DIR}"

SBATCH_COMMAND=(
  sbatch
  --parsable
  --job-name="${JOB_NAME}"
  --output="${LOG_DIR}/${JOB_NAME}_%j.log"
  --error="${LOG_DIR}/${JOB_NAME}_%j.log"
  --nodes=1
  --ntasks=1
  --gpus="${GPU_COUNT}"
  --cpus-per-task="${CPUS}"
  --mem="${MEMORY}"
  --time="${TIME_LIMIT}"
  --partition="${PARTITION}"
  "${SCRIPT_DIR}/scripts/slurm_parallel_eval_worker.sh"
  --model_path "${MODEL_PATH}"
  --datasets "${DATASETS}"
  --output_root "${OUTPUT_ROOT}"
  --run_tag "${RUN_TAG}"
  --gpus "${GPU_COUNT}"
  --shards_per_dataset "${SHARDS_PER_DATASET}"
  --min_rows_per_shard "${MIN_ROWS_PER_SHARD}"
)
[[ -z "${MAX_SAMPLES}" ]] || SBATCH_COMMAND+=(--max_samples "${MAX_SAMPLES}")
[[ "${RESUME}" == "0" ]] || SBATCH_COMMAND+=(--resume)
[[ "${INCLUDE_MMLUPRO_500}" == "0" ]] || SBATCH_COMMAND+=(--include_mmlupro_500)
[[ "${SCORE_CODE}" == "1" ]] || SBATCH_COMMAND+=(--no_score_code)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[parallel-eval] dry run; no job submitted\n'
  quote_command "${SBATCH_COMMAND[@]}"
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
[[ -x "${SCRIPT_DIR}/scripts/slurm_parallel_eval_worker.sh" ]] \
  || fail "Slurm worker is not executable: ${SCRIPT_DIR}/scripts/slurm_parallel_eval_worker.sh"
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
JOB_ID="$("${SBATCH_COMMAND[@]}")"
printf '[parallel-eval] submitted job_id=%s gpus=%s shards_per_dataset=%s output=%s/%s\n' \
  "${JOB_ID}" "${GPU_COUNT}" "${SHARDS_PER_DATASET}" "${OUTPUT_ROOT}" "${RUN_TAG}"
