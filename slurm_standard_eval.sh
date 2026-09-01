#!/usr/bin/env bash
set -euo pipefail

export LANG=C
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

usage() {
  cat <<'USAGE'
Submit the canonical MOPD Step-60 evaluation: 10 datasets, K=8, DP=4 by default.

Usage:
  ./slurm_standard_eval.sh --model_path PATH [options]

Options:
  --model_path PATH    Raw global_step_60 checkpoint or HF reference model.
  --output_root PATH   Parent output directory.
  --run_tag TAG        Unique run identifier (default: timestamp).
  --gpus N             Persistent TP=1 workers: 2, 3, or 4 (default: 4).
  --reference_anchor   Allow a canonical Base/Teacher HF model without Step 60.
  --resume             Resume the exact same run tag and protocol.
  --dry_run            Print the sbatch command without submitting.
  -h, --help           Show this help.

Environment overrides:
  SLURM_STANDARD_PARTITION  default: compute
  SLURM_STANDARD_TIME       default: 3-00:00:00
  SLURM_STANDARD_MEMORY     default/max: 400G
  SLURM_STANDARD_CPUS       default: 64
USAGE
}

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

quote_command() {
  printf '%q ' "$@"
  printf '\n'
}

MODEL_PATH=""
OUTPUT_ROOT="${SCRIPT_DIR}/data/eval_data/results/standard10"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
GPU_COUNT=4
RESUME=0
DRY_RUN=0
REFERENCE_ANCHOR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path|--model-path) MODEL_PATH="${2:?$1 requires a value}"; shift 2 ;;
    --output_root|--output-root) OUTPUT_ROOT="${2:?$1 requires a value}"; shift 2 ;;
    --run_tag|--run-tag) RUN_TAG="${2:?$1 requires a value}"; shift 2 ;;
    --gpus) GPU_COUNT="${2:?--gpus requires a value}"; shift 2 ;;
    --reference_anchor|--reference-anchor) REFERENCE_ANCHOR=1; shift ;;
    --resume) RESUME=1; shift ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[[ -n "${MODEL_PATH}" ]] || fail "--model_path is required"
[[ -d "${MODEL_PATH}" ]] || fail "model path does not exist: ${MODEL_PATH}"
if [[ "${REFERENCE_ANCHOR}" == "0" ]]; then
  [[ "$(basename "${MODEL_PATH}")" == "global_step_60" ]] \
    || fail "standard evaluation requires global_step_60 or --reference-anchor"
fi
[[ -n "${RUN_TAG}" ]] || fail "run_tag cannot be empty"
[[ "${GPU_COUNT}" == "2" || "${GPU_COUNT}" == "3" || "${GPU_COUNT}" == "4" ]] \
  || fail "--gpus must be 2, 3, or 4"
[[ -z "${SLURM_JOB_ID:-}" ]] || fail "submit mode must run from a login shell"

PARTITION="${SLURM_STANDARD_PARTITION:-compute}"
TIME_LIMIT="${SLURM_STANDARD_TIME:-3-00:00:00}"
MEMORY="${SLURM_STANDARD_MEMORY:-400G}"
CPUS="${SLURM_STANDARD_CPUS:-64}"
[[ "${MEMORY}" =~ ^([1-9][0-9]*)G$ ]] \
  || fail "SLURM_STANDARD_MEMORY must be expressed in GiB, for example 400G"
(( BASH_REMATCH[1] <= 400 )) || fail "SLURM_STANDARD_MEMORY exceeds the 400G hard cap"
[[ "${CPUS}" =~ ^[1-9][0-9]*$ ]] || fail "SLURM_STANDARD_CPUS must be positive"

JOB_NAME="mopd_standard10_${RUN_TAG}"
JOB_NAME="$(printf '%s' "${JOB_NAME}" | sed -E 's/[^A-Za-z0-9._-]+/_/g' | cut -c1-100)"
LOG_DIR="${SCRIPT_DIR}/logs/standard_eval"
LOCAL_ARCHIVE_ROOT="${LOCAL_EVAL_ARCHIVE_ROOT:-/Users/linghuazhang/Desktop/Project/OPD/experiments_records/eval}"
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
  "${SCRIPT_DIR}/scripts/slurm_standard_eval_worker.sh"
  --model_path "${MODEL_PATH}"
  --output_root "${OUTPUT_ROOT}"
  --run_tag "${RUN_TAG}"
  --local_archive "${LOCAL_ARCHIVE_ROOT}/${RUN_TAG}"
  --gpus "${GPU_COUNT}"
)
[[ "${RESUME}" == "0" ]] || SBATCH_COMMAND+=(--resume)
[[ "${REFERENCE_ANCHOR}" == "0" ]] || SBATCH_COMMAND+=(--reference_anchor)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[standard10] dry run; no job submitted\n'
  quote_command "${SBATCH_COMMAND[@]}"
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
[[ -x "${SCRIPT_DIR}/scripts/slurm_standard_eval_worker.sh" ]] \
  || fail "standard worker is not executable"
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
JOB_ID="$("${SBATCH_COMMAND[@]}")"
printf '[standard10] submitted job_id=%s checkpoint=%s output=%s/%s dp=%s\n' \
  "${JOB_ID}" "${MODEL_PATH}" "${OUTPUT_ROOT}" "${RUN_TAG}" "${GPU_COUNT}"
