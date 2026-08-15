#!/usr/bin/env bash
set -euo pipefail

export LANG=C
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

usage() {
  cat <<'USAGE'
Submit one or more OPD checkpoint evaluations to a single Slurm GPU.

Usage:
  ./slurm_eval.sh --model_path PATH [--model_path PATH ...] [options]

The default suite is the held-out Math/Code/Science validation set used by the
three-domain training profiles:
  aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus,gpqa_diamond

Default sampling protocol:
  - temperature=1.0, top_p=1.0, max_new_tokens=16384
  - Math: 32 samples/problem; Code: 4 samples/problem
  - Science: 1 sample/problem (not specified by the Math/Code paper protocol)

Checkpoint handling:
  - A Hugging Face model directory is evaluated directly.
  - A verl global_step directory or actor directory is detected automatically.
  - FSDP actor shards are merged once into actor/hf before evaluation.
  - Repeated --model_path arguments run sequentially in one single-GPU job.
  - HumanEvalPlus/MBPPPlus scoring runs generated code in restricted Docker.
    LiveCodeBench is rejected in Docker mode until its scorer is isolated.

Options:
  --model_path PATH       Model/checkpoint path. Repeat for multiple models.
  --datasets NAMES        Comma-separated dataset keys (default: held-out suite).
  --output_root PATH      Parent output directory (default:
                          data/eval_data/results/slurm_eval under the code root).
  --run_tag TAG           Suite identifier (default: timestamp).
  --max_samples N         Maximum examples per dataset (default: all).
  --resume                Resume existing partial models and start untouched ones.
  --no_score_code         Generate Code answers without executing the scorer.
  --dry_run               Print the sbatch command without submitting.
  -h, --help              Show this help.

Environment overrides:
  SLURM_EVAL_PARTITION      default: compute
  SLURM_EVAL_TIME           default: 48:00:00
  SLURM_EVAL_MEMORY         default: 24G
  SLURM_EVAL_CPUS           default: 8
  SLURM_EVAL_PYTHON         default: mopd-verl Python on the remote cluster
  SLURM_EVAL_MAX_TOKENS     default: 16384
  SLURM_EVAL_TEMPERATURE    default: 1.0
  SLURM_EVAL_TOP_P          default: 1.0
  SLURM_EVAL_MATH_SAMPLES   default: 32
  SLURM_EVAL_CODE_SAMPLES   default: 4
  SLURM_EVAL_SCIENCE_SAMPLES default: 1
  SLURM_EVAL_BATCH_SIZE     default: 24
  SLURM_EVAL_GPU_MEMORY     default: 0.85
  MOPD_CODE_SANDBOX_IMAGE   default: verlai/verl:vllm023.dev1
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

validate_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer: ${value}"
}

MODEL_PATHS=()
DATASETS="aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus,gpqa_diamond"
OUTPUT_ROOT="${SCRIPT_DIR}/data/eval_data/results/slurm_eval"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
MAX_SAMPLES=""
RESUME=0
SCORE_CODE=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path|--model-path)
      [[ $# -ge 2 ]] || fail "$1 requires a value"
      MODEL_PATHS+=("$2")
      shift 2
      ;;
    --datasets)
      [[ $# -ge 2 ]] || fail "$1 requires a value"
      DATASETS="$2"
      shift 2
      ;;
    --output_root|--output-root)
      [[ $# -ge 2 ]] || fail "$1 requires a value"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run_tag|--run-tag)
      [[ $# -ge 2 ]] || fail "$1 requires a value"
      RUN_TAG="$2"
      shift 2
      ;;
    --max_samples|--max-samples)
      [[ $# -ge 2 ]] || fail "$1 requires a value"
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --no_score_code|--no-score-code)
      SCORE_CODE=0
      shift
      ;;
    --dry_run|--dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

[[ "${#MODEL_PATHS[@]}" -gt 0 ]] || fail "at least one --model_path is required"
[[ -n "${DATASETS}" ]] || fail "--datasets cannot be empty"
[[ -n "${RUN_TAG}" ]] || fail "--run_tag cannot be empty"
[[ -z "${SLURM_JOB_ID:-}" ]] \
  || fail "submit mode cannot run inside an allocation; use a login shell"
if [[ -n "${MAX_SAMPLES}" ]]; then
  validate_positive_integer "--max_samples" "${MAX_SAMPLES}"
fi
for model_path in "${MODEL_PATHS[@]}"; do
  [[ -d "${model_path}" ]] || fail "model path does not exist: ${model_path}"
done

PARTITION="${SLURM_EVAL_PARTITION:-compute}"
TIME_LIMIT="${SLURM_EVAL_TIME:-48:00:00}"
MEMORY="${SLURM_EVAL_MEMORY:-24G}"
CPUS="${SLURM_EVAL_CPUS:-8}"
validate_positive_integer "SLURM_EVAL_CPUS" "${CPUS}"

LOG_DIR="${SCRIPT_DIR}/logs/slurm_eval"
JOB_NAME="mopd_eval_${RUN_TAG}"
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
  --gpus=1
  --cpus-per-task="${CPUS}"
  --mem="${MEMORY}"
  --time="${TIME_LIMIT}"
  --partition="${PARTITION}"
  "${SCRIPT_DIR}/scripts/slurm_eval_worker.sh"
  --datasets "${DATASETS}"
  --output_root "${OUTPUT_ROOT}"
  --run_tag "${RUN_TAG}"
)
[[ -z "${MAX_SAMPLES}" ]] || SBATCH_COMMAND+=(--max_samples "${MAX_SAMPLES}")
[[ "${RESUME}" == "0" ]] || SBATCH_COMMAND+=(--resume)
[[ "${SCORE_CODE}" == "1" ]] || SBATCH_COMMAND+=(--no_score_code)
for model_path in "${MODEL_PATHS[@]}"; do
  SBATCH_COMMAND+=(--model_path "${model_path}")
done

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[slurm-eval] dry run; no job submitted\n'
  quote_command "${SBATCH_COMMAND[@]}"
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
[[ -x "${SCRIPT_DIR}/scripts/slurm_eval_worker.sh" ]] \
  || fail "Slurm worker is not executable: ${SCRIPT_DIR}/scripts/slurm_eval_worker.sh"
unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
JOB_ID="$("${SBATCH_COMMAND[@]}")"
printf '[slurm-eval] submitted job_id=%s gpu=1 models=%s output=%s/%s\n' \
  "${JOB_ID}" "${#MODEL_PATHS[@]}" "${OUTPUT_ROOT}" "${RUN_TAG}"
