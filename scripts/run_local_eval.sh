#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
Run the parquet-based OPD evaluation with Transformers or vLLM.

Usage:
  scripts/run_local_eval.sh --model-path PATH [options]

Options:
  --model-path PATH       Local model directory or Hugging Face model id.
  --datasets NAMES        Comma-separated datasets (default: aime24,aime25,
                          hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus).
                          Also supports ifeval, ifbench, and gpqa_diamond.
                          Use training_ceiling for the overlapping four-domain
                          training-performance ceiling sample, or select
                          training_math, training_code, training_if, and
                          training_science individually. Use training_full or
                          training_full_{math,code,if,science} for raw training
                          rows.
  --modes NAMES           Comma-separated modes: non_thinking,thinking
                          (default: non_thinking).
  --max-samples N         Maximum examples per dataset (default: all).
  --sample-offset N       Rows to skip in every selected parquet (default: 0).
  --max-new-tokens N      Generation limit for every selected mode (default: 8192).
  --num-samples N         Rollouts per prompt (default: 1; GRPO AIME paper eval: 32).
  --temperature FLOAT     Sampling temperature (default: 0; GRPO AIME paper eval: 1).
  --top-p FLOAT           Nucleus sampling threshold (default: 1.0).
  --seed N                Base generation seed (default: 42).
  --backend NAME          transformers or vllm (default: transformers).
  --tensor-parallel-size N
                          vLLM tensor-parallel GPU count (default: 1).
  --batch-size N          vLLM generation batch size (default: 8).
  --gpu-memory FLOAT      vLLM GPU memory utilization (default: 0.9).
  --max-model-len N       vLLM context length override.
  --max-num-batched-tokens N
                          vLLM scheduler token budget override.
  --max-num-seqs N        vLLM concurrent sequence limit override.
  --enforce-eager         Disable CUDA graphs in vLLM.
  --disable-chunked-prefill
                          Explicitly disable vLLM chunked prefill.
  --torch-dtype NAME      Model dtype (default: auto).
  --output-dir PATH       Result directory (default: data/eval_data/results/<run-id>).
  --run-id ID             Run identifier used in the report.
  --python PATH           Python executable (default: $PYTHON or python3).
  --score-code            Execute generated code for Code scoring; use only in
                          an isolated environment with the full verl rewards.
  --save-completions      Save full model completions in JSONL output.
  --resume                Resume an existing output directory from its validated
                          incremental JSONL prefix.
  --wandb-project NAME    Upload metrics/artifacts to this W&B project.
  --wandb-entity NAME     Optional W&B entity override.
  --wandb-group NAME      Optional W&B run group.
  --wandb-mode NAME       online, offline, or disabled (default: online).
  --wandb-upload-raw      Upload full rollout JSONL and report files as an artifact.
  --wandb-timeout-seconds N
                          Per-run upload timeout (default: 1800; 0 disables it).
  --wandb-env-file PATH   Env file parsed with the training launcher logic.
  --defer-wandb-upload    Write the local report but let a parent launcher upload it.
  --no-wandb              Disable W&B even when EVAL_WANDB_PROJECT is set.
  --dry-run               Validate inputs and print the command only.
  -h, --help              Show this help.

USAGE
}
MODEL_PATH="${MODEL_PATH:-}"
DATASETS="${DATASETS:-aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus}"
MODES="${MODES:-non_thinking}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"
BACKEND="${BACKEND:-transformers}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
ENFORCE_EAGER=0
DISABLE_CHUNKED_PREFILL=0
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
RUN_ID="${RUN_ID:-local_eval_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PYTHON_BIN="${PYTHON:-python3}"
SCORE_CODE=0
SAVE_COMPLETIONS=0
RESUME=0
DRY_RUN=0
NEEDS_TRAINING_CODE_SCORER=0
NEEDS_IF_SCORER=0
WANDB_PROJECT_NAME="${EVAL_WANDB_PROJECT:-}"
WANDB_ENTITY_NAME="${EVAL_WANDB_ENTITY:-}"
WANDB_GROUP_NAME="${EVAL_WANDB_GROUP:-}"
WANDB_MODE_NAME="${EVAL_WANDB_MODE:-online}"
WANDB_UPLOAD_RAW="${EVAL_WANDB_UPLOAD_RAW:-0}"
WANDB_ENABLED="${EVAL_WANDB_ENABLED:-1}"
WANDB_TIMEOUT_SECONDS="${EVAL_WANDB_TIMEOUT_SECONDS:-1800}"
WANDB_ENV_FILE="${EVAL_WANDB_ENV_FILE:-${CODE_DIR}/.env.local}"
DEFER_WANDB_UPLOAD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="${2:?--model-path requires a value}"; shift 2 ;;
    --datasets) DATASETS="${2:?--datasets requires a value}"; shift 2 ;;
    --modes) MODES="${2:?--modes requires a value}"; shift 2 ;;
    --max-samples) MAX_SAMPLES="${2:?--max-samples requires a value}"; shift 2 ;;
    --sample-offset) SAMPLE_OFFSET="${2:?--sample-offset requires a value}"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="${2:?--max-new-tokens requires a value}"; shift 2 ;;
    --num-samples) NUM_SAMPLES="${2:?--num-samples requires a value}"; shift 2 ;;
    --temperature) TEMPERATURE="${2:?--temperature requires a value}"; shift 2 ;;
    --top-p) TOP_P="${2:?--top-p requires a value}"; shift 2 ;;
    --seed) SEED="${2:?--seed requires a value}"; shift 2 ;;
    --backend) BACKEND="${2:?--backend requires a value}"; shift 2 ;;
    --tensor-parallel-size) TENSOR_PARALLEL_SIZE="${2:?--tensor-parallel-size requires a value}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:?--batch-size requires a value}"; shift 2 ;;
    --gpu-memory) GPU_MEMORY_UTILIZATION="${2:?--gpu-memory requires a value}"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="${2:?--max-model-len requires a value}"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="${2:?--max-num-batched-tokens requires a value}"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="${2:?--max-num-seqs requires a value}"; shift 2 ;;
    --enforce-eager) ENFORCE_EAGER=1; shift ;;
    --disable-chunked-prefill) DISABLE_CHUNKED_PREFILL=1; shift ;;
    --torch-dtype) TORCH_DTYPE="${2:?--torch-dtype requires a value}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
    --run-id) RUN_ID="${2:?--run-id requires a value}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?--python requires a value}"; shift 2 ;;
    --score-code) SCORE_CODE=1; shift ;;
    --save-completions) SAVE_COMPLETIONS=1; shift ;;
    --resume) RESUME=1; shift ;;
    --wandb-project) WANDB_PROJECT_NAME="${2:?--wandb-project requires a value}"; WANDB_ENABLED=1; shift 2 ;;
    --wandb-entity) WANDB_ENTITY_NAME="${2:?--wandb-entity requires a value}"; shift 2 ;;
    --wandb-group) WANDB_GROUP_NAME="${2:?--wandb-group requires a value}"; shift 2 ;;
    --wandb-mode) WANDB_MODE_NAME="${2:?--wandb-mode requires a value}"; shift 2 ;;
    --wandb-upload-raw) WANDB_UPLOAD_RAW=1; shift ;;
    --wandb-timeout-seconds) WANDB_TIMEOUT_SECONDS="${2:?--wandb-timeout-seconds requires a value}"; shift 2 ;;
    --wandb-env-file) WANDB_ENV_FILE="${2:?--wandb-env-file requires a value}"; shift 2 ;;
    --defer-wandb-upload) DEFER_WANDB_UPLOAD=1; shift ;;
    --no-wandb) WANDB_ENABLED=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "${MODEL_PATH}" ]] || { echo "--model-path is required" >&2; exit 2; }
[[ "${MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max-new-tokens must be a positive integer" >&2
  exit 2
}
[[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || { echo "--num-samples must be a positive integer" >&2; exit 2; }
[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "--seed must be a non-negative integer" >&2; exit 2; }
[[ "${BACKEND}" == "transformers" || "${BACKEND}" == "vllm" ]] || {
  echo "--backend must be transformers or vllm" >&2
  exit 2
}
[[ "${TENSOR_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--tensor-parallel-size must be a positive integer" >&2
  exit 2
}
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "--batch-size must be a positive integer" >&2; exit 2; }
for integer_option in MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS; do
  if [[ -n "${!integer_option}" && ! "${!integer_option}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${integer_option} must be a positive integer when provided" >&2
    exit 2
  fi
done
if [[ -n "${MAX_SAMPLES}" && ! "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-samples must be a positive integer" >&2
  exit 2
fi
[[ "${SAMPLE_OFFSET}" =~ ^[0-9]+$ ]] || {
  echo "--sample-offset must be a non-negative integer" >&2
  exit 2
}
[[ "${WANDB_MODE_NAME}" == "online" || "${WANDB_MODE_NAME}" == "offline" \
  || "${WANDB_MODE_NAME}" == "disabled" ]] || {
  echo "--wandb-mode must be online, offline, or disabled" >&2
  exit 2
}
[[ "${WANDB_UPLOAD_RAW}" == "0" || "${WANDB_UPLOAD_RAW}" == "1" ]] || {
  echo "EVAL_WANDB_UPLOAD_RAW must be 0 or 1" >&2
  exit 2
}
[[ "${WANDB_ENABLED}" == "0" || "${WANDB_ENABLED}" == "1" ]] || {
  echo "EVAL_WANDB_ENABLED must be 0 or 1" >&2
  exit 2
}
[[ "${WANDB_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] || {
  echo "--wandb-timeout-seconds must be a non-negative integer" >&2
  exit 2
}
[[ "${WANDB_ENABLED}" == "1" ]] || WANDB_PROJECT_NAME=""

IFS=',' read -r -a DATASET_NAMES <<< "${DATASETS}"
DATA_FILES=()
for name in "${DATASET_NAMES[@]}"; do
  name="${name//[[:space:]]/}"
  relative_paths=()
  case "${name}" in
    aime24) relative_paths=("data/eval_data/math/AIME24/test.parquet") ;;
    aime25) relative_paths=("data/eval_data/math/AIME25/test.parquet") ;;
    hmmt25feb) relative_paths=("data/eval_data/math/HMMT25Feb/test.parquet") ;;
    hmmt25nov) relative_paths=("data/eval_data/math/HMMT25Nov/test.parquet") ;;
    humaneval_plus) relative_paths=("data/eval_data/code/HumanEvalPlus/test.parquet") ;;
    mbpp_plus) relative_paths=("data/eval_data/code/MBPPPlus/test.parquet") ;;
    livecodebench) relative_paths=("data/eval_data/code/LiveCodeBench/test.parquet") ;;
    ifeval) relative_paths=("data/eval_data/if/IFEval/test.parquet") ;;
    ifbench) relative_paths=("data/eval_data/if/IFBench/test.parquet") ;;
    gpqa_diamond) relative_paths=("data/eval_data/science/GPQA/test.parquet") ;;
    training_ceiling)
      relative_paths=(
        "data/eval_training_data/math/test.parquet"
        "data/eval_training_data/code/test.parquet"
        "data/eval_training_data/if/test.parquet"
        "data/eval_training_data/science/test.parquet"
      )
      ;;
    training_math) relative_paths=("data/eval_training_data/math/test.parquet") ;;
    training_code) relative_paths=("data/eval_training_data/code/test.parquet") ;;
    training_if) relative_paths=("data/eval_training_data/if/test.parquet") ;;
    training_science) relative_paths=("data/eval_training_data/science/test.parquet") ;;
    training_full)
      relative_paths=(
        "data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"
        "data/G-OPD-Training-Data/Eurus/code_train.parquet"
        "data/G-OPD-Training-Data/IF/train.parquet"
        "data/G-OPD-Training-Data/Science/train.parquet"
      )
      ;;
    training_full_math) relative_paths=("data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet") ;;
    training_full_code) relative_paths=("data/G-OPD-Training-Data/Eurus/code_train.parquet") ;;
    training_full_if) relative_paths=("data/G-OPD-Training-Data/IF/train.parquet") ;;
    training_full_science) relative_paths=("data/G-OPD-Training-Data/Science/train.parquet") ;;
    *)
      echo "Unknown dataset '${name}'." >&2
      echo "Valid names: aime24 aime25 hmmt25feb hmmt25nov humaneval_plus mbpp_plus livecodebench ifeval ifbench gpqa_diamond training_ceiling training_math training_code training_if training_science training_full training_full_math training_full_code training_full_if training_full_science" >&2
      exit 2
      ;;
  esac
  for relative_path in "${relative_paths[@]}"; do
    if [[ "${relative_path}" == data/eval_training_data/code/* \
      || "${relative_path}" == data/G-OPD-Training-Data/Eurus/* ]]; then
      NEEDS_TRAINING_CODE_SCORER=1
    fi
    if [[ "${relative_path}" == data/eval_training_data/if/* \
      || "${relative_path}" == data/eval_data/if/* \
      || "${relative_path}" == data/G-OPD-Training-Data/IF/* ]]; then
      NEEDS_IF_SCORER=1
    fi
    data_file="${CODE_DIR}/${relative_path}"
    [[ -f "${data_file}" ]] || {
      echo "Missing eval data: ${data_file}" >&2
      if [[ "${relative_path}" == data/eval_training_data/* ]]; then
        echo "Run: python scripts/split_domain_eval_training_data.py --eval-size 10000 --seed 42 --overwrite" >&2
      elif [[ "${relative_path}" == data/G-OPD-Training-Data/* ]]; then
        echo "Run: DOWNLOAD_LCB=1 bash scripts/download_qwen1p7b_goosereason_data.sh" >&2
      else
        echo "Run eval/scripts/prepare_paper_eval_data.sh first." >&2
      fi
      exit 2
    }
    DATA_FILES+=("${data_file}")
  done
done

IFS=',' read -r -a MODE_NAMES <<< "${MODES}"
for index in "${!MODE_NAMES[@]}"; do
  mode="${MODE_NAMES[${index}]//[[:space:]]/}"
  MODE_NAMES[${index}]="${mode}"
  [[ "${mode}" == "thinking" || "${mode}" == "non_thinking" ]] || {
    echo "Unknown mode '${mode}'; use thinking or non_thinking." >&2
    exit 2
  }
done

OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/data/eval_data/results/${RUN_ID}}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MOPD_ALLOW_SIMPLE_SCORER_FALLBACK="${MOPD_ALLOW_SIMPLE_SCORER_FALLBACK:-1}"

CMD=(
  "${PYTHON_BIN}" -m eval.runner
  --model-path "${MODEL_PATH}"
  --data-files "${DATA_FILES[@]}"
  --output-dir "${OUTPUT_DIR}"
  --modes "${MODE_NAMES[@]}"
  --backend "${BACKEND}"
  --torch-dtype "${TORCH_DTYPE}"
  --max-new-tokens-thinking "${MAX_NEW_TOKENS}"
  --max-new-tokens-non-thinking "${MAX_NEW_TOKENS}"
  --num-samples "${NUM_SAMPLES}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --seed "${SEED}"
  --sample-offset-per-dataset "${SAMPLE_OFFSET}"
)
if [[ "${BACKEND}" == "vllm" ]]; then
  CMD+=(
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --batch-size "${BATCH_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  )
  [[ -z "${MAX_MODEL_LEN}" ]] || CMD+=(--max-model-len "${MAX_MODEL_LEN}")
  [[ -z "${MAX_NUM_BATCHED_TOKENS}" ]] || {
    CMD+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
  }
  [[ -z "${MAX_NUM_SEQS}" ]] || CMD+=(--max-num-seqs "${MAX_NUM_SEQS}")
  [[ "${ENFORCE_EAGER}" == "0" ]] || CMD+=(--enforce-eager)
  [[ "${DISABLE_CHUNKED_PREFILL}" == "0" ]] || CMD+=(--no-enable-chunked-prefill)
else
  CMD+=(--device-map auto)
fi
[[ -z "${MAX_SAMPLES}" ]] || CMD+=(--max-samples-per-dataset "${MAX_SAMPLES}")
[[ "${SCORE_CODE}" == "0" ]] || CMD+=(--score-code)
[[ "${SAVE_COMPLETIONS}" == "0" ]] || CMD+=(--save-completions)
[[ "${RESUME}" == "0" ]] || CMD+=(--resume)

printf '[local-eval] model: %s\n' "${MODEL_PATH}"
printf '[local-eval] datasets: %s\n' "${DATASETS}"
printf '[local-eval] modes: %s\n' "${MODES}"
printf '[local-eval] backend: %s\n' "${BACKEND}"
if [[ "${BACKEND}" == "vllm" ]]; then
  printf '[local-eval] tensor parallel size: %s\n' "${TENSOR_PARALLEL_SIZE}"
fi
printf '[local-eval] output: %s\n' "${OUTPUT_DIR}"
if [[ -n "${WANDB_PROJECT_NAME}" ]]; then
  printf '[local-eval] W&B: project=%s group=%s mode=%s raw_artifact=%s timeout=%ss env_file=%s\n' \
    "${WANDB_PROJECT_NAME}" "${WANDB_GROUP_NAME:-<none>}" "${WANDB_MODE_NAME}" \
    "${WANDB_UPLOAD_RAW}" "${WANDB_TIMEOUT_SECONDS}" "${WANDB_ENV_FILE}"
  [[ "${DEFER_WANDB_UPLOAD}" == "0" ]] || echo "[local-eval] W&B upload deferred to parent launcher."
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[local-eval] command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
}
REQUIRED_MODULES='import pandas, pyarrow, torch, transformers'
[[ "${BACKEND}" != "transformers" ]] || REQUIRED_MODULES+=', accelerate'
[[ "${BACKEND}" != "vllm" ]] || REQUIRED_MODULES+=', vllm'
"${PYTHON_BIN}" -c "${REQUIRED_MODULES}" || {
  echo "Missing dependencies for eval backend '${BACKEND}'." >&2
  exit 2
}
if [[ "${SCORE_CODE}" == "1" && "${NEEDS_TRAINING_CODE_SCORER}" == "1" ]]; then
  "${PYTHON_BIN}" -c 'from verl.utils.reward_score import default_compute_score' || {
    echo "Training-code scoring requires the complete vendored verl reward environment; simple Math fallback is disabled." >&2
    exit 2
  }
fi
if [[ "${NEEDS_IF_SCORER}" == "1" ]]; then
  export IFBENCH_REPO="${IFBENCH_REPO:-${CODE_DIR}/../temp/IFBench}"
  if ! "${PYTHON_BIN}" -c 'import verifiable_instructions.instructions_registry' >/dev/null 2>&1 \
    && [[ ! -f "${IFBENCH_REPO}/evaluation_lib.py" ]]; then
    echo "IF scoring requires verifiable_instructions or allenai/IFBench." >&2
    echo "Activate the environment from scripts/setup_training_env.sh, or run: IFBENCH_REPO=${IFBENCH_REPO} scripts/prepare_ifbench_runtime.sh" >&2
    exit 2
  fi
fi

mkdir -p "${OUTPUT_DIR}"
cd "${CODE_DIR}"
"${CMD[@]}"
"${PYTHON_BIN}" -m eval.report \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  --model-path "${MODEL_PATH}" \
  --status final

if [[ -n "${WANDB_PROJECT_NAME}" && "${DEFER_WANDB_UPLOAD}" == "0" ]]; then
  WANDB_COMMAND=(
    bash "${CODE_DIR}/scripts/upload_eval_result_to_wandb.sh"
    --python "${PYTHON_BIN}"
    --output-dir "${OUTPUT_DIR}"
    --project "${WANDB_PROJECT_NAME}"
    --mode "${WANDB_MODE_NAME}"
    --upload-raw "${WANDB_UPLOAD_RAW}"
    --timeout-seconds "${WANDB_TIMEOUT_SECONDS}"
    --env-file "${WANDB_ENV_FILE}"
  )
  [[ -z "${WANDB_ENTITY_NAME}" ]] || WANDB_COMMAND+=(--entity "${WANDB_ENTITY_NAME}")
  [[ -z "${WANDB_GROUP_NAME}" ]] || WANDB_COMMAND+=(--group "${WANDB_GROUP_NAME}")
  if ! "${WANDB_COMMAND[@]}"; then
    echo "[local-eval] W&B upload is pending; local results are complete." >&2
    echo "[local-eval] retry with: ${PYTHON_BIN} -m eval.wandb_upload --output-dir ${OUTPUT_DIR} --project ${WANDB_PROJECT_NAME} --env-file ${WANDB_ENV_FILE}" >&2
  fi
fi

printf '[local-eval] report: %s/README.md\n' "${OUTPUT_DIR}"
