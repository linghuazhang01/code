#!/usr/bin/env bash
set -euo pipefail

export LANG=C
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
CODE_DIR="$(cd "${CODE_DIR}" && pwd -P)"
REMOTE_PYTHON_DEFAULT="/home/shuang_qiu/env/miniconda3/envs/mopd-verl/bin/python"
ACTIVE_MERGE_DIR=""

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

validate_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
    || fail "${name} must be a positive integer: ${value}"
}

validate_non_negative_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] \
    || fail "${name} must be a non-negative integer: ${value}"
}

append_protocol_dataset() {
  local domain="$1"
  local dataset="$2"
  case "${domain}" in
    math)
      [[ -z "${MATH_DATASETS}" ]] || MATH_DATASETS+="," 
      MATH_DATASETS+="${dataset}"
      ;;
    code)
      [[ -z "${CODE_DATASETS}" ]] || CODE_DATASETS+="," 
      CODE_DATASETS+="${dataset}"
      ;;
    science)
      [[ -z "${SCIENCE_DATASETS}" ]] || SCIENCE_DATASETS+="," 
      SCIENCE_DATASETS+="${dataset}"
      ;;
    *) fail "unsupported protocol domain: ${domain}" ;;
  esac
}

split_protocol_datasets() {
  local raw_dataset dataset
  local -a requested_datasets
  IFS=',' read -r -a requested_datasets <<<"${DATASETS}"
  for raw_dataset in "${requested_datasets[@]}"; do
    dataset="${raw_dataset//[[:space:]]/}"
    case "${dataset}" in
      aime24|aime25|hmmt25feb|hmmt25nov)
        append_protocol_dataset math "${dataset}"
        ;;
      humaneval_plus|mbpp_plus|livecodebench)
        append_protocol_dataset code "${dataset}"
        ;;
      gpqa_diamond)
        append_protocol_dataset science "${dataset}"
        ;;
      "") fail "empty dataset name in --datasets" ;;
      *)
        fail "dataset is outside the Math/Code/Science sampling protocol: ${dataset}"
        ;;
    esac
  done
}

cleanup_active_merge() {
  if [[ -n "${ACTIVE_MERGE_DIR}" && "${ACTIVE_MERGE_DIR}" == */hf.merge-* ]]; then
    rm -rf -- "${ACTIVE_MERGE_DIR}"
  fi
}

trap cleanup_active_merge EXIT INT TERM

is_hf_model() {
  local model_dir="$1"
  [[ -f "${model_dir}/config.json" ]] || return 1
  [[ -f "${model_dir}/model.safetensors" \
    || -f "${model_dir}/model.safetensors.index.json" \
    || -f "${model_dir}/pytorch_model.bin" \
    || -f "${model_dir}/pytorch_model.bin.index.json" ]]
}

source_signature() {
  local actor_dir="$1"
  local metadata_file
  for metadata_file in \
    "${actor_dir}/fsdp_config.json" \
    "${actor_dir}"/model_world_size_*_rank_*.pt; do
    [[ -f "${metadata_file}" ]] || fail "missing FSDP source file: ${metadata_file}"
    stat --printf='%n:%s:%Y\n' "${metadata_file}"
  done | sort | sha256sum | cut -d' ' -f1
}

merged_model_is_reusable() {
  local actor_dir="$1"
  local merged_dir="$2"
  local marker="${merged_dir}/.mopd_merge_complete"
  local recorded_signature current_signature
  is_hf_model "${merged_dir}" || return 1
  [[ -f "${marker}" && -f "${merged_dir}/tokenizer_config.json" ]] || return 1
  recorded_signature="$(sed -n 's/^source_signature=//p' "${marker}")"
  [[ -n "${recorded_signature}" ]] || return 1
  current_signature="$(source_signature "${actor_dir}")"
  [[ "${recorded_signature}" == "${current_signature}" ]]
}

resolve_actor_dir() {
  local model_path="$1"
  if [[ -f "${model_path}/actor/fsdp_config.json" ]]; then
    printf '%s\n' "${model_path}/actor"
  elif [[ -f "${model_path}/fsdp_config.json" ]]; then
    printf '%s\n' "${model_path}"
  else
    return 1
  fi
}

model_label() {
  local model_path="${1%/}"
  local leaf parent label
  leaf="$(basename "${model_path}")"
  parent="$(basename "$(dirname "${model_path}")")"
  if [[ "${leaf}" == global_step_* ]]; then
    label="${parent}__${leaf}"
  else
    label="${leaf}"
  fi
  label="$(printf '%s' "${label}" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
  [[ -n "${label}" ]] || fail "could not derive a label from ${model_path}"
  printf '%s\n' "${label}"
}

merge_fsdp_checkpoint() {
  local actor_dir="$1"
  local python_bin="$2"
  local merged_dir="${actor_dir}/hf"
  local temp_dir signature

  command -v flock >/dev/null 2>&1 || fail "flock is required for safe checkpoint merging"
  exec 9>"${actor_dir}/.hf_merge.lock"
  flock 9
  if merged_model_is_reusable "${actor_dir}" "${merged_dir}"; then
    printf '[slurm-eval] reusing merged model: %s\n' "${merged_dir}" >&2
    printf '%s\n' "${merged_dir}"
    flock -u 9
    exec 9>&-
    return
  fi
  if [[ -e "${merged_dir}" ]]; then
    fail "unverified or incomplete merge directory exists: ${merged_dir}"
  fi

  temp_dir="${actor_dir}/hf.merge-${SLURM_JOB_ID:-manual}-$$"
  [[ ! -e "${temp_dir}" ]] || fail "temporary merge directory exists: ${temp_dir}"
  ACTIVE_MERGE_DIR="${temp_dir}"
  printf '[slurm-eval] merging FSDP shards: %s -> %s\n' "${actor_dir}" "${merged_dir}" >&2
  "${python_bin}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${actor_dir}" \
    --target_dir "${temp_dir}" >&2
  is_hf_model "${temp_dir}" || fail "FSDP merge did not produce a loadable model: ${temp_dir}"
  [[ -f "${temp_dir}/tokenizer_config.json" ]] \
    || fail "FSDP merge did not preserve tokenizer metadata: ${temp_dir}"
  signature="$(source_signature "${actor_dir}")"
  printf 'source_dir=%s\nsource_signature=%s\n' "${actor_dir}" "${signature}" \
    >"${temp_dir}/.mopd_merge_complete"
  mv "${temp_dir}" "${merged_dir}"
  ACTIVE_MERGE_DIR=""
  flock -u 9
  exec 9>&-
  printf '%s\n' "${merged_dir}"
}

resolve_eval_model() {
  local model_path="$1"
  local python_bin="$2"
  local actor_dir
  if is_hf_model "${model_path}"; then
    printf '%s\n' "${model_path}"
    return
  fi
  actor_dir="$(resolve_actor_dir "${model_path}")" \
    || fail "not a Hugging Face model or verl FSDP checkpoint: ${model_path}"
  merge_fsdp_checkpoint "${actor_dir}" "${python_bin}"
}

output_is_resumable() {
  local output_dir="$1"
  local unexpected_entry
  if [[ ! -e "${output_dir}" ]]; then
    return 1
  fi
  if [[ -d "${output_dir}" \
    && -f "${output_dir}/eval_run_config.json" \
    && -f "${output_dir}/thinking_eval_samples.jsonl" ]]; then
    return 0
  fi
  if [[ -d "${output_dir}" \
    && -f "${output_dir}/eval_run_config.json" \
    && ! -e "${output_dir}/thinking_eval_samples.jsonl" ]]; then
    unexpected_entry="$(find "${output_dir}" -mindepth 1 -maxdepth 1 \
      ! -name eval_run_config.json ! -name run.log -print -quit)"
    [[ -z "${unexpected_entry}" ]] \
      || fail "config-only output contains an unexpected entry: ${unexpected_entry}"
    : >"${output_dir}/thinking_eval_samples.jsonl"
    return 0
  fi
  if [[ -d "${output_dir}" && -z "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    return 1
  fi
  fail "output is not safely resumable: ${output_dir}"
}

MODEL_PATHS=()
DATASETS=""
OUTPUT_ROOT=""
RUN_TAG=""
MAX_SAMPLES=""
SAMPLE_OFFSET=0
RESUME=0
SCORE_CODE=1
MATH_DATASETS=""
CODE_DATASETS=""
SCIENCE_DATASETS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATHS+=("${2:?--model_path requires a value}"); shift 2 ;;
    --datasets) DATASETS="${2:?--datasets requires a value}"; shift 2 ;;
    --output_root) OUTPUT_ROOT="${2:?--output_root requires a value}"; shift 2 ;;
    --run_tag) RUN_TAG="${2:?--run_tag requires a value}"; shift 2 ;;
    --max_samples) MAX_SAMPLES="${2:?--max_samples requires a value}"; shift 2 ;;
    --sample_offset) SAMPLE_OFFSET="${2:?--sample_offset requires a value}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --no_score_code) SCORE_CODE=0; shift ;;
    *) fail "unknown worker argument: $1" ;;
  esac
done

[[ "${#MODEL_PATHS[@]}" -gt 0 ]] || fail "worker requires at least one --model_path"
[[ -n "${DATASETS}" && -n "${OUTPUT_ROOT}" && -n "${RUN_TAG}" ]] \
  || fail "worker requires datasets, output_root, and run_tag"
[[ -n "${SLURM_JOB_ID:-}" ]] || fail "worker must run inside a Slurm allocation"
split_protocol_datasets
[[ -z "${MAX_SAMPLES}" ]] || validate_positive_integer "--max_samples" "${MAX_SAMPLES}"
validate_non_negative_integer "--sample_offset" "${SAMPLE_OFFSET}"

EVAL_MAX_TOKENS="${SLURM_EVAL_MAX_TOKENS:-16384}"
EVAL_TEMPERATURE="${SLURM_EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${SLURM_EVAL_TOP_P:-1.0}"
MATH_SAMPLES="${SLURM_EVAL_MATH_SAMPLES:-32}"
CODE_SAMPLES="${SLURM_EVAL_CODE_SAMPLES:-4}"
SCIENCE_SAMPLES="${SLURM_EVAL_SCIENCE_SAMPLES:-1}"
EVAL_SEED="${SLURM_EVAL_SEED:-42}"
validate_positive_integer "SLURM_EVAL_MAX_TOKENS" "${EVAL_MAX_TOKENS}"
validate_positive_integer "SLURM_EVAL_MATH_SAMPLES" "${MATH_SAMPLES}"
validate_positive_integer "SLURM_EVAL_CODE_SAMPLES" "${CODE_SAMPLES}"
validate_positive_integer "SLURM_EVAL_SCIENCE_SAMPLES" "${SCIENCE_SAMPLES}"
validate_non_negative_integer "SLURM_EVAL_SEED" "${EVAL_SEED}"

PYTHON_BIN="${SLURM_EVAL_PYTHON:-${REMOTE_PYTHON_DEFAULT}}"
VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a VISIBLE_DEVICE_ARRAY <<<"${VISIBLE_DEVICES}"
[[ -n "${VISIBLE_DEVICES}" && "${#VISIBLE_DEVICE_ARRAY[@]}" == "1" ]] \
  || fail "expected exactly one Slurm GPU, got: ${VISIBLE_DEVICES:-none}"

THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_EVAL_CPUS:-8}}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"
unset ROCR_VISIBLE_DEVICES

[[ -x "${PYTHON_BIN}" ]] || fail "Python executable is not runnable: ${PYTHON_BIN}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
[[ -f "${CODE_DIR}/scripts/run_local_eval.sh" ]] \
  || fail "evaluation entrypoint is missing"
if [[ "${SCORE_CODE}" == "1" ]]; then
  command -v docker >/dev/null 2>&1 || fail "Docker is required for isolated Code scoring"
  export MOPD_CODE_SANDBOX=docker
  export MOPD_CODE_SANDBOX_IMAGE="${MOPD_CODE_SANDBOX_IMAGE:-verlai/verl:vllm023.dev1}"
  docker image inspect "${MOPD_CODE_SANDBOX_IMAGE}" >/dev/null
fi

SUITE_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
mkdir -p "${SUITE_ROOT}"
OUTPUT_LABELS=()
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  LABEL="$(model_label "${MODEL_PATH}")"
  for EXISTING_LABEL in "${OUTPUT_LABELS[@]}"; do
    [[ "${EXISTING_LABEL}" != "${LABEL}" ]] \
      || fail "model paths resolve to the same output label: ${LABEL}"
  done
  OUTPUT_LABELS+=("${LABEL}")
done

printf '[slurm-eval] job_id=%s host=%s visible_gpu=%s models=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "${VISIBLE_DEVICES}" "${#MODEL_PATHS[@]}"
printf '[slurm-eval] protocol temperature=%s top_p=%s max_new_tokens=%s math_n=%s code_n=%s science_n=%s\n' \
  "${EVAL_TEMPERATURE}" "${EVAL_TOP_P}" "${EVAL_MAX_TOKENS}" \
  "${MATH_SAMPLES}" "${CODE_SAMPLES}" "${SCIENCE_SAMPLES}"
SLURM_EVAL_SEED="${EVAL_SEED}" "${PYTHON_BIN}" - <<'PY'
import os

import torch

torch.cuda.manual_seed_all(int(os.environ["SLURM_EVAL_SEED"]))
device = torch.device("cuda:0")
witness = torch.ones((8, 8), device=device) @ torch.ones((8, 8), device=device)
print(
    "[slurm-eval] CUDA_WITNESS",
    tuple(witness.shape),
    torch.cuda.get_device_name(0),
    torch.cuda.get_device_properties(0).uuid,
    bool(torch.isfinite(witness).all()),
)
PY

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  EVAL_MODEL="$(resolve_eval_model "${MODEL_PATH}" "${PYTHON_BIN}")"
  LABEL="$(model_label "${MODEL_PATH}")"
  MODEL_OUTPUT_DIR="${SUITE_ROOT}/${LABEL}"
  PROTOCOL_DOMAINS=(math code science)
  PROTOCOL_DATASETS=("${MATH_DATASETS}" "${CODE_DATASETS}" "${SCIENCE_DATASETS}")
  PROTOCOL_SAMPLES=("${MATH_SAMPLES}" "${CODE_SAMPLES}" "${SCIENCE_SAMPLES}")
  for DOMAIN_INDEX in "${!PROTOCOL_DOMAINS[@]}"; do
    DOMAIN="${PROTOCOL_DOMAINS[${DOMAIN_INDEX}]}"
    DOMAIN_DATASETS="${PROTOCOL_DATASETS[${DOMAIN_INDEX}]}"
    NUM_SAMPLES="${PROTOCOL_SAMPLES[${DOMAIN_INDEX}]}"
    [[ -n "${DOMAIN_DATASETS}" ]] || continue
    OUTPUT_DIR="${MODEL_OUTPUT_DIR}/${DOMAIN}"
    EXTRA_ARGS=()
    [[ -z "${MAX_SAMPLES}" ]] || EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
    [[ "${SAMPLE_OFFSET}" == "0" ]] || EXTRA_ARGS+=(--sample-offset "${SAMPLE_OFFSET}")
    [[ "${SCORE_CODE}" == "0" ]] || EXTRA_ARGS+=(--score-code)
    if [[ "${RESUME}" == "1" ]] && output_is_resumable "${OUTPUT_DIR}"; then
      EXTRA_ARGS+=(--resume)
    fi

    printf '[slurm-eval] evaluating domain=%s samples_per_problem=%s source=%s model=%s output=%s\n' \
      "${DOMAIN}" "${NUM_SAMPLES}" "${MODEL_PATH}" "${EVAL_MODEL}" "${OUTPUT_DIR}"
    bash "${CODE_DIR}/scripts/run_local_eval.sh" \
      --model-path "${EVAL_MODEL}" \
      --datasets "${DOMAIN_DATASETS}" \
      --modes non_thinking \
      --backend vllm \
      --tensor-parallel-size 1 \
      --batch-size "${SLURM_EVAL_BATCH_SIZE:-24}" \
      --gpu-memory "${SLURM_EVAL_GPU_MEMORY:-0.85}" \
      --max-model-len 18432 \
      --max-num-batched-tokens 32768 \
      --max-num-seqs 24 \
      --enforce-eager \
      --disable-chunked-prefill \
      --max-new-tokens "${EVAL_MAX_TOKENS}" \
      --num-samples "${NUM_SAMPLES}" \
      --temperature "${EVAL_TEMPERATURE}" \
      --top-p "${EVAL_TOP_P}" \
      --seed "${EVAL_SEED}" \
      --python "${PYTHON_BIN}" \
      --run-id "${LABEL}_${DOMAIN}_${RUN_TAG}" \
      --output-dir "${OUTPUT_DIR}" \
      --save-completions \
      --no-wandb \
      "${EXTRA_ARGS[@]}"
  done
done

touch "${SUITE_ROOT}/SUCCESS"
printf '[slurm-eval] complete: %s\n' "${SUITE_ROOT}"
