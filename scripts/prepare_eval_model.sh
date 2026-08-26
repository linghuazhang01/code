#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

is_hf_model() {
  local model_dir="$1"
  [[ -f "${model_dir}/config.json" ]] || return 1
  [[ -f "${model_dir}/model.safetensors" \
    || -f "${model_dir}/model.safetensors.index.json" \
    || -f "${model_dir}/pytorch_model.bin" \
    || -f "${model_dir}/pytorch_model.bin.index.json" ]]
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
  local recorded_signature
  is_hf_model "${merged_dir}" || return 1
  [[ -f "${marker}" && -f "${merged_dir}/tokenizer_config.json" ]] || return 1
  recorded_signature="$(sed -n 's/^source_signature=//p' "${marker}")"
  [[ -n "${recorded_signature}" ]] || return 1
  [[ "${recorded_signature}" == "$(source_signature "${actor_dir}")" ]]
}

MODEL_PATH=""
PYTHON_BIN="${SLURM_EVAL_PYTHON:-python3}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="${2:?--model-path requires a value}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?--python requires a value}"; shift 2 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[[ -n "${MODEL_PATH}" ]] || fail "--model-path is required"
[[ -x "${PYTHON_BIN}" ]] || fail "Python executable is not runnable: ${PYTHON_BIN}"
if is_hf_model "${MODEL_PATH}"; then
  printf '%s\n' "${MODEL_PATH}"
  exit 0
fi

ACTOR_DIR="$(resolve_actor_dir "${MODEL_PATH}")" \
  || fail "not a Hugging Face model or verl FSDP checkpoint: ${MODEL_PATH}"
MERGED_DIR="${ACTOR_DIR}/hf"
LOCK_FILE="${ACTOR_DIR}/.hf_merge.lock"
command -v flock >/dev/null 2>&1 || fail "flock is required for safe checkpoint merging"
exec 9>"${LOCK_FILE}"
flock 9

if merged_model_is_reusable "${ACTOR_DIR}" "${MERGED_DIR}"; then
  printf '[prepare-eval-model] reusing merged model: %s\n' "${MERGED_DIR}" >&2
  printf '%s\n' "${MERGED_DIR}"
  exit 0
fi
[[ ! -e "${MERGED_DIR}" ]] || fail "unverified or incomplete merge directory exists: ${MERGED_DIR}"

TEMP_DIR="${ACTOR_DIR}/hf.merge-${SLURM_JOB_ID:-manual}-$$"
[[ ! -e "${TEMP_DIR}" ]] || fail "temporary merge directory exists: ${TEMP_DIR}"
cleanup() {
  if [[ -n "${TEMP_DIR:-}" && "${TEMP_DIR}" == */hf.merge-* && -d "${TEMP_DIR}" ]]; then
    rm -rf -- "${TEMP_DIR}"
  fi
}
trap cleanup EXIT INT TERM

printf '[prepare-eval-model] merging FSDP shards: %s -> %s\n' \
  "${ACTOR_DIR}" "${MERGED_DIR}" >&2
"${PYTHON_BIN}" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${TEMP_DIR}" >&2
is_hf_model "${TEMP_DIR}" || fail "FSDP merge did not produce a loadable model: ${TEMP_DIR}"
[[ -f "${TEMP_DIR}/tokenizer_config.json" ]] \
  || fail "FSDP merge did not preserve tokenizer metadata: ${TEMP_DIR}"
SIGNATURE="$(source_signature "${ACTOR_DIR}")"
printf 'source_dir=%s\nsource_signature=%s\n' "${ACTOR_DIR}" "${SIGNATURE}" \
  >"${TEMP_DIR}/.mopd_merge_complete"
mv "${TEMP_DIR}" "${MERGED_DIR}"
TEMP_DIR=""
printf '%s\n' "${MERGED_DIR}"
