#!/usr/bin/env bash
set -euo pipefail

export LANG=C
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
CODE_DIR="$(cd "${CODE_DIR}" && pwd -P)"
REMOTE_ROOT="$(cd "${CODE_DIR}/.." && pwd -P)"
PYTHON_BIN="${SLURM_EVAL_PYTHON:-/home/shuang_qiu/env/miniconda3/envs/mopd-verl/bin/python}"
G_OPD_DIR="${G_OPD_DIR:-${REMOTE_ROOT}/G-OPD}"

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

MODEL_PATH=""
OUTPUT_ROOT=""
RUN_TAG=""
LOCAL_ARCHIVE=""
RESUME=0
GPU_COUNT=""
REFERENCE_ANCHOR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="${2:?--model_path requires a value}"; shift 2 ;;
    --output_root) OUTPUT_ROOT="${2:?--output_root requires a value}"; shift 2 ;;
    --run_tag) RUN_TAG="${2:?--run_tag requires a value}"; shift 2 ;;
    --local_archive) LOCAL_ARCHIVE="${2:?--local_archive requires a value}"; shift 2 ;;
    --gpus) GPU_COUNT="${2:?--gpus requires a value}"; shift 2 ;;
    --reference_anchor|--reference-anchor) REFERENCE_ANCHOR=1; shift ;;
    --resume) RESUME=1; shift ;;
    *) fail "unknown standard worker argument: $1" ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "standard worker must run inside Slurm"
[[ -n "${MODEL_PATH}" && -n "${OUTPUT_ROOT}" && -n "${RUN_TAG}" ]] \
  || fail "model_path, output_root, and run_tag are required"
[[ -d "${MODEL_PATH}" ]] || fail "checkpoint is missing: ${MODEL_PATH}"
if [[ "${REFERENCE_ANCHOR}" == "0" ]]; then
  [[ "$(basename "${MODEL_PATH}")" == "global_step_60" ]] \
    || fail "standard worker requires global_step_60 or --reference-anchor"
fi
[[ -x "${PYTHON_BIN}" ]] || fail "Python executable is not runnable: ${PYTHON_BIN}"

IFS=',' read -r -a VISIBLE_DEVICE_ARRAY <<<"${CUDA_VISIBLE_DEVICES:-}"
[[ "${GPU_COUNT}" == "2" || "${GPU_COUNT}" == "3" || "${GPU_COUNT}" == "4" ]] \
  || fail "standard evaluation requires an explicit GPU count of 2, 3, or 4"
[[ "${#VISIBLE_DEVICE_ARRAY[@]}" == "${GPU_COUNT}" ]] \
  || fail "standard evaluation expected ${GPU_COUNT} Slurm GPUs"
declare -A SEEN_GPUS=()
for device in "${VISIBLE_DEVICE_ARRAY[@]}"; do
  [[ -n "${device}" && -z "${SEEN_GPUS[${device}]:-}" ]] \
    || fail "standard evaluation requires ${GPU_COUNT} unique GPU IDs"
  SEEN_GPUS["${device}"]=1
done

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REMOTE_ROOT}/hf_home}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONINTMAXSTRDIGITS=0
STANDARD_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
if [[ "${RESUME}" == "1" && -f "${STANDARD_ROOT}/STANDARD_SUCCESS" ]]; then
  printf '[standard10] already complete: %s\n' "${STANDARD_ROOT}"
  exit 0
fi

EVAL_MODEL="$(
  bash "${CODE_DIR}/scripts/prepare_eval_model.sh" \
    --model-path "${MODEL_PATH}" \
    --python "${PYTHON_BIN}"
)"
COMMON_MANIFEST_ARGS=(
  --suite-root "${STANDARD_ROOT}"
  --model-path "${MODEL_PATH}"
  --eval-model-path "${EVAL_MODEL}"
  --run-tag "${RUN_TAG}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --remote-host "$(hostname)"
  --local-archive "${LOCAL_ARCHIVE}"
  --gopd-dir "${G_OPD_DIR}"
  --gpu-count "${GPU_COUNT}"
)
[[ "${REFERENCE_ANCHOR}" == "0" ]] || COMMON_MANIFEST_ARGS+=(--reference-anchor)
INITIALIZE_ARGS=("${COMMON_MANIFEST_ARGS[@]}")
[[ "${RESUME}" == "0" ]] || INITIALIZE_ARGS+=(--resume)
"${PYTHON_BIN}" -m eval.standard_suite initialize "${INITIALIZE_ARGS[@]}"

"${PYTHON_BIN}" - <<'PY' "${G_OPD_DIR}"
from pathlib import Path
import sys

from transformers import AutoTokenizer

from eval.standard_suite import LCB_SOURCE_SHA256, file_sha256

root = Path(sys.argv[1]) / "code_eval/coding/LiveCodeBench/code_generation_lite"
for release in ("v5", "v6"):
    source = root / f"test{release[1:]}.jsonl"
    if not source.is_file():
        raise FileNotFoundError(source)
    if file_sha256(source) != LCB_SOURCE_SHA256[release]:
        raise ValueError(f"LiveCodeBench {release} source hash mismatch: {source}")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-4B",
    local_files_only=True,
    trust_remote_code=True,
)
if len(tokenizer) != 151669:
    raise ValueError(f"Unexpected Qwen3 formatter tokenizer size: {len(tokenizer)}")
print("[standard10] LCB_SOURCE_WITNESS v5=167 v6=175")
print(f"[standard10] LCB_TOKENIZER_WITNESS class={type(tokenizer).__name__} size={len(tokenizer)}")
PY

PARALLEL_ARGS=(
  --model_path "${MODEL_PATH}"
  --eval_model_path "${EVAL_MODEL}"
  --gopd_dir "${G_OPD_DIR}"
  --datasets "aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus,lcb_v5,lcb_v6,gpqa_diamond,mmlupro_500_seed42"
  --output_root "${STANDARD_ROOT}"
  --run_tag parallel
  --gpus "${GPU_COUNT}"
  --shards_per_dataset 16
  --min_rows_per_shard 1
  --standard_protocol
)
[[ "${REFERENCE_ANCHOR}" == "0" ]] || PARALLEL_ARGS+=(--reference_anchor)
[[ "${RESUME}" == "0" ]] || PARALLEL_ARGS+=(--resume)
SLURM_EVAL_MATH_SAMPLES=8 \
SLURM_EVAL_CODE_SAMPLES=8 \
SLURM_EVAL_SCIENCE_SAMPLES=8 \
SLURM_EVAL_SEED=42 \
MOPD_ALLOW_SIMPLE_SCORER_FALLBACK=0 \
  bash "${CODE_DIR}/scripts/slurm_parallel_eval_worker.sh" "${PARALLEL_ARGS[@]}"

"${PYTHON_BIN}" -m eval.standard_suite finalize "${COMMON_MANIFEST_ARGS[@]}"
printf '[standard10] complete output=%s\n' "${STANDARD_ROOT}"
