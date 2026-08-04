#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${CODE_DIR}/../models/Qwen3-1.7B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${CODE_DIR}/../models/Nemotron-Research-GooseReason-4B-Instruct}"
GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CODE_DIR}/data/eval_data/results/two_model_training_ceiling/${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Match the GooseReason training config's validation-inference settings.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-24}"
GPU_MEMORY="${GPU_MEMORY:-0.6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-18432}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

for flag in RESUME DRY_RUN; do
  [[ "${!flag}" == "0" || "${!flag}" == "1" ]] || {
    echo "${flag} must be 0 or 1: ${!flag}" >&2
    exit 2
  }
done

[[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "NUM_SAMPLES must be a positive integer: ${NUM_SAMPLES}" >&2
  exit 2
}
"${PYTHON_BIN}" - "${NUM_SAMPLES}" "${TEMPERATURE}" <<'PY'
import math
import sys

num_samples = int(sys.argv[1])
temperature = float(sys.argv[2])
if not math.isfinite(temperature) or temperature < 0:
    raise SystemExit("TEMPERATURE must be a finite non-negative number.")
if num_samples > 1 and temperature <= 0:
    raise SystemExit(
        "NUM_SAMPLES>1 requires TEMPERATURE>0 to avoid duplicate greedy rollouts."
    )
PY

export IFBENCH_REPO="${IFBENCH_REPO:-${CODE_DIR}/../temp/IFBench}"
export MOPD_ALLOW_SIMPLE_SCORER_FALLBACK=0
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

validate_training_ceiling() {
  "${PYTHON_BIN}" - "${CODE_DIR}/data/eval_training_data" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
manifest_path = data_dir / "manifest.json"
regenerate_command = (
    "python scripts/split_domain_eval_training_data.py "
    "--eval-size 10000 --seed 42 --overwrite"
)

if not manifest_path.is_file():
    raise SystemExit(
        f"Missing training ceiling manifest: {manifest_path}\n"
        f"Generate it with: {regenerate_command}"
    )

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
config = manifest.get("config", {})
if config.get("eval_size") != 10_000 or config.get("seed") != 42:
    raise SystemExit(
        "Training ceiling must use eval_size=10000 and seed=42.\n"
        f"Regenerate it with: {regenerate_command}"
    )

domains = {entry.get("domain"): entry for entry in manifest.get("domains", [])}
for domain in ("math", "code", "if", "science"):
    entry = domains.get(domain)
    parquet_path = data_dir / domain / "test.parquet"
    if entry is None or entry.get("eval_rows") != 10_000:
        raise SystemExit(
            f"Invalid or missing {domain} entry in {manifest_path}.\n"
            f"Regenerate it with: {regenerate_command}"
        )
    if not parquet_path.is_file():
        raise SystemExit(
            f"Missing training ceiling parquet: {parquet_path}\n"
            f"Regenerate it with: {regenerate_command}"
        )
    expected_hash = entry.get("eval_sha256")
    actual_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    if not expected_hash or actual_hash != expected_hash:
        raise SystemExit(
            f"Hash mismatch for {parquet_path}.\n"
            f"Regenerate the deterministic split with: {regenerate_command}"
        )

print(f"Validated deterministic training ceiling: {manifest_path}")
PY
}

validate_training_ceiling

if [[ "${DRY_RUN}" == "0" ]]; then
  "${CODE_DIR}/scripts/prepare_ifbench_runtime.sh"
fi

run_model() {
  local model_label="$1"
  local model_path="$2"
  local output_dir="${OUTPUT_ROOT}/${model_label}"
  local run_id="${model_label}_training_ceiling_${RUN_TAG}"
  local extra_args=()

  [[ -z "${MAX_SAMPLES}" ]] || extra_args+=(--max-samples "${MAX_SAMPLES}")
  [[ "${RESUME}" == "0" ]] || extra_args+=(--resume)
  [[ "${DRY_RUN}" == "0" ]] || extra_args+=(--dry-run)

  echo "[two-model-training-ceiling] model=${model_label} path=${model_path} output=${output_dir}"
  local command=(
    "${CODE_DIR}/scripts/run_local_eval.sh"
      --model-path "${model_path}" \
      --datasets training_ceiling \
      --modes non_thinking \
      --backend vllm \
      --tensor-parallel-size 1 \
      --batch-size "${BATCH_SIZE}" \
      --gpu-memory "${GPU_MEMORY}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --enforce-eager \
      --disable-chunked-prefill \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --num-samples "${NUM_SAMPLES}" \
      --temperature "${TEMPERATURE}" \
      --top-p "${TOP_P}" \
      --seed "${SEED}" \
      --python "${PYTHON_BIN}" \
      --run-id "${run_id}" \
      --output-dir "${output_dir}" \
      --score-code \
      --save-completions \
      "${extra_args[@]}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}"
    return
  fi
  if [[ "${RESUME}" == "0" && -d "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty output directory: ${output_dir}" >&2
    echo "Use a new RUN_TAG, or set RESUME=1 with identical settings." >&2
    exit 2
  fi
  mkdir -p "${output_dir}"
  local tee_args=()
  [[ "${RESUME}" == "0" ]] || tee_args+=(-a)
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}" 2>&1 \
    | tee "${tee_args[@]}" "${output_dir}/run.log"
}

run_model qwen3_1p7b "${STUDENT_MODEL_PATH}"
run_model goosereason_4b "${TEACHER_MODEL_PATH}"

echo "[two-model-training-ceiling] complete: ${OUTPUT_ROOT}"
