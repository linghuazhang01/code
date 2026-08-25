#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="$(cd "${EVAL_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${1:-}}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required." >&2
  exit 2
fi

MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
SAFE_MODEL_NAME="${MODEL_NAME//[^A-Za-z0-9_.-]/_}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/data/eval_data/results/standard_ood/${SAFE_MODEL_NAME}_${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"

DATA_FILE="${CODE_DIR}/data/eval_data/science/MMLU-Pro/subsets/openprm_style_500_seed42/test.parquet"
MANIFEST_FILE="${CODE_DIR}/data/eval_data/science/MMLU-Pro/subsets/openprm_style_500_seed42/manifest.json"
EXPECTED_DATA_SHA256="9db4fb82f4fc59ab4514b2f3a2fe54928b3fc9d11a483bf678958261b8f6a4a6"
EXPECTED_SELECTED_IDS_SHA256="ea1c19950afe4ac82a3b32c8afb39b50fa032a64e096fed61364e1d0c1c81760"

for flag in DRY_RUN RESUME; do
  [[ "${!flag}" == "0" || "${!flag}" == "1" ]] || {
    echo "${flag} must be 0 or 1: ${!flag}" >&2
    exit 2
  }
done

"${PYTHON_BIN}" - \
  "${DATA_FILE}" \
  "${MANIFEST_FILE}" \
  "${EXPECTED_DATA_SHA256}" \
  "${EXPECTED_SELECTED_IDS_SHA256}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


data_file = Path(sys.argv[1])
manifest_file = Path(sys.argv[2])
expected_data_sha256 = sys.argv[3]
expected_selected_ids_sha256 = sys.argv[4]

if not data_file.is_file() or not manifest_file.is_file():
    raise SystemExit(f"Missing pinned MMLU-Pro-500 artifact under {data_file.parent}")

actual_data_sha256 = hashlib.sha256(data_file.read_bytes()).hexdigest()
if actual_data_sha256 != expected_data_sha256:
    raise SystemExit(f"MMLU-Pro-500 data SHA-256 mismatch: {actual_data_sha256}")

manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
selection = manifest.get("selection", {})
subset = manifest.get("subset", {})
selected_ids = selection.get("selected_ids")
if not isinstance(selected_ids, list):
    raise SystemExit("MMLU-Pro-500 manifest is missing selected_ids")
selected_ids_payload = json.dumps(
    selected_ids,
    ensure_ascii=False,
    separators=(",", ":"),
).encode("utf-8")
actual_selected_ids_sha256 = hashlib.sha256(selected_ids_payload).hexdigest()

expected_fields = {
    "dataset": manifest.get("dataset") == "MMLU-Pro",
    "seed": selection.get("seed") == 42,
    "sample_size": selection.get("sample_size") == 500,
    "subset_rows": subset.get("rows") == 500,
    "subset_sha256": subset.get("sha256") == expected_data_sha256,
    "selected_ids_sha256": actual_selected_ids_sha256 == expected_selected_ids_sha256,
}
invalid = [name for name, valid in expected_fields.items() if not valid]
if invalid:
    raise SystemExit(f"Invalid MMLU-Pro-500 manifest fields: {', '.join(invalid)}")
PY

COMMAND=(
  "${CODE_DIR}/eval/scripts/run_official_eval.sh"
  --domains science
  --datasets mmlupro_500_seed42
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --tensor-parallel-size 1
  --gpu-memory-utilization 0.85
  --max-model-len 18432
  --max-tokens 16384
  --temperature 1.0
  --top-p 1.0
  --num-samples 4
  --seed 42
  --enable-thinking false
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[standard-ood] CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ "${RESUME}" == "1" && -f "${OUTPUT_DIR}/SUCCESS" ]]; then
  echo "[standard-ood] already complete, skipping: ${OUTPUT_DIR}"
  exit 0
fi
if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty output directory: ${OUTPUT_DIR}" >&2
  echo "Use a new OUTPUT_DIR; incomplete MMLU-Pro runs are not resumable." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" - \
  "${OUTPUT_DIR}/standard_ood_manifest.json" \
  "${MODEL_PATH}" \
  "${DATA_FILE}" \
  "${EXPECTED_DATA_SHA256}" \
  "${EXPECTED_SELECTED_IDS_SHA256}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


payload = {
    "benchmark": "MMLU-Pro-500",
    "dataset_key": "mmlupro_500_seed42",
    "model_path": sys.argv[2],
    "data_file": sys.argv[3],
    "data_sha256": sys.argv[4],
    "selected_ids_sha256": sys.argv[5],
    "protocol": {
        "questions": 500,
        "rollouts_per_question": 4,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": 16384,
        "max_model_len": 18432,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "enable_thinking": False,
    },
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"

"${COMMAND[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
touch "${OUTPUT_DIR}/SUCCESS"
echo "[standard-ood] complete: ${OUTPUT_DIR}"
