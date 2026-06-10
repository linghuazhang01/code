#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/download_mopd_data.sh

Downloads the OPD training parquet data and stages evaluation parquet data.

Environment knobs:
  DATASET_ID=Keven16/G-OPD-Training-Data
  DATA_DIR=$CODE_DIR/data/G-OPD-Training-Data
  EVAL_DOMAIN_DIR=$CODE_DIR/eval/domains
  GOPD_REPO_URL=https://github.com/RUCBM/G-OPD.git
  GOPD_REF=37371a4c31ad7947746200d234161769191f4748
  EVAL_SOURCE_DIR=$DATA_DIR/.eval-source/G-OPD
  DOWNLOAD_LCB=0
  LCB_DIR=$DATA_DIR/.eval-source/LiveCodeBench
  HF_ENDPOINT=https://hf-mirror.com
  HF_XET_HIGH_PERFORMANCE=1
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ID="${DATASET_ID:-Keven16/G-OPD-Training-Data}"
DATA_DIR="${DATA_DIR:-${CODE_DIR}/data/G-OPD-Training-Data}"
EVAL_DOMAIN_DIR="${EVAL_DOMAIN_DIR:-${CODE_DIR}/eval/domains}"
GOPD_REPO_URL="${GOPD_REPO_URL:-https://github.com/RUCBM/G-OPD.git}"
GOPD_REF="${GOPD_REF:-37371a4c31ad7947746200d234161769191f4748}"
EVAL_SOURCE_DIR="${EVAL_SOURCE_DIR:-${DATA_DIR}/.eval-source/G-OPD}"
DOWNLOAD_LCB="${DOWNLOAD_LCB:-0}"
LCB_DIR="${LCB_DIR:-${DATA_DIR}/.eval-source/LiveCodeBench}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

export HF_ENDPOINT
export HF_XET_HIGH_PERFORMANCE
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

mkdir -p "${DATA_DIR}"

ensure_huggingface_hub() {
  python - <<'PY' >/dev/null 2>&1 || python -m pip install --upgrade "huggingface_hub>=0.30.0,<1.0" hf_xet
import importlib.metadata as metadata

version = metadata.version("huggingface_hub")
parts = [int(part) for part in version.split(".")[:2]]
major, minor = parts[0], parts[1] if len(parts) > 1 else 0
if major >= 1 or (major == 0 and minor < 30):
    raise SystemExit(1)
PY
}

ensure_parquet_support() {
  python - <<'PY' >/dev/null 2>&1 || python -m pip install "pandas>=2.0" "pyarrow>=19.0.0"
import pandas
import pyarrow
PY
}

ensure_huggingface_hub
ensure_parquet_support
python - "${DATASET_ID}" "${DATA_DIR}" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

dataset_id = sys.argv[1]
target_dir = Path(sys.argv[2])
target_dir.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=dataset_id, repo_type="dataset", local_dir=str(target_dir))
PY

eval_source_ready() {
  local required_files=(
    "data/aime24/test.jsonl"
    "data/aime25/test.jsonl"
    "data/hmmt25_feb/test.jsonl"
    "data/hmmt25_nov/test.jsonl"
    "code_eval/data/HumanEvalPlus.jsonl"
    "code_eval/data/MbppPlus.jsonl"
  )
  local relative_path
  for relative_path in "${required_files[@]}"; do
    if [[ ! -f "${EVAL_SOURCE_DIR}/${relative_path}" ]]; then
      return 1
    fi
  done
}

prepare_eval_source() {
  if eval_source_ready; then
    echo "Using cached G-OPD eval sources: ${EVAL_SOURCE_DIR}"
    return
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "git is required to download G-OPD evaluation sources." >&2
    return 1
  fi

  if [[ -d "${EVAL_SOURCE_DIR}/.git" ]]; then
    git -C "${EVAL_SOURCE_DIR}" sparse-checkout init --cone
    git -C "${EVAL_SOURCE_DIR}" sparse-checkout set \
      data/aime24 \
      data/aime25 \
      data/hmmt25_feb \
      data/hmmt25_nov \
      code_eval/data
    git -C "${EVAL_SOURCE_DIR}" fetch --depth 1 origin "${GOPD_REF}"
    git -C "${EVAL_SOURCE_DIR}" checkout --detach FETCH_HEAD
  else
    if [[ -e "${EVAL_SOURCE_DIR}" ]]; then
      echo "Incomplete non-git eval source directory: ${EVAL_SOURCE_DIR}" >&2
      echo "Move it aside and rerun this script." >&2
      return 1
    fi
    mkdir -p "$(dirname "${EVAL_SOURCE_DIR}")"
    git clone \
      --filter=blob:none \
      --no-checkout \
      "${GOPD_REPO_URL}" \
      "${EVAL_SOURCE_DIR}"
    git -C "${EVAL_SOURCE_DIR}" sparse-checkout init --cone
    git -C "${EVAL_SOURCE_DIR}" sparse-checkout set \
      data/aime24 \
      data/aime25 \
      data/hmmt25_feb \
      data/hmmt25_nov \
      code_eval/data
    git -C "${EVAL_SOURCE_DIR}" fetch --depth 1 origin "${GOPD_REF}"
    git -C "${EVAL_SOURCE_DIR}" checkout --detach FETCH_HEAD
  fi

  if ! eval_source_ready; then
    echo "G-OPD evaluation source checkout is incomplete: ${EVAL_SOURCE_DIR}" >&2
    return 1
  fi
}

prepare_eval_source

EVAL_SOURCE_DIR="${EVAL_SOURCE_DIR}" EVAL_DOMAIN_DIR="${EVAL_DOMAIN_DIR}" python - <<'PY'
import os
from pathlib import Path

from eval.data_prep.paper_eval import (
    PAPER_CODE_EVAL_SPECS,
    PAPER_MATH_EVAL_SPECS,
    evalplus_jsonl_to_verl_parquet,
    math_eval_jsonl_to_verl_parquet,
)

source_root = Path(os.environ["EVAL_SOURCE_DIR"])
output_root = Path(os.environ["EVAL_DOMAIN_DIR"])

for _, (data_source, source_path, output_path) in PAPER_MATH_EVAL_SPECS.items():
    count = math_eval_jsonl_to_verl_parquet(
        source_root / source_path,
        output_root / output_path,
        data_source,
    )
    print(f"{data_source}: {count} rows -> {output_root / output_path}")

for _, (data_source, source_path, output_path) in PAPER_CODE_EVAL_SPECS.items():
    count = evalplus_jsonl_to_verl_parquet(
        source_root / source_path,
        output_root / output_path,
        data_source,
    )
    print(f"{data_source}: {count} rows -> {output_root / output_path}")
PY

if [[ "${DOWNLOAD_LCB}" == "1" ]]; then
  mkdir -p "${LCB_DIR}"
  python - "${LCB_DIR}" <<'PY'
import sys

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="livecodebench/code_generation_lite",
    repo_type="dataset",
    local_dir=sys.argv[1],
    allow_patterns=["test*.jsonl"],
)
PY
  LCB_DIR="${LCB_DIR}" EVAL_DOMAIN_DIR="${EVAL_DOMAIN_DIR}" python - <<'PY'
import os
from pathlib import Path

from eval.data_prep.paper_eval import lcb_jsonl_to_verl_parquet

source_root = Path(os.environ["LCB_DIR"])
output_path = Path(os.environ["EVAL_DOMAIN_DIR"]) / "code/data/LiveCodeBench/test.parquet"
source_paths = sorted(source_root.glob("test*.jsonl"))
if not source_paths:
    raise SystemExit(f"No LiveCodeBench test shards found in {source_root}")
count = lcb_jsonl_to_verl_parquet(source_paths, output_path)
print(f"LiveCodeBench: {count} rows -> {output_path}")
PY
fi

required_files=(
  "DeepMath-103K/train_filtered_level6.parquet"
  "Eurus/code_train.parquet"
)

missing=0
for relative_path in "${required_files[@]}"; do
  if [[ ! -f "${DATA_DIR}/${relative_path}" ]]; then
    echo "Missing required data file: ${DATA_DIR}/${relative_path}" >&2
    missing=1
  fi
done

eval_required_files=(
  "math/data/AIME24/test.parquet"
  "math/data/AIME25/test.parquet"
  "math/data/HMMT25Feb/test.parquet"
  "math/data/HMMT25Nov/test.parquet"
  "code/data/HumanEvalPlus/test.parquet"
  "code/data/MBPPPlus/test.parquet"
)
if [[ "${DOWNLOAD_LCB}" == "1" ]]; then
  eval_required_files+=("code/data/LiveCodeBench/test.parquet")
fi
for relative_path in "${eval_required_files[@]}"; do
  if [[ ! -f "${EVAL_DOMAIN_DIR}/${relative_path}" ]]; then
    echo "Missing required eval data file: ${EVAL_DOMAIN_DIR}/${relative_path}" >&2
    missing=1
  fi
done

if [[ "${missing}" == "1" ]]; then
  exit 2
fi

echo "Training data ready: ${DATA_DIR}"
echo "Evaluation data ready: ${EVAL_DOMAIN_DIR}"
