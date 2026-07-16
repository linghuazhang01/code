#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/download_mopd_data.sh

Downloads the OPD training parquet data and stages evaluation parquet data.

Environment knobs:
  DATASET_ID=icemoon28/MOPD-Training-Data
  DATASET_REVISION=main
  DATA_DIR=$CODE_DIR/data/G-OPD-Training-Data
  EVAL_DATA_DIR=$CODE_DIR/data/eval_data
  PYTHON_BIN=<auto-detected python or python3>
  GOPD_REPO_URL=http://github.com/RUCBM/G-OPD.git
  GOPD_REF=37371a4c31ad7947746200d234161769191f4748
  EVAL_SOURCE_DIR=$DATA_DIR/.eval-source/G-OPD
  DOWNLOAD_LCB=0
  LCB_DIR=$DATA_DIR/.eval-source/LiveCodeBench
  LCB_REVISION=48d36ed304dca42cf8ab20e941262ccd096518a3
  LCB_SHA256=bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5
  REQUIRE_4DOMAIN_TRAIN_DATA=1
  PULL_REPO_LFS_FALLBACK=1
  GIT_LFS_TIMEOUT_SECONDS=300
  HF_XET_HIGH_PERFORMANCE=1
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ID="${DATASET_ID:-icemoon28/MOPD-Training-Data}"
DATASET_REVISION="${DATASET_REVISION:-main}"
DATA_DIR="${DATA_DIR:-${CODE_DIR}/data/G-OPD-Training-Data}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${CODE_DIR}/data/eval_data}"
PYTHON_BIN="${PYTHON_BIN:-}"
GOPD_REPO_URL="${GOPD_REPO_URL:-http://github.com/RUCBM/G-OPD.git}"
GOPD_REF="${GOPD_REF:-37371a4c31ad7947746200d234161769191f4748}"
EVAL_SOURCE_DIR="${EVAL_SOURCE_DIR:-${DATA_DIR}/.eval-source/G-OPD}"
DOWNLOAD_LCB="${DOWNLOAD_LCB:-0}"
LCB_DIR="${LCB_DIR:-${DATA_DIR}/.eval-source/LiveCodeBench}"
LCB_REVISION="${LCB_REVISION:-48d36ed304dca42cf8ab20e941262ccd096518a3}"
LCB_SHA256="${LCB_SHA256:-bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5}"
REQUIRE_4DOMAIN_TRAIN_DATA="${REQUIRE_4DOMAIN_TRAIN_DATA:-1}"
PULL_REPO_LFS_FALLBACK="${PULL_REPO_LFS_FALLBACK:-1}"
GIT_LFS_TIMEOUT_SECONDS="${GIT_LFS_TIMEOUT_SECONDS:-300}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

export HF_XET_HIGH_PERFORMANCE
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "python or python3 is required." >&2
    exit 1
  fi
fi

mkdir -p "${DATA_DIR}"

is_lfs_pointer() {
  local file_path="$1"
  [[ -f "${file_path}" ]] || return 1
  head -c 96 "${file_path}" | grep -q "version https://git-lfs.github.com/spec"
}

ensure_huggingface_hub() {
  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1 || "${PYTHON_BIN}" -m pip install --upgrade "huggingface_hub>=0.30.0,<1.0" hf_xet
import importlib.metadata as metadata

version = metadata.version("huggingface_hub")
parts = [int(part) for part in version.split(".")[:2]]
major, minor = parts[0], parts[1] if len(parts) > 1 else 0
if major >= 1 or (major == 0 and minor < 30):
    raise SystemExit(1)
PY
}

ensure_parquet_support() {
  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1 || "${PYTHON_BIN}" -m pip install "pandas>=2.0" "pyarrow>=19.0.0"
import pandas
import pyarrow
PY
}

ensure_huggingface_hub
ensure_parquet_support
"${PYTHON_BIN}" - "${DATASET_ID}" "${DATASET_REVISION}" "${DATA_DIR}" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

dataset_id = sys.argv[1]
dataset_revision = sys.argv[2]
target_dir = Path(sys.argv[3])
target_dir.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=dataset_id,
    repo_type="dataset",
    revision=dataset_revision,
    local_dir=str(target_dir),
)
PY

pull_repo_lfs_fallback() {
  if [[ "${PULL_REPO_LFS_FALLBACK}" != "1" || "${REQUIRE_4DOMAIN_TRAIN_DATA}" != "1" ]]; then
    return
  fi

  local if_path="${DATA_DIR}/IF/train.parquet"
  local science_path="${DATA_DIR}/Science/train.parquet"
  if ! is_lfs_pointer "${if_path}" && ! is_lfs_pointer "${science_path}"; then
    return
  fi

  if ! command -v git >/dev/null 2>&1 || ! git -C "${CODE_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "IF/science training files are Git LFS pointers, but this is not a git checkout." >&2
    return
  fi
  if ! git -C "${CODE_DIR}" lfs version >/dev/null 2>&1; then
    echo "IF/science training files are Git LFS pointers, but git-lfs is unavailable." >&2
    echo "Install git-lfs or provide real parquet files at ${if_path} and ${science_path}." >&2
    return
  fi
  if ! [[ "${GIT_LFS_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "GIT_LFS_TIMEOUT_SECONDS must be a non-negative integer: ${GIT_LFS_TIMEOUT_SECONDS}" >&2
    return 2
  fi

  local include_paths="data/G-OPD-Training-Data/IF/train.parquet,data/G-OPD-Training-Data/Science/train.parquet"
  echo "IF/science training files are Git LFS pointers; pulling repo LFS objects."
  if [[ "${GIT_LFS_TIMEOUT_SECONDS}" == "0" ]] || ! command -v timeout >/dev/null 2>&1; then
    git -C "${CODE_DIR}" lfs pull --include "${include_paths}"
  else
    timeout "${GIT_LFS_TIMEOUT_SECONDS}" git -C "${CODE_DIR}" lfs pull --include "${include_paths}" || {
      local status=$?
      if [[ "${status}" == "124" ]]; then
        echo "git lfs pull timed out after ${GIT_LFS_TIMEOUT_SECONDS}s." >&2
      fi
      return "${status}"
    }
  fi
}

pull_repo_lfs_fallback

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

EVAL_SOURCE_DIR="${EVAL_SOURCE_DIR}" EVAL_DATA_DIR="${EVAL_DATA_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from eval.data_prep.paper_eval import (
    PAPER_CODE_EVAL_SPECS,
    PAPER_MATH_EVAL_SPECS,
    evalplus_jsonl_to_verl_parquet,
    math_eval_jsonl_to_verl_parquet,
)

source_root = Path(os.environ["EVAL_SOURCE_DIR"])
output_root = Path(os.environ["EVAL_DATA_DIR"])

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
  "${PYTHON_BIN}" - "${LCB_DIR}" "${LCB_REVISION}" <<'PY'
import sys

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="livecodebench/code_generation_lite",
    repo_type="dataset",
    local_dir=sys.argv[1],
    revision=sys.argv[2],
    allow_patterns=["test6.jsonl"],
)
PY
  LCB_DIR="${LCB_DIR}" LCB_REVISION="${LCB_REVISION}" LCB_SHA256="${LCB_SHA256}" EVAL_DATA_DIR="${EVAL_DATA_DIR}" "${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from eval.data_prep.paper_eval import lcb_jsonl_to_verl_parquet

source_root = Path(os.environ["LCB_DIR"])
output_path = Path(os.environ["EVAL_DATA_DIR"]) / "code/LiveCodeBench/test.parquet"
source_paths = [source_root / "test6.jsonl"]
if not source_paths[0].is_file():
    raise SystemExit(f"No LiveCodeBench v6 shard found in {source_root}")
digest = hashlib.sha256()
with source_paths[0].open("rb") as handle:
    while chunk := handle.read(8 * 1024 * 1024):
        digest.update(chunk)
source_sha256 = digest.hexdigest()
if source_sha256 != os.environ["LCB_SHA256"]:
    raise SystemExit(f"LiveCodeBench v6 SHA-256 mismatch: {source_sha256}")
count = lcb_jsonl_to_verl_parquet(source_paths, output_path)
manifest = {
    "dataset": "livecodebench/code_generation_lite",
    "evaluation_tests": "public+private",
    "release_version": "v6",
    "revision": os.environ["LCB_REVISION"],
    "rows": count,
    "source_file": source_paths[0].name,
    "source_sha256": source_sha256,
}
(output_path.parent / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"LiveCodeBench: {count} rows -> {output_path}")
PY
fi

required_files=(
  "DeepMath-103K/train_filtered_level6.parquet"
  "Eurus/code_train.parquet"
)
if [[ "${REQUIRE_4DOMAIN_TRAIN_DATA}" == "1" ]]; then
  required_files+=(
    "IF/train.parquet"
    "Science/train.parquet"
  )
fi

missing=0
for relative_path in "${required_files[@]}"; do
  if [[ ! -f "${DATA_DIR}/${relative_path}" ]]; then
    echo "Missing required data file: ${DATA_DIR}/${relative_path}" >&2
    missing=1
  fi
done

eval_required_files=(
  "math/AIME24/test.parquet"
  "math/AIME25/test.parquet"
  "math/HMMT25Feb/test.parquet"
  "math/HMMT25Nov/test.parquet"
  "code/HumanEvalPlus/test.parquet"
  "code/MBPPPlus/test.parquet"
)
if [[ "${DOWNLOAD_LCB}" == "1" ]]; then
  eval_required_files+=("code/LiveCodeBench/test.parquet")
fi
for relative_path in "${eval_required_files[@]}"; do
  if [[ ! -f "${EVAL_DATA_DIR}/${relative_path}" ]]; then
    echo "Missing required eval data file: ${EVAL_DATA_DIR}/${relative_path}" >&2
    missing=1
  fi
done

if [[ "${missing}" == "1" ]]; then
  exit 2
fi

echo "Training data ready: ${DATA_DIR}"
echo "Evaluation data ready: ${EVAL_DATA_DIR}"
