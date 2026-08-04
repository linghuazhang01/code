#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/download_qwen1p7b_goosereason_data.sh

Download or validate the math, code, IF, and science parquet assets referenced by
the Qwen3-1.7B non-thinking to GooseReason-4B-Instruct distillation configs.

Environment knobs:
  DATA_DIR=$CODE_DIR/data/G-OPD-Training-Data
  EVAL_DATA_DIR=$CODE_DIR/data/eval_data
  DATASET_ID=icemoon28/MOPD-Training-Data
  DATASET_REVISION=main
  GOPD_REPO_URL=https://github.com/RUCBM/G-OPD.git
  GOPD_REF=37371a4c31ad7947746200d234161769191f4748
  PYTHON_BIN=<auto-detected python or python3>
  DOWNLOAD_DATA=1
  DOWNLOAD_LCB=0

The GPQA parquet is versioned under data/eval_data/science/GPQA and is checked
alongside the assets prepared by scripts/download_mopd_data.sh.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${CODE_DIR}/data/G-OPD-Training-Data}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${CODE_DIR}/data/eval_data}"
DATASET_ID="${DATASET_ID:-icemoon28/MOPD-Training-Data}"
DATASET_REVISION="${DATASET_REVISION:-main}"
GOPD_REPO_URL="${GOPD_REPO_URL:-https://github.com/RUCBM/G-OPD.git}"
GOPD_REF="${GOPD_REF:-37371a4c31ad7947746200d234161769191f4748}"
PYTHON_BIN="${PYTHON_BIN:-}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
DOWNLOAD_LCB="${DOWNLOAD_LCB:-0}"

for binary_flag in DOWNLOAD_DATA DOWNLOAD_LCB; do
  if [[ "${!binary_flag}" != "0" && "${!binary_flag}" != "1" ]]; then
    echo "${binary_flag} must be 0 or 1: ${!binary_flag}" >&2
    exit 2
  fi
done

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

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
  DATA_DIR="${DATA_DIR}" \
  EVAL_DATA_DIR="${EVAL_DATA_DIR}" \
  DATASET_ID="${DATASET_ID}" \
  DATASET_REVISION="${DATASET_REVISION}" \
  GOPD_REPO_URL="${GOPD_REPO_URL}" \
  GOPD_REF="${GOPD_REF}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  DOWNLOAD_LCB="${DOWNLOAD_LCB}" \
  REQUIRE_4DOMAIN_TRAIN_DATA=1 \
    bash "${SCRIPT_DIR}/download_mopd_data.sh"
else
  echo "Data download skipped; validating existing assets."
fi

"${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1 || \
  "${PYTHON_BIN}" -m pip install "pyarrow>=19.0.0"
import pyarrow.parquet
PY

"${PYTHON_BIN}" - "${DATA_DIR}" "${EVAL_DATA_DIR}" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq


def validate_parquet(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing required parquet asset: {path}")
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:
        raise SystemExit(f"Unreadable parquet asset {path}: {exc}") from exc
    row_count = parquet_file.metadata.num_rows
    if row_count <= 0:
        raise SystemExit(f"Empty parquet asset: {path}")
    columns = set(parquet_file.schema_arrow.names)
    if "prompt" not in columns:
        raise SystemExit(f"Parquet asset lacks required prompt column: {path}")
    print(f"parquet ready: {path} rows={row_count}")


train_root = Path(sys.argv[1])
eval_root = Path(sys.argv[2])
required_paths = [
    train_root / "DeepMath-103K/train_filtered_level6.parquet",
    train_root / "Eurus/code_train.parquet",
    train_root / "IF/train.parquet",
    train_root / "Science/train.parquet",
    eval_root / "math/AIME24/test.parquet",
    eval_root / "math/AIME25/test.parquet",
    eval_root / "math/HMMT25Feb/test.parquet",
    eval_root / "math/HMMT25Nov/test.parquet",
    eval_root / "code/HumanEvalPlus/test.parquet",
    eval_root / "code/MBPPPlus/test.parquet",
    eval_root / "if/IFBench/test.parquet",
    eval_root / "science/GPQA/test.parquet",
]
for required_path in required_paths:
    validate_parquet(required_path)
PY

echo "Qwen1.7B/GooseReason four-domain training and validation data ready."
