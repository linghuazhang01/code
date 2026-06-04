#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/download_mopd_data.sh

Downloads the OPD training/evaluation parquet data into this checkout.

Environment knobs:
  DATASET_ID=Keven16/G-OPD-Training-Data
  DATA_DIR=$CODE_DIR/data/G-OPD-Training-Data
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
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

export HF_ENDPOINT
export HF_XET_HIGH_PERFORMANCE
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

ensure_huggingface_hub
python - "${DATASET_ID}" "${DATA_DIR}" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

dataset_id = sys.argv[1]
target_dir = Path(sys.argv[2])
target_dir.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=dataset_id, repo_type="dataset", local_dir=str(target_dir))
PY

required_files=(
  "DeepMath-103K/train_filtered_level6.parquet"
  "Eurus/code_train.parquet"
  "PaperEval/AIME24/test.parquet"
  "PaperEval/AIME25/test.parquet"
  "PaperEval/HMMT25Feb/test.parquet"
  "PaperEval/HMMT25Nov/test.parquet"
  "PaperEval/HumanEvalPlus/test.parquet"
  "PaperEval/MBPPPlus/test.parquet"
  "PaperEval/LiveCodeBench/test.parquet"
)

missing=0
for relative_path in "${required_files[@]}"; do
  if [[ ! -f "${DATA_DIR}/${relative_path}" ]]; then
    echo "Missing required data file: ${DATA_DIR}/${relative_path}" >&2
    missing=1
  fi
done

if [[ "${missing}" == "1" ]]; then
  exit 2
fi

echo "Data ready: ${DATA_DIR}"
