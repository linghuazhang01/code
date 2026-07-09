#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/download_qwen30b_teacher.sh

Downloads or verifies the Qwen3-30B-A3B teacher model used by the Qwen30B
distillation smoke configs.

Defaults:
  MODEL_ROOT=<parent of OPD-code>/models
  QWEN30B_MODEL_ID=Qwen/Qwen3-30B-A3B
  QWEN30B_DIR_NAME=Qwen3-30B-A3B
  DOWNLOAD_QWEN30B=1
  REQUIRE_QWEN30B=1
  MODEL_BACKEND=huggingface
  PYTHON_BIN=<auto-detected python or python3>
  HF_HOME=$MODEL_ROOT/.hf_home
  HF_XET_CHUNK_CACHE_SIZE_BYTES=0
  MIN_FREE_GB=0

Examples:
  scripts/download_qwen30b_teacher.sh

  MODEL_BACKEND=modelscope scripts/download_qwen30b_teacher.sh

  MODEL_ROOT=/root/autodl-tmp/opd_mopd/models \
    scripts/download_qwen30b_teacher.sh

  DOWNLOAD_QWEN30B=0 REQUIRE_QWEN30B=1 \
    scripts/download_qwen30b_teacher.sh

Backend:
  MODEL_BACKEND=huggingface | modelscope

Notes:
  - Hugging Face authentication, if needed, is read from the standard HF_TOKEN
    environment variable or existing huggingface-cli login state.
  - Set MIN_FREE_GB to a positive integer to fail early when the target
    filesystem has less free space than requested.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ROOT="${MODEL_ROOT:-$(cd "${CODE_DIR}/.." && pwd)/models}"
MODEL_BACKEND="${MODEL_BACKEND:-huggingface}"
PYTHON_BIN="${PYTHON_BIN:-}"
HF_HOME="${HF_HOME:-${MODEL_ROOT}/.hf_home}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
HF_XET_CHUNK_CACHE_SIZE_BYTES="${HF_XET_CHUNK_CACHE_SIZE_BYTES:-0}"
QWEN30B_MODEL_ID="${QWEN30B_MODEL_ID:-Qwen/Qwen3-30B-A3B}"
QWEN30B_DIR_NAME="${QWEN30B_DIR_NAME:-Qwen3-30B-A3B}"
DOWNLOAD_QWEN30B="${DOWNLOAD_QWEN30B:-1}"
REQUIRE_QWEN30B="${REQUIRE_QWEN30B:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-0}"

export HF_XET_HIGH_PERFORMANCE
export HF_HOME
export HF_XET_CHUNK_CACHE_SIZE_BYTES
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

mkdir -p "${MODEL_ROOT}"

ensure_python_bin() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    return
  fi

  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "python or python3 is required." >&2
    exit 1
  fi
}

check_disk_space() {
  if [[ "${MIN_FREE_GB}" == "0" ]]; then
    return
  fi
  if ! [[ "${MIN_FREE_GB}" =~ ^[0-9]+$ ]]; then
    echo "MIN_FREE_GB must be a non-negative integer: ${MIN_FREE_GB}" >&2
    return 2
  fi

  local available_kb
  available_kb="$(df -Pk "${MODEL_ROOT}" | awk 'NR == 2 {print $4}')"
  local required_kb=$((MIN_FREE_GB * 1024 * 1024))
  if ((available_kb < required_kb)); then
    echo "Insufficient free space under ${MODEL_ROOT}: need ${MIN_FREE_GB} GB" >&2
    return 1
  fi
}

ensure_huggingface_hub() {
  ensure_python_bin
  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1 || "${PYTHON_BIN}" -m pip install --upgrade "huggingface_hub>=0.30.0,<1.0" hf_xet
import importlib.metadata as metadata

version = metadata.version("huggingface_hub")
parts = [int(part) for part in version.split(".")[:2]]
major, minor = parts[0], parts[1] if len(parts) > 1 else 0
if major >= 1 or (major == 0 and minor < 30):
    raise SystemExit(1)
PY
}

download_huggingface() {
  local repo_id="$1"
  local target_dir="$2"
  ensure_huggingface_hub
  "${PYTHON_BIN}" - "${repo_id}" "${target_dir}" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
target_dir = Path(sys.argv[2])
target_dir.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo_id, local_dir=str(target_dir))
PY
}

download_modelscope() {
  local repo_id="$1"
  local target_dir="$2"
  ensure_python_bin
  "${PYTHON_BIN}" - "${repo_id}" "${target_dir}" <<'PY'
import sys
from pathlib import Path

try:
    from modelscope import snapshot_download
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"modelscope is required: {exc}")

repo_id = sys.argv[1]
target_dir = Path(sys.argv[2])
target_dir.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id, local_dir=str(target_dir))
PY
}

download_repo() {
  local repo_id="$1"
  local target_dir="$2"
  if [[ -z "${repo_id}" ]]; then
    echo "Missing model repo id for ${target_dir}" >&2
    return 2
  fi

  case "${MODEL_BACKEND}" in
    huggingface)
      download_huggingface "${repo_id}" "${target_dir}"
      ;;
    modelscope)
      download_modelscope "${repo_id}" "${target_dir}"
      ;;
    *)
      echo "Unsupported MODEL_BACKEND=${MODEL_BACKEND}" >&2
      return 2
      ;;
  esac
}

validate_model_dir() {
  local label="$1"
  local model_dir="$2"
  if [[ ! -f "${model_dir}/config.json" ]]; then
    echo "Missing ${label} model config: ${model_dir}/config.json" >&2
    return 1
  fi
  echo "${label} ready: ${model_dir}"
}

qwen30b_dir="${MODEL_ROOT}/${QWEN30B_DIR_NAME}"

check_disk_space

if [[ "${DOWNLOAD_QWEN30B}" == "1" ]]; then
  download_repo "${QWEN30B_MODEL_ID}" "${qwen30b_dir}"
else
  echo "Qwen30B download skipped: ${qwen30b_dir}"
fi

if [[ "${REQUIRE_QWEN30B}" == "1" ]]; then
  validate_model_dir "Qwen3-30B-A3B teacher" "${qwen30b_dir}"
else
  echo "Qwen3-30B-A3B teacher validation skipped: ${qwen30b_dir}"
fi

echo "Qwen30B teacher model ready: ${qwen30b_dir}"
