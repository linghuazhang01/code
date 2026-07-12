#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/download_mopd_models.sh

Downloads or verifies reusable MOPD model directories. The current Qwen30B
four-domain asset flow calls this script for the Qwen3-4B student/base model
and uses scripts/download_qwen30b_teacher.sh for the Qwen3-30B-A3B teacher.

Defaults:
  MODEL_ROOT=<parent of OPD-code>/models
  STUDENT_MODEL_ID=Qwen/Qwen3-0.6B
  STUDENT_DIR_NAME=Qwen3-0.6B
  DOWNLOAD_STUDENT=1
  DOWNLOAD_BASE_4B=1
  DOWNLOAD_REASONING_TEACHER=0
  DOWNLOAD_REASONING_BASE_14B=0
  REQUIRE_STUDENT=$DOWNLOAD_STUDENT
  REQUIRE_BASE_4B=$DOWNLOAD_BASE_4B
  REQUIRE_REASONING_TEACHER=$DOWNLOAD_REASONING_TEACHER

Model directories prepared by default:
  $MODEL_ROOT/Qwen3-0.6B
  $MODEL_ROOT/Qwen3-4B

Python:
  PYTHON_BIN=<auto-detected python or python3>

Default 4B base hub id:
  BASE_4B_MODEL_ID=Qwen/Qwen3-4B

Default General-Reasoner 14B hub ids:
  BASE_14B_MODEL_ID=Qwen/Qwen3-14B
  REASONING_TEACHER_MODEL_ID=TIGER-Lab/General-Reasoner-Qwen3-14B

Set DOWNLOAD_BASE_4B=0 to skip downloading and validating the 4B base model.
Set DOWNLOAD_REASONING_TEACHER=1 to prepare General-Reasoner-Qwen3-14B.
Set DOWNLOAD_REASONING_BASE_14B=1 only when the reasoning teacher needs the
Qwen3-14B base checkpoint separately.
For a General-Reasoner-only MOPD run, set DOWNLOAD_STUDENT=0,
REQUIRE_STUDENT=0.

Backend:
  MODEL_BACKEND=huggingface | modelscope
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ROOT="${MODEL_ROOT:-$(cd "${CODE_DIR}/.." && pwd)/models}"
PYTHON_BIN="${PYTHON_BIN:-}"
MODEL_BACKEND="${MODEL_BACKEND:-huggingface}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

DOWNLOAD_STUDENT="${DOWNLOAD_STUDENT:-1}"
DOWNLOAD_BASE_4B="${DOWNLOAD_BASE_4B:-1}"
DOWNLOAD_REASONING_TEACHER="${DOWNLOAD_REASONING_TEACHER:-0}"
DOWNLOAD_REASONING_BASE_14B="${DOWNLOAD_REASONING_BASE_14B:-0}"
REQUIRE_STUDENT="${REQUIRE_STUDENT:-${DOWNLOAD_STUDENT}}"
REQUIRE_BASE_4B="${REQUIRE_BASE_4B:-${DOWNLOAD_BASE_4B}}"
REQUIRE_REASONING_TEACHER="${REQUIRE_REASONING_TEACHER:-${DOWNLOAD_REASONING_TEACHER}}"

STUDENT_MODEL_ID="${STUDENT_MODEL_ID:-Qwen/Qwen3-0.6B}"
STUDENT_DIR_NAME="${STUDENT_DIR_NAME:-Qwen3-0.6B}"
BASE_4B_MODEL_ID="${BASE_4B_MODEL_ID:-Qwen/Qwen3-4B}"
BASE_4B_DIR_NAME="${BASE_4B_DIR_NAME:-Qwen3-4B}"
BASE_14B_MODEL_ID="${BASE_14B_MODEL_ID:-Qwen/Qwen3-14B}"
BASE_14B_DIR_NAME="${BASE_14B_DIR_NAME:-Qwen3-14B}"
REASONING_TEACHER_MODEL_ID="${REASONING_TEACHER_MODEL_ID:-TIGER-Lab/General-Reasoner-Qwen3-14B}"
REASONING_TEACHER_DIR_NAME="${REASONING_TEACHER_DIR_NAME:-General-Reasoner-Qwen3-14B}"

export HF_XET_HIGH_PERFORMANCE
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

student_dir="${MODEL_ROOT}/${STUDENT_DIR_NAME}"
base_4b_dir="${MODEL_ROOT}/${BASE_4B_DIR_NAME}"
base_14b_dir="${MODEL_ROOT}/${BASE_14B_DIR_NAME}"
reasoning_teacher_dir="${MODEL_ROOT}/${REASONING_TEACHER_DIR_NAME}"

if [[ "${DOWNLOAD_STUDENT}" == "1" ]]; then
  download_repo "${STUDENT_MODEL_ID}" "${student_dir}"
fi

if [[ "${DOWNLOAD_BASE_4B}" == "1" ]]; then
  download_repo "${BASE_4B_MODEL_ID}" "${base_4b_dir}"
fi

if [[ "${DOWNLOAD_REASONING_BASE_14B}" == "1" ]]; then
  download_repo "${BASE_14B_MODEL_ID}" "${base_14b_dir}"
fi

if [[ "${DOWNLOAD_REASONING_TEACHER}" == "1" ]]; then
  download_repo "${REASONING_TEACHER_MODEL_ID}" "${reasoning_teacher_dir}"
fi

if [[ "${REQUIRE_STUDENT}" == "1" ]]; then
  validate_model_dir "student" "${student_dir}"
else
  echo "student validation skipped: ${student_dir}"
fi

if [[ "${REQUIRE_BASE_4B}" == "1" ]]; then
  validate_model_dir "4B base" "${base_4b_dir}"
else
  echo "4B base skipped: ${base_4b_dir}"
fi

if [[ "${DOWNLOAD_REASONING_BASE_14B}" == "1" ]]; then
  validate_model_dir "14B base" "${base_14b_dir}"
else
  echo "14B base skipped: ${base_14b_dir}"
fi

if [[ "${REQUIRE_REASONING_TEACHER}" == "1" ]]; then
  validate_model_dir "reasoning teacher" "${reasoning_teacher_dir}"
fi

echo "Models ready: ${MODEL_ROOT}"
