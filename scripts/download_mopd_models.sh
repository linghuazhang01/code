#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/download_mopd_models.sh

Downloads or verifies the model directories used by the current single-A800
config plus the optional 4B-base student run.

Defaults:
  MODEL_ROOT=<parent of OPD-code>/models
  STUDENT_MODEL_ID=Qwen/Qwen3-0.6B
  STUDENT_DIR_NAME=Qwen3-0.6B
  DOWNLOAD_STUDENT=1
  DOWNLOAD_BASE_4B=1
  DOWNLOAD_TEACHERS=1

Model directories prepared by default:
  $MODEL_ROOT/Qwen3-4B
  $MODEL_ROOT/Qwen3-4B-Non-Thinking-RL-Math-Step500
  $MODEL_ROOT/Qwen3-4B-Non-Thinking-RL-Code-Step300

Default 4B base hub id:
  BASE_4B_MODEL_ID=Qwen/Qwen3-4B

Default teacher hub ids:
  MATH_TEACHER_MODEL_ID=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
  CODE_TEACHER_MODEL_ID=Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300

Set DOWNLOAD_BASE_4B=0 to skip downloading and validating the 4B base model.
Set DOWNLOAD_TEACHERS=0 to only verify existing teacher directories.

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
MODEL_BACKEND="${MODEL_BACKEND:-huggingface}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

DOWNLOAD_STUDENT="${DOWNLOAD_STUDENT:-1}"
DOWNLOAD_BASE_4B="${DOWNLOAD_BASE_4B:-1}"
DOWNLOAD_TEACHERS="${DOWNLOAD_TEACHERS:-1}"

STUDENT_MODEL_ID="${STUDENT_MODEL_ID:-Qwen/Qwen3-0.6B}"
STUDENT_DIR_NAME="${STUDENT_DIR_NAME:-Qwen3-0.6B}"
BASE_4B_MODEL_ID="${BASE_4B_MODEL_ID:-Qwen/Qwen3-4B}"
BASE_4B_DIR_NAME="${BASE_4B_DIR_NAME:-Qwen3-4B}"
MATH_TEACHER_MODEL_ID="${MATH_TEACHER_MODEL_ID:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}"
MATH_TEACHER_DIR_NAME="${MATH_TEACHER_DIR_NAME:-Qwen3-4B-Non-Thinking-RL-Math-Step500}"
CODE_TEACHER_MODEL_ID="${CODE_TEACHER_MODEL_ID:-Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300}"
CODE_TEACHER_DIR_NAME="${CODE_TEACHER_DIR_NAME:-Qwen3-4B-Non-Thinking-RL-Code-Step300}"

export HF_ENDPOINT
export HF_XET_HIGH_PERFORMANCE
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

mkdir -p "${MODEL_ROOT}"

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

download_huggingface() {
  local repo_id="$1"
  local target_dir="$2"
  ensure_huggingface_hub
  python - "${repo_id}" "${target_dir}" <<'PY'
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
  python - "${repo_id}" "${target_dir}" <<'PY'
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
math_teacher_dir="${MODEL_ROOT}/${MATH_TEACHER_DIR_NAME}"
code_teacher_dir="${MODEL_ROOT}/${CODE_TEACHER_DIR_NAME}"

if [[ "${DOWNLOAD_STUDENT}" == "1" ]]; then
  download_repo "${STUDENT_MODEL_ID}" "${student_dir}"
fi

if [[ "${DOWNLOAD_BASE_4B}" == "1" ]]; then
  download_repo "${BASE_4B_MODEL_ID}" "${base_4b_dir}"
fi

if [[ "${DOWNLOAD_TEACHERS}" == "1" ]]; then
  download_repo "${MATH_TEACHER_MODEL_ID}" "${math_teacher_dir}"
  download_repo "${CODE_TEACHER_MODEL_ID}" "${code_teacher_dir}"
fi

validate_model_dir "student" "${student_dir}"

if [[ "${DOWNLOAD_BASE_4B}" == "1" ]]; then
  validate_model_dir "4B base" "${base_4b_dir}"
else
  echo "4B base skipped: ${base_4b_dir}"
fi

teacher_missing=0
validate_model_dir "math teacher" "${math_teacher_dir}" || teacher_missing=1
validate_model_dir "code teacher" "${code_teacher_dir}" || teacher_missing=1

if [[ "${teacher_missing}" == "1" ]]; then
  cat >&2 <<EOF
Teacher model directories are missing.

Either place the checkpoints under:
  ${math_teacher_dir}
  ${code_teacher_dir}

or rerun with:
  scripts/download_mopd_models.sh

To override the default teacher hub ids, run:
  DOWNLOAD_TEACHERS=1 \\
  MATH_TEACHER_MODEL_ID=<math-teacher-hub-id> \\
  CODE_TEACHER_MODEL_ID=<code-teacher-hub-id> \\
  scripts/download_mopd_models.sh
EOF
  exit 2
fi

echo "Models ready: ${MODEL_ROOT}"
