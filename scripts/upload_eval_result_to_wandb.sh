#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR=""
PROJECT="mopd-eval"
ENTITY=""
GROUP=""
MODE="online"
UPLOAD_RAW=1
TIMEOUT_SECONDS=1800
ENV_FILE="${CODE_DIR}/.env.local"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="${2:?--python requires a value}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a value}"; shift 2 ;;
    --project) PROJECT="${2:?--project requires a value}"; shift 2 ;;
    --entity) ENTITY="${2:?--entity requires a value}"; shift 2 ;;
    --group) GROUP="${2:?--group requires a value}"; shift 2 ;;
    --mode) MODE="${2:?--mode requires a value}"; shift 2 ;;
    --upload-raw) UPLOAD_RAW="${2:?--upload-raw requires 0 or 1}"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="${2:?--timeout-seconds requires a value}"; shift 2 ;;
    --env-file) ENV_FILE="${2:?--env-file requires a value}"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${OUTPUT_DIR}" ]] || { echo "--output-dir is required" >&2; exit 2; }
[[ "${MODE}" == "online" || "${MODE}" == "offline" || "${MODE}" == "disabled" ]] || {
  echo "--mode must be online, offline, or disabled" >&2
  exit 2
}
[[ "${UPLOAD_RAW}" == "0" || "${UPLOAD_RAW}" == "1" ]] || {
  echo "--upload-raw must be 0 or 1" >&2
  exit 2
}
[[ "${TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] || {
  echo "--timeout-seconds must be a non-negative integer" >&2
  exit 2
}

UPLOAD_COMMAND=(
  "${PYTHON_BIN}" -m eval.wandb_upload
  --output-dir "${OUTPUT_DIR}"
  --project "${PROJECT}"
  --mode "${MODE}"
  --env-file "${ENV_FILE}"
)
[[ -z "${ENTITY}" ]] || UPLOAD_COMMAND+=(--entity "${ENTITY}")
[[ -z "${GROUP}" ]] || UPLOAD_COMMAND+=(--group "${GROUP}")
if [[ "${UPLOAD_RAW}" == "1" ]]; then
  UPLOAD_COMMAND+=(--upload-raw)
else
  UPLOAD_COMMAND+=(--no-upload-raw)
fi

"${PYTHON_BIN}" - "${TIMEOUT_SECONDS}" "${CODE_DIR}" "${OUTPUT_DIR}" \
  "${UPLOAD_COMMAND[@]}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

timeout_seconds = int(sys.argv[1])
code_dir = sys.argv[2]
output_dir = Path(sys.argv[3])
command = sys.argv[4:]
try:
    completed = subprocess.run(
        command,
        cwd=code_dir,
        check=False,
        timeout=None if timeout_seconds == 0 else timeout_seconds,
    )
except subprocess.TimeoutExpired:
    status_path = output_dir / "wandb_upload_status.json"
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        payload = {}
    payload["state"] = "upload_pending"
    payload["error"] = f"W&B upload exceeded {timeout_seconds} seconds"
    status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(124)
raise SystemExit(completed.returncode)
PY
