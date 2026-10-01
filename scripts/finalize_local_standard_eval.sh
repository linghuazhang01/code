#!/usr/bin/env bash
# Finish official Code scoring before publishing canonical suite completion.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_ROOT="${1:?suite root required}"
PYTHON_BIN="${PYTHON:?Python executable required}"
MODEL_LABEL="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["label"])' "${SUITE_ROOT}/suite_manifest.json")"
RECORDS="${SUITE_ROOT}/${MODEL_LABEL}/code/prompt_response_records.jsonl"
OUTPUT="${SUITE_ROOT}/official_evalplus"
attempt=1
while [[ -e "${OUTPUT}" ]]; do
  attempt=$((attempt + 1))
  OUTPUT="${SUITE_ROOT}/official_evalplus_attempt_${attempt}"
done
[[ -f "${RECORDS}" ]] || { echo "Missing raw Code responses: ${RECORDS}" >&2; exit 1; }
[[ -f "${SUITE_ROOT}/MERGE_SUCCESS" ]] || { echo "Missing MERGE_SUCCESS" >&2; exit 1; }
# The historical filename is reused as a plain CPU process, never submitted.
CUDA_VISIBLE_DEVICES="" bash "${SCRIPT_DIR}/slurm_evalplus_rescore_worker.sh" \
  "${MODEL_LABEL}" "${RECORDS}" "${OUTPUT}" \
  > "${SUITE_ROOT}/logs/official_evalplus.log" 2>&1
[[ -f "${OUTPUT}/SUCCESS" && -f "${OUTPUT}/humaneval/SUCCESS" && -f "${OUTPUT}/mbpp/SUCCESS" ]]
"${PYTHON_BIN}" - "${SUITE_ROOT}/suite_manifest.json" "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
manifest["status"] = "complete"
manifest["official_evalplus"] = Path(sys.argv[2]).name
path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
PY
touch "${SUITE_ROOT}/SUCCESS"
