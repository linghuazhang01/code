#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="$(cd "${EVAL_DIR}/.." && pwd)"

REMOTE_ROOT="${REMOTE_ROOT:-$(cd "${CODE_DIR}/.." && pwd)}"
DEFAULT_G_OPD_DIR="${REMOTE_ROOT}/G-OPD"
if [[ ! -d "${DEFAULT_G_OPD_DIR}" && -d "${REMOTE_ROOT}/../G-OPD" ]]; then
  DEFAULT_G_OPD_DIR="$(cd "${REMOTE_ROOT}/../G-OPD" && pwd)"
fi
G_OPD_DIR="${G_OPD_DIR:-${DEFAULT_G_OPD_DIR}}"
HF_HOME="${HF_HOME:-${CODE_DIR}/hf_home}"
LCB_DIR="${LCB_DIR:-${G_OPD_DIR}/code_eval/coding/LiveCodeBench/code_generation_lite}"
LCB_REVISION="${LCB_REVISION:-48d36ed304dca42cf8ab20e941262ccd096518a3}"
LCB_SHA256="${LCB_SHA256:-bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5}"
DOWNLOAD_LCB="${DOWNLOAD_LCB:-1}"

if [[ -f "${CODE_DIR}/logs/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CODE_DIR}/logs/env.sh"
elif [[ -f "${REMOTE_ROOT}/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${REMOTE_ROOT}/env.sh"
fi

export CODE_DIR
export LCB_DIR
export LCB_REVISION
export LCB_SHA256
export HF_HOME="${HF_HOME}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export PYTHONPATH="${CODE_DIR}:${G_OPD_DIR}/verl:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

python "${CODE_DIR}/scripts/apply_gopd_audit_patch.py" "${G_OPD_DIR}"

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

if [[ "${DOWNLOAD_LCB}" == "1" ]]; then
  mkdir -p "${LCB_DIR}"
  ensure_huggingface_hub
  LCB_DIR="${LCB_DIR}" LCB_REVISION="${LCB_REVISION}" python - <<'PY'
import os
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="livecodebench/code_generation_lite",
    repo_type="dataset",
    local_dir=os.environ["LCB_DIR"],
    revision=os.environ["LCB_REVISION"],
    allow_patterns=["test6.jsonl"],
)
source = Path(os.environ["LCB_DIR"]) / "test6.jsonl"
digest = hashlib.sha256()
with source.open("rb") as handle:
    while chunk := handle.read(8 * 1024 * 1024):
        digest.update(chunk)
if digest.hexdigest() != os.environ["LCB_SHA256"]:
    raise SystemExit(f"LiveCodeBench v6 SHA-256 mismatch: {digest.hexdigest()}")
PY
fi

python -m mopd_verl.prepare_data prepare-paper-eval \
  --gopd-dir "${G_OPD_DIR}" \
  --output-root "${CODE_DIR}/data/eval_data"

python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import pandas as pd

root = Path(os.environ["CODE_DIR"]) / "data/eval_data"
paths = {
    "AIME24": root / "math/AIME24/test.parquet",
    "AIME25": root / "math/AIME25/test.parquet",
    "HMMT25Feb": root / "math/HMMT25Feb/test.parquet",
    "HMMT25Nov": root / "math/HMMT25Nov/test.parquet",
    "HumanEvalPlus": root / "code/HumanEvalPlus/test.parquet",
    "MBPPPlus": root / "code/MBPPPlus/test.parquet",
    "LiveCodeBench-v6": root / "code/LiveCodeBench/test.parquet",
}
for name, path in paths.items():
    print(f"{name}\t{len(pd.read_parquet(path))}\t{path}")

lcb_source = Path(os.environ["LCB_DIR"]) / "test6.jsonl"
digest = hashlib.sha256()
with lcb_source.open("rb") as handle:
    while chunk := handle.read(8 * 1024 * 1024):
        digest.update(chunk)
source_sha256 = digest.hexdigest()
if source_sha256 != os.environ["LCB_SHA256"]:
    raise SystemExit(f"LiveCodeBench v6 SHA-256 mismatch: {source_sha256}")
lcb_output = paths["LiveCodeBench-v6"]
manifest = {
    "dataset": "livecodebench/code_generation_lite",
    "evaluation_tests": "public+private",
    "release_version": "v6",
    "revision": os.environ["LCB_REVISION"],
    "rows": len(pd.read_parquet(lcb_output)),
    "source_file": lcb_source.name,
    "source_sha256": source_sha256,
}
(lcb_output.parent / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

python - <<'PY'
import os
from pathlib import Path

gopd = Path(os.environ["G_OPD_DIR"])
paths = {
    "HumanEval+": gopd / "code_eval/data/HumanEvalPlus.jsonl",
    "MBPP+": gopd / "code_eval/data/MbppPlus.jsonl",
}
for name, path in paths.items():
    count = sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
    print(f"{name}\t{count}\t{path}")

lcb_dir = gopd / "code_eval/coding/LiveCodeBench/code_generation_lite"
for path in sorted(lcb_dir.glob("test6.jsonl")):
    print(f"LCB shard\t{path.name}\t{path}")
PY
