#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="$(cd "${EVAL_DIR}/.." && pwd)"

REMOTE_ROOT="${REMOTE_ROOT:-$(cd "${CODE_DIR}/.." && pwd)}"
G_OPD_DIR="${G_OPD_DIR:-${REMOTE_ROOT}/G-OPD}"
HF_HOME="${HF_HOME:-${CODE_DIR}/hf_home}"
LCB_DIR="${LCB_DIR:-${G_OPD_DIR}/code_eval/coding/LiveCodeBench/code_generation_lite}"
DOWNLOAD_LCB="${DOWNLOAD_LCB:-1}"

if [[ -f "${CODE_DIR}/logs/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CODE_DIR}/logs/env.sh"
elif [[ -f "${REMOTE_ROOT}/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${REMOTE_ROOT}/env.sh"
fi

export CODE_DIR
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
  LCB_DIR="${LCB_DIR}" python - <<'PY'
import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="livecodebench/code_generation_lite",
    repo_type="dataset",
    local_dir=os.environ["LCB_DIR"],
    allow_patterns=["test*.jsonl"],
)
PY
fi

python -m mopd_verl.prepare_data prepare-paper-eval \
  --gopd-dir "${G_OPD_DIR}" \
  --output-root "${CODE_DIR}/data/eval_data"

python - <<'PY'
import os
from pathlib import Path
import pandas as pd

root = Path(os.environ["CODE_DIR"]) / "data/eval_data"
paths = {
    "AIME24": root / "math/data/AIME24/test.parquet",
    "AIME25": root / "math/data/AIME25/test.parquet",
    "HMMT25Feb": root / "math/data/HMMT25Feb/test.parquet",
    "HMMT25Nov": root / "math/data/HMMT25Nov/test.parquet",
    "HumanEvalPlus": root / "code/data/HumanEvalPlus/test.parquet",
    "MBPPPlus": root / "code/data/MBPPPlus/test.parquet",
    "LiveCodeBench": root / "code/data/LiveCodeBench/test.parquet",
}
for name, path in paths.items():
    print(f"{name}\t{len(pd.read_parquet(path))}\t{path}")
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
for path in sorted(lcb_dir.glob("test*.jsonl")):
    print(f"LCB shard\t{path.name}\t{path}")
PY
