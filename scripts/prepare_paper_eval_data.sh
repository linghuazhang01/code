#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/opd_mopd}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
ENV_NAME="${ENV_NAME:-mopd-verl}"
G_OPD_DIR="${G_OPD_DIR:-${REMOTE_ROOT}/G-OPD}"
HF_HOME="${HF_HOME:-${REMOTE_ROOT}/hf_home}"
LCB_DIR="${LCB_DIR:-${G_OPD_DIR}/code_eval/coding/LiveCodeBench/code_generation_lite}"
DOWNLOAD_LCB="${DOWNLOAD_LCB:-1}"

if [[ -f "${REMOTE_ROOT}/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${REMOTE_ROOT}/env.sh"
fi

export PATH="${CONDA_ROOT}/bin:${PATH}"
export HF_HOME="${HF_HOME}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTHONPATH="${CODE_DIR}:${G_OPD_DIR}/verl:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python "${CODE_DIR}/scripts/apply_gopd_audit_patch.py" "${G_OPD_DIR}"

if [[ "${DOWNLOAD_LCB}" == "1" ]]; then
  mkdir -p "${LCB_DIR}"
  huggingface-cli download livecodebench/code_generation_lite \
    --repo-type dataset \
    --local-dir "${LCB_DIR}" \
    --include "test*.jsonl"
fi

python -m mopd_verl.prepare_data prepare-paper-eval --gopd-dir "${G_OPD_DIR}"

python - <<'PY'
import os
from pathlib import Path
import pandas as pd

root = Path(os.environ["G_OPD_DIR"]) / "G-OPD-Training-Data"
paths = {
    "AIME24": root / "PaperEval/AIME24/test.parquet",
    "AIME25": root / "PaperEval/AIME25/test.parquet",
    "HMMT25Feb": root / "PaperEval/HMMT25Feb/test.parquet",
    "HMMT25Nov": root / "PaperEval/HMMT25Nov/test.parquet",
    "HumanEvalPlus": root / "PaperEval/HumanEvalPlus/test.parquet",
    "MBPPPlus": root / "PaperEval/MBPPPlus/test.parquet",
    "LiveCodeBench": root / "PaperEval/LiveCodeBench/test.parquet",
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
