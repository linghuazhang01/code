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
LCB_V5_SHARD0_SHA256="${LCB_V5_SHARD0_SHA256:-2cafe2a842652f6aca997755a8150c348fbe25c040c0fb2ac7e63e400e10e5cb}"
LCB_V5_SHARD1_SHA256="${LCB_V5_SHARD1_SHA256:-3558c5766089965eda005c39647ccf0b42be2bffe35665fecfaaa90d355b5d59}"
LCB_V6_SHA256="${LCB_V6_SHA256:-${LCB_SHA256:-bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5}}"
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
export LCB_V5_SHARD0_SHA256
export LCB_V5_SHARD1_SHA256
export LCB_V6_SHA256
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
from eval.data_prep.paper_eval import lcb_source_parquet_to_jsonl

snapshot_download(
    repo_id="livecodebench/code_generation_lite",
    repo_type="dataset",
    local_dir=os.environ["LCB_DIR"],
    revision=os.environ["LCB_REVISION"],
    allow_patterns=["v5/*.parquet", "test6.jsonl"],
)
root = Path(os.environ["LCB_DIR"])
expected = {
    root / "v5/test-00000-of-00002.parquet": os.environ["LCB_V5_SHARD0_SHA256"],
    root / "v5/test-00001-of-00002.parquet": os.environ["LCB_V5_SHARD1_SHA256"],
    root / "test6.jsonl": os.environ["LCB_V6_SHA256"],
}
for source, expected_sha256 in expected.items():
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise SystemExit(f"LiveCodeBench SHA-256 mismatch for {source}: {digest.hexdigest()}")
compatibility_rows = lcb_source_parquet_to_jsonl(
    [
        root / "v5/test-00000-of-00002.parquet",
        root / "v5/test-00001-of-00002.parquet",
    ],
    root / "test5.jsonl",
)
if compatibility_rows != 167:
    raise SystemExit(f"Expected 167 LiveCodeBench v5 rows, found {compatibility_rows}")
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

from eval.data_prep.code_prompt_validation import prompt_column_sha256
from eval.data_prep.paper_eval import PAPER_CODE_EVAL_SPECS
from eval.domains.code.prompting import (
    EVALPLUS_PROMPT_TEMPLATE,
    LCB_QWEN3_PROMPT_TEMPLATE,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

root = Path(os.environ["CODE_DIR"]) / "data/eval_data"
paths = {
    "AIME24": root / "math/AIME24/test.parquet",
    "AIME25": root / "math/AIME25/test.parquet",
    "HMMT25Feb": root / "math/HMMT25Feb/test.parquet",
    "HMMT25Nov": root / "math/HMMT25Nov/test.parquet",
    "HumanEvalPlus": root / "code/HumanEvalPlus/test.parquet",
    "MBPPPlus": root / "code/MBPPPlus/test.parquet",
    "LiveCodeBench-v5": root / "code/LiveCodeBench-v5/test.parquet",
    "LiveCodeBench-v6": root / "code/LiveCodeBench/test.parquet",
}
for name, path in paths.items():
    print(f"{name}\t{len(pd.read_parquet(path))}\t{path}")

gopd_root = Path(os.environ["G_OPD_DIR"])
for _, (data_source, source_path, output_path) in PAPER_CODE_EVAL_SPECS.items():
    source_file = gopd_root / source_path
    destination = root / output_path
    manifest = {
        "data_source": data_source,
        "dataset": data_source,
        "enable_thinking": False,
        "prompt_template": "gopd_evalplus_qwen_chat",
        "prompt_template_sha256": hashlib.sha256(
            EVALPLUS_PROMPT_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "rows": len(pd.read_parquet(destination)),
        "source_file": source_file.name,
        "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "user_content_sha256": prompt_column_sha256(
            pd.read_parquet(destination, columns=["prompt"])["prompt"]
        ),
    }
    destination.with_name("manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

lcb_root = Path(os.environ["LCB_DIR"])
lcb_specs = {
    "v5": {
        "output": paths["LiveCodeBench-v5"],
        "sources": [
            lcb_root / "v5/test-00000-of-00002.parquet",
            lcb_root / "v5/test-00001-of-00002.parquet",
        ],
        "runner_compatibility_file": lcb_root / "test5.jsonl",
    },
    "v6": {
        "output": paths["LiveCodeBench-v6"],
        "sources": [lcb_root / "test6.jsonl"],
    },
}
for release_version, spec in lcb_specs.items():
    source_files = []
    for source in spec["sources"]:
        source_files.append(
            {
                "name": str(source.relative_to(lcb_root)),
                "sha256": file_sha256(source),
            }
        )
    output = spec["output"]
    manifest = {
        "chat_template_enable_thinking": False,
        "chat_template_tokenizer": "Qwen/Qwen3-4B",
        "data_source": f"LiveCodeBench-{release_version}",
        "dataset": "livecodebench/code_generation_lite",
        "enable_thinking": False,
        "evaluation_tests": "public+private",
        "prompt_template": "gopd_qwen3_non_thinking",
        "prompt_template_sha256": hashlib.sha256(
            LCB_QWEN3_PROMPT_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "release_version": release_version,
        "revision": os.environ["LCB_REVISION"],
        "rows": len(pd.read_parquet(output)),
        "source_files": source_files,
        "user_content_sha256": prompt_column_sha256(
            pd.read_parquet(output, columns=["prompt"])["prompt"]
        ),
    }
    compatibility_file = spec.get("runner_compatibility_file")
    if compatibility_file is not None:
        manifest["runner_compatibility_file"] = {
            "derived_from": [source["name"] for source in source_files],
            "name": compatibility_file.name,
            "rows": sum(
                1
                for line in compatibility_file.open(encoding="utf-8")
                if line
            ),
            "sha256": file_sha256(compatibility_file),
        }
    (output.parent / "manifest.json").write_text(
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
for path in [*sorted((lcb_dir / "v5").glob("*.parquet")), lcb_dir / "test6.jsonl"]:
    print(f"LCB shard\t{path.name}\t{path}")
PY
