"""Local GPU selection, dispatch, and provenance without GPU dependencies."""

import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


CODE_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = CODE_DIR / "scripts/run_local_math_parallel_eval.sh"


def run_launcher(tmp_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), "--model_path", str(tmp_path), *args],
        env={**os.environ, "PYTHON": sys.executable, "GPU_IDS": "0,1,2,3"},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("count,ids", [(2, "4,5"), (3, "1,2,3"), (4, "0,1,2,3")])
@pytest.mark.parametrize("prefix", [["--eval", "--local"], ["--local", "--eval"]])
def test_start_standard_passthrough(tmp_path: Path, count: int, ids: str, prefix: list[str]) -> None:
    result = subprocess.run(
        ["bash", str(CODE_DIR / "start.sh"), *prefix,
         "--model_path", str(tmp_path), "--gpus", str(count), "--gpu_ids", ids,
         "--standard_protocol", "--gopd_dir", str(tmp_path), "--dry_run"],
        env={**os.environ, "PYTHON": sys.executable},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"--worker-count {count}" in result.stdout
    assert "--math-samples 8 --code-samples 8 --science-samples 8" in result.stdout
    plan = shlex.split(result.stdout.splitlines()[1])
    datasets = plan[plan.index("--datasets") + 1].split(",")
    assert len(datasets) == 10
    assert datasets[-4:] == ["lcb_v5", "lcb_v6", "gpqa_diamond", "mmlupro_500_seed42"]


def test_default_four_workers(tmp_path: Path) -> None:
    result = run_launcher(tmp_path, ["--dry_run"])
    assert result.returncode == 0, result.stderr
    assert "--worker-count 4" in result.stdout


@pytest.mark.parametrize("count,ids", [
    ("1", "0"), ("5", "0,1,2,3,4"), ("x", "0,1"),
    ("2", "0,1,2"), ("3", "4,5"), ("2", "4,4"),
    ("2", "-1,2"), ("2", "a,2"), ("2", "1,2,"),
    ("2", ",1,2"), ("2", "1,,2"), ("2", "1, 2"),
    ("2", "1,01"), ("2", "1,2\n3,4"),
])
def test_invalid_gpu_selection(tmp_path: Path, count: str, ids: str) -> None:
    result = run_launcher(tmp_path, ["--gpus", count, "--gpu_ids", ids, "--dry_run"])
    assert result.returncode == 2
    assert "GPU IDs" in result.stderr or "--gpu" in result.stderr


@pytest.mark.parametrize("ids", ["4,5", "1,2,3", "0,1,2,3"])
def test_worker_mapping_and_manifest(tmp_path: Path, ids: str) -> None:
    # A fake Python executable exercises the actual shell launch/wait/merge flow.
    fake_python = tmp_path / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nset -eu\n'
        'if [[ "$2" == "eval.parallel_worker" ]]; then\n'
        '  printf "gpu=%s args=%s\\n" "$CUDA_VISIBLE_DEVICES" "$*"\n'
        'fi\n'
    )
    fake_python.chmod(0o755)
    count = len(ids.split(","))
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--model_path", str(tmp_path),
         "--output_root", str(tmp_path / "output"), "--run_tag", "test",
         "--gpus", str(count), "--gpu_ids", ids],
        env={**os.environ, "PYTHON": str(fake_python)},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    suite = tmp_path / "output/test"
    manifest = (suite / "RUN_MANIFEST.md").read_text()
    assert f"DP={count}, {count} persistent TP=1" in manifest
    assert f"through gpu_worker_{count - 1}.log" in manifest
    assert f"GPU IDs: {ids}" in manifest
    logs = sorted((suite / "logs").glob("gpu_worker_*.log"))
    assert len(logs) == count
    for worker_id, (log, gpu_id) in enumerate(zip(logs, ids.split(","))):
        assert f"gpu={gpu_id} " in log.read_text()
        assert f"--worker-id {worker_id}" in log.read_text()
    assert (suite / "COMPLETED_AT_UTC").exists()
