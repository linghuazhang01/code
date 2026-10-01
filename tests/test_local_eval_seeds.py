"""Verify seed changes reach task manifests without changing canonical runs."""

import os
from pathlib import Path
import shlex
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval.parallel_eval import build_manifest, resume_signature


CODE_DIR = Path(__file__).resolve().parents[1]


def launch(tmp_path: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(CODE_DIR / "start.sh"), "--eval", "--local",
         "--model_path", str(tmp_path), "--dry_run", *arguments],
        env={**os.environ, "PYTHON": sys.executable, "GPU_IDS": "0,1,2,3"},
        capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize("seed", [43, 44])
def test_custom_seed_reaches_planner(tmp_path: Path, seed: int) -> None:
    result = launch(tmp_path, ["--datasets", "hmmt25nov", "--seed", str(seed),
                               "--seed_sequence_offset", "48"])
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[1])
    assert command[command.index("--base-seed") + 1] == str(seed)
    assert command[command.index("--seed-sequence-offset") + 1] == "48"


@pytest.mark.parametrize("argument,value", [
    ("--seed", "-1"), ("--seed", "43.5"), ("--seed", "bad"),
    ("--seed", "043"), ("--seed_sequence_offset", "-1"),
])
def test_invalid_seed_rejected(tmp_path: Path, argument: str, value: str) -> None:
    result = launch(tmp_path, [argument, value])
    assert result.returncode == 2
    assert "must be a non-negative integer" in result.stderr


@pytest.mark.parametrize("arguments", [
    ["--seed", "43"], ["--seed_sequence_offset", "48"],
])
def test_canonical_protocol_rejects_seed_changes(tmp_path: Path, arguments: list[str]) -> None:
    result = launch(tmp_path, ["--standard_protocol", *arguments])
    assert result.returncode == 2
    assert "requires seed 42 and seed_sequence_offset 0" in result.stderr


def test_default_seed_stays_42(tmp_path: Path) -> None:
    result = launch(tmp_path, [])
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[1])
    assert command[command.index("--base-seed") + 1] == "42"
    assert command[command.index("--seed-sequence-offset") + 1] == "0"


def test_subset_preserves_historical_shard_seed_positions(tmp_path: Path) -> None:
    for dataset in ["AIME24", "AIME25", "HMMT25Feb", "HMMT25Nov"]:
        path = tmp_path / "data/eval_data/math" / dataset / "test.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"row_id": list(range(30))}), path)
    options = dict(
        code_dir=tmp_path, suite_root=tmp_path / "output", run_tag="test",
        model_path="/checkpoint/global_step_60", shards_per_dataset=16,
        min_rows_per_shard=1, math_samples=8, code_samples=8, science_samples=8,
        max_samples_per_dataset=None, include_mmlupro_500=False, worker_count=2,
    )
    original = build_manifest(
        **options, base_seed=42,
        dataset_keys=["aime24", "aime25", "hmmt25feb", "hmmt25nov"],
    )
    old_tasks = [task for task in original["tasks"] if task["dataset"] == "hmmt25nov"]
    for seed in [43, 44]:
        subset = build_manifest(
            **options, base_seed=seed, dataset_keys=["hmmt25nov"], seed_sequence_offset=48,
        )
        assert subset["generation"]["base_seed"] == seed
        assert subset["generation"]["seed_sequence_offset"] == 48
        assert subset["expected_records_total"] == 240
        for old, new in zip(old_tasks, subset["tasks"], strict=True):
            assert new["source_start"] == old["source_start"]
            assert new["source_end_exclusive"] == old["source_end_exclusive"]
            assert new["generation_seed"] - old["generation_seed"] == seed - 42
    assert "seed_sequence_offset" not in original["generation"]
    assert original["generation"] == {
        "math_samples": 8, "code_samples": 8, "science_samples": 8,
        "mmlupro_samples": 8, "base_seed": 42,
        "standard_shard_seed_rule": "base_seed + task_sequence * 1000003",
        "mmlupro_seed_rule": "fixed base_seed for every prompt shard",
        "max_new_tokens": 16384, "temperature": 1.0, "top_p": 1.0,
    }
    unchanged = build_manifest(
        **options, base_seed=42, seed_sequence_offset=0,
        dataset_keys=["aime24", "aime25", "hmmt25feb", "hmmt25nov"],
    )
    assert resume_signature(original) == resume_signature(unchanged)
