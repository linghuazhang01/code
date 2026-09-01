from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE_MANAGER = (
    ROOT
    / "skills"
    / "cityu-a100-task-queue"
    / "scripts"
    / "queue_manager.py"
)


def _run_queue(state_file: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(QUEUE_MANAGER),
            "--state-file",
            str(state_file),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _add_task(state_file: Path, title: str, command: str) -> str:
    result = _run_queue(
        state_file,
        "add",
        "--title",
        title,
        "--gpu-counts",
        "3,4",
        "--command",
        command,
        "--json",
    )
    assert result.returncode == 0, result.stderr
    return str(json.loads(result.stdout)["task_id"])


def test_dual_gpu_task_requires_adaptive_command(tmp_path: Path) -> None:
    result = _run_queue(
        tmp_path / "queue.json",
        "add",
        "--title",
        "fixed placement",
        "--gpu-counts",
        "3,4",
        "--command",
        "echo fixed",
    )

    assert result.returncode == 2
    assert "must branch on OPD_SLOT_GPU_COUNT" in result.stderr


def test_json_list_omits_command_body(tmp_path: Path) -> None:
    state_file = tmp_path / "queue.json"
    _add_task(
        state_file,
        "adaptive",
        'case "$OPD_SLOT_GPU_COUNT" in 3|4) true ;; esac',
    )

    result = _run_queue(state_file, "list", "--json")

    assert result.returncode == 0, result.stderr
    task = json.loads(result.stdout)["tasks"][0]
    assert "command" not in task


def test_explicit_claim_cannot_bypass_lifo(tmp_path: Path) -> None:
    state_file = tmp_path / "queue.json"
    older_task_id = _add_task(
        state_file,
        "older",
        'case "$OPD_SLOT_GPU_COUNT" in 3|4) true ;; esac',
    )
    newer_task_id = _add_task(
        state_file,
        "newer",
        'case "$OPD_SLOT_GPU_COUNT" in 3|4) true ;; esac',
    )

    rejected = _run_queue(
        state_file,
        "claim",
        "--gpu-count",
        "3",
        "--slurm-job-id",
        "100",
        "--task-id",
        older_task_id,
    )
    assert rejected.returncode == 2
    assert "not the newest compatible pending task" in rejected.stderr

    claimed = _run_queue(
        state_file,
        "claim",
        "--gpu-count",
        "3",
        "--slurm-job-id",
        "100",
        "--json",
    )
    assert claimed.returncode == 0, claimed.stderr
    assert json.loads(claimed.stdout)["task_id"] == newer_task_id
