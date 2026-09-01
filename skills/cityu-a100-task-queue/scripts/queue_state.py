"""Immutable queue state and atomic persistence helpers."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


SCHEMA_VERSION = 1
ACTIVE_STATUSES = ("pending", "claimed", "running")
TERMINAL_STATUSES = ("completed", "failed", "canceled")
VALID_STATUSES = ACTIVE_STATUSES + TERMINAL_STATUSES
SUPPORTED_GPU_COUNTS = (3, 4)


@dataclass(frozen=True)
class QueueTask:
    task_id: str
    sequence: int
    title: str
    command: str
    remote_cwd: str
    gpu_counts: tuple[int, ...]
    status: str
    created_at: str
    updated_at: str
    slurm_job_id: str | None = None
    allocation_gpu_count: int | None = None
    node: str | None = None
    note: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QueueTask":
        task = cls(
            task_id=str(value["task_id"]),
            sequence=int(value["sequence"]),
            title=str(value["title"]),
            command=str(value["command"]),
            remote_cwd=str(value["remote_cwd"]),
            gpu_counts=tuple(int(item) for item in value["gpu_counts"]),
            status=str(value["status"]),
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
            slurm_job_id=_optional_string(value.get("slurm_job_id")),
            allocation_gpu_count=_optional_int(value.get("allocation_gpu_count")),
            node=_optional_string(value.get("node")),
            note=_optional_string(value.get("note")),
        )
        validate_task(task)
        return task

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["gpu_counts"] = list(self.gpu_counts)
        return value


@dataclass(frozen=True)
class QueueState:
    version: int
    next_sequence: int
    tasks: tuple[QueueTask, ...]

    @classmethod
    def empty(cls) -> "QueueState":
        return cls(version=SCHEMA_VERSION, next_sequence=1, tasks=())

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "next_sequence": self.next_sequence,
            "tasks": [task.to_mapping() for task in self.tasks],
        }


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_state_file() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / ".codex" / "state" / "cityu-a100-task-queue.json"


def validate_task(task: QueueTask) -> None:
    if task.status not in VALID_STATUSES:
        raise ValueError(f"invalid task status: {task.status}")
    if not task.title.strip() or not task.command.strip():
        raise ValueError("task title and command must be non-empty")
    if not task.gpu_counts or any(
        count not in SUPPORTED_GPU_COUNTS for count in task.gpu_counts
    ):
        raise ValueError("gpu_counts must contain only 3 and/or 4")
    if len(task.gpu_counts) > 1 and "OPD_SLOT_GPU_COUNT" not in task.command:
        raise ValueError(
            "a 3/4-GPU task command must branch on OPD_SLOT_GPU_COUNT"
        )
    if task.status in ("claimed", "running") and not task.slurm_job_id:
        raise ValueError(f"{task.status} task requires slurm_job_id")


def load_state(path: Path) -> QueueState:
    if not path.exists():
        return QueueState.empty()
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value.get("version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported queue schema: {value.get('version')}")
    tasks = tuple(QueueTask.from_mapping(item) for item in value.get("tasks", []))
    return QueueState(
        version=SCHEMA_VERSION,
        next_sequence=int(value["next_sequence"]),
        tasks=tasks,
    )


def write_state(path: Path, state: QueueState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(state.to_mapping(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


@contextmanager
def locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def replace_task(state: QueueState, updated: QueueTask) -> QueueState:
    tasks = tuple(
        updated if task.task_id == updated.task_id else task for task in state.tasks
    )
    return replace(state, tasks=tasks)


def find_task(state: QueueState, task_id: str) -> QueueTask:
    for task in state.tasks:
        if task.task_id == task_id:
            return task
    raise ValueError(f"unknown task_id: {task_id}")


def newest_pending(state: QueueState, gpu_count: int) -> QueueTask | None:
    candidates = (
        task
        for task in state.tasks
        if task.status == "pending" and gpu_count in task.gpu_counts
    )
    return max(candidates, key=lambda item: item.sequence, default=None)
