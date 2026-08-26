"""Non-blocking, model-only Hugging Face uploads."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Callable, MutableMapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mopd_verl.huggingface_checkpoint import (
    HuggingFaceCheckpointConfig,
    _huggingface_model_dir,
    _upload_huggingface_model,
    _validate_huggingface_model_dir,
    checkpoint_upload_completed,
)


@dataclass(frozen=True)
class HuggingFaceUploadResult:
    """Completed asynchronous model upload."""

    step: int
    commit_url: str


class AsyncHuggingFaceUploader:
    """Queue model-only Hub uploads without blocking training steps."""

    def __init__(
        self,
        config: HuggingFaceCheckpointConfig,
        *,
        environ: MutableMapping[str, str] | None = None,
        api_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._config = config
        self._environ = environ
        self._api_factory = api_factory
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="huggingface-model-upload",
        )
        self._futures: dict[int, Future[str]] = {}
        self._closed = False

    def schedule(self, checkpoint_dir: str | Path, step: int) -> bool:
        """Stage and queue one upload, returning before network I/O completes."""

        if self._closed:
            raise RuntimeError("Hugging Face uploader is already closed.")
        if not self._config.includes_step(step):
            raise ValueError(
                f"Step {step} is not configured for Hugging Face upload."
            )
        if step in self._futures:
            return False
        local_dir = Path(checkpoint_dir)
        staged_model_dir = _staged_model_dir(local_dir.parent, step)
        if checkpoint_upload_completed(self._config, checkpoint_dir, step):
            if staged_model_dir.is_dir():
                shutil.rmtree(staged_model_dir)
            return False

        if staged_model_dir.is_dir():
            _validate_huggingface_model_dir(staged_model_dir)
        else:
            model_dir = _huggingface_model_dir(local_dir)
            staged_model_dir = _stage_huggingface_model(
                model_dir,
                local_dir,
                step,
            )
        self._futures[step] = self._executor.submit(
            _upload_staged_huggingface_model,
            self._config,
            local_dir,
            staged_model_dir,
            step,
            self._environ,
            self._api_factory,
        )
        return True

    def schedule_retained(self, checkpoint_root: str | Path) -> tuple[int, ...]:
        """Queue retained failed-upload snapshots before training resumes."""

        root = Path(checkpoint_root)
        staging_root = root / ".huggingface_uploads"
        if not staging_root.is_dir():
            return ()

        retained_steps: list[int] = []
        for staged_model_dir in staging_root.iterdir():
            step = _step_from_staged_model_dir(staged_model_dir)
            if step is None or not self._config.includes_step(step):
                continue
            retained_steps.append(step)

        scheduled: list[int] = []
        for step in sorted(retained_steps):
            if self.schedule(root / f"global_step_{step}", step):
                scheduled.append(step)
        return tuple(scheduled)

    def wait(self) -> tuple[HuggingFaceUploadResult, ...]:
        """Wait after training for all queued uploads and surface failures."""

        self._closed = True
        self._executor.shutdown(wait=True)
        completed: list[HuggingFaceUploadResult] = []
        failures: list[tuple[int, Exception]] = []
        for step, future in sorted(self._futures.items()):
            try:
                completed.append(
                    HuggingFaceUploadResult(step=step, commit_url=future.result())
                )
            except Exception as exc:  # noqa: BLE001 - drain every queued upload.
                failures.append((step, exc))
        if failures:
            failed_steps = ", ".join(str(step) for step, _ in failures)
            raise RuntimeError(
                f"Hugging Face model upload failed for steps: {failed_steps}. "
                "The staged model directories were retained for retry."
            ) from failures[0][1]
        return tuple(completed)


def _upload_staged_huggingface_model(
    config: HuggingFaceCheckpointConfig,
    checkpoint_dir: Path,
    staged_model_dir: Path,
    step: int,
    environ: MutableMapping[str, str] | None,
    api_factory: Callable[[str], Any] | None,
) -> str:
    commit_url = _upload_huggingface_model(
        config,
        checkpoint_dir,
        staged_model_dir,
        step,
        environ=environ,
        api_factory=api_factory,
    )
    shutil.rmtree(staged_model_dir)
    return commit_url


def _stage_huggingface_model(
    model_dir: Path,
    checkpoint_dir: Path,
    step: int,
) -> Path:
    staging_root = checkpoint_dir.parent / ".huggingface_uploads"
    staging_root.mkdir(parents=True, exist_ok=True)
    staged_model_dir = _staged_model_dir(checkpoint_dir.parent, step)
    if staged_model_dir.exists():
        raise FileExistsError(
            f"Hugging Face upload snapshot already exists: {staged_model_dir}"
        )

    temporary_dir = staging_root / f".{staged_model_dir.name}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copytree(model_dir, temporary_dir, copy_function=_hardlink_file)
        temporary_dir.replace(staged_model_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return staged_model_dir


def _staged_model_dir(checkpoint_root: Path, step: int) -> Path:
    return checkpoint_root / ".huggingface_uploads" / f"global_step_{step}"


def _step_from_staged_model_dir(path: Path) -> int | None:
    prefix = "global_step_"
    if not path.is_dir() or not path.name.startswith(prefix):
        return None
    raw_step = path.name.removeprefix(prefix)
    if not raw_step.isdigit():
        return None
    return int(raw_step)


def _hardlink_file(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError as exc:
        raise RuntimeError(
            "The checkpoint filesystem must support hard links for non-blocking "
            "Hugging Face uploads; refusing to copy model weights synchronously."
        ) from exc
