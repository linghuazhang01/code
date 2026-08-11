"""Filesystem layout helpers shared by MOPD audit writers."""

from __future__ import annotations

from pathlib import Path


def step_jsonl_dir(
    output_dir: str | Path,
    step: int,
    *,
    create: bool = False,
) -> Path:
    """Return ``<audit_root>/step_XXXXXX/jsonls`` for one optimizer step."""

    if int(step) < 0:
        raise ValueError(f"Audit step must be non-negative, got {step}.")
    directory = Path(output_dir) / f"step_{int(step):06d}" / "jsonls"
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory
