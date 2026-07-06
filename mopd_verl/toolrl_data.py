"""Legacy ToolRL GRPO data adapter compatibility shims."""

from __future__ import annotations

from typing import Any


def _legacy_toolrl_removed() -> RuntimeError:
    return RuntimeError(
        "Legacy ToolRL GRPO adapters were removed when grpo/ was reset for "
        "M2RL-style IF/Science GRPO. Use grpo.data.m2rl or restore the legacy "
        "adapter from temp/grpo_legacy_backup_*/grpo if ToolRL is still needed."
    )


def toolrl_frame_to_verl(*_: Any, **__: Any) -> Any:
    raise _legacy_toolrl_removed()


def toolrl_to_verl_parquet(*_: Any, **__: Any) -> Any:
    raise _legacy_toolrl_removed()


__all__ = ["toolrl_frame_to_verl", "toolrl_to_verl_parquet"]
