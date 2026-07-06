"""Legacy ToolRL GRPO reward adapter compatibility shim."""

from __future__ import annotations

from typing import Any


def compute_score(*_: Any, **__: Any) -> dict[str, float]:
    raise RuntimeError(
        "Legacy ToolRL GRPO reward adapters were removed when grpo/ was reset "
        "for M2RL-style IF/Science GRPO. Use grpo.rewards.m2rl.compute_score "
        "for current GRPO runs."
    )


__all__ = ["compute_score"]
