"""Legacy General-Reasoner GRPO reward adapter compatibility shim."""

from __future__ import annotations

from typing import Any


def compute_score(*_: Any, **__: Any) -> dict[str, float]:
    raise RuntimeError(
        "Legacy General-Reasoner GRPO reward adapters were removed when grpo/ "
        "was reset for M2RL-style IF/Science GRPO. Use grpo.rewards.m2rl "
        "for current GRPO reward scoring."
    )


__all__ = ["compute_score"]
