"""Removed General-Reasoner reward adapter compatibility shim."""

from __future__ import annotations

from typing import Any


def compute_score(*_: Any, **__: Any) -> dict[str, float]:
    raise RuntimeError(
        "The legacy General-Reasoner reward adapter is unavailable. "
        "Use mopd_verl.m2rl_reward for IF/Science validation scoring."
    )


__all__ = ["compute_score"]
