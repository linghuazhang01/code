"""Data preparation helpers for M2RL-style GRPO recipes."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_M2RL_EXPORTS = {
    "M2RLSchemaReport",
    "m2rl_frame_to_verl",
    "m2rl_to_verl_parquet",
    "validate_m2rl_frame",
    "validate_m2rl_parquet",
}
_NEMOTRON_EXPORTS = {
    "SplitOutputs",
    "normalize_nemotron_record",
    "prepare_nemotron_rl_data",
}

__all__ = sorted(_M2RL_EXPORTS | _NEMOTRON_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _M2RL_EXPORTS:
        return getattr(import_module("grpo.data.m2rl"), name)
    if name in _NEMOTRON_EXPORTS:
        return getattr(import_module("grpo.data.nemotron"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
