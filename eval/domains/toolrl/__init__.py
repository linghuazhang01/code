"""ToolRL evaluation metadata."""

from __future__ import annotations

TOOLRL_EVAL_DATASETS = frozenset({"BFCL", "API-Bank", "Bamboogle"})


def is_toolrl_dataset(data_source: str) -> bool:
    return data_source in TOOLRL_EVAL_DATASETS or data_source.startswith(("ToolRL", "toolrl"))
