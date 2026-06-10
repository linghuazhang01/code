"""General-Reasoner/WebInstruct evaluation metadata."""

from __future__ import annotations

GREASONER_DATASETS = frozenset({"general-reasoner", "WebInstructVerified", "TIGER-Lab/WebInstruct-verified"})


def is_greasoner_dataset(data_source: str) -> bool:
    return data_source in GREASONER_DATASETS or data_source.startswith(("general-reasoner", "WebInstruct"))
