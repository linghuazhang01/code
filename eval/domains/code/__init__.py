"""Code-domain metadata helpers for paper-eval datasets."""

from __future__ import annotations

CODE_DATASETS = frozenset({"HumanEvalPlus", "MBPPPlus", "LiveCodeBench"})


def is_code_dataset(data_source: str) -> bool:
    return data_source in CODE_DATASETS
