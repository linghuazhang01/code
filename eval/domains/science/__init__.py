"""Science evaluation dataset metadata."""

from __future__ import annotations

SCIENCE_DATASETS = frozenset(
    {
        "MMLU-Pro",
        "SuperGPQA",
        "m2rl_gpqa_diamond",
        "m2rl_hle",
        "mmlupro",
        "supergpqa",
    }
)


def is_science_dataset(data_source: str) -> bool:
    normalized = data_source.strip().lower().replace("_", "-")
    return data_source in SCIENCE_DATASETS or normalized in {
        "mmlu-pro",
        "mmlupro",
        "supergpqa",
        "m2rl-gpqa-diamond",
        "m2rl-hle",
    }
