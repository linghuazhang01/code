"""Search-domain metadata and answer extraction helpers."""

from __future__ import annotations

import re

SEARCH_DATA_SOURCE_PREFIXES = ("searchR1_", "searchqa", "SearchQA")


def is_search_dataset(data_source: str) -> bool:
    return data_source.startswith(SEARCH_DATA_SOURCE_PREFIXES)


def extract_search_answer(text: str) -> str:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, flags=re.DOTALL))
    if not matches:
        return ""
    return matches[-1].group(1).strip()
