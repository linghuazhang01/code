"""Math-domain answer extraction helpers for reporting and fallback scoring."""

from __future__ import annotations

import re


def _extract_braced_content(text: str, start: int) -> str | None:
    depth = 0
    content_start = None
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
            if content_start is None:
                content_start = index + 1
        elif char == "}":
            depth -= 1
            if depth == 0 and content_start is not None:
                return text[content_start:index]
    return None


def extract_boxed_answer(text: str) -> str | None:
    marker = "\\boxed"
    start = text.rfind(marker)
    if start < 0:
        return None
    brace = text.find("{", start + len(marker))
    if brace < 0:
        return None
    return _extract_braced_content(text, brace)


def extract_final_answer(text: str) -> str:
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed.strip()

    answer_patterns = [
        r"(?i)final answer\s*(?:is|:)?\s*([^\n]+)",
        r"(?i)answer\s*(?:is|:)?\s*([^\n]+)",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()

    numeric_matches = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text.replace(",", ""))
    if numeric_matches:
        return numeric_matches[-1].strip()

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return non_empty_lines[-1] if non_empty_lines else ""


def normalize_answer(value: str) -> str:
    normalized = value.strip()
    normalized = normalized.replace("\\,", "")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\(", "").replace("\\)", "")
    normalized = normalized.replace("\\[", "").replace("\\]", "")
    normalized = re.sub(r"\\text\{([^{}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip(".")
    return normalized.lower()


def simple_score_math_answer(completion: str, ground_truth: str) -> tuple[float, str]:
    prediction = extract_final_answer(completion)
    score = float(normalize_answer(prediction) == normalize_answer(ground_truth))
    return score, prediction
