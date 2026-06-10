"""Shared ToolRL official benchmark helpers."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from typing import Any

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
RESPONSE_OPEN = "<response>"
RESPONSE_CLOSE = "</response>"


def extract_between(text: str, open_tag: str, close_tag: str) -> str:
    if open_tag not in text or close_tag not in text:
        return ""
    return text.split(open_tag, 1)[1].split(close_tag, 1)[0].strip()


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    payload = extract_between(text, TOOL_OPEN, TOOL_CLOSE)
    if not payload:
        return []
    calls: list[dict[str, Any]] = []
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(stripped)
        if isinstance(parsed, Mapping):
            calls.append(dict(parsed))
    return calls


def normalize_answer(answer: Any) -> dict[str, Any]:
    if isinstance(answer, list):
        answer = answer[0] if answer else {}
    if not isinstance(answer, Mapping):
        return {}
    return dict(answer)


def exact_tool_match(predicted: dict[str, Any], answer: dict[str, Any]) -> bool:
    name = predicted.get("name")
    parameters = predicted.get("parameters")
    if name is None or parameters is None:
        name = answer.get("name")
        parameters = predicted
    return name == answer.get("name") and parameters == answer.get("parameters")


def score_single_tool_call(tool_calls: list[dict[str, Any]], answer: Any) -> int:
    normalized_answer = normalize_answer(answer)
    if not normalized_answer:
        return 0
    return int(any(exact_tool_match(tool_call, normalized_answer) for tool_call in tool_calls))


def response_text(output: str) -> str:
    response = extract_between(output, RESPONSE_OPEN, RESPONSE_CLOSE)
    if response:
        return response
    if THINK_CLOSE in output:
        return output.split(THINK_CLOSE, 1)[1].strip()
    return output.strip()
