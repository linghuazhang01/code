"""ToolRL/RLLA reward adapter for verl custom_reward_function."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
RESPONSE_OPEN = "<response>"
RESPONSE_CLOSE = "</response>"


def _env_enabled(name: str) -> bool:
    return str(os.getenv(name, "0")) == "1"


def _match_score(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_items = list(left)
    right_items = list(right)
    if left_items == right_items:
        return 1.0
    if _env_enabled("REFINEDREWARD"):
        return 0.0
    if not left_items or not right_items:
        return 0.0

    left_counts = Counter(left_items)
    right_counts = Counter(right_items)
    intersection = sum(min(left_counts[key], right_counts[key]) for key in left_counts.keys() & right_counts.keys())
    max_possible = len(left_items) + len(right_items) - intersection
    return intersection / max_possible if max_possible > 0 else 0.0


def _scheduled_bounds(
    *,
    step: int,
    max_reward: float,
    min_reward: float,
    max1_step30_max3: bool = False,
    schedule_reward: bool = False,
    for_format: bool = False,
) -> tuple[float, float]:
    if max1_step30_max3 and _env_enabled("MAX1STEP30MAX3"):
        if for_format and step >= 30:
            return max_reward / 2, min_reward / 2
        if not for_format and step < 30:
            return max_reward / 3, min_reward / 3

    if schedule_reward and _env_enabled("SCHEDULEREWARD"):
        if for_format:
            max_reward = 2 - (2 - max_reward) * step / 150
            min_reward = -2 + (2 + min_reward) * step / 150
            return max(max_reward, 1.0), min(min_reward, -1.0)
        max_reward = (max_reward - 2) * step / 150 + 2
        min_reward = (min_reward + 2) * step / 150 - 2
        return min(max_reward, 3.0), max(min_reward, -3.0)

    return max_reward, min_reward


def _single_tag(text: str, open_tag: str, close_tag: str) -> bool:
    return text.count(open_tag) == 1 and text.count(close_tag) == 1


def _format_reward(response: str, answer: str, step: int) -> float:
    max_reward, min_reward = _scheduled_bounds(
        step=step,
        max_reward=1.0,
        min_reward=0.0,
        max1_step30_max3=True,
        schedule_reward=True,
        for_format=True,
    )
    target_has_response = RESPONSE_OPEN in answer
    target_has_tool = TOOL_OPEN in answer

    if target_has_response and not target_has_tool:
        pattern = rf"^{THINK_OPEN}.*?{THINK_CLOSE}\n{RESPONSE_OPEN}.*?{RESPONSE_CLOSE}$"
        is_valid = bool(re.search(pattern, response, re.DOTALL)) and _single_tag(response, RESPONSE_OPEN, RESPONSE_CLOSE)
    elif not target_has_response and target_has_tool:
        pattern = rf"^{THINK_OPEN}.*?{THINK_CLOSE}\n{TOOL_OPEN}\n.*?\n{TOOL_CLOSE}$"
        is_valid = bool(re.search(pattern, response, re.DOTALL)) and _single_tag(response, TOOL_OPEN, TOOL_CLOSE)
    elif target_has_response and target_has_tool:
        pattern = rf"^{THINK_OPEN}.*?{THINK_CLOSE}\n{TOOL_OPEN}\n.*?\n{TOOL_CLOSE}\n{RESPONSE_OPEN}.*?{RESPONSE_CLOSE}$"
        is_valid = (
            bool(re.search(pattern, response, re.DOTALL))
            and _single_tag(response, TOOL_OPEN, TOOL_CLOSE)
            and _single_tag(response, RESPONSE_OPEN, RESPONSE_CLOSE)
        )
    else:
        pattern = rf"^{THINK_OPEN}.*?{THINK_CLOSE}$"
        is_valid = bool(re.search(pattern, response, re.DOTALL))

    return max_reward if is_valid else min_reward


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    if TOOL_OPEN not in text or TOOL_CLOSE not in text:
        return []
    payload = text.split(TOOL_OPEN, 1)[1].split(TOOL_CLOSE, 1)[0].strip()
    calls: list[dict[str, Any]] = []
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if not isinstance(item, Mapping):
            raise ValueError("Tool call line is not a JSON object.")
        calls.append(dict(item))
    return calls


def _tool_call_reward(
    gt_tools: list[dict[str, Any]],
    predicted_tools: list[dict[str, Any]],
    max_reward: float,
    min_reward: float,
) -> float:
    if gt_tools == predicted_tools:
        return max_reward
    if _env_enabled("COARSEREWARD"):
        return min_reward

    gt_names = [tool.get("name") for tool in gt_tools]
    predicted_names = [tool.get("name") for tool in predicted_tools]
    score = _match_score(gt_names, predicted_names)
    local_max = 1.0
    used_predicted: set[int] = set()

    for gt_tool in gt_tools:
        gt_name = gt_tool.get("name")
        gt_params = dict(gt_tool.get("parameters") or {})
        local_max += 1.0 if _env_enabled("INTERMEDIATEREWARD") else 1.0 + len(gt_params)
        best_score = 0.0
        best_index = -1
        for index, predicted_tool in enumerate(predicted_tools):
            if index in used_predicted or predicted_tool.get("name") != gt_name:
                continue
            if _env_enabled("INTERMEDIATEREWARD"):
                if gt_tool == predicted_tool:
                    best_score = 1.0
                    best_index = index
                    break
                continue

            predicted_params = dict(predicted_tool.get("parameters") or {})
            param_score = _match_score(gt_params.keys(), predicted_params.keys())
            correctness = sum(1.0 for key, value in gt_params.items() if predicted_params.get(key) == value)
            candidate_score = param_score + correctness
            if candidate_score > best_score:
                best_score = candidate_score
                best_index = index

        if best_index >= 0:
            used_predicted.add(best_index)
            score += best_score

    return (max_reward - min_reward) * score / local_max + min_reward


def _correctness_reward(response: str, answer: str, step: int) -> float:
    if _env_enabled("CORRECTMAX1"):
        max_reward, min_reward = 1.0, -1.0
    else:
        max_reward, min_reward = 3.0, -3.0
    max_reward, min_reward = _scheduled_bounds(
        step=step,
        max_reward=max_reward,
        min_reward=min_reward,
        max1_step30_max3=True,
        schedule_reward=True,
    )
    if TOOL_OPEN not in answer:
        return 0.0
    try:
        gt_tools = _extract_tool_calls(answer)
        predicted_tools = _extract_tool_calls(response)
    except (json.JSONDecodeError, ValueError):
        return min_reward
    if not predicted_tools:
        return min_reward
    return _tool_call_reward(gt_tools, predicted_tools, max_reward, min_reward)


def _length_reward(response: str) -> float:
    if THINK_OPEN not in response or THINK_CLOSE not in response:
        return 0.0
    thought = response.split(THINK_OPEN, 1)[1].split(THINK_CLOSE, 1)[0].strip()
    max_reward_len = 512
    return min(round(len(thought.split()) / max_reward_len, 2), 1.0)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, float]:
    """Return ToolRL's format, tool correctness, and optional length reward."""

    step = int((extra_info or {}).get("global_step", (extra_info or {}).get("step", 0)) or 0)
    response = solution_str.strip()
    format_score = _format_reward(response, ground_truth, step)
    correctness_score = _correctness_reward(response, ground_truth, step)
    length_score = _length_reward(response) if _env_enabled("WITHLENGTH") else 0.0
    score = format_score + correctness_score + length_score
    return {
        "score": float(score),
        "toolrl_format": float(format_score),
        "toolrl_correctness": float(correctness_score),
        "toolrl_length": float(length_score),
    }
