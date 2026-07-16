"""Reward router for mixed MOPD math/code and IF/Science training."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from mopd_verl.m2rl_reward import compute_score as compute_m2rl_score


def _normalize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _uses_m2rl_reward(data_source: str, extra_info: Any) -> bool:
    metadata = _normalize_metadata(extra_info)
    raw_reward_type = metadata.get("rm_type") or metadata.get("reward_type") or data_source
    reward_type = str(raw_reward_type or "").lower()
    return (
        "ifbench" in reward_type
        or "gpqa" in reward_type
        or "science" in reward_type
        or "instruction_following" in reward_type
        or reward_type in {"if", "instruction_following", "instruction-following", "science"}
    )


def _compute_default_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | str | None,
    **kwargs: Any,
) -> dict[str, float]:
    from verl.utils.reward_score import default_compute_score

    result = default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    # Return a dict with a consistent set of keys so that every item in a
    # mixed batch contributes the same entries to reward_extra_info.
    # The NaiveRewardManager aggregates per-item results into a
    # defaultdict(list) and later merges it into batch.non_tensor_batch.
    # If different items return different keys, the lengths won't match
    # and DataProto.check_consistency() will raise an AssertionError.
    if isinstance(result, dict):
        score = float(result.get("score", next(iter(result.values()))))
    else:
        score = float(result)
    return {
        "score": score,
        "m2rl_gpqa": 0.0,
        "m2rl_ifbench": 0.0,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | str | None = None,
    **kwargs: Any,
) -> Any:
    """Route M2RL rows to M2RL rewards and keep existing verl rewards for others."""

    if _uses_m2rl_reward(data_source, extra_info):
        return compute_m2rl_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=_normalize_metadata(extra_info),
            **kwargs,
        )
    return _compute_default_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
