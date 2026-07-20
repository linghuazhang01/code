"""Adapters from eval samples to the original project reward scorers."""

from __future__ import annotations

import inspect
import os
from typing import Any

from eval.common import EvalSample, remove_think_block
from eval.domains.math import extract_final_answer, simple_score_math_answer
from eval.domains.search import extract_search_answer

SCORER_NAME = "verl.utils.reward_score.default_compute_score"


def _load_default_compute_score() -> Any:
    try:
        from verl.utils.reward_score import default_compute_score
    except ModuleNotFoundError as exc:
        if os.environ.get("MOPD_ALLOW_SIMPLE_SCORER_FALLBACK") == "1":
            return None
        raise RuntimeError(
            "Original verl reward scorer is unavailable. Set PYTHONPATH to include "
            "code/third_party/verl or run inside the mopd-verl environment."
        ) from exc
    return default_compute_score


def score_with_project_reward(
    data_source: str,
    completion: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    *,
    allow_simple_math_fallback: bool = True,
) -> tuple[float, list[dict[str, Any]] | None]:
    rm_type = str((extra_info or {}).get("rm_type") or "").lower()
    if rm_type in {"ifbench", "gpqa"}:
        from mopd_verl.m2rl_reward import compute_score as compute_m2rl_score

        result = compute_m2rl_score(data_source, completion, ground_truth, extra_info=extra_info)
        return float(result["score"]), [result]
    if data_source in {"HumanEvalPlus", "MBPPPlus", "LiveCodeBench"}:
        from mopd_verl.code_reward import compute_score as compute_code_score

        score, metadata = compute_code_score(data_source, completion, ground_truth)
        return float(score), metadata
    compute_score = _load_default_compute_score()
    if compute_score is None:
        if not allow_simple_math_fallback:
            raise RuntimeError(
                "The project reward scorer is unavailable, and the simple Math "
                f"fallback is invalid for data_source={data_source!r}. Run inside "
                "the complete mopd-verl environment."
            )
        score, _ = simple_score_math_answer(completion, ground_truth)
        return score, [{"scorer": "simple_math_fallback"}]

    signature = inspect.signature(compute_score)
    supports_extra_info = "extra_info" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )
    if supports_extra_info:
        result = compute_score(data_source, completion, ground_truth, extra_info=extra_info)
    else:
        result = compute_score(data_source, completion, ground_truth)
    if isinstance(result, dict):
        score_value = result.get("score", result.get("reward", result.get("accuracy", 0.0)))
        return float(score_value), [result]
    if isinstance(result, tuple):
        return float(result[0]), [{"raw": repr(result[1:])}]
    return float(result), None


def score_completion(
    sample: EvalSample,
    completion: str,
    *,
    score_code: bool,
) -> tuple[float | None, str, list[dict[str, Any]] | None]:
    if sample.ability == "code" and not score_code:
        return None, "", None
    completion_for_scoring = remove_think_block(completion)
    if sample.ability == "math":
        prediction = extract_final_answer(completion_for_scoring)
    elif sample.ability == "reasoning":
        prediction = extract_final_answer(completion_for_scoring)
    elif sample.ability == "search":
        prediction = extract_search_answer(completion_for_scoring)
    elif sample.ability == "tool":
        return None, "", [{"scorer": "external_tool_eval_required"}]
    elif sample.ability == "if":
        prediction = completion_for_scoring
    elif sample.ability == "science":
        prediction = completion_for_scoring
    else:
        prediction = ""
    if str(sample.extra_info.get("rm_type") or "").lower() == "hle":
        return None, prediction, [{"scorer": "official_hle_judge_required"}]
    score, metadata = score_with_project_reward(
        sample.dataset,
        completion_for_scoring,
        sample.ground_truth,
        sample.extra_info,
        allow_simple_math_fallback=sample.ability in {"math", "reasoning"},
    )
    return score, prediction, metadata
