"""Bounded multiprocessing for mixed rule-based reward scoring."""

from __future__ import annotations

import logging
import multiprocessing
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


LOGGER = logging.getLogger(__name__)
_BASE_REWARD = {
    "score": 0.0,
    "m2rl_gpqa": 0.0,
    "m2rl_ifbench": 0.0,
}


@dataclass(frozen=True)
class RewardTask:
    """One independently scoreable rollout."""

    data_source: str
    solution_str: str
    ground_truth: Any
    extra_info: Any


def _normalize_result(result: Any) -> dict[str, float]:
    if isinstance(result, dict):
        score = float(result.get("score", next(iter(result.values()))))
        gpqa_score = float(result.get("m2rl_gpqa", 0.0))
        ifbench_score = float(result.get("m2rl_ifbench", 0.0))
    else:
        score = float(result)
        gpqa_score = 0.0
        ifbench_score = 0.0
    return {
        "score": score,
        "m2rl_gpqa": gpqa_score,
        "m2rl_ifbench": ifbench_score,
        "reward_timeout": 0.0,
        "reward_error": 0.0,
    }


def _fallback_result(*, timed_out: bool) -> dict[str, float]:
    return {
        **_BASE_REWARD,
        "reward_timeout": float(timed_out),
        "reward_error": float(not timed_out),
    }


def _score_task(task: RewardTask) -> dict[str, float]:
    # This function lives in an importable module so it remains pickle-safe
    # when the custom reward entry point is loaded dynamically by verl.
    from mopd_verl.mixed_reward import compute_score

    result = compute_score(
        data_source=task.data_source,
        solution_str=task.solution_str,
        ground_truth=task.ground_truth,
        extra_info=task.extra_info,
    )
    return _normalize_result(result)


def _collect_ready_results(
    pending: dict[int, Any],
    results: list[dict[str, float] | None],
) -> int:
    error_count = 0
    for index, async_result in list(pending.items()):
        if not async_result.ready():
            continue
        try:
            results[index] = async_result.get()
        except Exception:  # noqa: BLE001 - one bad reward must not stop training
            results[index] = _fallback_result(timed_out=False)
            error_count += 1
        del pending[index]
    return error_count


def compute_score_batched(
    data_sources: Sequence[str],
    solution_strs: Sequence[str],
    ground_truths: Sequence[Any],
    extra_infos: Sequence[Any],
    max_workers: int = 32,
    batch_timeout_seconds: float = 120.0,
) -> list[dict[str, float]]:
    """Score a batch in child processes and hard-bound pathological graders."""

    sizes = {
        len(data_sources),
        len(solution_strs),
        len(ground_truths),
        len(extra_infos),
    }
    if len(sizes) != 1:
        raise ValueError("Batched reward inputs must have equal lengths.")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    if batch_timeout_seconds <= 0:
        raise ValueError("batch_timeout_seconds must be positive.")

    tasks = [
        RewardTask(
            data_source=str(data_source),
            solution_str=str(solution_str),
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources,
            solution_strs,
            ground_truths,
            extra_infos,
            strict=True,
        )
    ]
    if not tasks:
        return []

    context = multiprocessing.get_context("spawn")
    pool = context.Pool(processes=min(max_workers, len(tasks)))
    pending = {
        index: pool.apply_async(_score_task, (task,))
        for index, task in enumerate(tasks)
    }
    results: list[dict[str, float] | None] = [None] * len(tasks)
    deadline = time.monotonic() + batch_timeout_seconds
    error_count = 0

    try:
        while pending and time.monotonic() < deadline:
            error_count += _collect_ready_results(pending, results)
            if pending:
                time.sleep(0.01)
        # Capture tasks that crossed into ready state exactly at the deadline.
        error_count += _collect_ready_results(pending, results)
    finally:
        if pending:
            pool.terminate()
        else:
            pool.close()
        pool.join()

    timeout_count = len(pending)
    for index in pending:
        results[index] = _fallback_result(timed_out=True)

    if timeout_count or error_count:
        LOGGER.warning(
            "Batched reward completed with %d timeout(s) and %d error(s) out of %d samples.",
            timeout_count,
            error_count,
            len(tasks),
        )

    return [
        result if result is not None else _fallback_result(timed_out=True)
        for result in results
    ]
