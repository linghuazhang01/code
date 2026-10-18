from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from mopd_verl import mixed_reward


def test_ifbench_training_rows_use_instruction_following_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_m2rl_score(**kwargs: Any) -> dict[str, float]:
        captured.update(kwargs)
        return {"score": 1.0, "m2rl_gpqa": 0.0, "m2rl_ifbench": 1.0}

    monkeypatch.setattr(mixed_reward, "compute_m2rl_score", fake_m2rl_score)

    result = mixed_reward.compute_score(
        data_source="m2rl_ifbench",
        solution_str="compliant response",
        ground_truth="",
        extra_info='{"rm_type": "ifbench", "domain": "if"}',
    )

    assert result["m2rl_ifbench"] == 1.0
    assert captured["data_source"] == "m2rl_ifbench"
    assert captured["extra_info"] == {"rm_type": "ifbench", "domain": "if"}


@pytest.mark.parametrize("data_source", ["HMMT25Feb", "HMMT25Nov"])
def test_hmmt_validation_rows_use_math_verify_reward(
    monkeypatch: pytest.MonkeyPatch,
    data_source: str,
) -> None:
    captured: dict[str, str] = {}

    def fake_math_verify_score(
        solution_str: str,
        ground_truth: Any,
    ) -> float:
        captured["solution_str"] = solution_str
        captured["ground_truth"] = str(ground_truth)
        return 1.0

    monkeypatch.setattr(
        mixed_reward,
        "_compute_math_verify_score",
        fake_math_verify_score,
    )

    result = mixed_reward.compute_score(
        data_source=data_source,
        solution_str="The answer is \\boxed{42}.",
        ground_truth=42,
        extra_info={},
    )

    assert result == {
        "score": 1.0,
        "m2rl_gpqa": 0.0,
        "m2rl_ifbench": 0.0,
    }
    assert captured == {
        "solution_str": "The answer is \\boxed{42}.",
        "ground_truth": "42",
    }


def test_math_verify_adapter_converts_ground_truth_to_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verl.utils.reward_score import math_verify

    captured: dict[str, str] = {}

    def fake_compute_score(model_output: str, ground_truth: str) -> bool:
        captured["model_output"] = model_output
        captured["ground_truth"] = ground_truth
        return True

    monkeypatch.setattr(math_verify, "compute_score", fake_compute_score)

    result = mixed_reward._compute_math_verify_score(
        "The answer is \\boxed{42}.",
        42,
    )

    assert result == 1.0
    assert captured == {
        "model_output": "The answer is \\boxed{42}.",
        "ground_truth": "42",
    }


def test_batched_reward_entrypoint_forwards_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mopd_verl import batched_reward

    captured: dict[str, Any] = {}

    def fake_score_batched(**kwargs: Any) -> list[dict[str, float]]:
        captured.update(kwargs)
        return [{"score": 1.0}]

    monkeypatch.setattr(batched_reward, "compute_score_batched", fake_score_batched)

    result = mixed_reward.compute_score_batched(
        data_sources=["DeepMath"],
        solution_strs=["answer"],
        ground_truths=["answer"],
        extra_infos=[{}],
        max_workers=7,
        batch_timeout_seconds=12.5,
    )

    assert result == [{"score": 1.0}]
    assert captured["max_workers"] == 7
    assert captured["batch_timeout_seconds"] == 12.5


def test_batched_reward_normalizes_all_metric_keys() -> None:
    from mopd_verl.batched_reward import _fallback_result, _normalize_result

    assert _normalize_result({"score": 0.5, "m2rl_gpqa": 1.0}) == {
        "score": 0.5,
        "m2rl_gpqa": 1.0,
        "m2rl_ifbench": 0.0,
        "reward_timeout": 0.0,
        "reward_error": 0.0,
    }
    assert _fallback_result(timed_out=True) == {
        "score": 0.0,
        "m2rl_gpqa": 0.0,
        "m2rl_ifbench": 0.0,
        "reward_timeout": 1.0,
        "reward_error": 0.0,
    }


def test_batched_reward_terminates_pool_at_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mopd_verl import batched_reward

    async_result = SimpleNamespace(ready=lambda: False)
    pool = MagicMock()
    pool.apply_async.return_value = async_result

    monkeypatch.setattr(batched_reward, "_create_pool", lambda _processes: pool)
    monotonic_values = iter([0.0, 1.0])
    monkeypatch.setattr(
        batched_reward.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    result = batched_reward.compute_score_batched(
        data_sources=["DeepMath"],
        solution_strs=["pathological answer"],
        ground_truths=["0"],
        extra_infos=[{}],
        max_workers=1,
        batch_timeout_seconds=0.5,
    )

    assert result[0]["score"] == 0.0
    assert result[0]["reward_timeout"] == 1.0
    pool.terminate.assert_called_once_with()
    pool.close.assert_not_called()
    pool.join.assert_called_once_with()


def test_batched_reward_pool_workers_are_not_daemonic() -> None:
    from mopd_verl.batched_reward import _NonDaemonSpawnProcess

    process = _NonDaemonSpawnProcess()
    process.daemon = True  # multiprocessing.Pool requests daemonic workers

    assert process.daemon is False


def _worker_start_method() -> str:
    import multiprocessing

    return multiprocessing.get_start_method()


def test_batched_reward_pool_workers_fork_their_child_processes() -> None:
    from mopd_verl.batched_reward import _create_pool

    pool = _create_pool(1)
    try:
        start_method = pool.apply(_worker_start_method)
    finally:
        pool.terminate()
        pool.join()

    assert start_method == "fork"


_CODE_TESTS = {"inputs": ["1 2\n", "5 7\n"], "outputs": ["3\n", "12\n"]}
_CODE_SOLUTIONS = {
    "correct": "```python\na, b = map(int, input().split())\nprint(a + b)\n```",
    "wrong": "```python\na, b = map(int, input().split())\nprint(a - b)\n```",
}


def test_batched_reward_scores_code_rollouts_inside_pool_workers() -> None:
    # Regression: prime_code starts a Manager and a child Process per sample,
    # which fails inside a daemonic Pool worker and used to score every Code
    # rollout 0.
    pytest.importorskip("verl.utils.reward_score.prime_code")
    from mopd_verl import batched_reward

    result = batched_reward.compute_score_batched(
        data_sources=["taco", "taco"],
        solution_strs=[_CODE_SOLUTIONS["correct"], _CODE_SOLUTIONS["wrong"]],
        ground_truths=[json.dumps(_CODE_TESTS)] * 2,
        extra_infos=[{}, {}],
        max_workers=2,
        batch_timeout_seconds=120.0,
    )

    assert [item["score"] for item in result] == [1.0, 0.0]
    assert all(item["reward_timeout"] == 0.0 for item in result)
    assert all(item["reward_error"] == 0.0 for item in result)
