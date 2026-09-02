from __future__ import annotations

from typing import Any

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
