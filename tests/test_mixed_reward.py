from __future__ import annotations

from typing import Any

from pytest import MonkeyPatch

from mopd_verl import mixed_reward


def test_ifbench_training_rows_use_instruction_following_reward(
    monkeypatch: MonkeyPatch,
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
