from __future__ import annotations

from types import SimpleNamespace

from pytest import MonkeyPatch

from mopd_verl import m2rl_reward


def test_partial_instruction_registry_coverage_uses_official_fallback(
    monkeypatch: MonkeyPatch,
) -> None:
    registry = SimpleNamespace(INSTRUCTION_DICT={"known": object})
    monkeypatch.setattr(
        m2rl_reward,
        "_ensure_verifiable_instruction_registry",
        lambda: registry,
    )

    result = m2rl_reward._compute_verifiable_instruction_reward(
        response="response",
        instruction_ids=["known", "ifbench:new_constraint"],
        kwargs_list=[{}, {}],
        metadata={},
    )

    assert result is None
