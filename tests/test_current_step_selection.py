"""Independent examples for current-step type/position head and tail ranking."""

from dataclasses import replace

import pytest

from mopd_verl.domain_gradient.current_step_selection import (
    CurrentStepSelection,
    compact_token_positions,
    select_current_step,
)
from mopd_verl.domain_gradient.occurrence import Position


def _positions(
    tokens: list[int], losses: list[float], confidence: list[float] | None = None,
) -> list[Position]:
    return [
        Position("math", 0, 0, column, token, loss,
                 confidence[column] if confidence is not None else None)
        for column, (token, loss) in enumerate(zip(tokens, losses, strict=True))
    ]


def _select(
    ranks: list, *, unit: str = "token_id", head: float = .2, tail: float = .2,
    mode: str = "top_loss", tail_mode: str = "bottom_loss", minimum: float = 0,
) -> CurrentStepSelection:
    return select_current_step(
        ranks, {"math": [300, 301, 302]}, {"math": head}, {"math": tail},
        minimum, True, unit=unit, head_modes={"math": mode}, tail_mode=tail_mode,
    )


def test_types_aggregate_across_ranks_and_use_whole_id_budgets() -> None:
    # ID300 globally has loss (10 + 0 + 0)/3, ID301 has (5 + 5)/2.
    ranks = [
        ({"math": 4}, _positions([300, 301, 302], [10., 5., 1.]), None),
        ({"math": 6}, _positions([300, 300, 301, 302], [0., 0., 5., 1.]), None),
    ]
    result = _select(ranks, head=.1, tail=.1)
    assert result.head == {"math": frozenset({(301,)})}
    assert result.tail == {"math": frozenset({(302,)})}
    assert result.metrics["math/current_step/head/budget_count"] == 1
    assert result.metrics["math/current_step/head/selected_count"] == 2
    assert result.metrics["math/current_step/head/overshoot"] == 1
    assert result.metrics["math/current_step/tail/coverage"] == .2
    positional = _select(ranks, unit="occurrence", head=.1, tail=.1)
    assert positional.head["math"] == {(0, 0, 0, 0)}
    assert positional.tail["math"] == {(1, 0, 0, 0)}


def test_compacted_statistics_equal_occurrence_input_with_unequal_counts() -> None:
    ranks = [
        ({"math": 5}, _positions([300, 300, 301], [8., 2., 5.], [-.1, -.9, -2.]), None),
        ({"math": 7}, _positions([300, 301, 301, 302], [2., 1., 9., .1], [-2., -.1, -2., -4.]), None),
    ]
    compact = [(counts, compact_token_positions(p), error) for counts, p, error in ranks]
    assert len(compact[0][1]) == 2
    for mode in ("top_loss", "top_teacher_confidence", "top_loss_teacher_confidence"):
        raw_result = _select(ranks, mode=mode, tail_mode="bottom_loss_teacher_confidence")
        reduced_result = _select(compact, mode=mode, tail_mode="bottom_loss_teacher_confidence")
        assert raw_result.head == reduced_result.head
        assert raw_result.tail == reduced_result.tail
        assert raw_result.metrics == pytest.approx(reduced_result.metrics)


def test_strict_count_gate_missing_ids_and_zero_tail_budget() -> None:
    ranks = [({"math": 10}, _positions([300, 300, 301], [1., 1., 100.]), None)]
    result = _select(ranks, minimum=1, tail=0)
    assert result.head["math"] == {(300,)}
    assert result.tail["math"] == set()
    assert result.metrics["math/current_step/eligible_unit_count"] == 1
    assert (301,) not in result.head["math"] | result.tail["math"]
    assert (302,) not in result.head["math"] | result.tail["math"]
    gated = _select(ranks, minimum=2)
    assert gated.head["math"] == gated.tail["math"] == set()


def test_head_priority_shortfall_and_deterministic_ties() -> None:
    ranks = [({"math": 10}, _positions([302, 301, 300], [1., 1., 1.]), None)]
    result = _select(ranks, head=.2, tail=.5)
    assert result.head["math"] == {(300,), (301,)}
    assert result.tail["math"] == {(302,)}
    assert result.metrics["math/current_step/tail/shortfall"] == 4
    assert not result.head["math"] & result.tail["math"]


def test_confidence_is_chosen_logp_and_can_drive_head_and_tail_independently() -> None:
    ranks = [({"math": 5}, _positions([300, 301, 302], [10., 1., 5.], [-4., -.1, -2.]), None)]
    tc = _select(ranks, mode="top_teacher_confidence", tail_mode="bottom_teacher_confidence")
    assert tc.head["math"] == {(301,)}
    assert tc.tail["math"] == {(300,)}
    mixed = _select(ranks, mode="top_loss", tail_mode="bottom_teacher_confidence")
    assert mixed.head["math"] == {(300,)}
    assert mixed.tail["math"] == {(302,)}  # The lowest confidence ID is reserved for head.


def test_composite_score_hand_oracle_and_shared_eligible_normalization() -> None:
    # Symmetric endpoints: normalized loss=(0,.5,1), confidence=(1,.5,0).
    # L+C+L*C gives (1,1.25,1): ID301 is head; ID300 wins the tail tie.
    ranks = [({"math": 5}, _positions([300, 301, 302], [0., 1., 2.], [0., -1., -2.]), None)]
    result = _select(ranks, mode="top_loss_teacher_confidence", tail_mode="bottom_loss_teacher_confidence")
    assert result.head["math"] == {(301,)}
    assert result.tail["math"] == {(300,)}
    metrics = result.metrics
    for signal in ("loss", "teacher_logp"):
        for quantile in ("q02", "q98"):
            suffix = f"loss_teacher_confidence_{signal}_{quantile}"
            assert metrics["math/current_step/head/" + suffix] == metrics["math/current_step/tail/" + suffix]


def test_domain_budgets_and_ranking_are_independent() -> None:
    math_positions = _positions([300, 301], [2., 1.])
    code_positions = [replace(p, domain="code", row=1, score=3. - p.score) for p in math_positions]
    result = select_current_step(
        [({"math": 10, "code": 20}, math_positions + code_positions, None)],
        {"math": [300, 301], "code": [300, 301]},
        {"math": .1, "code": .05}, {"math": .1, "code": 0},
        0, True, unit="token_id", head_modes={"math": "top_loss", "code": "top_loss"},
        tail_mode="bottom_loss",
    )
    assert result.head == {"math": {(300,)}, "code": {(301,)}}
    assert result.tail == {"math": {(301,)}, "code": set()}


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), .1])
def test_bad_teacher_confidence_fails_instead_of_becoming_low_confidence(bad: float | None) -> None:
    positions = [Position("math", 0, 0, 0, 300, 1., bad)]
    with pytest.raises(ValueError, match="chosen-token logp"):
        _select([({"math": 5}, positions, None)], mode="top_teacher_confidence")
    # The loss-only path has no confidence dependency.
    assert _select([({"math": 5}, positions, None)]).head["math"] == {(300,)}


def test_remote_data_error_fails_before_selection() -> None:
    with pytest.raises(ValueError, match="current-step scoring failed: missing teacher"):
        _select([({"math": 5}, [], None), ({"math": 0}, [], "missing teacher")])
