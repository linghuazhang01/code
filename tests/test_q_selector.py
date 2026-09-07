"""Independent occurrence-level contract tests for the globally normalized Q."""

from __future__ import annotations

import math
from typing import Any, Sequence

import pytest
import torch

from mopd_verl.domain_gradient.control_top_loss import (
    initial_online_control_selection_state,
    update_online_control_selection,
)
from mopd_verl.domain_gradient.control_top_loss_runtime import (
    global_candidate_loss_statistics_with_valid_counts as aggregate,
)


MODE = "top_q_loss_entropy"
Row = tuple[str, int, float, float, bool]


def _batches(rows: Sequence[Row], sizes: Sequence[int]) -> dict[str, Any]:
    assert sum(sizes) == len(rows)
    result: dict[str, Any] = {
        name: [] for name in (
            "token_id_batches", "loss_batches", "mask_batches",
            "label_batches", "student_entropy_batches",
        )
    }
    offset = 0
    for size in sizes:
        chunk = rows[offset:offset + size]
        offset += size
        result["token_id_batches"].append(torch.tensor([[r[1]] for r in chunk]))
        result["loss_batches"].append(torch.tensor(
            [[r[2]] for r in chunk], dtype=torch.float64, requires_grad=True,
        ))
        result["student_entropy_batches"].append(torch.tensor(
            [[r[3]] for r in chunk], dtype=torch.float64, requires_grad=True,
        ))
        result["mask_batches"].append(torch.tensor(
            [[r[4]] for r in chunk], dtype=torch.bool,
        ))
        result["label_batches"].append(tuple(r[0] for r in chunk))
    return result


def _run(
    rows: Sequence[Row], candidates: dict[str, tuple[int, ...]],
    sizes: Sequence[int] | None = None,
) -> Any:
    return aggregate(
        **_batches(rows, sizes if sizes is not None else (len(rows),)),
        domains=tuple(candidates), domain_candidate_token_ids=candidates,
        selection_mode=MODE,
    )


def _oracle(
    rows: Sequence[Row], candidates: dict[str, tuple[int, ...]],
) -> tuple[dict[str, dict[int, tuple[float, int]]], dict[str, int], dict[str, float]]:
    """Compute each Q directly; do not reuse production scoring or packing."""
    scores, counts, sums = {}, {}, {}
    for domain, allowed in candidates.items():
        valid = [r for r in rows if r[0] == domain and r[4]]
        mean_l = math.fsum(abs(r[2]) for r in valid) / len(valid)
        mean_h = math.fsum(r[3] for r in valid) / len(valid)
        occurrences = [
            (r[1], abs(r[2]) / mean_l + r[3] / mean_h
             + abs(r[2]) * r[3] / (mean_l * mean_h))
            for r in valid
        ]
        scores[domain] = {
            token: (math.fsum(q for tid, q in occurrences if tid == token),
                    sum(tid == token for tid, _ in occurrences))
            for token in allowed if any(tid == token for tid, _ in occurrences)
        }
        counts[domain] = len(valid)
        sums[domain] = math.fsum(q for _, q in occurrences)
    return scores, counts, sums


def _assert_oracle(
    result: Any, rows: Sequence[Row], candidates: dict[str, tuple[int, ...]],
) -> None:
    scores, counts, sums = _oracle(rows, candidates)
    assert result.valid_token_counts == counts
    assert result.valid_score_sums == pytest.approx(sums)
    assert set(result.by_domain) == set(scores)
    for domain, expected in scores.items():
        assert set(result.by_domain[domain]) == set(expected)
        for token, (score, count) in expected.items():
            actual_score, actual_count = result.by_domain[domain][token]
            assert actual_count == count
            assert actual_score == pytest.approx(score)


@pytest.mark.parametrize("sizes", [(8,), (1, 7), (3, 2, 3), (1,) * 8])
def test_brute_force_and_microbatch_repartition(sizes: tuple[int, ...]) -> None:
    rows = [
        ("math", 10, -1.0, 8.0, True), ("code", 20, 3.0, 1.0, True),
        ("math", 10, 7.0, 2.0, True), ("math", 20, 20.0, 4.0, True),
        ("code", 10, -40.0, 2.0, True), ("math", 999, 2.0, 20.0, True),
        ("code", 999, 5.0, 9.0, True), ("code", 20, 1.0, 7.0, True),
    ]
    # The other domain's candidate and out-of-pool 999 enter denominators only.
    candidates = {"math": (10, 77), "code": (20,)}
    _assert_oracle(_run(rows, candidates, sizes), rows, candidates)


def test_cross_term_is_occurrence_product_over_product_of_means() -> None:
    rows = [("math", 10, -1.0, 1.0, True), ("math", 10, 9.0, 9.0, True)]
    result = _run(rows, {"math": (10,)})
    # mean(LH)=41, mean(L)*mean(H)=25; product-of-token-means is also wrong.
    assert result.by_domain["math"][10] == pytest.approx((7.28, 2))
    assert result.valid_score_sums["math"] == pytest.approx(7.28)
    assert result.by_domain["math"][10][0] / 2 > 3.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -3.0])
def test_masked_invalid_values_are_ignored(bad: float) -> None:
    rows = [("math", 10, 2.0, 4.0, True), ("math", 10, bad, bad, False)]
    _assert_oracle(_run(rows, {"math": (10,)}), rows, {"math": (10,)})


def test_negative_entropy_error_includes_global_counts() -> None:
    with pytest.raises(ValueError, match="'negative_finite_entropy': 1.0"):
        _run([("math", 10, 1.0, -0.125, True)], {"math": (10,)})


def test_missing_entropy_is_an_explicit_error() -> None:
    batches = _batches([("math", 10, 1.0, 1.0, True)], (1,))
    del batches["student_entropy_batches"]
    with pytest.raises(ValueError, match="(?i)entropy"):
        aggregate(**batches, domains=("math",), candidate_token_ids=(10,),
                  selection_mode=MODE)


@pytest.mark.parametrize("token", [10, 999])
@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_entropy_including_outside_pool_raises(token: int, bad: float) -> None:
    with pytest.raises(ValueError, match="(?i)entropy|non.?finite|non.?negative"):
        _run([("math", 10, 1.0, 1.0, True), ("math", token, 1.0, bad, True)],
             {"math": (10,)})


@pytest.mark.parametrize("loss,entropy", [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0)])
def test_zero_domain_denominator_raises(loss: float, entropy: float) -> None:
    with pytest.raises(ValueError, match="(?i)mean|denominator|positive|zero"):
        _run([("math", 10, loss, entropy, True)], {"math": (10,)})


@pytest.mark.parametrize("masked", [False, True])
def test_empty_domain_raises(masked: bool) -> None:
    rows = [("math", 10, 1.0, 1.0, True)]
    if masked:
        rows.append(("code", 10, float("nan"), float("nan"), False))
    with pytest.raises(ValueError, match="(?i)empty|valid|count|domain"):
        _run(rows, {"math": (10,), "code": (10,)})


def _moments(rows: Sequence[Row], candidates: dict[str, tuple[int, ...]]) -> torch.Tensor:
    """Independent wire-format oracle: L, H, LH, N; all-valid; error sentinel."""
    union = sorted({token for tokens in candidates.values() for token in tokens})
    packed = torch.zeros((len(candidates), 4, len(union) + 2), dtype=torch.float64)
    for d, (domain, allowed) in enumerate(candidates.items()):
        for label, token, loss, entropy, valid in rows:
            if label != domain or not valid:
                continue
            values = torch.tensor([abs(loss), entropy, abs(loss) * entropy, 1.0],
                                  dtype=torch.float64)
            packed[d, :, -2] += values
            if token in allowed:
                packed[d, :, union.index(token)] += values
    return packed


def test_distributed_pools_moments_before_normalization_and_detaches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = {"math": (10,), "code": (20,)}
    local = [("math", 10, -1.0, 1.0, True)]
    remote = [("math", 10, 9.0, 3.0, True), ("math", 999, 30.0, 8.0, True),
              ("code", 20, 2.0, 4.0, True)]
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    calls = []

    def all_reduce(packed: torch.Tensor, op: Any = None, **kwargs: Any) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        assert not packed.requires_grad
        assert packed.grad_fn is None
        torch.testing.assert_close(packed, _moments(local, candidates))
        calls.append(packed.clone())
        packed.add_(_moments(remote, candidates))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    batches = _batches(local, (1,))
    result = aggregate(**batches, domains=tuple(candidates),
                       domain_candidate_token_ids=candidates, selection_mode=MODE)
    assert len(calls) == 1
    _assert_oracle(result, local + remote, candidates)
    local_q = _oracle(local, {"math": (10,)})[0]["math"][10][0]
    remote_q = _oracle(remote, {"math": (10,)})[0]["math"][10][0]
    assert result.by_domain["math"][10][0] != pytest.approx(local_q + remote_q)
    for name in ("loss_batches", "student_entropy_batches"):
        assert all(t.grad is None for t in batches[name])


def test_remote_error_sentinel_is_checked_after_one_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    calls = []

    def all_reduce(packed: torch.Tensor, op: Any = None, **kwargs: Any) -> None:
        assert packed.shape == (1, 4, 3)
        calls.append(op)
        packed[0, 0, -1] += 1.0

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    with pytest.raises(ValueError):
        _run([("math", 10, 1.0, 1.0, True)], {"math": (10,)})
    assert calls == [torch.distributed.ReduceOp.SUM]


def test_local_invalid_entropy_reaches_collective_before_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    calls = []

    def all_reduce(packed: torch.Tensor, op: Any = None, **kwargs: Any) -> None:
        assert packed.shape == (1, 4, 3)
        assert packed[0, 0, -1] > 0
        assert not packed.requires_grad
        calls.append(op)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    with pytest.raises(ValueError, match="(?i)entropy|non.?negative|invalid"):
        _run([("math", 10, 1.0, 1.0, True), ("math", 999, 1.0, -1.0, True)],
             {"math": (10,)})
    assert calls == [torch.distributed.ReduceOp.SUM]


def test_joint_top_p_strict_gate_and_next_step_fixed_state() -> None:
    state = initial_online_control_selection_state(
        ("math",), (10, 20, 30, 40), audit_interval_steps=1, window_steps=1,
        min_mean_occurrences_per_step=1.0, strict_occurrence_gate=True,
        top_k=1, budget_mode="top_p", top_p=0.5, selection_mode=MODE,
        weight_mode="fixed", candidate_token_groups={
            "math": {"control": (10, 20), "structure": (30, 40)},
        },
    )
    # At t, 10 fails the strict >1 gate despite the highest Q. A joint budget
    # must take 20, 30, and 40 to cover 5 of all 10 valid occurrences.
    outcome, next_state = update_online_control_selection(
        state, {"math": {10: (100.0, 1), 20: (18.0, 2),
                         30: (16.0, 2), 40: (2.0, 2)}},
        step=1, valid_token_counts={"math": 10}, valid_score_sums={"math": 140.0},
    )
    assert state.active_map() == {"math": ()}
    assert outcome.audit_triggered
    assert next_state.active_map() == {"math": (20, 30, 40)}
    detail = outcome.domain_results[0]
    assert detail.eligible_token_count == 3
    assert detail.target_occurrence_count == 5
    assert detail.selected_occurrence_count == 6
    # Fixed4 is configured by the consumer: fixed state emits no Q-derived
    # overrides (in particular no 1+Q weights), preserving that fixed weight.
    assert next_state.weight_mode == "fixed"
    assert next_state.active_weight_map() == {"math": {}}
    _, after_next_step = update_online_control_selection(
        next_state, {"math": {10: (60.0, 6), 20: (1.0, 2)}},
        step=2, valid_token_counts={"math": 10}, valid_score_sums={"math": 63.0},
    )
    assert next_state.active_map() == {"math": (20, 30, 40)}
    assert after_next_step.active_map() == {"math": (10,)}
    assert after_next_step.active_weight_map() == {"math": {}}
