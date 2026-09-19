"""Occurrence Loss+TC contract, including real actor-world normalization."""

from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import test_occurrence_selection as existing
from mopd_verl.domain_gradient.control_loss_teacher_confidence import (
    loss_teacher_confidence_type_scores,
)
from mopd_verl.domain_gradient.occurrence import (
    Position, occurrence_mask, prepare_occurrence_masks, select_positions,
)


TC = "top_loss_teacher_confidence"
PREFIX = "science/occurrence/"
FORWARD = "mopd_verl.full_gradient.actor_loss.build_actor_micro_batch_loss"
TEACHER = "mopd_verl.full_gradient.loss_support.selected_teacher_log_prob"


def _oracle(positions: list[Position]) -> tuple[torch.Tensor, dict[str, float]]:
    losses = torch.tensor([abs(p.score) for p in positions], dtype=torch.float64)
    return loss_teacher_confidence_type_scores(
        loss_abs_sums=losses,
        teacher_logp_sums=torch.tensor([p.teacher_logp for p in positions], dtype=torch.float64),
        counts=torch.ones_like(losses),
        normalization_mask=torch.ones_like(losses, dtype=torch.bool),
    )


def _positions(losses: list[float], logps: list[float]) -> list[Position]:
    return [Position("science", 0, 0, i, 300, loss, logp)
            for i, (loss, logp) in enumerate(zip(losses, logps, strict=True))]


def _audit() -> SimpleNamespace:
    cfg = replace(
        existing.config(), control_token_online_min_mean_occurrences_per_step=0,
        control_token_online_strict_occurrence_gate=False,
        control_token_online_selection_mode_by_domain=(("science", TC),),
        control_token_online_top_p_by_domain=tuple((d, .25) for d in ("math", "code", "science")),
    )
    return SimpleNamespace(config=cfg, actor=SimpleNamespace(
        actor_module=torch.nn.Linear(1, 1),
        config={"policy_loss": asdict(existing.load_config(existing.OVERLAY).actor)},
    ))


def _batch(domains: tuple[str, ...] = ("science",)) -> SimpleNamespace:
    return SimpleNamespace(
        batch={"responses": torch.tensor([[300, 300, 300, 99999]] * len(domains)),
               "response_mask": torch.ones(len(domains), 4)},
        non_tensor_batch={"domain": list(domains)},
    )


def _result(batch: SimpleNamespace, losses: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(selector_token_loss=torch.tensor(losses),
                           selector_token_loss_mask=batch.batch["response_mask"].clone(),
                           configured_token_loss=torch.tensor(losses).flip(-1))


def test_math_code_raw_unchanged_science_confidence_flips_ranking() -> None:
    domains = ("math", "code", "science")
    positions = [Position(d, 0, row, col, 300, loss, logp)
                 for row, d in enumerate(domains)
                 for col, (loss, logp) in enumerate(((10., -100.), (8., 0.), (-20., -100.)))]
    args = ([({d: 3 for d in domains}, positions, None)],
            {d: [300] for d in domains}, {d: 1 / 3 for d in domains}, 0, False)
    raw, _, _ = select_positions(*args)
    mixed, _, _ = select_positions(*args, selection_modes={"science": TC})
    assert raw == {(0, 0, row, 0) for row in range(3)}
    assert mixed == {(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 1)}


def test_duplicate_ids_are_independent_occurrences_and_exact_unit_count_oracle() -> None:
    positions = _positions([0., -8., 10., 4., 7.], [-100., 1., -100., -5., -.1])
    scores, expected_metrics = _oracle(positions)
    # Verify the full ordering via every prefix, not just the top occurrence.
    order = sorted(range(len(positions)), key=lambda i: (-scores[i].item(), i))
    assert order[0] != 0  # ID-mean broadcasting would tie all five positions.
    for budget in range(1, 6):
        chosen, _, metrics = select_positions(
            [({"science": 5}, positions, None)], {"science": [300]},
            {"science": budget / 5}, 0, False, {"science": TC},
        )
        assert chosen == {(0, 0, 0, i) for i in order[:budget]}
        for key, value in expected_metrics.items():
            assert metrics[PREFIX + key] == pytest.approx(value, abs=1e-12)
    assert expected_metrics["loss_teacher_confidence_teacher_logp_q98"] <= 0


@pytest.mark.parametrize("strict,expected_count", [(True, 3), (False, 5)])
def test_gate_and_candidate_exclusions_precede_normalization(strict: bool, expected_count: int) -> None:
    eligible = _positions([1., 4., 3.], [-8., -1., -3.])
    boundary = [Position("science", 0, 0, i + 3, 301, 1e6, -1e6) for i in range(2)]
    excluded = [Position("science", 0, 0, 5, 99999, 1e12, -1e12)]
    positions = eligible + boundary + excluded
    selected, means, metrics = select_positions(
        [({"science": 20}, positions, None)], {"science": [300, 301]},
        {"science": .5}, 2, strict, {"science": TC},
    )
    _, oracle_metrics = _oracle(eligible if strict else eligible + boundary)
    assert len(selected) == expected_count
    assert (0, 0, 0, 5) not in selected
    assert metrics[PREFIX + "budget_count"] == 10
    assert metrics[PREFIX + "valid_count"] == 20
    assert means["science"] == 1 + 3 * expected_count / 20
    for key, value in oracle_metrics.items():
        assert metrics[PREFIX + key] == pytest.approx(value)


def test_constant_signals_tie_by_rank_batch_row_column_not_token_id() -> None:
    positions = [Position("science", b, r, c, token, 2., -.5)
                 for b, r, c, token in ((1, 0, 0, 1), (0, 1, 0, 2),
                                         (0, 0, 1, 3), (0, 0, 0, 4))]
    ranks = [({"science": 4}, positions, None)] * 2
    order = [(rank, p.batch, p.row, p.column) for rank in range(2)
             for p in reversed(positions)]
    for budget in range(1, 9):
        selected, _, _ = select_positions(
            ranks, {"science": [1, 2, 3, 4]}, {"science": budget / 8},
            0, False, {"science": TC},
        )
        assert selected == set(order[:budget])


@pytest.mark.parametrize("count,positions,minimum", [
    (0, [], 0), (10, [], 0), (10, _positions([1.], [-1.]), 2),
])
def test_empty_or_fully_gated_pool(count: int, positions: list[Position], minimum: int) -> None:
    chosen, means, metrics = select_positions(
        [({"science": count}, positions, None)], {"science": [300]},
        {"science": .5}, minimum, True, {"science": TC},
    )
    assert chosen == set()
    assert means == {"science": 1.}
    assert metrics[PREFIX + "eligible_count"] == 0
    assert all(value == 0 for key, value in metrics.items() if "loss_teacher_confidence" in key)


def test_prepare_uses_chosen_logp_not_entropy_and_only_valid_tc_rows() -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.full_gradient.loss_support import selected_teacher_log_prob

        audit = _audit()
        batch = _batch(("math", "code", "science"))
        batch.batch["response_mask"][:, 3] = 0
        batch.batch["ref_log_prob"] = torch.tensor([
            [float("nan")] * 4, [float("inf")] * 4, [-100., 0., -100., float("nan")],
        ])
        batch.batch["teacher_entropy"] = torch.tensor([[100., 0., 0., 0.]] * 3)
        raw = [[10., 8., 0., float("nan")]] * 3
        with patch(FORWARD, return_value=_result(batch, raw)), \
                patch(TEACHER, wraps=selected_teacher_log_prob) as fetch:
            metrics = prepare_occurrence_masks(audit, [batch], [1.], on_policy=True, temperature=1.)
        fetch.assert_called_once()
        weights = occurrence_mask(audit, batch)
        assert weights.argmax(dim=1).tolist() == [0, 0, 1]
        assert torch.equal(weights[:, 3], torch.zeros(3))
        assert torch.allclose(weights.sum(dim=1), torch.full((3,), 3.))
        assert metrics[PREFIX + "selected_count"] == 1


def test_prepare_raw_domains_do_not_fetch_teacher_logp() -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        audit, batch = _audit(), _batch(("math", "code"))
        with patch(FORWARD, return_value=_result(batch, [[10., 8., 0., 1.]] * 2)), \
                patch(TEACHER, side_effect=AssertionError("unexpected teacher fetch")) as fetch:
            prepare_occurrence_masks(audit, [batch], [1.], on_policy=True, temperature=1.)
        fetch.assert_not_called()
        assert occurrence_mask(audit, batch).argmax(dim=1).tolist() == [0, 0]


@pytest.mark.parametrize("failure", ["missing", "shape", "nan", "inf", "-inf"])
@pytest.mark.parametrize("column", [0, 3], ids=["candidate", "valid-Other"])
def test_prepare_teacher_errors_are_gathered_before_failure(failure: str, column: int) -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        audit, batch = _audit(), _batch()
        audit._occurrence_masks = {123: "stale"}
        if failure != "missing":
            logp = torch.full((1, 4), -1.)
            if failure == "shape":
                logp = logp[:, :2]
            else:
                logp[0, column] = float(failure)
            batch.batch["ref_log_prob"] = logp
        payloads = []

        def gather(output: list, payload: tuple) -> None:
            payloads.append(payload)
            output[:] = [({"science": 0}, [], None), payload]

        with patch(FORWARD, return_value=_result(batch, [[1., 2., 3., 4.]])), \
                patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.get_world_size", return_value=2), \
                patch("torch.distributed.all_gather_object", side_effect=gather):
            with pytest.raises(ValueError, match="occurrence scoring failed: .*teacher chosen-token logp"):
                prepare_occurrence_masks(audit, [batch], [1.], on_policy=True, temperature=1.)
        assert len(payloads) == 1 and payloads[0][2]
        assert audit._occurrence_masks == {}


def _gloo_worker(rank: int, rendezvous: str, invalid: bool) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30),
    )
    try:
        with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
            audit, batch = _audit(), _batch()
            audit.config = replace(audit.config, control_token_online_top_p_by_domain=(("science", .125),))
            # Rank-local minmax ties the two local maxima at 3; global scores favor rank 1.
            losses = [[1., 2., 3.], [10., 20., 30.]]
            logps = [[-100., -90., -80.], [-3., -2., -1.]]
            batch.batch["ref_log_prob"] = torch.tensor([logps[rank] + [-1.]])
            if invalid and rank == 1:
                batch.batch["ref_log_prob"][0, 3] = float("nan")
            result = _result(batch, [losses[rank] + [10000.]])
            with patch(FORWARD, return_value=result):
                if invalid:
                    with pytest.raises(ValueError, match="occurrence scoring failed: .*teacher"):
                        prepare_occurrence_masks(audit, [batch], [1.], on_policy=True, temperature=1.)
                    assert audit._occurrence_masks == {}
                    return
                metrics = prepare_occurrence_masks(audit, [batch], [1.], on_policy=True, temperature=1.)
            positions = [_positions(losses[r], logps[r]) for r in range(2)]
            global_scores, oracle_metrics = _oracle(positions[0] + positions[1])
            local_scores = torch.cat([_oracle(p)[0] for p in positions])
            assert global_scores.argmax().item() == 5
            assert local_scores.argmax().item() == 2
            for key, value in oracle_metrics.items():
                assert metrics[PREFIX + key] == pytest.approx(value)
            weights = occurrence_mask(audit, batch)
            denominator = 1 + 3 / 8
            expected = torch.full((1, 4), 1 / denominator)
            if rank == 1:
                expected[0, 2] *= 4
            assert torch.allclose(weights, expected)
            assert metrics[PREFIX + "budget_count"] == 1
            total = weights.sum()
            torch.distributed.all_reduce(total)
            assert total.item() == pytest.approx(8.)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason="requires torch.distributed Gloo")
@pytest.mark.parametrize("invalid", [False, True], ids=["global-normalization", "collective-Other-error"])
def test_real_two_rank_gloo(tmp_path: Path, invalid: bool) -> None:
    torch.multiprocessing.spawn(
        _gloo_worker, args=("file://" + str(tmp_path / "rendezvous"), invalid), nprocs=2,
    )
