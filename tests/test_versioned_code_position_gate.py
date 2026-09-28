"""Dynamic Code C/S source and application positions use one policy."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import test_current_step_runtime as runtime
import test_versioned_next_step as existing
import test_versioned_shared_selection as shared

from mopd_verl.domain_gradient.versioned_next_step import (
    next_step_candidate_statistics,
)


def _code_batch() -> object:
    batch = runtime._batch([[704, 1590, 704, 1590]], ["code"])
    batch.batch["mopd_control_position_mask"] = torch.tensor(
        [[1, 1, 0, 1]], dtype=torch.bool
    )
    batch.batch["mopd_structure_position_mask"] = torch.tensor(
        [[0, 1, 0, 0]], dtype=torch.bool
    )
    batch.batch["ref_log_prob"] = torch.tensor([[-5., -3., -0.01, -5.]])
    return batch


@pytest.mark.parametrize("mode", ["top_loss", "top_teacher_confidence"])
@pytest.mark.parametrize("gate", [False, True])
def test_code_gate_aligns_source_selection_and_next_step_weighting(
    tmp_path: Path, mode: str, gate: bool,
) -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        logger = shared._logger(
            tmp_path, "position_fixed", mode, code_gate=gate,
        )
        config = existing.DomainGradientConfig.from_meta(
            logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
        )
        batch = _code_batch()
        raw = torch.tensor([[1., 2., 100., 20.]])
        valid = batch.batch["response_mask"]
        if mode == "top_loss":
            statistics = next_step_candidate_statistics(
                config, [batch], [raw], [valid],
            )
            expected_sums = (
                {704: (1., 1), 1590: (2., 1)} if gate else
                {704: (101., 2), 1590: (22., 2)}
            )
            assert statistics.by_domain["code"] == expected_sums
        else:
            from mopd_verl.domain_gradient.versioned_teacher_confidence import (
                next_step_teacher_confidence_statistics,
            )

            statistics = next_step_teacher_confidence_statistics(
                config, [batch], [batch.batch["ref_log_prob"]],
            )
            assert {
                token_id: count
                for token_id, (_, count) in statistics.by_domain["code"].items()
            } == ({704: 1, 1590: 1} if gate else {704: 2, 1590: 2})
        assert statistics.valid_token_counts["code"] == 4

        actor = existing._actor()
        first = shared._audit(actor, logger, 1)
        torch.testing.assert_close(
            first.training_gradient_mask(batch), torch.ones(1, 4),
        )
        with patch(
            "mopd_verl.domain_gradient.audit.build_actor_micro_batch_loss",
            side_effect=AssertionError("unexpected scoring forward"),
        ):
            first.run_before_training(
                [batch], [1.], on_policy=True, temperature=1.,
            )
            first.observe_completed_step(
                [batch], [raw], [valid],
                selector_token_loss_batches=[raw],
                selector_token_loss_mask_batches=[valid],
            )
        selected = 1590 if gate else 704
        assert first._online_control_selection_state.active_map()["code"] == (
            selected,
        )
        restored = existing._actor()
        restored.actor_optimizer.param_groups[0].update(
            actor.actor_optimizer.param_groups[0]
        )
        second = shared._audit(restored, logger, 2)
        expected = (
            torch.tensor([[1., 4., 1., 1.]]) if gate
            else torch.tensor([[4., 1., 4., 1.]])
        )
        expected /= expected.mean()
        torch.testing.assert_close(second.training_gradient_mask(batch), expected)


def test_ungated_code_does_not_require_position_masks(tmp_path: Path) -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        logger = shared._logger(
            tmp_path, "position_fixed", "top_loss", code_gate=False,
        )
        config = existing.DomainGradientConfig.from_meta(
            logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
        )
        batch = _code_batch()
        del batch.batch["mopd_control_position_mask"]
        del batch.batch["mopd_structure_position_mask"]
        raw = torch.tensor([[1., 2., 100., 20.]])
        statistics = next_step_candidate_statistics(
            config, [batch], [raw], [batch.batch["response_mask"]],
        )
        assert statistics.by_domain["code"] == {
            704: (101., 2), 1590: (22., 2),
        }
        first = shared._audit(existing._actor(), logger, 1)
        torch.testing.assert_close(
            first.training_gradient_mask(batch), torch.ones(1, 4),
        )


@pytest.mark.parametrize(
    "missing_mask", ["mopd_control_position_mask", "mopd_structure_position_mask"]
)
def test_gated_code_requires_both_aligned_masks(
    tmp_path: Path, missing_mask: str,
) -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        logger = shared._logger(
            tmp_path, "position_fixed", "top_loss", code_gate=True,
        )
        config = existing.DomainGradientConfig.from_meta(
            logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
        )
        batch = _code_batch()
        del batch.batch[missing_mask]
        raw = torch.ones(1, 4)
        with pytest.raises(ValueError, match=missing_mask):
            next_step_candidate_statistics(
                config, [batch], [raw], [batch.batch["response_mask"]],
            )
        first = shared._audit(existing._actor(), logger, 1)
        with pytest.raises(ValueError, match=missing_mask):
            first.training_gradient_mask(batch)


def test_gated_code_empty_eligible_pool_has_budget_shortfall(tmp_path: Path) -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        logger = shared._logger(
            tmp_path, "position_fixed", "top_loss", code_gate=True,
        )
        batch = _code_batch()
        batch.batch["mopd_control_position_mask"] = torch.zeros(1, 4).bool()
        batch.batch["mopd_structure_position_mask"] = torch.zeros(1, 4).bool()
        config = existing.DomainGradientConfig.from_meta(
            logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
        )
        raw = torch.tensor([[1., 2., 100., 20.]])
        valid = batch.batch["response_mask"]
        statistics = next_step_candidate_statistics(
            config, [batch], [raw], [valid],
        )
        assert statistics.valid_token_counts["code"] == 4
        assert statistics.by_domain["code"] == {}
        actor = existing._actor()
        first = shared._audit(actor, logger, 1)
        metrics = first.observe_completed_step(
            [batch], [raw], [valid],
            selector_token_loss_batches=[raw],
            selector_token_loss_mask_batches=[valid],
        )
        assert first._online_control_selection_state.active_map()["code"] == ()
        assert metrics["code/token_weight/top_p_target_occurrence_count"] == 1.0
        assert metrics["code/token_weight/top_p_selected_occurrence_count"] == 0.0
        assert metrics["code/token_weight/top_p_occurrence_shortfall"] == 1.0
        restored = existing._actor()
        restored.actor_optimizer.param_groups[0].update(
            actor.actor_optimizer.param_groups[0]
        )
        second = shared._audit(restored, logger, 2)
        torch.testing.assert_close(
            second.training_gradient_mask(batch), torch.ones(1, 4),
        )


def _gated_confidence_rank_worker(rank: int, rendezvous: str, run_dir: str) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
            from mopd_verl.domain_gradient.versioned_teacher_confidence import (
                next_step_teacher_confidence_statistics,
            )

            config = replace(
                existing._domain(Path(run_dir)),
                versioned_cs_selection_mode_by_domain=(
                    ("math", "position_fixed"),
                    ("code", "top_teacher_confidence"),
                ),
                code_cs_position_gate_enabled=True,
            )
            batch = _code_batch()
            batch.batch["ref_log_prob"] = torch.tensor(
                [[-0.1, -5., -0.01, -5.]] if rank == 0
                else [[-5., -0.1, -0.01, -5.]]
            )
            result = next_step_teacher_confidence_statistics(
                config, [batch], [batch.batch["ref_log_prob"]],
            )
            assert result.valid_token_counts["code"] == 8
            assert {token_id: count for token_id, (_, count) in
                    result.by_domain["code"].items()} == {704: 2, 1590: 2}
            control_sum, _ = result.by_domain["code"][704]
            structure_sum, _ = result.by_domain["code"][1590]
            assert control_sum == pytest.approx(structure_sum)
    finally:
        torch.distributed.destroy_process_group()


def test_gated_code_confidence_reduces_eligible_counts_across_ranks(
    tmp_path: Path,
) -> None:
    torch.multiprocessing.spawn(
        _gated_confidence_rank_worker,
        args=("file://" + str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
    )


def _gated_toploss_rank_worker(rank: int, rendezvous: str, run_dir: str) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
            logger = shared._logger(
                Path(run_dir), "position_fixed", "top_loss", code_gate=True,
            )
            config = existing.DomainGradientConfig.from_meta(
                logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
            )
            batch = _code_batch()
            raw = torch.tensor(
                [[1., 2., 100., 20.]] if rank == 0
                else [[3., 4., 200., 40.]]
            )
            valid = batch.batch["response_mask"]
            result = next_step_candidate_statistics(
                config, [batch], [raw], [valid],
            )
            assert result.valid_token_counts["code"] == 8
            assert result.by_domain["code"] == {
                704: (4., 2), 1590: (6., 2),
            }
            if rank == 1:
                del batch.batch["mopd_structure_position_mask"]
            with pytest.raises(ValueError, match="at least one rank"):
                next_step_candidate_statistics(
                    config, [batch], [raw], [valid],
                )
            if rank == 1:
                batch.batch["mopd_structure_position_mask"] = torch.tensor(
                    [[0, 1, 0, 0]], dtype=torch.bool,
                )
                raw[0, 0] = float("nan")
            with pytest.raises(ValueError, match="at least one rank"):
                next_step_candidate_statistics(
                    config, [batch], [raw], [valid],
                )
    finally:
        torch.distributed.destroy_process_group()


def test_gated_code_toploss_reduces_and_synchronizes_mask_errors(
    tmp_path: Path,
) -> None:
    torch.multiprocessing.spawn(
        _gated_toploss_rank_worker,
        args=("file://" + str(tmp_path / "rendezvous-toploss"), str(tmp_path)),
        nprocs=2,
    )
