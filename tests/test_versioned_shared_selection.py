"""Versioned C+S uses one selector and one Top-P budget per dynamic domain."""

from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

import test_current_step_runtime as runtime
import test_versioned_next_step as existing

from mopd_verl.domain_gradient.versioned_next_step import (
    next_step_candidate_statistics,
    versioned_selector_candidate_map,
)
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config


def _batch(logp: list[float] | None = None) -> object:
    batch = runtime._batch([[704, 1590, 704, 1590, 9000]], ["math"])
    batch.batch["mopd_control_position_mask"] = torch.ones(1, 5).bool()
    batch.batch["mopd_structure_position_mask"] = torch.zeros(1, 5).bool()
    if logp is not None:
        batch.batch["ref_log_prob"] = torch.tensor([logp])
    return batch


def _logger(
    tmp_path: Path, math_mode: str, code_mode: str,
    *, code_gate: bool | None = None,
) -> object:
    gate_override = (
        {} if code_gate is None
        else {"code_cs_position_gate_enabled": code_gate}
    )
    return existing._logger(
        tmp_path,
        versioned_cs_selection_mode_by_domain={
            "math": math_mode,
            "code": code_mode,
        },
        control_token_online_min_mean_occurrences_per_step=0,
        control_token_online_top_p_by_domain={"math": 0.2, "code": 0.2},
        **gate_override,
    )


def _audit(actor: object, logger: object, step: int) -> object:
    from mopd_verl.domain_gradient.audit import DomainGradientAudit

    return DomainGradientAudit(
        actor, logger.full_gradient_meta("train", step)["mopd_full_gradient"],
    )


@pytest.mark.parametrize("mode", ["top_loss", "top_teacher_confidence"])
def test_config_expands_one_shared_pool_and_budget(tmp_path: Path, mode: str) -> None:
    path = tmp_path / "shared.yaml"
    path.write_text(yaml.safe_dump({
        "extends": str(existing.CONFIG.resolve()),
        "audit": {
            "versioned_cs_selection_mode_by_domain": {
                "math": mode, "code": "position_fixed",
            },
        },
    }))
    config = load_config(path)
    assert config.audit.control_token_online_top_p_by_domain == {
        "math": 0.05, "code": 0.01,
    }
    assert config.audit.versioned_cs_selection_mode_by_domain == {
        "math": mode, "code": "position_fixed",
    }
    assert any(
        item.startswith("+mopd_audit.versioned_cs_selection_mode_by_domain=")
        for item in build_command(config)
    )
    runtime_config = existing.DomainGradientConfig.from_meta(
        existing.MOPDAuditLogger({"mopd_audit": asdict(config.audit)})
        .full_gradient_meta("train", 1)["mopd_full_gradient"]
    )
    pools = versioned_selector_candidate_map(runtime_config)
    assert len(pools["math"]) == 128  # Frozen V4 C119 + S9.
    assert len(pools["code"]) == 153  # Position mode keeps C-only ranking.


@pytest.mark.parametrize(
    ("filename", "expected_modes", "expected_gate"),
    [
        (
            "mopd_math_code_next_step_token_v4_shared_toploss_m05_c01_fixed4_4gpu_colocated.yaml",
            {"math": "top_loss", "code": "top_loss"},
            False,
        ),
        (
            "mopd_math_code_next_step_token_v4_math_teacherconf_code_toploss_m05_c01_fixed4_4gpu_colocated.yaml",
            {"math": "top_teacher_confidence", "code": "top_loss"},
            False,
        ),
        (
            "mopd_math_code_next_step_token_v4_shared_toploss_code_gated_m05_c01_fixed4_4gpu_colocated.yaml",
            {"math": "top_loss", "code": "top_loss"},
            True,
        ),
    ],
)
def test_prepared_profiles_use_one_budget_per_domain(
    filename: str, expected_modes: dict[str, str], expected_gate: bool,
) -> None:
    config = load_config(existing.CONFIG.parent / filename)
    assert config.audit.versioned_cs_selection_mode_by_domain == expected_modes
    assert config.audit.code_cs_position_gate_enabled is expected_gate
    assert config.audit.control_token_online_top_p_by_domain == {
        "math": 0.05, "code": 0.01,
    }
    assert config.audit.control_token_online_selection_timing == "next_step"
    assert config.audit.control_token_loss_weight == 4.0
    assert config.audit.structure_token_loss_weight == 4.0
    assert not config.huggingface_checkpoint.enabled
    runtime_config = existing.DomainGradientConfig.from_meta(
        existing.MOPDAuditLogger({"mopd_audit": asdict(config.audit)})
        .full_gradient_meta("train", 1)["mopd_full_gradient"]
    )
    pools = versioned_selector_candidate_map(runtime_config)
    assert {domain: len(pool) for domain, pool in pools.items()} == {
        "math": 128, "code": 198,
    }


def test_v5_shared_pool_uses_frozen_v5_membership(tmp_path: Path) -> None:
    baseline = existing.CONFIG.parent / (
        "mopd_math_code_next_step_token_v5_toploss_"
        "m05_c01_fixed4_4gpu_colocated.yaml"
    )
    path = tmp_path / "v5-shared.yaml"
    path.write_text(yaml.safe_dump({
        "extends": str(baseline.resolve()),
        "audit": {
            "versioned_cs_selection_mode_by_domain": {
                "math": "top_loss", "code": "top_teacher_confidence",
            },
        },
    }))
    config = load_config(path)
    runtime_config = existing.DomainGradientConfig.from_meta(
        existing.MOPDAuditLogger({"mopd_audit": asdict(config.audit)})
        .full_gradient_meta("train", 1)["mopd_full_gradient"]
    )
    pools = versioned_selector_candidate_map(runtime_config)
    assert {domain: len(pool) for domain, pool in pools.items()} == {
        "math": 70, "code": 117,
    }


def test_shared_pool_rejects_control_only_candidate_groups(tmp_path: Path) -> None:
    config = existing._domain(tmp_path)
    controls = config.effective_domain_candidate_map()
    grouped = replace(
        config,
        versioned_cs_selection_mode_by_domain=(
            ("math", "top_loss"), ("code", "position_fixed"),
        ),
        domain_control_token_candidate_ids=(),
        domain_control_token_candidate_groups=tuple(
            (domain, (("all", token_ids),))
            for domain, token_ids in controls.items()
        ),
    )
    with pytest.raises(ValueError, match="Control-only candidate groups"):
        grouped.validate()


def test_toploss_shares_budget_and_lags_both_categories(tmp_path: Path) -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        logger = _logger(tmp_path, "top_loss", "position_fixed")
        actor = existing._actor()
        first = _audit(actor, logger, 1)
        batch = _batch()
        torch.testing.assert_close(
            first.training_gradient_mask(batch), torch.ones(1, 5),
        )
        raw = torch.tensor([[1., 10., 1., 10., 0.]])
        configured = torch.tensor([[100., 1., 100., 1., 0.]])
        with patch(
            "mopd_verl.domain_gradient.audit.build_actor_micro_batch_loss",
            side_effect=AssertionError("unexpected scoring forward"),
        ):
            first.run_before_training([batch], [1.], on_policy=True, temperature=1.)
            first.observe_completed_step(
                [batch], [configured], [batch.batch["response_mask"]],
                selector_token_loss_batches=[raw],
                selector_token_loss_mask_batches=[batch.batch["response_mask"]],
            )
        # One shared Top-P target: Structure wins, Control is not also selected.
        assert first._online_control_selection_state.active_map()["math"] == (1590,)
        restored = existing._actor()
        restored.actor_optimizer.param_groups[0].update(
            actor.actor_optimizer.param_groups[0]
        )
        second = _audit(restored, logger, 2)
        expected = torch.tensor([[1., 4., 1., 4., 1.]])
        expected /= expected.mean()
        torch.testing.assert_close(second.training_gradient_mask(batch), expected)


def test_teacher_confidence_ranks_shared_pool_not_structure_alone(
    tmp_path: Path,
) -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        logger = _logger(tmp_path, "top_teacher_confidence", "position_fixed")
        actor = existing._actor()
        first = _audit(actor, logger, 1)
        batch = _batch([-0.1, -5., -0.1, -5., -5.])
        torch.testing.assert_close(
            first.training_gradient_mask(batch), torch.ones(1, 5),
        )
        assert first._online_control_selection_state.selection_mode_map() == {
            "math": "top_teacher_confidence", "code": "top_loss",
        }
        # Structure has greater raw loss; Control has higher teacher confidence.
        raw = torch.tensor([[1., 10., 1., 10., 0.]])
        first.observe_completed_step(
            [batch], [raw], [batch.batch["response_mask"]],
            selector_token_loss_batches=[raw],
            selector_token_loss_mask_batches=[batch.batch["response_mask"]],
        )
        assert first._online_control_selection_state.active_map()["math"] == (704,)
        restored = existing._actor()
        restored.actor_optimizer.param_groups[0].update(
            actor.actor_optimizer.param_groups[0]
        )
        second = _audit(restored, logger, 2)
        expected = torch.tensor([[4., 1., 4., 1., 1.]])
        expected /= expected.mean()
        torch.testing.assert_close(second.training_gradient_mask(batch), expected)


def test_teacher_confidence_requires_source_signal(tmp_path: Path) -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        logger = _logger(tmp_path, "top_teacher_confidence", "position_fixed")
        first = _audit(existing._actor(), logger, 1)
        batch = _batch()
        raw = torch.ones(1, 5)
        with pytest.raises(ValueError, match="teacher chosen-token logp"):
            first.observe_completed_step(
                [batch], [raw], [batch.batch["response_mask"]],
                selector_token_loss_batches=[raw],
                selector_token_loss_mask_batches=[batch.batch["response_mask"]],
            )


def test_shared_code_pool_default_counts_all_valid_occurrences(tmp_path: Path) -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        logger = _logger(tmp_path, "position_fixed", "top_loss")
        config = existing.DomainGradientConfig.from_meta(
            logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
        )
        batch = runtime._batch([[704, 1590, 704, 1590]], ["code"])
        batch.batch["mopd_control_position_mask"] = torch.tensor([[1, 1, 0, 1]])
        batch.batch["mopd_structure_position_mask"] = torch.zeros(1, 4).bool()
        raw = torch.tensor([[1., 2., 100., 2.]])
        statistics = next_step_candidate_statistics(
            config, [batch], [raw], [batch.batch["response_mask"]],
        )
        assert statistics.by_domain["code"][704] == (101., 2)
        assert statistics.by_domain["code"][1590] == (4., 2)
        assert statistics.valid_token_counts["code"] == 4


def _confidence_rank_worker(rank: int, rendezvous: str, run_dir: str) -> None:
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
                    ("math", "top_teacher_confidence"),
                    ("code", "position_fixed"),
                ),
            )
            batch = _batch(
                [-0.1, -5., -0.1] if rank == 0 else [-3., -0.2, -3.]
            )
            for key in (
                "responses", "response_mask", "mopd_control_position_mask",
                "mopd_structure_position_mask",
            ):
                batch.batch[key] = batch.batch[key][:, :3]
            result = next_step_teacher_confidence_statistics(
                config, [batch], [batch.batch["ref_log_prob"]],
            )
            assert result.valid_token_counts["math"] == 6
            control_sum, control_count = result.by_domain["math"][704]
            structure_sum, structure_count = result.by_domain["math"][1590]
            assert control_count == 4
            assert structure_count == 2
            assert control_sum / control_count > structure_sum / structure_count
    finally:
        torch.distributed.destroy_process_group()


def test_teacher_confidence_statistics_reduce_across_ranks(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _confidence_rank_worker,
        args=("file://" + str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
    )
