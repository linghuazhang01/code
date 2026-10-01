"""Current-step V4 can select positioned Structure without weighting Control."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

import test_current_step_runtime as runtime

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.occurrence_config import (
    current_step_candidate_map,
    effective_code_cs_position_policy,
)
from mopd_verl.settings import load_config
from mopd_verl.verl_audit import MOPDAuditLogger


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/token_selection/math_code/taxonomy"
LEGACY_CONFIG = CONFIG_DIR / (
    "mopd_math_code_current_step_toploss_m05_c01_"
    "code_structure_fixed4_8gpu_6s2t.yaml"
)
V4_CONFIG = CONFIG_DIR / (
    "mopd_math_code_current_step_token_v4_toploss_m05_c01_"
    "code_structure_fixed4_8gpu_colocated.yaml"
)


def _domain_config(path: Path) -> DomainGradientConfig:
    config = load_config(path)
    metadata = MOPDAuditLogger(
        {"mopd_audit": asdict(config.audit)}
    ).full_gradient_meta("train", 1)["mopd_full_gradient"]
    return DomainGradientConfig.from_meta(metadata)


def test_profiles_are_current_step_eight_gpu_and_bounded() -> None:
    legacy = load_config(LEGACY_CONFIG)
    versioned = load_config(V4_CONFIG)
    for config in (legacy, versioned):
        assert config.audit.control_token_online_selection_timing == "current_step"
        assert config.data.train_batch_size == 528
        assert config.actor.ppo_mini_batch_size == 528
        assert config.runtime.slurm_allocation_gpus == 8
        assert config.trainer.total_training_steps == 60
        assert config.huggingface_checkpoint.steps == (60,)
        assert not config.huggingface_checkpoint.private
    assert legacy.worker_placement.actor_rollout.n_gpus_per_node == 6
    assert legacy.worker_placement.ref_policy.n_gpus_per_node == 2
    assert versioned.worker_placement.actor_rollout.n_gpus_per_node == 8
    assert versioned.audit.versioned_cs_selection_mode_by_domain == {
        "math": "position_fixed",
        "code": "structure_only",
    }
    assert versioned.audit.structure_token_position_profile == "none"
    assert versioned.audit.code_cs_position_gate_enabled is False


def test_v4_structure_only_uses_frozen_code_structure_candidates() -> None:
    config = _domain_config(V4_CONFIG)
    candidates = current_step_candidate_map(config)
    controls = config.effective_domain_candidate_map()
    structures = config.effective_domain_structure_map()
    assert candidates["math"] == controls["math"]
    assert candidates["code"] == structures["code"]
    assert len(candidates["math"]) == 119
    assert len(candidates["code"]) == 45


def test_v4_structure_only_scores_and_weights_unpositioned_structure() -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(
        torch
    ):
        config = replace(
            _domain_config(V4_CONFIG),
            control_token_online_min_mean_occurrences_per_step=0,
        )
        audit = runtime._audit()
        audit.config = config
        assert effective_code_cs_position_policy(config) == "structure_only_ungated"
        batch = runtime._batch([[396, 526, 707, 9000]], ["code"])
        runtime._prepare(audit, [batch], [runtime._result(batch, [[9.0, 1.0, 100.0, 0.0]])])
        torch.testing.assert_close(
            runtime.occurrence_mask(audit, batch),
            torch.tensor([[4 / 7, 4 / 7, 16 / 7, 4 / 7]]),
        )


def test_v4_math_control_and_fixed_structure_are_unpositioned() -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(
        torch
    ):
        config = replace(
            _domain_config(V4_CONFIG),
            control_token_online_min_mean_occurrences_per_step=0,
        )
        audit = runtime._audit()
        audit.config = config
        batch = runtime._batch([[258, 1590, 9000]], ["math"])
        runtime._prepare(
            audit,
            [batch],
            [runtime._result(batch, [[9.0, 1.0, 0.0]])],
        )
        torch.testing.assert_close(
            runtime.occurrence_mask(audit, batch),
            torch.tensor([[4 / 3, 4 / 3, 1 / 3]]),
        )


def test_structure_only_rejects_next_step(tmp_path: Path) -> None:
    config = _domain_config(V4_CONFIG)
    with pytest.raises(ValueError, match="requires current_step"):
        replace(config, control_token_online_selection_timing="next_step").validate()
