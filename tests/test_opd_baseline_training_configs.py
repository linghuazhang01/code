from __future__ import annotations

from pathlib import Path

import pytest

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import MOPDConfig, load_config


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "configs" / "baselines"

BASELINE_SLUGS = (
    "entropy_topk32_rho50",
    "tip_topk32_rho50",
    "fire_topk32_matched",
    "exopd_native_lambda1p25",
    "eopd_native",
    "fire_opd_native",
)

TOPOLOGIES = {
    "4gpu_b525": {
        "allocation_gpus": 4,
        "actor_gpus": 3,
        "teacher_gpus": 1,
        "batch_size": 525,
        "tensor_parallel_size": 1,
    },
    "8gpu_b528": {
        "allocation_gpus": 8,
        "actor_gpus": 6,
        "teacher_gpus": 2,
        "batch_size": 528,
        "tensor_parallel_size": 2,
    },
}


def _config_path(baseline_slug: str, topology_slug: str) -> Path:
    return (
        BASELINE_DIR
        / f"qwen4b_30b_{baseline_slug}_{topology_slug}.yaml"
    )


def _load(baseline_slug: str, topology_slug: str) -> MOPDConfig:
    return load_config(_config_path(baseline_slug, topology_slug))


@pytest.mark.parametrize("baseline_slug", BASELINE_SLUGS)
@pytest.mark.parametrize("topology_slug", tuple(TOPOLOGIES))
def test_standalone_baseline_topology_and_batch(
    baseline_slug: str,
    topology_slug: str,
) -> None:
    expected = TOPOLOGIES[topology_slug]
    path = _config_path(baseline_slug, topology_slug)
    config = load_config(path)

    assert path.is_file()
    assert config.runtime.slurm_allocation_gpus == expected["allocation_gpus"]
    assert config.worker_placement.separate_ref_policy
    assert (
        config.worker_placement.actor_rollout.n_gpus_per_node
        == expected["actor_gpus"]
    )
    assert (
        config.worker_placement.ref_policy.n_gpus_per_node
        == expected["teacher_gpus"]
    )
    assert config.trainer.n_gpus_per_node == expected["actor_gpus"]
    assert config.data.train_batch_size == expected["batch_size"]
    assert config.actor.ppo_mini_batch_size == expected["batch_size"]
    assert config.rollout.tensor_model_parallel_size == expected[
        "tensor_parallel_size"
    ]
    assert config.data.train_batch_size % expected["actor_gpus"] == 0
    assert config.data.train_batch_size % 3 == 0
    assert config.model.student_path == "../models/Qwen3-4B"
    assert set(config.model.domain_teacher_paths) == {
        "math",
        "code",
        "science",
    }

    command = format_command(build_command(config))
    assert f"data.train_batch_size={expected['batch_size']}" in command
    assert (
        "actor_rollout_ref.actor.ppo_mini_batch_size="
        f"{expected['batch_size']}"
    ) in command
    assert f"trainer.n_gpus_per_node={expected['actor_gpus']}" in command
    assert (
        "+actor_rollout_ref.worker_placement.actor_rollout."
        f"n_gpus_per_node={expected['actor_gpus']}"
    ) in command
    assert (
        "+actor_rollout_ref.worker_placement.ref_policy."
        f"n_gpus_per_node={expected['teacher_gpus']}"
    ) in command


@pytest.mark.parametrize("topology_slug", tuple(TOPOLOGIES))
def test_budget_matched_baseline_objectives(topology_slug: str) -> None:
    entropy = _load("entropy_topk32_rho50", topology_slug)
    tip = _load("tip_topk32_rho50", topology_slug)
    fire = _load("fire_topk32_matched", topology_slug)

    assert entropy.actor.distill_loss_builder == "topk_kl"
    assert entropy.actor.token_baseline_method == "entropy"
    assert entropy.actor.token_baseline_retention_ratio == 0.5
    assert entropy.actor.token_baseline_selection_mode == "topk"

    assert tip.actor.distill_loss_builder == "topk_kl"
    assert tip.actor.token_baseline_method == "tip_topk32"
    assert tip.actor.token_baseline_retention_ratio == 0.5
    assert tip.rollout.gpu_memory_utilization == 0.8

    assert fire.actor.distill_loss_builder == "topk_kl"
    assert fire.actor.token_baseline_method == "fire_opd"
    assert not fire.actor.fire_opd_filter_trajectories
    assert fire.rollout.gpu_memory_utilization == 0.8

    for config in (entropy, tip, fire):
        assert config.actor.topk_distill_enabled
        assert config.actor.topk_distill_k == 32
        assert config.actor.topk_distill_kl_direction == "reverse"
        assert config.actor.distill_mode == "topk_renormalized_reverse_kl"


@pytest.mark.parametrize("topology_slug", tuple(TOPOLOGIES))
def test_native_baseline_objectives(topology_slug: str) -> None:
    exopd = _load("exopd_native_lambda1p25", topology_slug)
    eopd = _load("eopd_native", topology_slug)
    fire = _load("fire_opd_native", topology_slug)

    assert exopd.actor.distill_loss_builder == "exopd"
    assert exopd.actor.distill_mode == "chosen_token_policy_gradient"
    assert exopd.actor.only_reverse_kl_advantages
    assert exopd.actor.lambda_vals == 1.25
    assert exopd.model.student_base_path == exopd.model.student_path
    assert not exopd.actor.topk_distill_enabled

    assert eopd.actor.distill_loss_builder == "eopd"
    assert eopd.actor.distill_mode == "chosen_token_policy_gradient"
    assert eopd.actor.eopd_entropy_threshold == 0.8
    assert eopd.actor.eopd_forward_kl_weight == 1.0
    assert eopd.actor.eopd_topk_k == 16
    assert not eopd.actor.topk_distill_enabled

    assert fire.actor.distill_loss_builder == "policy_gradient"
    assert fire.actor.distill_mode == "chosen_token_policy_gradient"
    assert fire.actor.token_baseline_method == "fire_opd"
    assert fire.actor.fire_opd_teacher_confidence_alpha == 1.0
    assert fire.actor.fire_opd_student_confusion_beta == 1.0
    assert fire.actor.fire_opd_trajectory_drop_ratio == 0.2
    assert fire.actor.fire_opd_filter_trajectories
    assert not fire.actor.topk_distill_enabled


def test_standalone_baseline_run_identifiers_are_unique() -> None:
    configs = [
        _load(baseline_slug, topology_slug)
        for baseline_slug in BASELINE_SLUGS
        for topology_slug in TOPOLOGIES
    ]
    experiment_names = [config.trainer.experiment_name for config in configs]
    local_dirs = [config.trainer.default_local_dir for config in configs]
    wandb_run_ids = [config.runtime.wandb_run_id for config in configs]

    assert len(experiment_names) == len(set(experiment_names))
    assert len(local_dirs) == len(set(local_dirs))
    assert None not in wandb_run_ids
    assert len(wandb_run_ids) == len(set(wandb_run_ids))
