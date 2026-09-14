from __future__ import annotations

from pathlib import Path

import pytest

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "baselines"


CASES = (
    ("qwen1p7b_30b_opd_native_4gpu_b528_colocated.yaml", "policy_gradient"),
    ("qwen1p7b_30b_eopd_native_4gpu_b528_colocated.yaml", "eopd"),
    ("qwen1p7b_30b_fire_opd_native_4gpu_b528_colocated.yaml", "policy_gradient"),
)


EIGHT_GPU_CASES = (
    ("qwen1p7b_30b_opd_native_8gpu_b528.yaml", "policy_gradient", "none"),
    ("qwen1p7b_30b_eopd_native_8gpu_b528.yaml", "eopd", "none"),
    (
        "qwen1p7b_30b_fire_opd_native_8gpu_b528.yaml",
        "policy_gradient",
        "fire_opd",
    ),
)


@pytest.mark.parametrize(("filename", "loss_builder"), CASES)
def test_qwen1p7b_colocated_baseline_topology_and_batch(
    filename: str,
    loss_builder: str,
) -> None:
    config = load_config(CONFIG_DIR / filename)
    command = format_command(build_command(config))

    assert config.runtime.slurm_allocation_gpus == 4
    assert config.data.train_batch_size == 528
    assert config.actor.ppo_mini_batch_size == 528
    assert config.worker_placement.separate_ref_policy is False
    assert config.worker_placement.actor_rollout.n_gpus_per_node == 4
    assert config.worker_placement.ref_policy.n_gpus_per_node is None
    assert config.trainer.n_gpus_per_node == 4
    assert config.rollout.tensor_model_parallel_size == 1
    assert config.model.student_path == "../mopd/models/Qwen3-1.7B"
    assert config.model.primary_teacher_path == (
        "../mopd/models/Qwen3-30B-A3B-Instruct-2507"
    )
    assert config.actor.distill_loss_builder == loss_builder
    assert not config.teacher_performance.enabled
    assert "separate_ref_policy=false" in command
    assert "actor_rollout_ref.worker_placement.actor_rollout.n_gpus_per_node=4" in command
    assert "actor_rollout_ref.ref.fsdp_config.fsdp_size=4" in command


def test_colocated_baseline_method_contracts() -> None:
    opd = load_config(CONFIG_DIR / CASES[0][0])
    eopd = load_config(CONFIG_DIR / CASES[1][0])
    fire = load_config(CONFIG_DIR / CASES[2][0])

    assert opd.actor.token_baseline_method == "none"
    assert eopd.actor.eopd_entropy_threshold == 0.8
    assert eopd.actor.eopd_forward_kl_weight == 1.0
    assert eopd.actor.eopd_topk_k == 16
    assert fire.actor.token_baseline_method == "fire_opd"
    assert fire.actor.fire_opd_filter_trajectories


@pytest.mark.parametrize(
    ("filename", "loss_builder", "token_baseline_method"), EIGHT_GPU_CASES
)
def test_qwen1p7b_six_student_two_teacher_baseline(
    filename: str,
    loss_builder: str,
    token_baseline_method: str,
) -> None:
    config = load_config(CONFIG_DIR / filename)
    command = format_command(build_command(config))

    assert config.runtime.slurm_allocation_gpus == 8
    assert config.data.train_batch_size == 528
    assert config.actor.ppo_mini_batch_size == 528
    assert config.worker_placement.separate_ref_policy is True
    assert config.worker_placement.actor_rollout.n_gpus_per_node == 6
    assert config.worker_placement.ref_policy.n_gpus_per_node == 2
    assert config.trainer.n_gpus_per_node == 6
    assert config.rollout.tensor_model_parallel_size == 2
    assert config.actor.distill_loss_builder == loss_builder
    assert config.actor.token_baseline_method == token_baseline_method
    assert config.teacher_performance.enabled
    assert config.teacher_performance.topk_logprob_chunk_size == 1024
    assert "separate_ref_policy=true" in command
    assert "actor_rollout_ref.worker_placement.actor_rollout.n_gpus_per_node=6" in command
    assert "actor_rollout_ref.worker_placement.ref_policy.n_gpus_per_node=2" in command
    assert "trainer.n_gpus_per_node=6" in command
    assert "ref.teacher_performance.topk_logprob_chunk_size=1024" in command
