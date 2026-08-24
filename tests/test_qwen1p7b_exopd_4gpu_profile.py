from __future__ import annotations

from pathlib import Path

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "baselines"
    / "qwen1p7b_30b_exopd_native_lambda1p25_4gpu_b525.yaml"
)


def test_formal_four_gpu_exopd_profile_preserves_method_contract() -> None:
    config = load_config(CONFIG_PATH)

    assert config.actor.distill_loss_builder == "exopd"
    assert config.actor.distill_mode == "chosen_token_policy_gradient"
    assert config.actor.lambda_vals == 1.25
    assert not config.actor.topk_distill_enabled
    assert config.model.gopd_reference_path == config.model.student_path


def test_formal_four_gpu_exopd_profile_has_three_plus_one_topology() -> None:
    config = load_config(CONFIG_PATH)
    command = format_command(build_command(config))

    assert config.runtime.slurm_allocation_gpus == 4
    assert config.worker_placement.actor_rollout.n_gpus_per_node == 3
    assert config.worker_placement.ref_policy.n_gpus_per_node == 1
    assert config.trainer.n_gpus_per_node == 3
    assert config.rollout.tensor_model_parallel_size == 1
    assert config.rollout.gpu_memory_utilization == 0.6
    assert config.data.train_batch_size == 525
    assert config.actor.ppo_mini_batch_size == 525
    assert config.data.train_batch_size % 3 == 0
    assert "policy_loss.distill_loss_builder=exopd" in command
    assert "policy_loss.lambda_vals=1.25" in command
    assert "trainer.n_gpus_per_node=3" in command
    assert "actor_rollout_ref.ref.model.path=" in command
    assert "trainer.resume_mode=disable" in command
