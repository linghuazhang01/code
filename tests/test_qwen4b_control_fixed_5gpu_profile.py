from pathlib import Path

from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / (
    "mopd_qwen4b_30b_a3b_instruct_2507_5gpu_math_code_science_"
    "topk32_control_fixed_w4_b528.yaml"
)


def test_fixed_weight_five_gpu_profile_contract() -> None:
    config = load_config(CONFIG_PATH)

    assert config.data.train_batch_size == 528
    assert config.actor.ppo_mini_batch_size == 528
    assert config.data.train_batch_size % 4 == 0
    assert config.data.train_batch_size % 3 == 0
    assert config.worker_placement.actor_rollout.n_gpus_per_node == 4
    assert config.worker_placement.ref_policy.n_gpus_per_node == 1
    assert config.trainer.n_gpus_per_node == 4
    assert config.rollout.tensor_model_parallel_size == 2
    assert config.rollout.gpu_memory_utilization == 0.6

    assert config.actor.distill_mode == "topk_renormalized_reverse_kl"
    assert config.audit.loss_variance_signal == (
        "topk_renormalized_reverse_kl"
    )
    assert config.audit.control_token_loss_weighting_enabled
    assert config.audit.control_token_loss_weight == 4.0
    assert not config.audit.control_token_speed_weighting_enabled
    assert config.audit.control_token_speed_initial_weight == 1.0
    assert config.audit.control_token_normalize_per_domain

    assert config.runtime.slurm_allocation_gpus == 5
    assert config.runtime.cuda_visible_devices is None
    assert [
        len(config.audit.domain_control_token_ids[domain])
        for domain in ("math", "code", "science")
    ] == [44, 30, 27]
