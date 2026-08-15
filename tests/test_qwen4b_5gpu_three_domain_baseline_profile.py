from pathlib import Path

from mopd_verl.domain_sampling import allocate_domain_batch_counts
from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / (
    "mopd_qwen4b_30b_a3b_instruct_2507_5gpu_math_code_science_"
    "topk32_baseline_b528.yaml"
)


def test_five_gpu_three_domain_baseline_contract() -> None:
    config = load_config(CONFIG_PATH)

    assert config.data.train_batch_size == 528
    assert config.actor.ppo_mini_batch_size == 528
    assert config.data.domain_sampling_weights == {
        "math": 1.0,
        "code": 1.0,
        "science": 1.0,
    }
    assert allocate_domain_batch_counts(
        config.data.train_batch_size,
        config.data.domain_sampling_weights,
    ) == {"math": 176, "code": 176, "science": 176}

    assert config.worker_placement.separate_ref_policy
    assert config.worker_placement.actor_rollout.n_gpus_per_node == 4
    assert config.worker_placement.ref_policy.n_gpus_per_node == 1
    assert config.trainer.n_gpus_per_node == 4
    assert config.runtime.slurm_allocation_gpus == 5
    assert config.runtime.cuda_visible_devices is None
    assert config.rollout.tensor_model_parallel_size == 2
    assert config.rollout.gpu_memory_utilization == 0.6

    assert config.rollout_correction.rollout_is == "null"
    assert config.rollout_correction.rollout_rs is None
    assert not config.rollout_correction.bypass_mode
    assert not config.domain_budgeting.enabled
    assert not config.audit.dynamic_domain_loss_weighting_enabled
    assert not config.audit.control_token_loss_weighting_enabled
    assert config.audit.control_token_ids == []
    assert config.audit.domain_control_token_ids == {}
    assert not config.audit.control_token_normalize_per_domain
    assert not config.audit.control_token_phase_gate_enabled
    assert not config.audit.control_token_span_weighting_enabled
    assert not config.audit.control_token_speed_weighting_enabled
    assert not config.audit.all_domain_shared_token_loss_weighting_enabled

    artifact_paths = (
        config.runtime.wandb_run_id,
        config.audit.output_dir,
        config.paper_eval.output_dir,
        config.trainer.experiment_name,
        config.trainer.default_local_dir,
    )
    for artifact_path in artifact_paths:
        assert "5gpu" in artifact_path
        assert "b528" in artifact_path
        assert "6gpu" not in artifact_path
        assert "b525" not in artifact_path

    rendered = format_command(build_command(config))
    for expected_override in (
        "data.train_batch_size=528",
        "actor_rollout_ref.actor.ppo_mini_batch_size=528",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "trainer.n_gpus_per_node=4",
        "+actor_rollout_ref.worker_placement.actor_rollout.n_gpus_per_node=4",
        "+actor_rollout_ref.worker_placement.ref_policy.n_gpus_per_node=1",
        "algorithm.rollout_correction.rollout_is=null",
    ):
        assert expected_override in rendered
    assert "+mopd_domain_budgeting.enabled=true" not in rendered
