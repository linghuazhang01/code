from pathlib import Path

from mopd_verl.domain_sampling import allocate_domain_batch_counts
from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / (
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code_science_"
    "topk32_baseline_b525.yaml"
)


def test_six_gpu_three_domain_baseline_contract() -> None:
    config = load_config(CONFIG_PATH)

    assert config.runtime.python_bin == (
        "/home/shuang_qiu/env/miniconda3/envs/mopd-verl/bin/python"
    )
    assert config.data.train_batch_size == 525
    assert config.actor.ppo_mini_batch_size == 525
    assert config.data.domain_sampling_weights == {
        "math": 1.0,
        "code": 1.0,
        "science": 1.0,
    }
    assert allocate_domain_batch_counts(
        config.data.train_batch_size,
        config.data.domain_sampling_weights,
    ) == {"math": 175, "code": 175, "science": 175}

    assert config.worker_placement.separate_ref_policy
    assert config.worker_placement.actor_rollout.n_gpus_per_node == 5
    assert config.worker_placement.ref_policy.n_gpus_per_node == 1
    assert config.trainer.n_gpus_per_node == 5
    assert config.runtime.slurm_allocation_gpus == 6
    assert config.runtime.cuda_visible_devices is None
    assert config.rollout.tensor_model_parallel_size == 1
    assert config.rollout_correction.rollout_is == "null"
    assert config.rollout_correction.rollout_rs is None
    assert not config.rollout_correction.bypass_mode

    assert config.actor.distill_mode == "topk_renormalized_reverse_kl"
    assert config.actor.topk_distill_k == 32
    assert not config.audit.dynamic_domain_loss_weighting_enabled
    assert not config.audit.control_token_loss_weighting_enabled
    assert config.audit.control_token_ids == []
    assert config.audit.domain_control_token_ids == {}
    assert not config.audit.control_token_normalize_per_domain
    assert not config.audit.control_token_phase_gate_enabled
    assert not config.audit.control_token_span_weighting_enabled
    assert not config.audit.control_token_speed_weighting_enabled
    assert not config.audit.all_domain_shared_token_loss_weighting_enabled
    assert not config.domain_budgeting.enabled

    for artifact_path in (
        config.audit.output_dir,
        config.paper_eval.output_dir,
        config.trainer.experiment_name,
        config.trainer.default_local_dir,
    ):
        assert "baseline" in artifact_path
        for forbidden_name in ("control", "fixed", "reweight", "dynamic"):
            assert forbidden_name not in artifact_path

    rendered = format_command(build_command(config))
    assert "algorithm.rollout_correction.rollout_is=null" in rendered
    assert "algorithm.rollout_correction.rollout_is=token" not in rendered
    assert "+mopd_domain_budgeting.enabled=true" not in rendered
