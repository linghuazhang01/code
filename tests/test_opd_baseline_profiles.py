from __future__ import annotations

from pathlib import Path

from mopd_verl.config_profiles import list_config_profiles
from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config
from mopd_verl.token_baselines import token_baseline_method


ROOT = Path(__file__).resolve().parents[1]
BASELINE_MATRIX = ROOT / "configs" / "baselines" / "opd_baselines.yaml"
CANONICAL_DIR = ROOT / "configs" / "baselines" / "canonical"


def test_baseline_matrix_profiles_and_rendered_overrides() -> None:
    expected_profiles = (
        "topk32_uniform",
        "entropy_topk32_rho50",
        "entropy_sample_topk32_rho50",
        "tip_topk32_rho50",
        "fire_token_topk32",
        "opd_native",
        "gopd_native_lambda0p5",
        "exopd_native_lambda1p25",
        "eopd_native",
        "fire_opd_native",
    )
    assert list_config_profiles(BASELINE_MATRIX) == expected_profiles

    configs = {
        profile: load_config(f"{BASELINE_MATRIX}::{profile}")
        for profile in expected_profiles
    }
    for config in configs.values():
        assert config.model.student_path == "../models/Qwen3-1.7B"
        assert set(config.model.domain_teacher_paths.values()) == {
            "../models/Qwen3-30B-A3B-Instruct-2507"
        }
        assert not config.actor.multi_teacher_distill
    rendered_configs = {
        profile: format_command(build_command(config))
        for profile, config in configs.items()
    }
    tip = configs["tip_topk32_rho50"]
    fire = configs["fire_opd_native"]
    gopd = configs["gopd_native_lambda0p5"]
    exopd = configs["exopd_native_lambda1p25"]
    eopd = configs["eopd_native"]

    assert token_baseline_method(tip.actor) == "tip_topk32"
    assert tip.actor.distill_loss_builder == "topk_kl"
    assert tip.actor.loss_agg_mode == "seq-mean-token-mean"
    assert fire.actor.token_baseline_method == "fire_opd"
    assert fire.actor.fire_opd_filter_trajectories
    assert gopd.actor.distill_loss_builder == "gopd"
    assert gopd.actor.lambda_vals == 0.5
    assert gopd.model.gopd_reference_path == gopd.model.student_path
    assert exopd.actor.distill_loss_builder == "exopd"
    assert exopd.actor.lambda_vals == 1.25
    assert exopd.model.gopd_reference_path == exopd.model.student_path
    assert eopd.actor.eopd_topk_k == 16
    assert eopd.rollout_correction.rollout_is is None

    rendered_tip = rendered_configs["tip_topk32_rho50"]
    rendered_fire = rendered_configs["fire_opd_native"]
    rendered_gopd = rendered_configs["gopd_native_lambda0p5"]
    rendered_exopd = rendered_configs["exopd_native_lambda1p25"]
    assert "policy_loss.token_baseline_method=tip_topk32" in rendered_tip
    assert "policy_loss.token_baseline_retention_ratio=0.5" in rendered_tip
    assert "policy_loss.fire_opd_filter_trajectories=true" in rendered_fire
    assert "policy_loss.distill_loss_builder=gopd" in rendered_gopd
    assert "policy_loss.lambda_vals=0.5" in rendered_gopd
    assert "+actor_rollout_ref.model.base_model_path=../models/Qwen3-1.7B" in (
        rendered_gopd
    )
    assert "policy_loss.distill_loss_builder=exopd" in rendered_exopd
    assert "+actor_rollout_ref.model.base_model_path=../models/Qwen3-1.7B" in (
        rendered_exopd
    )


def test_runtime_dataflow_carries_entropy_and_fire_filter_weights() -> None:
    actor_source = (
        ROOT / "third_party" / "verl" / "verl" / "workers" / "actor" / "dp_actor.py"
    ).read_text(encoding="utf-8")
    trainer_source = (
        ROOT
        / "third_party"
        / "verl"
        / "verl"
        / "trainer"
        / "ppo"
        / "ray_trainer.py"
    ).read_text(encoding="utf-8")
    policy_config_source = (
        ROOT
        / "third_party"
        / "verl"
        / "verl"
        / "workers"
        / "config"
        / "actor.py"
    ).read_text(encoding="utf-8")
    actor_yaml = (
        ROOT
        / "third_party"
        / "verl"
        / "verl"
        / "trainer"
        / "config"
        / "actor"
        / "actor.yaml"
    ).read_text(encoding="utf-8")

    assert '"student_entropy"' in actor_source
    assert '"token_baseline_trajectory_weight"' in actor_source
    assert "PRECOMPUTED_KEYS" in actor_source
    assert "token_baseline_requires_teacher_entropy" in trainer_source
    assert "fire_opd_trajectory_weights" in trainer_source
    assert "build_precomputed_baseline_tensors" in trainer_source
    for field_name in (
        "token_baseline_method",
        "token_baseline_retention_ratio",
        "token_baseline_selection_mode",
        "fire_opd_teacher_confidence_alpha",
        "fire_opd_student_confusion_beta",
        "fire_opd_trajectory_drop_ratio",
        "fire_opd_filter_trajectories",
    ):
        assert field_name in policy_config_source
        assert field_name in actor_yaml


def test_each_canonical_baseline_has_one_direct_config() -> None:
    expected = {
        "topk32_uniform.yaml",
        "entropy_topk32_rho50.yaml",
        "entropy_sample_topk32_rho50.yaml",
        "tip_topk32_rho50.yaml",
        "tip_native_full_vocab_rho50.yaml",
        "fire_token_topk32.yaml",
        "opd_native.yaml",
        "gopd_native_lambda0p5.yaml",
        "exopd_native_lambda1p25.yaml",
        "eopd_native.yaml",
        "fire_opd_native.yaml",
    }
    assert {path.name for path in CANONICAL_DIR.glob("*.yaml")} == expected
    for filename in expected:
        config = load_config(CANONICAL_DIR / filename)
        assert config.data.train_files
        assert config.model.student_path == "../models/Qwen3-1.7B"
        assert set(config.model.domain_teacher_paths.values()) == {
            "../models/Qwen3-30B-A3B-Instruct-2507"
        }


def test_native_tip_config_is_full_vocab_and_colocated() -> None:
    config = load_config(CANONICAL_DIR / "tip_native_full_vocab_rho50.yaml")
    command = format_command(build_command(config))

    assert config.actor.distill_loss_builder == "tip_full_vocab"
    assert not config.actor.topk_distill_enabled
    assert config.actor.token_baseline_method == "none"
    assert config.actor.tip_native_retention_ratio == 0.5
    assert config.actor.tip_native_entropy_clip_quantile == 0.98
    assert config.actor.tip_native_chunk_size == 512
    assert config.actor.lr_scheduler_type == "cosine"
    assert config.rollout.n == 1
    assert config.data.train_batch_size == 24
    assert set(config.data.domain_train_files) == {"math", "code", "science"}
    assert config.model.student_path == "../models/Qwen3-1.7B"
    assert config.model.primary_teacher_path == (
        "../models/Qwen3-30B-A3B-Instruct-2507"
    )
    assert not config.worker_placement.separate_ref_policy
    assert not config.model.use_remove_padding
    assert "policy_loss.distill_loss_builder=tip_full_vocab" in command
    assert "policy_loss.tip_native_entropy_clip_quantile=0.98" in command
    assert "actor.optim.lr_scheduler_type=cosine" in command
