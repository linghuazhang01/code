from __future__ import annotations

from pathlib import Path

from mopd_verl.config_profiles import list_config_profiles
from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config
from mopd_verl.token_baselines import token_baseline_method


ROOT = Path(__file__).resolve().parents[1]
BASELINE_MATRIX = ROOT / "configs" / "baselines" / "opd_baselines.yaml"


def test_baseline_matrix_profiles_and_rendered_overrides() -> None:
    expected_profiles = (
        "topk32_uniform",
        "entropy_topk32_rho50",
        "entropy_sample_topk32_rho50",
        "tip_topk32_rho50",
        "fire_token_topk32",
        "opd_native",
        "exopd_native_lambda1p25",
        "eopd_native",
        "fire_opd_native",
    )
    assert list_config_profiles(BASELINE_MATRIX) == expected_profiles

    configs = {
        profile: load_config(f"{BASELINE_MATRIX}::{profile}")
        for profile in expected_profiles
    }
    rendered_configs = {
        profile: format_command(build_command(config))
        for profile, config in configs.items()
    }
    tip = configs["tip_topk32_rho50"]
    fire = configs["fire_opd_native"]
    exopd = configs["exopd_native_lambda1p25"]
    eopd = configs["eopd_native"]

    assert token_baseline_method(tip.actor) == "tip_topk32"
    assert tip.actor.distill_loss_builder == "topk_kl"
    assert fire.actor.token_baseline_method == "fire_opd"
    assert fire.actor.fire_opd_filter_trajectories
    assert exopd.actor.distill_loss_builder == "exopd"
    assert exopd.actor.lambda_vals == 1.25
    assert exopd.model.student_base_path == exopd.model.student_path
    assert eopd.actor.eopd_topk_k == 16

    rendered_tip = rendered_configs["tip_topk32_rho50"]
    rendered_fire = rendered_configs["fire_opd_native"]
    rendered_exopd = rendered_configs["exopd_native_lambda1p25"]
    assert "policy_loss.token_baseline_method=tip_topk32" in rendered_tip
    assert "policy_loss.token_baseline_retention_ratio=0.5" in rendered_tip
    assert "policy_loss.fire_opd_filter_trajectories=true" in rendered_fire
    assert "policy_loss.distill_loss_builder=exopd" in rendered_exopd
    assert "+actor_rollout_ref.model.base_model_path=../models/Qwen3-4B" in (
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
    assert "token_baseline_requires_teacher_entropy" in trainer_source
    assert "fire_opd_trajectory_weights" in trainer_source
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
