from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
        "token_baseline_normalize_weights",
        "fire_opd_teacher_confidence_alpha",
        "fire_opd_student_confusion_beta",
        "fire_opd_trajectory_drop_ratio",
        "fire_opd_filter_trajectories",
    ):
        assert field_name in policy_config_source
        assert field_name in actor_yaml
