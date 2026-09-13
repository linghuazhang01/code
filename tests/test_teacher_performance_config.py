"""Check teacher-only performance settings reach worker Hydra overrides."""

from dataclasses import replace

import pytest

from mopd_verl.launch import build_overrides
from mopd_verl.settings import load_config
from mopd_verl.teacher_performance_config import TeacherPerformanceConfig


@pytest.mark.parametrize("values", [
    {"enabled": "false"}, {"expandable_segments": "false"},
    {"expandable_segments": 1}, {"topk_logprob_chunk_size": 0},
    {"topk_logprob_chunk_size": True}, {"max_micro_batch_size": 33},
    {"max_tokens": 57345}, {"memory_margin_gib": 0},
    {"memory_margin_gib": float("nan")}, {"moe_dispatch": "unknown"},
])
def test_invalid_tuning_fails_before_launch(values: dict) -> None:
    with pytest.raises(ValueError):
        TeacherPerformanceConfig(**values)


def test_teacher_settings_reach_ref_without_changing_actor() -> None:
    config = load_config("configs/token_selection/math/v3/"
        "top32kl_next_step_expanded_pruned_v3_unified_topp0p05_i1_w1_lossratio_5gpu_4a1t_b256_r20260907.yaml")
    overrides = build_overrides(config)
    assert "actor_rollout_ref.rollout.enforce_eager=False" in overrides
    assert "actor_rollout_ref.rollout.max_num_seqs=64" in overrides
    assert "+actor_rollout_ref.ref.teacher_performance.topk_logprob_chunk_size=1024" in overrides
    assert "+actor_rollout_ref.ref.teacher_performance.max_micro_batch_size=32" in overrides
    assert "+actor_rollout_ref.ref.teacher_performance.max_tokens=57344" in overrides
    assert "+actor_rollout_ref.ref.teacher_performance.expandable_segments=true" in overrides
    changed = replace(config, teacher_performance=replace(config.teacher_performance, enabled=False))
    original_actor = [value for value in overrides if value.startswith("actor_rollout_ref.actor.")]
    changed_actor = [value for value in build_overrides(changed) if value.startswith("actor_rollout_ref.actor.")]
    assert original_actor == changed_actor
    assert "+actor_rollout_ref.ref.teacher_performance.enabled=false" in build_overrides(changed)
    assert "+actor_rollout_ref.ref.teacher_performance.expandable_segments=true" in build_overrides(changed)
    allocator_disabled = replace(config, teacher_performance=replace(config.teacher_performance, expandable_segments=False))
    assert "+actor_rollout_ref.ref.teacher_performance.expandable_segments=false" in build_overrides(allocator_disabled)
    assert "+actor_rollout_ref.ref.teacher_performance.enabled=true" in build_overrides(allocator_disabled)
