"""Resolved contracts for the EOPD teacher optimization candidates."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mopd_verl.launch import build_overrides
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "plan" / "eopd-ref-logp-optimization-20260915" / "configs"


def test_stable_sort_candidate_is_a_single_system_change() -> None:
    baseline = load_config(
        ROOT / "configs" / "baselines" / "qwen1p7b_30b_eopd_native_8gpu_b528.yaml"
    )
    candidate = load_config(CANDIDATES / "eopd_ref_stable_sort_4513.yaml")

    assert candidate.data == baseline.data
    assert candidate.model == baseline.model
    assert candidate.actor == baseline.actor
    assert candidate.rollout == baseline.rollout
    assert candidate.worker_placement == baseline.worker_placement
    assert candidate.teacher_performance == baseline.teacher_performance
    assert candidate.teacher_performance.moe_dispatch == "stable_sort"
    assert not candidate.teacher_performance.fused_statistics


def test_stock_control_disables_only_moe_dispatch() -> None:
    baseline = load_config(
        ROOT / "configs" / "baselines" / "qwen1p7b_30b_eopd_native_8gpu_b528.yaml"
    )
    control = load_config(CANDIDATES / "eopd_ref_stock_control.yaml")

    assert control.data == baseline.data
    assert control.model == baseline.model
    assert control.actor == baseline.actor
    assert control.rollout == baseline.rollout
    assert control.worker_placement == baseline.worker_placement
    assert control.teacher_performance.moe_dispatch == "stock"
    assert (
        replace(control.teacher_performance, moe_dispatch="stable_sort")
        == baseline.teacher_performance
    )


def test_primary_candidate_changes_only_teacher_systems_path() -> None:
    baseline = load_config(
        ROOT / "configs" / "baselines" / "qwen1p7b_30b_eopd_native_8gpu_b528.yaml"
    )
    candidate = load_config(CANDIDATES / "eopd_ref_kernel_tuned.yaml")
    overrides = build_overrides(candidate)

    assert candidate.worker_placement == baseline.worker_placement
    assert candidate.data == baseline.data
    assert candidate.model == baseline.model
    assert candidate.actor == baseline.actor
    assert candidate.rollout == baseline.rollout
    assert candidate.teacher_performance.fused_statistics
    assert candidate.teacher_performance.compact_topk_ids
    assert candidate.teacher_performance.adaptive_topk_chunk_size == 256
    assert (
        "+actor_rollout_ref.ref.teacher_performance.fused_statistics=true" in overrides
    )
    assert (
        "+actor_rollout_ref.ref.teacher_performance.compact_topk_ids=true" in overrides
    )
    assert (
        "+actor_rollout_ref.ref.teacher_performance.adaptive_topk_chunk_size=256"
        in overrides
    )


def test_prefetch_candidate_is_an_isolated_fsdp_followup() -> None:
    candidate = load_config(CANDIDATES / "eopd_ref_forward_prefetch.yaml")
    overrides = build_overrides(candidate)

    assert "actor_rollout_ref.ref.fsdp_config.fsdp_size=2" in overrides
    assert "actor_rollout_ref.ref.fsdp_config.forward_prefetch=True" in overrides
    assert candidate.worker_placement.ref_policy.n_gpus_per_node == 2


def test_replica_candidate_has_a_conservative_memory_smoke_envelope() -> None:
    candidate = load_config(CANDIDATES / "eopd_ref_replica_experimental.yaml")
    overrides = build_overrides(candidate)

    assert "actor_rollout_ref.ref.fsdp_config.fsdp_size=1" in overrides
    assert "actor_rollout_ref.ref.fsdp_config.param_offload=False" in overrides
    assert candidate.teacher_performance.topk_logprob_chunk_size == 256
    assert candidate.teacher_performance.adaptive_topk_chunk_size is None
    assert candidate.teacher_performance.max_micro_batch_size == 8
    assert candidate.teacher_performance.max_tokens == 16384
