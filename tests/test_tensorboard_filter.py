from __future__ import annotations

from mopd_verl.tensorboard_filter import filter_tensorboard_metrics


def test_core_filter_drops_retired_domain_sum_training_audit_metrics() -> None:
    metrics = {
        "global/audit/training_gradient_from_domain_sum_requested": 1.0,
        "global/audit/training_gradient_from_domain_sum_applied": 1.0,
        "global/audit/training_gradient_from_domain_sum_skipped_backward": 1.0,
        "global/audit/training_gradient_from_domain_sum_restore/rel_l2": 0.0,
    }

    assert filter_tensorboard_metrics(metrics, "core") == {}


def test_core_filter_keeps_domain_gradient_reliability_metrics() -> None:
    metrics = {
        "global/audit/domain_gradient_source_step": 2.0,
        "global/audit/domain_gradient_peak_cpu_vector_bytes_per_rank": 128.0,
        "global/audit/domain_gradient_peak_cpu_vector_bytes_actor_group_total": 256.0,
        "global/full_grad_closure/domain_sum_vs_audit_total/diff_norm": 0.01,
        "global/full_grad_closure/domain_sum_vs_audit_total/passed": 1.0,
        "global/full_grad_training_parity/audit_total_vs_training_total/passed": 1.0,
    }

    assert filter_tensorboard_metrics(metrics, "core") == metrics
