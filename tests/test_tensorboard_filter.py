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
        (
            "global/pre_reweight_full_grad_closure/"
            "domain_sum_vs_pre_reweight_audit_total/diff_norm"
        ): 0.01,
        (
            "global/pre_reweight_full_grad_closure/"
            "domain_sum_vs_pre_reweight_audit_total/passed"
        ): 1.0,
        "global/full_grad_training_parity/audit_total_vs_training_total/passed": 1.0,
    }

    assert filter_tensorboard_metrics(metrics, "core") == metrics


def test_core_filter_keeps_signed_logp_vocab_cosines() -> None:
    metrics = {
        "global/logp_vocab_cosine/math_vs_code/token_count_cosine": 0.9,
        "global/logp_vocab_cosine/math_vs_code/logp_sum_cosine": -0.2,
        "global/logp_vocab_cosine/math_vs_code/logp_mean_cosine": -0.1,
    }

    assert filter_tensorboard_metrics(metrics, "core") == metrics


def test_core_filter_keeps_configured_loss_gap_tail_and_dynamic_metrics() -> None:
    metrics = {
        "math/loss/token_opd_loss_sum": 3.0,
        "math/loss/token_opd_loss_p95": 1.9,
        "math/teacher/teacher_student_gap_p05": -0.4,
        "math/teacher/teacher_student_gap_sum": 0.0,
        "math/token_grad/tail_grad_signed_projection_share": 0.7,
        "math/token_grad/tail_token_fraction": 0.1,
        "math/token_grad/top_p1_grad_signed_projection_share": 1.0,
        "math/token_grad/top_p1_token_fraction": 1.0,
        "math/dynamic_weight/applied_gradient_weight": 1.2,
        "math/dynamic_weight/bounded_target_gradient_weight": 1.3,
        "math/dynamic_weight/ema_grad_norm": 2.5,
        "math/dynamic_weight/weighted_grad_norm": 3.0,
        "global/teacher/teacher_student_gap_p95": 0.8,
    }

    assert filter_tensorboard_metrics(metrics, "core") == metrics


def test_core_filter_drops_retired_token_conflict_metrics() -> None:
    metrics = {
        "math/token_conflict/comparison_token_count": 8.0,
        "math/token_conflict/combined_diff_mean": 0.4,
    }

    assert filter_tensorboard_metrics(metrics, "core") == {}
