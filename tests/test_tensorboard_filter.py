from __future__ import annotations

from mopd_verl.tensorboard_filter import filter_tensorboard_metrics


def test_core_filter_keeps_global_and_per_domain_adaptive_threshold_counts() -> None:
    metrics = {
        "actor/adaptive_threshold_pass_token_count": 42.0,
        "actor/adaptive_threshold_eligible_token_count": 100.0,
        "actor/adaptive_threshold_pass_token_fraction": 0.42,
        "actor/adaptive_domain/math/threshold_pass_token_count": 17.0,
        "actor/adaptive_domain/code/threshold_pass_token_fraction": 0.25,
    }

    assert filter_tensorboard_metrics(metrics, "core") == metrics


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


def test_core_filter_keeps_token_weight_amplification_metrics() -> None:
    metrics = {
        "global/token_weight/raw_configured_loss_abs_mass": 10.0,
        "global/token_weight/effective_configured_loss_abs_mass": 15.0,
        "global/token_weight/effective_to_raw_abs_loss_mass_ratio": 1.5,
        "global/token_weight/mean_token_gradient_multiplier": 1.4,
        "global/token_weight/gradient_multiplier_mean_abs_error": 0.0,
        "global/token_weight/amplified_token_occurrence_fraction": 0.3,
        "global/token_weight/control_token_id_count": 44.0,
        "global/token_weight/control_occurrence_count": 12.0,
        "global/token_weight/shared_occurrence_count": 10.0,
        "global/token_weight/control_shared_overlap_occurrence_count": 4.0,
        "global/token_weight/token_weighted_to_raw_abs_loss_mass_ratio": 2.0,
        "math/token_weight/cumulative_abs_loss_mass": 120.0,
        "math/token_weight/observed_token_type_count": 500.0,
        "math/token_weight/eligible_selection_score_p50": 1.3,
        "math/token_weight/eligible_selection_score_p90": 2.4,
        "math/token_weight/selected_selection_score_mean": 2.2,
        "math/token_weight/selected_selection_score_std": 0.3,
    }

    assert filter_tensorboard_metrics(metrics, "core") == metrics


def test_core_filter_keeps_control_speed_metrics() -> None:
    metrics = {
        "global/control_speed/enabled": 1.0,
        "global/control_speed/window_steps": 5.0,
        "global/control_speed/update_interval_steps": 2.0,
        "math/control_speed/control_gap_raw": 0.12,
        "math/control_speed/control_gap_ema": 0.10,
        "math/control_speed/optimization_speed": 0.01,
        "math/control_speed/speed_reference_step": 8.0,
        "math/control_speed/speed_computed_this_step": 1.0,
        "math/control_speed/weight_update_triggered": 1.0,
        "math/control_speed/control_weight_applied_raw": 2.0,
        "math/control_speed/control_weight_mapped_from_speed": 3.0,
        "math/control_speed/control_weight_next": 3.0,
        "math/control_speed/control_weight_applied_normalized": 1.8,
        "math/control_speed/control_occurrence_count": 256.0,
        "math/control_speed/observation_available": 1.0,
        "math/control_speed/minimum_occurrences_met": 1.0,
        "math/control_speed/state_observation_count": 9.0,
    }

    assert filter_tensorboard_metrics(metrics, "core") == metrics
