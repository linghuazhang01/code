"""TensorBoard tag naming helpers for MOPD audit metrics."""

from __future__ import annotations

from typing import Any


def safe_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _is_domain_data_metric(key: str) -> bool:
    return key in {
        "domain_sample_count",
        "domain_token_count",
        "domain_token_frac",
    }


def _is_domain_loss_metric(key: str) -> bool:
    return key in {
        "advantage_mean",
        "high_variance_sample_rate",
        "opd_loss_mean",
        "opd_loss_variance",
        "sample_loss_variance_mean",
    }


def _is_domain_teacher_metric(key: str) -> bool:
    return key in {"teacher_confidence_mean", "teacher_student_gap_mean"}


def domain_metric_category(key: str) -> str:
    if _is_domain_data_metric(key):
        return "data"
    if _is_domain_loss_metric(key):
        return "loss"
    if _is_domain_teacher_metric(key):
        return "teacher"
    if key.startswith("calibration"):
        return "calibration"
    if key == "duplicate_rate":
        return "coverage"
    return "misc"


def global_metric_category(key: str) -> str:
    if key in {"gpu_seconds_step", "tokens_per_second", "memory_peak_step", "step_seconds"}:
        return "cost"
    if key in {"total_tokens", "total_samples", "domain_mix_entropy"}:
        return "data"
    if key.startswith("audit_"):
        return "audit"
    return "misc"
