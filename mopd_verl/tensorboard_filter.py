"""TensorBoard scalar pruning for MOPD audit metrics."""

from __future__ import annotations

from typing import Any

from mopd_verl.audit_math import finite_float

UNPRUNED_MODES = {"", "none", "off", "false", "0", "full", "all"}
DIRECT_AUDIT_CATEGORIES = {
    "audit",
    "advantage",
    "cost",
    "full_grad",
    "full_grad_alignment",
    "full_grad_contribution",
    "full_grad_conflict",
    "full_grad_cost",
    "loss",
    "length",
    "optimization",
    "reward",
    "sample_grad",
    "sample_grad_contribution",
    "sample_grad_cos",
    "teacher",
    "calibration",
    "coverage",
}

CORE_DOMAIN_DATA = {"domain_sample_count", "domain_token_count", "domain_token_frac"}
CORE_DOMAIN_LOSS = {
    "advantage_mean",
    "high_variance_sample_rate",
    "sample_opd_loss_mean",
    "sample_opd_loss_std",
    "sample_opd_loss_variance",
    "token_opd_loss_mean",
    "token_opd_loss_std",
    "token_opd_loss_variance",
}
CORE_DOMAIN_ADVANTAGE = {"positive_frac"}
CORE_DOMAIN_LENGTH = {"response_mean", "response_p95", "response_clip_ratio"}
CORE_SAMPLE_GRAD = {"norm_mean", "norm_p50", "norm_p95", "norm_max", "norm_cv", "sample_count"}
CORE_SAMPLE_GRAD_COS = {"domain_cos_mean", "domain_cos_p05", "domain_cos_negative_frac", "sample_count"}
CORE_SAMPLE_GRAD_CONTRIBUTION = {
    "projection_share_mean",
    "projection_share_min",
    "projection_share_max",
    "projection_share_negative_frac",
    "top1_abs_share",
}
CORE_GLOBAL_LOSS = {
    "sample_opd_loss_mean",
    "sample_opd_loss_std",
    "sample_opd_loss_variance",
    "token_opd_loss_mean",
    "token_opd_loss_std",
    "token_opd_loss_variance",
}
CORE_GLOBAL_OPTIMIZATION = {"learning_rate"}
CORE_DOMAIN_TEACHER = {"teacher_confidence_mean", "teacher_student_gap_mean"}
CORE_DOMAIN_REWARD = {"training_accuracy", "training_reward_mean"}
CORE_DOMAIN_COVERAGE = {"duplicate_rate"}
CORE_FULL_GRAD = {"grad_norm", "sample_count"}
CORE_GLOBAL_FULL_GRAD = {"total_grad_norm"}
CORE_FULL_GRAD_ALIGNMENT = {"full_grad_cosine_domain_total"}
CORE_FULL_GRAD_CONTRIBUTION = {"signed_projection_share"}
CORE_CONFLICT = {
    "conflict_magnitude_i_k",
    "full_grad_cosine_train_i_k",
}
CORE_AUDIT = {
    "error",
    "full_gradient_autograd_unavailable",
    "full_gradient_domain_sequential_available",
    "full_gradient_domain_sequential_unsupported",
    "full_gradient_replicated_all_reduce",
    "full_gradient_true_backward_fallback",
    "sample_gradient_distributed_unsupported",
    "sample_gradient_distributed_world_size",
    "sample_gradient_zero_norm_count",
    "wall_time_step",
}
CORE_GLOBAL_DATA = {"domain_mix_entropy", "total_samples", "total_tokens"}
CORE_GLOBAL_COST = {"gpu_seconds_step", "memory_peak_step", "step_seconds", "tokens_per_second"}
CORE_ACTOR = {
    "actor/entropy",
    "actor/grad_norm",
    "actor/lr",
    "actor/pg_clipfrac",
    "actor/pg_loss",
    "actor/ppo_kl",
}
CORE_ROLLOUT_CORR = {
    "kl",
    "ppl_ratio",
    "rollout_is_catastrophic_token_fraction",
    "rollout_is_eff_sample_size",
    "rollout_is_veto_fraction",
}
CORE_LENGTH = {"clip_ratio", "max", "mean"}
CORE_TIMING_SECONDS = {"gen", "step", "testing", "update_actor"}
CORE_TIMING_PER_TOKEN = {"gen", "update_actor"}
CORE_PERF = {"max_memory_allocated_gb", "throughput", "time_per_step", "total_num_tokens"}
CORE_GLOBAL_SEQLEN = {"max", "mean", "minmax_diff"}
CORE_TRAINING = {"epoch", "global_step"}


def is_direct_audit_metric_key(key: str) -> bool:
    parts = _parts(key)
    return len(parts) >= 2 and parts[1] in DIRECT_AUDIT_CATEGORIES


def filter_tensorboard_metrics(metrics: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode.lower() in UNPRUNED_MODES:
        return metrics
    if not metrics:
        return {}

    filtered: dict[str, float] = {}
    for key, value in metrics.items():
        numeric = finite_float(value)
        if numeric is not None and keep_core_metric(str(key)):
            filtered[str(key)] = numeric
    return filtered


def keep_core_metric(key: str) -> bool:
    parts = _parts(key)
    if not parts:
        return False
    root = parts[0]
    category = parts[1] if len(parts) > 1 else ""
    metric = parts[-1]

    if key.startswith("val-core/") or key in CORE_ACTOR:
        return True
    if root == "critic":
        return len(parts) == 3 and parts[1] in {"advantages", "returns", "rewards", "score"} and metric == "mean"
    if root == "rollout_corr":
        return metric in CORE_ROLLOUT_CORR
    if root == "response_length_non_aborted":
        return metric == "mean"
    if root in {"prompt_length", "response_length"}:
        return metric in CORE_LENGTH
    if key == "response/aborted_ratio":
        return True
    if root == "timing_s":
        return metric in CORE_TIMING_SECONDS
    if root == "timing_per_token_ms":
        return metric in CORE_TIMING_PER_TOKEN
    if root == "perf":
        return metric in CORE_PERF
    if root == "global_seqlen":
        return metric in CORE_GLOBAL_SEQLEN
    if root == "training":
        return metric in CORE_TRAINING
    if root == "global":
        return _keep_global(category, metric, parts)
    return _keep_domain(category, metric, parts)


def _keep_global(category: str, metric: str, parts: list[str]) -> bool:
    if category == "audit":
        return metric in CORE_AUDIT
    if category == "cost":
        return metric in CORE_GLOBAL_COST
    if category == "full_grad_cost":
        return metric in {"backward_seconds", "max_memory_allocated_gb"}
    if category == "full_grad":
        return metric in CORE_GLOBAL_FULL_GRAD
    if category == "full_grad_alignment":
        return metric in CORE_FULL_GRAD_ALIGNMENT
    if category == "full_grad_contribution":
        return metric in CORE_FULL_GRAD_CONTRIBUTION
    if category == "data":
        return metric in CORE_GLOBAL_DATA
    if category == "full_grad_conflict":
        return metric in CORE_CONFLICT
    if category == "loss":
        return metric in CORE_GLOBAL_LOSS
    if category == "optimization":
        return metric in CORE_GLOBAL_OPTIMIZATION
    if category == "validation":
        return False
    if category == "validation_gain":
        return not _contains_audit_category(parts[2:])
    if category == "validation_gain_stats":
        return metric in {"mean", "variance"} and not _contains_audit_category(parts[2:])
    return False


def _keep_domain(category: str, metric: str, parts: list[str]) -> bool:
    if category == "data":
        return metric in CORE_DOMAIN_DATA
    if category == "loss":
        return metric in CORE_DOMAIN_LOSS
    if category == "advantage":
        return metric in CORE_DOMAIN_ADVANTAGE
    if category == "length":
        return metric in CORE_DOMAIN_LENGTH
    if category == "sample_grad":
        return metric in CORE_SAMPLE_GRAD
    if category == "sample_grad_cos":
        return metric in CORE_SAMPLE_GRAD_COS
    if category == "sample_grad_contribution":
        return metric in CORE_SAMPLE_GRAD_CONTRIBUTION
    if category == "full_grad":
        return metric in CORE_FULL_GRAD
    if category == "teacher":
        return metric in CORE_DOMAIN_TEACHER
    if category == "reward":
        return metric in CORE_DOMAIN_REWARD
    if category == "calibration":
        return metric == "calibration_error"
    if category == "coverage":
        return metric in CORE_DOMAIN_COVERAGE
    if category == "validation":
        return False
    if category == "validation_gain":
        return not _contains_audit_category(parts[2:])
    if category == "validation_gain_stats":
        return metric in {"mean", "variance"} and not _contains_audit_category(parts[2:])
    return False


def _contains_audit_category(parts: list[str]) -> bool:
    return any(part in DIRECT_AUDIT_CATEGORIES for part in parts)


def _parts(key: str) -> list[str]:
    return [part for part in key.replace("\\", "/").split("/") if part]
