"""Fail-closed contract for current-minibatch occurrence selection."""

from typing import Any


def validate_occurrence_config(config: Any, actor: Any = None) -> None:
    unit = getattr(config, "control_token_online_selection_unit", "token_id")
    if unit not in {"token_id", "occurrence"}:
        raise ValueError(
            "control_token_online_selection_unit must be token_id or occurrence"
        )
    if unit == "token_id":
        return
    expected = {
        "control_token_online_selection_enabled": True,
        "control_token_normalize_per_domain": True,
        "control_token_online_candidate_scope": "configured",
        "control_token_online_audit_interval_steps": 1,
        "control_token_online_window_steps": 1,
        "control_token_online_budget_mode": "top_p",
        "control_token_online_weight_mode": "fixed",
        "control_token_online_top_k_per_group": None,
        "control_token_loss_ratio_alpha": 1.0,
    }
    for key, value in expected.items():
        if getattr(config, key) != value:
            raise ValueError(f"occurrence requires {key}={value!r}")
    for key in (
        "control_token_adaptive_neighborhood_enabled",
        "control_token_phase_gate_enabled",
        "control_token_span_weighting_enabled",
        "control_token_speed_weighting_enabled",
        "all_domain_shared_token_weighting_enabled",
        "dynamic_weighting_enabled",
        "all_domain_shared_token_loss_weighting_enabled",
        "dynamic_domain_loss_weighting_enabled",
        "token_gradient_enabled",
    ):
        if getattr(config, key, False):
            raise ValueError(f"occurrence does not support {key}")
    supported = {"top_loss", "top_loss_teacher_confidence"}
    if config.control_token_online_selection_mode not in supported:
        raise ValueError("occurrence requires TopLoss or Loss+TC selection")
    selection_modes = dict(config.control_token_online_selection_mode_by_domain)
    if any(mode not in supported for mode in selection_modes.values()):
        raise ValueError("occurrence requires per-domain TopLoss or Loss+TC")
    for suffix, expected_mode in (("weight", "fixed"),):
        modes = dict(getattr(config, f"control_token_online_{suffix}_mode_by_domain"))
        if any(mode != expected_mode for mode in modes.values()):
            raise ValueError(f"occurrence requires per-domain {expected_mode}")
    weight = getattr(config, "control_token_loss_weight", None)
    if weight is None:
        weight = config.control_token_weight
    enabled = getattr(config, "control_token_loss_weighting_enabled", None)
    if enabled is None:
        enabled = config.control_token_weighting_enabled
    if weight != 4.0 or not enabled:
        raise ValueError("occurrence requires enabled Fixed4 weighting")
    from mopd_verl.domain_gradient.frozen_taxonomy import (
        CONTROL_TOKEN_IDS,
        STRUCTURE_TOKEN_IDS,
    )

    groups = dict(config.domain_control_token_candidate_groups)
    ids = dict(config.domain_control_token_candidate_ids)
    configured = set(config.control_token_candidate_ids)
    for values in ids.values():
        configured.update(values)
    for domain_groups in groups.values():
        for values in dict(domain_groups).values():
            configured.update(values)
    if not configured or not configured <= CONTROL_TOKEN_IDS | STRUCTURE_TOKEN_IDS:
        raise ValueError("occurrence requires configured C+S candidate IDs")
    if actor is None:
        return

    def get(key: str, default: Any = None) -> Any:
        return (
            actor.get(key, default)
            if hasattr(actor, "get")
            else getattr(actor, key, default)
        )

    requirements = {
        "distill_loss_builder": "topk_kl",
        "distill_mode": "topk_renormalized_reverse_kl",
        "topk_distill_enabled": True,
        "topk_distill_kl_direction": "reverse",
        "topk_distill_support_source": "teacher",
        "topk_distill_tail_bucket": False,
        "token_baseline_method": "none",
        "teacher_prefix_enabled": False,
        "region_dpo_enabled": False,
    }
    for key, value in requirements.items():
        if get(key, value) != value:
            raise ValueError(f"occurrence requires pure RKL: {key}={value!r}")
    for key in ("topk_distill_loss_by_domain", "token_category_loss_by_domain"):
        if get(key, {}):
            raise ValueError(f"occurrence does not support mixed KL / {key}")


def validate_occurrence_actor(actor: Any) -> None:
    """Actor-level guards also apply to direct Hydra launches."""

    def get(key: str, default: Any = None) -> Any:
        return (
            actor.get(key, default)
            if hasattr(actor, "get")
            else getattr(actor, key, default)
        )

    if get("ppo_epochs", 1) != 1 or get("ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError(
            "occurrence requires one PPO epoch and sequence parallel size 1"
        )
    if get("entropy_coeff", 0) != 0 or (
        get("use_kl_loss", False) and get("kl_loss_coef", 0) != 0
    ):
        raise ValueError("occurrence does not support auxiliary entropy/KL loss")
