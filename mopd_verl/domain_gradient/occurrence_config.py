"""Fail-closed contracts for current-step token and occurrence selection."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from mopd_verl.domain_gradient.structure_positions import (
    STRUCTURE_POSITION_PROFILE,
)
from mopd_verl.domain_gradient.token_taxonomy_registry import (
    TOKEN_TAXONOMY_ARTIFACT_SHA256,
    TOKEN_TAXONOMY_VERSIONS,
    normalize_token_taxonomy_version,
    token_taxonomy,
)


def uses_current_step_selection(config: Any) -> bool:
    """Legacy occurrence profiles already select within the current step."""
    return (
        getattr(config, "control_token_online_selection_timing", "next_step")
        == "current_step"
        or getattr(config, "control_token_online_selection_unit", "token_id")
        == "occurrence"
    )


def uses_versioned_next_step_selection(config: Any) -> bool:
    """Versioned C+S profiles can reuse the preceding production loss."""
    return (
        normalize_token_taxonomy_version(
            getattr(config, "token_taxonomy_version", "legacy")
        ) in TOKEN_TAXONOMY_VERSIONS
        and getattr(config, "structure_token_loss_weighting_enabled", False)
        and getattr(config, "control_token_online_selection_timing", "next_step")
        == "next_step"
    )


def normalize_tail_top_p_by_domain(
    domains: Sequence[str], value: Any,
) -> tuple[tuple[str, float], ...]:
    """Allow partial domain overrides and zero to disable a domain's tail."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError("control_token_tail_top_p_by_domain must be a mapping")
    items = value.items() if isinstance(value, Mapping) else value
    normalized = {}
    for domain, raw in items:
        domain = str(domain)
        if domain not in domains or domain in normalized:
            raise ValueError("Tail budget contains an unknown or duplicate domain")
        fraction = float(raw)
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Tail budgets must be finite and in [0, 1]")
        normalized[domain] = fraction
    return tuple((d, normalized[d]) for d in domains if d in normalized)


def normalize_versioned_cs_selection_mode_by_domain(
    domains: Sequence[str], value: Any,
) -> tuple[tuple[str, str], ...]:
    """Default existing profiles to fixed-S, Control-only selection."""

    if value is None:
        value = ()
    if isinstance(value, (str, bytes)):
        raise TypeError("versioned_cs_selection_mode_by_domain must be a mapping")
    items = value.items() if isinstance(value, Mapping) else value
    normalized: dict[str, str] = {}
    for domain, raw_mode in items:
        domain = str(domain)
        if domain not in domains or domain in normalized:
            raise ValueError(
                "V4/V5 C+S selection contains an unknown or duplicate domain"
            )
        mode = str(raw_mode)
        if mode not in {
            "position_fixed",
            "structure_only",
            "top_loss",
            "top_teacher_confidence",
        }:
            raise ValueError(f"Unsupported V4/V5 C+S selection mode: {mode!r}")
        normalized[domain] = mode
    if normalized and set(normalized) != set(domains):
        raise ValueError("V4/V5 C+S selection modes must cover every domain")
    return tuple(
        (domain, normalized.get(domain, "position_fixed")) for domain in domains
    )


def effective_code_cs_position_policy(config: Any) -> str | None:
    """Resolve Code position eligibility for versioned dynamic selectors."""

    setting = getattr(config, "code_cs_position_gate_enabled", None)
    if setting is not None and not isinstance(setting, bool):
        raise TypeError("code_cs_position_gate_enabled must be bool or null")
    version = normalize_token_taxonomy_version(
        getattr(config, "token_taxonomy_version", "legacy")
    )
    if version not in TOKEN_TAXONOMY_VERSIONS or not getattr(
        config, "control_token_online_selection_enabled", False
    ):
        if setting is not None:
            raise ValueError(
                "code_cs_position_gate_enabled requires active V4/V5 selection"
            )
        return None
    modes = dict(
        normalize_versioned_cs_selection_mode_by_domain(
            config.domains,
            getattr(config, "versioned_cs_selection_mode_by_domain", ()),
        )
    )
    code_mode = modes.get("code", "position_fixed")
    if uses_current_step_selection(config):
        if code_mode == "structure_only":
            return (
                "structure_only_ungated"
                if setting is False
                else "structure_only_gated"
            )
        if setting is not None:
            raise ValueError(
                "code_cs_position_gate_enabled requires current-step Code "
                "structure_only selection"
            )
        return None
    if not uses_versioned_next_step_selection(config):
        if setting is not None:
            raise ValueError(
                "code_cs_position_gate_enabled requires V4/V5 Next-Step selection"
            )
        return None
    if code_mode == "position_fixed":
        if setting is not None:
            raise ValueError(
                "code_cs_position_gate_enabled requires dynamic Code C+S selection"
            )
        return "position_fixed"
    return "dynamic_gated" if setting else "dynamic_ungated"


def uses_ungated_current_step_code_structure(config: Any) -> bool:
    """Return whether Code Structure-only selection ignores position masks."""

    return (
        uses_current_step_selection(config)
        and config.versioned_cs_selection_mode_map().get("code") == "structure_only"
        and getattr(config, "code_cs_position_gate_enabled", None) is False
    )


def _validate_tail_config(config: Any) -> None:
    timing = getattr(config, "control_token_online_selection_timing", "next_step")
    if timing not in {"next_step", "current_step"}:
        raise ValueError("selection timing must be next_step or current_step")
    fraction = getattr(config, "control_token_tail_top_p", 0.0)
    weight = getattr(config, "control_token_tail_weight", 1.0)
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (fraction, weight)):
        raise ValueError("Tail top_p and weight must be finite and in [0, 1]")
    overrides = normalize_tail_top_p_by_domain(
        config.domains, getattr(config, "control_token_tail_top_p_by_domain", ()),
    )
    mode = getattr(config, "control_token_tail_selection_mode", "bottom_loss")
    if mode not in {
        "bottom_loss", "bottom_teacher_confidence", "bottom_loss_teacher_confidence",
    }:
        raise ValueError("Unsupported tail selection mode")
    modes = {config.control_token_online_selection_mode}
    modes.update(dict(config.control_token_online_selection_mode_by_domain).values())
    if timing != "current_step" and (
        fraction != 0 or overrides or weight != 1 or mode != "bottom_loss"
        or "top_teacher_confidence" in modes
    ):
        raise ValueError("Tail selection and pure teacher confidence require current_step")


def _as_domain_map(value: Any) -> dict[str, tuple[int, ...]]:
    if value is None:
        return {}
    items = value.items() if isinstance(value, Mapping) else value
    return {
        str(domain): tuple(int(token_id) for token_id in token_ids)
        for domain, token_ids in items
    }


def _candidate_map(config: Any) -> dict[str, tuple[int, ...]]:
    groups = config.domain_control_token_candidate_groups
    if groups:
        group_map = dict(groups)
        return {
            str(domain): tuple(
                sorted(
                    {
                        int(token_id)
                        for token_ids in dict(domain_groups).values()
                        for token_id in token_ids
                    }
                )
            )
            for domain, domain_groups in group_map.items()
        }
    ids = _as_domain_map(config.domain_control_token_candidate_ids)
    if ids:
        return ids
    shared = tuple(int(value) for value in config.control_token_candidate_ids)
    return {str(domain): shared for domain in config.domains} if shared else {}


def _validate_versioned_taxonomy(config: Any) -> str:
    version = normalize_token_taxonomy_version(
        getattr(config, "token_taxonomy_version", "legacy")
    )
    structure_enabled = bool(
        getattr(config, "structure_token_loss_weighting_enabled", False)
    )
    structure_modes = dict(
        normalize_versioned_cs_selection_mode_by_domain(
            config.domains,
            getattr(config, "versioned_cs_selection_mode_by_domain", ()),
        )
    )
    if version == "legacy":
        if (
            structure_enabled
            or any(mode != "position_fixed" for mode in structure_modes.values())
        ):
            raise ValueError(
                "Explicit C+S selection requires token_v4 or token_v5."
            )
        return version
    if version not in TOKEN_TAXONOMY_VERSIONS:
        raise ValueError(f"Unsupported token taxonomy version: {version!r}")
    current_step = uses_current_step_selection(config)
    if current_step and any(
        mode not in {"position_fixed", "structure_only"}
        for mode in structure_modes.values()
    ):
        raise ValueError(
            "Token V4/V5 current_step supports position_fixed or "
            "structure_only selection."
        )
    if not current_step and any(
        mode == "structure_only" for mode in structure_modes.values()
    ):
        raise ValueError(
            "Token V4/V5 structure_only selection requires current_step."
        )
    if (
        current_step or config.control_token_online_selection_enabled
    ) and not structure_enabled:
        raise ValueError("Token V4/V5 requires enabled Structure weighting.")
    if getattr(config, "token_taxonomy_artifact_sha256", "") != (
        TOKEN_TAXONOMY_ARTIFACT_SHA256
    ):
        raise ValueError("Token V4/V5 taxonomy artifact SHA256 mismatch.")
    weight = float(getattr(config, "structure_token_loss_weight", 1.0))
    if not math.isfinite(weight) or weight not in {4.0, 8.0}:
        raise ValueError("Token V4/V5 Structure weighting requires Fixed4 or Fixed8.")
    position_profile = getattr(
        config, "structure_token_position_profile", "none"
    )
    allowed_position_profiles = {STRUCTURE_POSITION_PROFILE}
    if current_step:
        allowed_position_profiles.add("none")
    if position_profile not in allowed_position_profiles:
        raise ValueError(
            "Token V4/V5 requires the frozen position profile, or 'none' "
            "for current-step selection."
        )
    expected = token_taxonomy(version)
    if set(config.domains) != set(expected):
        raise ValueError("Token V4/V5 currently supports exactly math and code.")
    candidates = _candidate_map(config)
    structures = _as_domain_map(
        getattr(config, "domain_structure_token_ids", {})
    )
    for domain, definition in expected.items():
        if len(candidates.get(domain, ())) != len(set(candidates.get(domain, ()))):
            raise ValueError(f"{version}/{domain} Control IDs contain duplicates.")
        if len(structures.get(domain, ())) != len(set(structures.get(domain, ()))):
            raise ValueError(f"{version}/{domain} Structure IDs contain duplicates.")
        if set(candidates.get(domain, ())) != set(definition.control):
            raise ValueError(
                f"{version}/{domain} candidate IDs must equal frozen Control."
            )
        if set(structures.get(domain, ())) != set(definition.structure):
            raise ValueError(
                f"{version}/{domain} Structure IDs must equal the frozen artifact."
            )
    if set(candidates) != set(expected) or set(structures) != set(expected):
        raise ValueError("Token V4/V5 maps must contain exactly math and code.")
    return version


def current_step_candidate_map(config: Any) -> dict[str, tuple[int, ...]]:
    """Return the exact candidate pool used by current-step selection."""

    candidates = config.effective_domain_candidate_map()
    version = normalize_token_taxonomy_version(
        getattr(config, "token_taxonomy_version", "legacy")
    )
    if version not in TOKEN_TAXONOMY_VERSIONS:
        return candidates
    modes = dict(
        normalize_versioned_cs_selection_mode_by_domain(
            config.domains,
            getattr(config, "versioned_cs_selection_mode_by_domain", ()),
        )
    )
    structures = config.effective_domain_structure_map()
    return {
        domain: (
            structures[domain]
            if modes[domain] == "structure_only"
            else candidates[domain]
        )
        for domain in config.domains
    }


def validate_occurrence_config(config: Any, actor: Any = None) -> None:
    _validate_tail_config(config)
    taxonomy_version = _validate_versioned_taxonomy(config)
    effective_code_cs_position_policy(config)
    unit = getattr(config, "control_token_online_selection_unit", "token_id")
    if unit not in {"token_id", "occurrence"}:
        raise ValueError(
            "control_token_online_selection_unit must be token_id or occurrence"
        )
    versioned_next_step = uses_versioned_next_step_selection(config)
    if not uses_current_step_selection(config) and not versioned_next_step:
        return
    if versioned_next_step and any(
        mode != "position_fixed"
        for _, mode in normalize_versioned_cs_selection_mode_by_domain(
            config.domains,
            getattr(config, "versioned_cs_selection_mode_by_domain", ()),
        )
    ) and config.domain_control_token_candidate_groups:
        raise ValueError(
            "Shared V4/V5 C+S TopP cannot use Control-only candidate groups."
        )
    selection_modes = dict(config.control_token_online_selection_mode_by_domain)
    if versioned_next_step and (
        unit != "token_id"
        or config.control_token_online_selection_mode != "top_loss"
        or any(mode != "top_loss" for mode in selection_modes.values())
    ):
        raise ValueError("Token V4/V5 next_step requires token_id TopLoss selection.")
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
    supported = {"top_loss", "top_loss_teacher_confidence", "top_teacher_confidence"}
    if config.control_token_online_selection_mode not in supported:
        raise ValueError("current-step selection requires TopLoss, TC, or Loss+TC")
    selection_modes = dict(config.control_token_online_selection_mode_by_domain)
    if any(mode not in supported for mode in selection_modes.values()):
        raise ValueError("current-step requires per-domain TopLoss, TC, or Loss+TC")
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
    allowed_weights = {4.0} if taxonomy_version == "legacy" else {4.0, 8.0}
    if weight not in allowed_weights or not enabled:
        raise ValueError("current-step requires enabled Fixed4 (or versioned Fixed8) weighting")
    if taxonomy_version != "legacy" and weight != config.structure_token_loss_weight:
        raise ValueError("Token V4/V5 Control and Structure raw weights must match.")
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
    if not configured:
        raise ValueError("occurrence requires configured candidate IDs")
    if taxonomy_version == "legacy" and not (
        configured <= CONTROL_TOKEN_IDS | STRUCTURE_TOKEN_IDS
    ):
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
