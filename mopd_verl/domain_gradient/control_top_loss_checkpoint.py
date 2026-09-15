"""Checkpoint migration and validation for online Control selection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from mopd_verl.domain_gradient.control_selection_budget import (
    CONFIGURED_CANDIDATE_SCOPE,
    normalize_candidate_scope,
    normalize_top_p_by_domain,
    normalize_valid_token_counts,
    validate_candidate_scope_contract,
)
from mopd_verl.domain_gradient.control_selection_scoring import (
    FIXED_ONLINE_WEIGHT_MODE,
    LOSS_RATIO_ONLINE_WEIGHT_MODE,
    PAIRED_ONLINE_WEIGHT_MODE,
    TOP_K_BUDGET_MODE,
    TOP_LOSS_SELECTION_MODE,
    TOP_P_BUDGET_MODE,
    normalize_online_budget_mode,
    normalize_online_selection_mode_by_domain,
    normalize_online_weight_mode,
    normalize_online_weight_mode_by_domain,
    normalize_selection_mode,
    validate_online_control_mode_contracts,
    validate_scaled_loss_ratio_weight,
)


StateT = TypeVar("StateT")


def restore_online_control_selection_state(
    state_type: type[StateT],
    value: Mapping[str, Any],
) -> StateT:
    """Restore one state while migrating legacy checkpoint schemas."""

    if int(value.get("schema_version", 1)) >= 10 and "loss_ratio_alpha" not in value:
        raise ValueError(
            "Schema 10+ online Control checkpoints require loss_ratio_alpha."
        )
    if int(value.get("schema_version", 1)) >= 12 and (
        "candidate_scope" not in value or "candidate_vocab_size" not in value
    ):
        raise ValueError(
            "Schema 12+ online Control checkpoints require candidate scope "
            "and vocabulary size."
        )
    domains = tuple(str(item) for item in value.get("domains", ()))
    raw_domain_candidates = value.get("domain_candidate_token_ids")
    if raw_domain_candidates is None:
        legacy_candidates = tuple(
            int(item) for item in value.get("candidate_token_ids", ())
        )
        domain_candidates = tuple((domain, legacy_candidates) for domain in domains)
    else:
        domain_candidates = tuple(
            (
                str(domain),
                tuple(int(token_id) for token_id in token_ids),
            )
            for domain, token_ids in raw_domain_candidates
        )
    raw_candidate_groups = value.get("domain_candidate_token_groups", ())
    domain_candidate_groups = tuple(
        (
            str(domain),
            tuple(
                (
                    str(group),
                    tuple(int(token_id) for token_id in token_ids),
                )
                for group, token_ids in groups
            ),
        )
        for domain, groups in raw_candidate_groups
    )
    budget_mode = normalize_online_budget_mode(
        value.get("budget_mode", TOP_K_BUDGET_MODE)
    )
    candidate_scope = normalize_candidate_scope(
        value.get("candidate_scope", CONFIGURED_CANDIDATE_SCOPE)
    )
    raw_candidate_vocab_size = value.get("candidate_vocab_size")
    candidate_vocab_size = (
        None if raw_candidate_vocab_size is None else int(raw_candidate_vocab_size)
    )
    top_p_by_domain = normalize_top_p_by_domain(
        domains,
        value.get("top_p_by_domain", ()),
    )
    selection_mode = normalize_selection_mode(
        value.get("selection_mode", TOP_LOSS_SELECTION_MODE)
    )
    weight_mode = normalize_online_weight_mode(
        value.get("weight_mode", FIXED_ONLINE_WEIGHT_MODE)
    )
    if domains:
        selection_mode_by_domain = normalize_online_selection_mode_by_domain(
            domains,
            value.get(
                "selection_mode_by_domain",
                value.get("control_token_online_selection_mode_by_domain", {}),
            ),
            selection_mode,
        )
        weight_mode_by_domain = normalize_online_weight_mode_by_domain(
            domains,
            value.get(
                "weight_mode_by_domain",
                value.get("control_token_online_weight_mode_by_domain", {}),
            ),
            weight_mode,
        )
    elif (
        value.get("selection_mode_by_domain")
        or value.get("weight_mode_by_domain")
        or value.get("control_token_online_selection_mode_by_domain")
        or value.get("control_token_online_weight_mode_by_domain")
    ):
        raise ValueError(
            "Online Control per-domain mode maps require configured domains."
        )
    else:
        selection_mode_by_domain = ()
        weight_mode_by_domain = ()
    history = tuple(
        (
            int(step),
            tuple(
                (
                    str(domain),
                    tuple(
                        (int(token_id), float(loss_sum), int(count))
                        for token_id, loss_sum, count in statistics
                    ),
                )
                for domain, statistics in domain_rows
            ),
        )
        for step, domain_rows in value.get("history", ())
    )
    raw_valid_token_count_history = value.get("valid_token_count_history")
    valid_token_count_history = tuple(
        (
            int(step),
            tuple((str(domain), int(count)) for domain, count in domain_counts),
        )
        for step, domain_counts in (raw_valid_token_count_history or ())
    )
    reset_legacy_top_p_history = bool(
        history
        and raw_valid_token_count_history is None
        and budget_mode == TOP_P_BUDGET_MODE
    )
    if reset_legacy_top_p_history:
        history = ()
    elif history and raw_valid_token_count_history is None:
        valid_token_count_history = tuple(
            (step, tuple((domain, 0) for domain in domains)) for step, _ in history
        )
    raw_valid_score_sum_history = value.get("valid_score_sum_history")
    valid_score_sum_history = tuple(
        (
            int(step),
            tuple((str(domain), float(score_sum)) for domain, score_sum in domain_sums),
        )
        for step, domain_sums in (raw_valid_score_sum_history or ())
    )
    if reset_legacy_top_p_history:
        valid_score_sum_history = ()
    elif history and raw_valid_score_sum_history is None:
        valid_score_sum_history = tuple(
            (step, tuple((domain, 0.0) for domain in domains)) for step, _ in history
        )
    active = tuple(
        (
            str(domain),
            tuple(int(token_id) for token_id in token_ids),
        )
        for domain, token_ids in value.get("active_token_ids", ())
    )
    raw_active_weights = value.get("active_token_weights")
    active_weights: tuple[tuple[str, tuple[tuple[int, float], ...]], ...]
    if raw_active_weights is None:
        active_weights = tuple((domain, ()) for domain in domains)
    else:
        active_weights = tuple(
            (
                str(domain),
                tuple(
                    (int(token_id), float(weight)) for token_id, weight in token_weights
                ),
            )
            for domain, token_weights in raw_active_weights
        )
    if reset_legacy_top_p_history:
        active = tuple((domain, ()) for domain in domains)
        active_weights = tuple((domain, ()) for domain in domains)
    raw_observed = value.get("last_observed_step")
    raw_audit = value.get("last_audit_step")
    state = cast(Any, state_type)(
        domains=domains,
        domain_candidate_token_ids=domain_candidates,
        audit_interval_steps=int(value.get("audit_interval_steps", 0)),
        window_steps=int(value.get("window_steps", 0)),
        min_mean_occurrences_per_step=float(
            value.get("min_mean_occurrences_per_step", 0.0)
        ),
        strict_occurrence_gate=bool(value.get("strict_occurrence_gate", False)),
        top_k=int(value.get("top_k", 0)),
        budget_mode=budget_mode,
        top_p=float(value.get("top_p", 1.0)),
        candidate_scope=candidate_scope,
        candidate_vocab_size=candidate_vocab_size,
        top_p_by_domain=top_p_by_domain,
        selection_mode=selection_mode,
        weight_mode=weight_mode,
        selection_mode_by_domain=selection_mode_by_domain,
        weight_mode_by_domain=weight_mode_by_domain,
        loss_ratio_alpha=float(value.get("loss_ratio_alpha", 1.0)),
        history=history,
        valid_token_count_history=valid_token_count_history,
        valid_score_sum_history=valid_score_sum_history,
        active_token_ids=active,
        active_token_weights=active_weights,
        last_observed_step=(None if raw_observed is None else int(raw_observed)),
        last_audit_step=None if raw_audit is None else int(raw_audit),
        update_count=int(value.get("update_count", 0)),
        domain_candidate_token_groups=domain_candidate_groups,
        top_k_per_group=(
            None
            if value.get("top_k_per_group") is None
            else int(value["top_k_per_group"])
        ),
    )
    state_view = cast(Any, state)
    _validate_restored_state(
        state_view,
        raw_valid_token_count_history=raw_valid_token_count_history,
        raw_valid_score_sum_history=raw_valid_score_sum_history,
    )
    return state


def _validate_restored_state(
    state: Any,
    *,
    raw_valid_token_count_history: object,
    raw_valid_score_sum_history: object,
) -> None:
    """Validate history alignment, budgets, and variable token weights."""

    effective_selection_modes, effective_weight_modes = (
        validate_online_control_mode_contracts(
            state.domains,
            selection_mode=state.selection_mode,
            weight_mode=state.weight_mode,
            selection_mode_by_domain=state.selection_mode_by_domain,
            weight_mode_by_domain=state.weight_mode_by_domain,
            interval=state.audit_interval_steps,
            window=state.window_steps,
            loss_ratio_alpha=state.loss_ratio_alpha,
        )
    )
    validate_candidate_scope_contract(
        state.candidate_scope,
        candidate_source_count=int(
            any(token_ids for _, token_ids in state.domain_candidate_token_ids)
        ),
        selection_modes=tuple(mode for _, mode in effective_selection_modes),
        candidate_vocab_size=state.candidate_vocab_size,
        top_k_per_group=state.top_k_per_group,
        require_full_vocab_size=True,
    )
    candidate_domains = tuple(domain for domain, _ in state.domain_candidate_token_ids)
    if len(candidate_domains) != len(set(candidate_domains)) or set(
        candidate_domains
    ) != set(state.domains):
        raise ValueError(
            "Checkpointed online Control candidate domains must exactly "
            "match configured domains."
        )
    for _, domain_statistics in state.history:
        history_domains = tuple(domain for domain, _ in domain_statistics)
        if len(history_domains) != len(set(history_domains)) or set(
            history_domains
        ) != set(state.domains):
            raise ValueError(
                "Checkpointed online Control history domains must exactly "
                "match configured domains."
            )
        if state.candidate_vocab_size is not None and any(
            token_id < 0 or token_id >= state.candidate_vocab_size
            for _, statistics in domain_statistics
            for token_id, _, _ in statistics
        ):
            raise ValueError(
                "Checkpointed online Control history token IDs exceed the "
                "candidate vocabulary."
            )
    if (
        effective_selection_modes != state.selection_mode_by_domain
        or effective_weight_modes != state.weight_mode_by_domain
    ):
        raise ValueError("Checkpointed online Control mode maps are not normalized.")
    history_steps = tuple(step for step, _ in state.history)
    count_steps = tuple(step for step, _ in state.valid_token_count_history)
    score_sum_steps = tuple(step for step, _ in state.valid_score_sum_history)
    if history_steps != count_steps or history_steps != score_sum_steps:
        raise ValueError(
            "Checkpointed online Control valid-token count and score-sum "
            "history must align with selector history."
        )
    if any(
        len(domain_counts) != len(state.domains)
        or {domain for domain, _ in domain_counts} != set(state.domains)
        or any(count < 0 for _, count in domain_counts)
        for _, domain_counts in state.valid_token_count_history
    ):
        raise ValueError(
            "Checkpointed online Control valid-token counts must be "
            "non-negative and exactly match domains."
        )
    if any(
        len(domain_sums) != len(state.domains)
        or {domain for domain, _ in domain_sums} != set(state.domains)
        or any(
            not math.isfinite(score_sum) or score_sum < 0.0
            for _, score_sum in domain_sums
        )
        for _, domain_sums in state.valid_score_sum_history
    ):
        raise ValueError(
            "Checkpointed online Control valid score sums must be finite, "
            "non-negative, and exactly match domains."
        )
    if (
        raw_valid_token_count_history is not None
        and state.budget_mode == TOP_P_BUDGET_MODE
    ):
        for (_, domain_statistics), (_, domain_counts) in zip(
            state.history,
            state.valid_token_count_history,
            strict=True,
        ):
            normalize_valid_token_counts(
                domains=state.domains,
                budget_mode=state.budget_mode,
                valid_token_counts=dict(domain_counts),
                statistics=domain_statistics,
            )
    if not math.isfinite(state.top_p) or not 0.0 < state.top_p <= 1.0:
        raise ValueError(
            "Checkpointed online Control top_p must be finite and in (0, 1]."
        )
    normalized_top_p_by_domain = normalize_top_p_by_domain(
        state.domains,
        state.top_p_by_domain,
    )
    if normalized_top_p_by_domain != state.top_p_by_domain:
        raise ValueError(
            "Checkpointed online Control per-domain Top-P map is not normalized."
        )
    if state.budget_mode == TOP_P_BUDGET_MODE and state.top_k_per_group is not None:
        raise ValueError(
            "Checkpointed online Control top_k_per_group requires Top-K mode."
        )
    if (
        state.has_weight_mode(LOSS_RATIO_ONLINE_WEIGHT_MODE)
        and state.history
        and raw_valid_score_sum_history is None
    ):
        raise ValueError(
            "Checkpointed loss-ratio state requires valid score-sum history."
        )
    active_map = state.active_map()
    weight_map = state.active_weight_map()
    for domain in state.domains:
        domain_weight_mode = state.weight_mode_for_domain(domain)
        domain_weights = weight_map.get(domain, {})
        if state.candidate_vocab_size is not None and any(
            token_id < 0 or token_id >= state.candidate_vocab_size
            for token_id in active_map.get(domain, ())
        ):
            raise ValueError(
                "Checkpointed online Control active token IDs exceed the "
                "candidate vocabulary."
            )
        if domain_weight_mode == FIXED_ONLINE_WEIGHT_MODE and domain_weights:
            raise ValueError(
                "Checkpointed fixed online Control domains cannot store "
                "variable token weights."
            )
        if domain_weight_mode == FIXED_ONLINE_WEIGHT_MODE:
            continue
        if set(active_map.get(domain, ())) != set(domain_weights):
            raise ValueError(
                "Checkpointed variable online Control weights must exactly "
                "match active token IDs."
            )
        weights = tuple(domain_weights.values())
        if domain_weight_mode == LOSS_RATIO_ONLINE_WEIGHT_MODE:
            for weight in weights:
                validate_scaled_loss_ratio_weight(weight)
        if domain_weight_mode == PAIRED_ONLINE_WEIGHT_MODE and any(
            not math.isfinite(weight) or not 1.0 <= weight <= 4.0 for weight in weights
        ):
            raise ValueError(
                "Checkpointed paired online Control weights must be finite "
                "and in [1, 4]."
            )
        if domain_weight_mode == LOSS_RATIO_ONLINE_WEIGHT_MODE and any(
            not math.isfinite(weight) or weight < min(1.0, state.loss_ratio_alpha)
            for weight in weights
        ):
            raise ValueError(
                "Checkpointed loss-ratio online Control weights must be "
                "finite and at least min(1, loss_ratio_alpha)."
            )
