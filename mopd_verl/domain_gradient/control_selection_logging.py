"""JSONL provenance for online Control-token selection."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from mopd_verl.audit_io import step_jsonl_dir
from mopd_verl.domain_gradient.control_selection_budget import (
    FULL_VOCABULARY_CANDIDATE_SCOPE,
)
from mopd_verl.domain_gradient.control_selection_types import (
    SelectionScoreDistribution,
)
from mopd_verl.domain_gradient.control_top_loss import (
    OnlineControlSelectionOutcome,
    OnlineControlSelectionState,
)


def _score_distribution_record(
    distribution: SelectionScoreDistribution | None,
) -> dict[str, float | int] | None:
    """Return a stable JSON representation for one score distribution."""

    if distribution is None:
        return None
    return {
        "count": distribution.count,
        "mean": distribution.mean,
        "std": distribution.std,
        "min": distribution.minimum,
        "p10": distribution.p10,
        "p50": distribution.p50,
        "p90": distribution.p90,
        "max": distribution.maximum,
    }


def append_online_control_selection_jsonl(
    *,
    output_dir: str,
    state: OnlineControlSelectionState,
    outcome: OnlineControlSelectionOutcome,
    control_weight: float,
    applied_token_ids: Mapping[str, Sequence[int]],
    applied_token_weights: Mapping[str, Mapping[int, float]],
    applied_token_occurrence_counts: Mapping[str, int],
    valid_token_counts: Mapping[str, int],
    q_normalization_stats: Mapping[str, Mapping[str, float]] | None = None,
    observed_candidate_token_ids: Mapping[str, Sequence[int]] | None = None,
) -> None:
    """Persist exact current- and next-step online selector membership."""

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    ):
        return
    destination = step_jsonl_dir(output_dir, outcome.observed_step, create=True)
    observed_candidates = {
        domain: tuple(int(token_id) for token_id in token_ids)
        for domain, token_ids in (observed_candidate_token_ids or {}).items()
    }
    configured_candidates = state.candidate_map()
    effective_candidates = (
        state.window_candidate_map()
        if state.candidate_scope == FULL_VOCABULARY_CANDIDATE_SCOPE
        else configured_candidates
    )
    effective_candidate_union = {
        token_id
        for token_ids in effective_candidates.values()
        for token_id in token_ids
    }
    record: dict[str, Any] = {
        "observed_step": outcome.observed_step,
        "q_normalization_stats": q_normalization_stats,
        "applies_from_step": outcome.observed_step + 1,
        "audit_triggered": outcome.audit_triggered,
        "duplicate_step": outcome.duplicate_step,
        "history_reset": outcome.history_reset,
        "window_fill_steps": outcome.window_fill_steps,
        "window_steps": state.window_steps,
        "audit_interval_steps": state.audit_interval_steps,
        "min_mean_occurrences_per_step": state.min_mean_occurrences_per_step,
        "strict_occurrence_gate": state.strict_occurrence_gate,
        "top_k": state.top_k,
        "top_k_per_group": state.top_k_per_group,
        "budget_mode": state.budget_mode,
        "top_p": state.top_p,
        "top_p_by_domain": state.top_p_map(),
        "top_p_basis": "selected_occurrences_over_valid_tokens",
        "candidate_scope": state.candidate_scope,
        "candidate_population": (
            "all_valid_response_tokens"
            if state.candidate_scope == FULL_VOCABULARY_CANDIDATE_SCOPE
            else "configured_token_ids"
        ),
        "candidate_universe_vocab_size": state.candidate_vocab_size,
        "selection_mode": state.selection_mode,
        "weight_mode": state.weight_mode,
        "control_token_online_selection_mode": state.selection_mode,
        "control_token_online_weight_mode": state.weight_mode,
        "selection_mode_by_domain": state.selection_mode_map(),
        "weight_mode_by_domain": state.weight_mode_map(),
        "control_token_online_selection_mode_by_domain": state.selection_mode_map(),
        "control_token_online_weight_mode_by_domain": state.weight_mode_map(),
        "loss_ratio_alpha": state.loss_ratio_alpha,
        "candidate_token_count": len(effective_candidate_union),
        "candidate_union_count": len(effective_candidate_union),
        "configured_candidate_union_count": len(state.candidate_token_ids),
        "observed_candidate_union_count": len(
            {
                token_id
                for token_ids in observed_candidates.values()
                for token_id in token_ids
            }
        ),
        "domain_candidate_token_counts": {
            domain: len(token_ids) for domain, token_ids in effective_candidates.items()
        },
        "configured_domain_candidate_token_counts": {
            domain: len(token_ids)
            for domain, token_ids in configured_candidates.items()
        },
        "observed_domain_candidate_token_counts": {
            domain: len(observed_candidates.get(domain, ())) for domain in state.domains
        },
        "domain_candidate_group_counts": {
            domain: {group: len(token_ids) for group, token_ids in groups.items()}
            for domain, groups in state.candidate_group_map().items()
        },
        "control_weight": float(control_weight),
        "applied_token_ids": {
            str(domain): [int(token_id) for token_id in token_ids]
            for domain, token_ids in applied_token_ids.items()
        },
        "applied_token_weights": {
            str(domain): {
                str(int(token_id)): float(weight)
                for token_id, weight in token_weights.items()
            }
            for domain, token_weights in applied_token_weights.items()
        },
        "applied_token_coverage": {
            str(domain): {
                "token_type_count": len(applied_token_ids.get(domain, ())),
                "occurrence_count": int(applied_token_occurrence_counts.get(domain, 0)),
                "valid_token_count": int(valid_token_counts.get(domain, 0)),
                "occurrence_fraction": (
                    float(applied_token_occurrence_counts.get(domain, 0))
                    / float(valid_token_counts[domain])
                    if valid_token_counts.get(domain, 0) > 0
                    else 0.0
                ),
            }
            for domain in state.domains
        },
        "next_active_token_ids": state.active_map(),
        "next_active_token_weights": state.active_weight_map(),
        "domains": {
            result.domain: {
                "top_p": state.top_p_for_domain(result.domain),
                "valid_token_count": result.valid_token_count,
                "eligible_token_count": result.eligible_token_count,
                "selected_occurrence_count": result.selected_occurrence_count,
                "selected_occurrence_fraction": result.selected_occurrence_fraction,
                "target_occurrence_count": result.target_occurrence_count,
                "top_p_target_reached": result.top_p_target_reached,
                "top_p_occurrence_shortfall": result.top_p_occurrence_shortfall,
                "eligible_selection_score_distribution": (
                    _score_distribution_record(result.eligible_score_distribution)
                ),
                "selected_selection_score_distribution": (
                    _score_distribution_record(result.selected_score_distribution)
                ),
                "selected_occurrence_mean_abs_loss": (
                    result.selected_occurrence_mean_abs_loss
                ),
                "other_occurrence_count": result.other_occurrence_count,
                "other_occurrence_mean_abs_loss": (
                    result.other_occurrence_mean_abs_loss
                ),
                "raw_selected_to_other_loss_ratio": (
                    result.raw_selected_to_other_loss_ratio
                ),
                "selected_raw_loss_ratio_weight": (
                    result.selected_raw_loss_ratio_weight
                ),
                "selected_unscaled_loss_ratio_weight": (
                    result.selected_unscaled_loss_ratio_weight
                ),
                "selected_tokens": [
                    {
                        "token_id": item.token_id,
                        "occurrence_count": item.occurrence_count,
                        "mean_occurrences_per_step": item.mean_occurrences_per_step,
                        "mean_abs_loss": item.mean_abs_loss,
                        "mean_selection_score": item.mean_selection_score,
                        "optimization_speed": item.optimization_speed,
                        "observed_step_count": item.observed_step_count,
                    }
                    for item in result.selected_tokens
                ],
            }
            for result in outcome.domain_results
        },
    }
    with (destination / "online_control_selection.jsonl").open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
