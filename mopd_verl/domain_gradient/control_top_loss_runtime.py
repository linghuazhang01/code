"""Torch aggregation and logging for online Control-token selection."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from mopd_verl.audit_io import step_jsonl_dir
from mopd_verl.domain_gradient.control_top_loss import (
    OnlineControlSelectionOutcome,
    OnlineControlSelectionState,
)
from mopd_verl.domain_gradient.control_selection_types import (
    SelectionScoreDistribution,
)
from mopd_verl.domain_gradient.control_selection_scoring import (
    PAIRED_SIGNAL_SELECTION_MODES,
    TOP_LOSS_SELECTION_MODE,
    TOP_TEACHER_CONFIDENCE_STUDENT_ENTROPY_SELECTION_MODE,
    normalize_selection_mode,
    paired_selection_bonus,
)


@dataclass(frozen=True)
class GlobalCandidateLossStatistics:
    """Globally reduced candidate scores and valid-token denominators."""

    by_domain: dict[str, dict[int, tuple[float, int]]]
    valid_token_counts: dict[str, int]


def global_candidate_loss_statistics_with_valid_counts(
    token_id_batches: Sequence[torch.Tensor],
    loss_batches: Sequence[torch.Tensor],
    mask_batches: Sequence[torch.Tensor],
    label_batches: Sequence[Sequence[str]],
    *,
    domains: Sequence[str],
    candidate_token_ids: Sequence[int] = (),
    domain_candidate_token_ids: Mapping[str, Sequence[int]] | None = None,
    selection_mode: str = TOP_LOSS_SELECTION_MODE,
    student_entropy_batches: Sequence[torch.Tensor] | None = None,
    teacher_entropy_batches: Sequence[torch.Tensor] | None = None,
) -> GlobalCandidateLossStatistics:
    """Aggregate the configured selector score and count in one all-reduce."""

    lengths = {
        len(token_id_batches),
        len(loss_batches),
        len(mask_batches),
        len(label_batches),
    }
    if lengths != {len(token_id_batches)} or not token_id_batches:
        raise ValueError(
            "Online Control observer batches must be non-empty and aligned."
        )
    mode = normalize_selection_mode(selection_mode)
    paired_mode = mode in PAIRED_SIGNAL_SELECTION_MODES
    if paired_mode and (
        student_entropy_batches is None
        or len(student_entropy_batches) != len(token_id_batches)
    ):
        raise ValueError(
            "Paired online Control selection requires one aligned Student "
            "entropy matrix per token batch."
        )
    if (
        mode == TOP_TEACHER_CONFIDENCE_STUDENT_ENTROPY_SELECTION_MODE
        and (
            teacher_entropy_batches is None
            or len(teacher_entropy_batches) != len(token_id_batches)
        )
    ):
        raise ValueError(
            "Teacher-confidence online Control selection requires one aligned "
            "Teacher entropy matrix per token batch."
        )
    normalized_domains = tuple(str(domain) for domain in domains)
    if domain_candidate_token_ids is not None:
        domain_candidates = {
            str(domain): tuple(sorted({int(item) for item in token_ids}))
            for domain, token_ids in domain_candidate_token_ids.items()
        }
    else:
        candidates = tuple(sorted({int(item) for item in candidate_token_ids}))
        domain_candidates = {domain: candidates for domain in normalized_domains}
    if set(domain_candidates) != set(normalized_domains):
        raise ValueError("Online Control domain candidates must exactly match domains.")
    candidates = tuple(
        sorted(
            {
                token_id
                for token_ids in domain_candidates.values()
                for token_id in token_ids
            }
        )
    )
    if not candidates:
        raise ValueError("Online Control candidates must be non-empty.")
    device = loss_batches[0].device
    packed = torch.zeros(
        (len(normalized_domains), 2, len(candidates) + 1),
        device=device,
        dtype=torch.float64,
    )
    candidate_tensor = torch.tensor(
        candidates,
        device=device,
        dtype=torch.long,
    )
    allowed = torch.zeros(
        (len(normalized_domains), len(candidates)),
        device=device,
        dtype=torch.bool,
    )
    candidate_indices = {token_id: index for index, token_id in enumerate(candidates)}
    for domain_index, domain in enumerate(normalized_domains):
        allowed[
            domain_index,
            [candidate_indices[item] for item in domain_candidates[domain]],
        ] = True
    for batch_index, (token_ids, losses, mask, labels) in enumerate(
        zip(
            token_id_batches,
            loss_batches,
            mask_batches,
            label_batches,
            strict=True,
        )
    ):
        if token_ids.shape != losses.shape or losses.shape != mask.shape:
            raise ValueError(
                "Online Control token IDs, configured loss, and mask must align."
            )
        if len(labels) != int(losses.shape[0]):
            raise ValueError("Online Control domain labels must align with batch rows.")
        valid = mask.to(device=device, dtype=torch.bool)
        loss_values = losses.to(device=device, dtype=torch.float64)
        if not torch.isfinite(loss_values[valid]).all():
            raise ValueError("Online Control configured loss must be finite.")
        if paired_mode:
            if student_entropy_batches is None:
                raise RuntimeError(
                    "Validated paired selection is missing Student entropy."
                )
            score_values = paired_selection_bonus(
                selection_mode=mode,
                student_entropy=student_entropy_batches[batch_index].to(
                    device=device
                ),
                response_mask=valid,
                configured_loss=loss_values,
                teacher_entropy=(
                    teacher_entropy_batches[batch_index].to(device=device)
                    if teacher_entropy_batches is not None
                    else None
                ),
            ).to(dtype=torch.float64)
        else:
            score_values = loss_values.abs()
        ids = token_ids.to(device=device, dtype=torch.long)
        for domain_index, domain in enumerate(normalized_domains):
            rows = torch.tensor(
                [str(label) == domain for label in labels],
                device=device,
                dtype=torch.bool,
            ).unsqueeze(-1)
            selected_mask = valid & rows
            packed[domain_index, 1, -1].add_(selected_mask.sum())
            selected_ids = ids[selected_mask]
            if selected_ids.numel() == 0:
                continue
            positions = torch.searchsorted(candidate_tensor, selected_ids)
            in_range = positions < candidate_tensor.numel()
            safe_positions = positions.clamp(max=candidate_tensor.numel() - 1)
            matched = (
                in_range
                & candidate_tensor[safe_positions].eq(selected_ids)
                & allowed[domain_index, safe_positions]
            )
            matched_positions = positions[matched]
            packed[domain_index, 0].scatter_add_(
                0,
                matched_positions,
                score_values[selected_mask][matched],
            )
            packed[domain_index, 1].scatter_add_(
                0,
                matched_positions,
                torch.ones_like(matched_positions, dtype=torch.float64),
            )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            packed,
            op=torch.distributed.ReduceOp.SUM,
        )
    packed_cpu = packed.cpu()
    return GlobalCandidateLossStatistics(
        by_domain={
            domain: {
                token_id: (
                    float(packed_cpu[domain_index, 0, candidate_index]),
                    int(packed_cpu[domain_index, 1, candidate_index]),
                )
                for candidate_index, token_id in enumerate(candidates)
                if token_id in set(domain_candidates[domain])
                if packed_cpu[domain_index, 1, candidate_index] > 0
            }
            for domain_index, domain in enumerate(normalized_domains)
        },
        valid_token_counts={
            domain: int(packed_cpu[domain_index, 1, -1])
            for domain_index, domain in enumerate(normalized_domains)
        },
    )


def global_candidate_loss_statistics(
    *args: Any,
    **kwargs: Any,
) -> dict[str, dict[int, tuple[float, int]]]:
    """Backward-compatible candidate-only view of the global statistics."""

    return global_candidate_loss_statistics_with_valid_counts(
        *args,
        **kwargs,
    ).by_domain


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
) -> None:
    """Persist exact current- and next-step online selector membership."""

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    ):
        return
    destination = step_jsonl_dir(output_dir, outcome.observed_step, create=True)
    record: dict[str, Any] = {
        "observed_step": outcome.observed_step,
        "applies_from_step": outcome.observed_step + 1,
        "audit_triggered": outcome.audit_triggered,
        "duplicate_step": outcome.duplicate_step,
        "history_reset": outcome.history_reset,
        "window_fill_steps": outcome.window_fill_steps,
        "window_steps": state.window_steps,
        "audit_interval_steps": state.audit_interval_steps,
        "min_mean_occurrences_per_step": (state.min_mean_occurrences_per_step),
        "strict_occurrence_gate": state.strict_occurrence_gate,
        "top_k": state.top_k,
        "top_k_per_group": state.top_k_per_group,
        "budget_mode": state.budget_mode,
        "top_p": state.top_p,
        "selection_mode": state.selection_mode,
        "weight_mode": state.weight_mode,
        "candidate_token_count": len(state.candidate_token_ids),
        "candidate_union_count": len(state.candidate_token_ids),
        "domain_candidate_token_counts": {
            domain: len(token_ids)
            for domain, token_ids in state.domain_candidate_token_ids
        },
        "domain_candidate_group_counts": {
            domain: {
                group: len(token_ids) for group, token_ids in groups.items()
            }
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
                "occurrence_count": int(
                    applied_token_occurrence_counts.get(domain, 0)
                ),
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
                "eligible_token_count": result.eligible_token_count,
                "eligible_selection_score_distribution": (
                    _score_distribution_record(
                        result.eligible_score_distribution
                    )
                ),
                "selected_selection_score_distribution": (
                    _score_distribution_record(
                        result.selected_score_distribution
                    )
                ),
                "selected_tokens": [
                    {
                        "token_id": item.token_id,
                        "occurrence_count": item.occurrence_count,
                        "mean_occurrences_per_step": (item.mean_occurrences_per_step),
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
