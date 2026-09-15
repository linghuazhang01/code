"""Torch aggregation and logging for online Control-token selection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from mopd_verl.domain_gradient.control_full_vocabulary import (
    global_full_vocabulary_loss_statistics,
)
from mopd_verl.domain_gradient.control_selection_budget import (
    CONFIGURED_CANDIDATE_SCOPE,
    FULL_VOCABULARY_CANDIDATE_SCOPE,
    normalize_candidate_scope,
)
from mopd_verl.domain_gradient.control_selection_logging import (
    append_online_control_selection_jsonl,
)
from mopd_verl.domain_gradient.control_loss_teacher_confidence import (
    loss_teacher_confidence_type_scores,
)
from mopd_verl.domain_gradient.control_selection_scoring import (
    PAIRED_SIGNAL_SELECTION_MODES,
    TOP_LOSS_TEACHER_CONFIDENCE_SELECTION_MODE,
    TOP_LOSS_SELECTION_MODE,
    TOP_Q_LOSS_ENTROPY_SELECTION_MODE,
    TOP_SPEED_SELECTION_MODE,
    TOP_TEACHER_CONFIDENCE_STUDENT_ENTROPY_SELECTION_MODE,
    normalize_online_selection_mode_by_domain,
    normalize_selection_mode,
    paired_selection_bonus,
)


@dataclass(frozen=True)
class GlobalCandidateLossStatistics:
    """Globally reduced candidate scores and valid-token denominators."""

    by_domain: dict[str, dict[int, tuple[float, int]]]
    valid_token_counts: dict[str, int]
    valid_score_sums: dict[str, float]
    q_normalization_stats: dict[str, dict[str, float]] | None = None


@torch.no_grad()
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
    selection_mode_by_domain: (
        Mapping[str, str] | Sequence[tuple[str, str]] | None
    ) = None,
    selection_loss_batches_by_domain: (
        Mapping[str, Sequence[torch.Tensor]] | None
    ) = None,
    selection_loss_mask_batches_by_domain: (
        Mapping[str, Sequence[torch.Tensor]] | None
    ) = None,
    student_entropy_batches: Sequence[torch.Tensor] | None = None,
    teacher_entropy_batches: Sequence[torch.Tensor] | None = None,
    teacher_log_prob_batches: Sequence[torch.Tensor] | None = None,
    normalization_min_occurrences: float = 0.0,
    normalization_strict_occurrence_gate: bool = False,
    candidate_scope: str = CONFIGURED_CANDIDATE_SCOPE,
    candidate_vocab_size: int | None = None,
) -> GlobalCandidateLossStatistics:
    """Aggregate mixed per-domain selector scores in one all-reduce."""

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
    normalized_domains = tuple(str(domain) for domain in domains)
    effective_modes = dict(
        normalize_online_selection_mode_by_domain(
            normalized_domains,
            selection_mode_by_domain,
            mode,
        )
    )
    normalized_candidate_scope = normalize_candidate_scope(candidate_scope)
    if normalized_candidate_scope == FULL_VOCABULARY_CANDIDATE_SCOPE:
        if set(effective_modes.values()) - {
            TOP_LOSS_SELECTION_MODE,
            TOP_SPEED_SELECTION_MODE,
        }:
            raise ValueError(
                "Full-vocabulary online Control statistics support only "
                "top_loss and top_speed selection."
            )
        if candidate_token_ids or any(
            token_ids for token_ids in (domain_candidate_token_ids or {}).values()
        ):
            raise ValueError(
                "Full-vocabulary online Control statistics cannot use "
                "configured candidate IDs."
            )
        if selection_loss_batches_by_domain or selection_loss_mask_batches_by_domain:
            raise ValueError(
                "Full-vocabulary online Control statistics cannot use "
                "alternate selector-loss batches."
            )
        if candidate_vocab_size is None:
            raise ValueError(
                "Full-vocabulary online Control statistics require a tokenizer "
                "vocabulary size."
            )
        full_statistics = global_full_vocabulary_loss_statistics(
            token_id_batches,
            loss_batches,
            mask_batches,
            label_batches,
            domains=normalized_domains,
            vocab_size=int(candidate_vocab_size),
        )
        return GlobalCandidateLossStatistics(
            by_domain=full_statistics.by_domain,
            valid_token_counts=full_statistics.valid_token_counts,
            valid_score_sums=full_statistics.valid_score_sums,
        )
    paired_mode = any(
        mode_value in PAIRED_SIGNAL_SELECTION_MODES
        for mode_value in effective_modes.values()
    )
    q_mode = any(
        mode_value == TOP_Q_LOSS_ENTROPY_SELECTION_MODE
        for mode_value in effective_modes.values()
    )
    loss_teacher_confidence_mode = any(
        mode_value == TOP_LOSS_TEACHER_CONFIDENCE_SELECTION_MODE
        for mode_value in effective_modes.values()
    )
    if (paired_mode or q_mode) and (
        student_entropy_batches is None
        or len(student_entropy_batches) != len(token_id_batches)
    ):
        raise ValueError(
            "Paired online Control selection requires one aligned Student "
            "entropy matrix per token batch."
        )
    if any(
        mode_value == TOP_TEACHER_CONFIDENCE_STUDENT_ENTROPY_SELECTION_MODE
        for mode_value in effective_modes.values()
    ) and (
        teacher_entropy_batches is None
        or len(teacher_entropy_batches) != len(token_id_batches)
    ):
        raise ValueError(
            "Teacher-confidence online Control selection requires one aligned "
            "Teacher entropy matrix per token batch."
        )
    if loss_teacher_confidence_mode and (
        teacher_log_prob_batches is None
        or len(teacher_log_prob_batches) != len(token_id_batches)
    ):
        raise ValueError(
            "Loss + teacher-confidence selection requires one aligned teacher "
            "chosen-token log-probability matrix per token batch."
        )
    if (
        not math.isfinite(normalization_min_occurrences)
        or normalization_min_occurrences < 0.0
    ):
        raise ValueError(
            "Normalization minimum occurrences must be finite and non-negative."
        )
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

    # Keep the established Q wire format for scalar-Q callers.  Mixed-mode
    # batches use the packed path below so every domain still participates in
    # one collective without changing Q's per-domain denominators.
    if (
        effective_modes
        and all(
            mode_value == TOP_Q_LOSS_ENTROPY_SELECTION_MODE
            for mode_value in effective_modes.values()
        )
        and not selection_loss_batches_by_domain
        and not selection_loss_mask_batches_by_domain
    ):
        from mopd_verl.domain_gradient.control_q_statistics import (
            global_q_statistics,
        )

        q = global_q_statistics(
            token_id_batches,
            loss_batches,
            mask_batches,
            label_batches,
            domains=normalized_domains,
            domain_candidates=domain_candidates,
            student_entropy_batches=student_entropy_batches,
        )
        return GlobalCandidateLossStatistics(
            q.by_domain,
            q.valid_token_counts,
            q.valid_score_sums,
            q.normalization,
        )

    mode_values = set(effective_modes.values())
    legacy_ordinary = (
        bool(effective_modes)
        and len(mode_values) == 1
        and mode_values.isdisjoint(
            {
                *PAIRED_SIGNAL_SELECTION_MODES,
                TOP_Q_LOSS_ENTROPY_SELECTION_MODE,
                TOP_LOSS_TEACHER_CONFIDENCE_SELECTION_MODE,
            }
        )
        and not selection_loss_batches_by_domain
        and not selection_loss_mask_batches_by_domain
    )

    def _domain_batches(
        source_map: Mapping[str, Sequence[torch.Tensor]] | None,
        domain: str,
        fallback: Sequence[torch.Tensor],
        name: str,
    ) -> Sequence[torch.Tensor]:
        if source_map is None:
            return fallback
        unknown = set(str(key) for key in source_map) - set(normalized_domains)
        if unknown:
            raise ValueError(f"{name} contains unknown domains: {sorted(unknown)}.")
        return source_map.get(domain, fallback)

    if selection_loss_mask_batches_by_domain is not None:
        unknown_masks = set(
            str(key) for key in selection_loss_mask_batches_by_domain
        ) - set(normalized_domains)
        if unknown_masks:
            raise ValueError(
                "selection_loss_mask_batches_by_domain contains unknown domains: "
                f"{sorted(unknown_masks)}."
            )
    if selection_loss_batches_by_domain is not None:
        unknown_losses = {str(key) for key in selection_loss_batches_by_domain} - set(
            normalized_domains
        )
        if unknown_losses:
            raise ValueError(
                "selection_loss_batches_by_domain contains unknown domains: "
                f"{sorted(unknown_losses)}."
            )

    selection_losses = {
        domain: _domain_batches(
            selection_loss_batches_by_domain,
            domain,
            loss_batches,
            "selection_loss_batches_by_domain",
        )
        for domain in normalized_domains
    }
    selection_masks = {
        domain: _domain_batches(
            selection_loss_mask_batches_by_domain,
            domain,
            mask_batches,
            "selection_loss_mask_batches_by_domain",
        )
        for domain in normalized_domains
    }
    if any(
        len(selection_losses[domain]) != len(token_id_batches)
        or len(selection_masks[domain]) != len(token_id_batches)
        for domain in normalized_domains
    ):
        raise ValueError(
            "Online Control per-domain selector loss and mask batches must be "
            "aligned with token batches."
        )
    device = loss_batches[0].device
    channels = 2 if legacy_ordinary else 16
    packed = torch.zeros(
        (len(normalized_domains), channels, len(candidates) + 1),
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
    for batch_index, (token_ids, base_loss, base_mask, labels) in enumerate(
        zip(token_id_batches, loss_batches, mask_batches, label_batches, strict=True)
    ):
        if token_ids.shape != base_loss.shape or base_loss.shape != base_mask.shape:
            raise ValueError(
                "Online Control token IDs, configured loss, and mask must align."
            )
        if len(labels) != int(base_loss.shape[0]):
            raise ValueError("Online Control domain labels must align with batch rows.")
        ids = token_ids.to(device=device, dtype=torch.long)
        if legacy_ordinary:
            legacy_loss_values = base_loss.to(device=device, dtype=torch.float64)
            legacy_valid = base_mask.to(device=device, dtype=torch.bool)
            if not torch.isfinite(legacy_loss_values[legacy_valid]).all():
                raise ValueError("Online Control configured loss must be finite.")
        for domain_index, domain in enumerate(normalized_domains):
            mode_for_domain = effective_modes[domain]
            losses = selection_losses[domain][batch_index]
            mask = selection_masks[domain][batch_index]
            if losses.shape != mask.shape or losses.shape != token_ids.shape:
                raise ValueError(
                    "Online Control per-domain selector loss and mask must align "
                    "with token IDs."
                )
            valid = mask.to(device=device, dtype=torch.bool)
            loss_values = losses.to(device=device, dtype=torch.float64)
            rows = torch.tensor(
                [str(label) == domain for label in labels],
                device=device,
                dtype=torch.bool,
            ).unsqueeze(-1)
            selected_mask = valid & rows
            if mode_for_domain == TOP_LOSS_TEACHER_CONFIDENCE_SELECTION_MODE:
                if teacher_log_prob_batches is None:
                    raise RuntimeError(
                        "Validated loss + teacher-confidence selection is "
                        "missing teacher chosen-token log-probabilities."
                    )
                teacher_logp = teacher_log_prob_batches[batch_index].to(
                    device=device,
                    dtype=torch.float64,
                )
                if teacher_logp.shape != losses.shape:
                    raise ValueError(
                        "Teacher chosen-token log-probabilities must align with "
                        "the configured token loss."
                    )
                loss_values = loss_values.abs()
                lval = loss_values[selected_mask]
                zval = teacher_logp[selected_mask].clamp(max=0.0)
                if not torch.isfinite(lval).all() or not torch.isfinite(zval).all():
                    raise ValueError(
                        "Loss + teacher-confidence selection requires finite "
                        "loss and chosen-token log-probability values."
                    )
                valid_moments = torch.stack((lval, zval, torch.ones_like(lval)))
                packed[domain_index, 12:15, -1].add_(valid_moments.sum(dim=1))
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
                packed[domain_index, 2:5].scatter_add_(
                    1,
                    positions[matched].expand(3, -1),
                    valid_moments[:, matched],
                )
                continue
            if mode_for_domain == TOP_Q_LOSS_ENTROPY_SELECTION_MODE:
                if student_entropy_batches is None:
                    raise RuntimeError(
                        "Validated Q selection is missing Student entropy."
                    )
                entropy = student_entropy_batches[batch_index].to(device=device)
                if entropy.shape != losses.shape:
                    raise ValueError("Q selector loss and student entropy must align.")
                loss_values = loss_values.abs()
                entropy_values = entropy.to(dtype=torch.float64)
                lval = loss_values[selected_mask]
                hval = entropy_values[selected_mask]
                product = lval * hval
                good = (
                    torch.isfinite(lval)
                    & torch.isfinite(hval)
                    & (hval >= 0)
                    & torch.isfinite(product)
                )
                packed[domain_index, 6, -1].add_((~good).sum())
                packed[domain_index, 7, -1].add_((~torch.isfinite(lval)).sum())
                packed[domain_index, 8, -1].add_((~torch.isfinite(hval)).sum())
                packed[domain_index, 9, -1].add_(
                    (torch.isfinite(hval) & (hval < 0)).sum()
                )
                moments = torch.stack((lval, hval, product, torch.ones_like(lval)))
                moments = torch.where(
                    good.unsqueeze(0), moments, torch.zeros_like(moments)
                )
                packed[domain_index, 12:16, -1].add_(moments.sum(dim=1))
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
                packed[domain_index, 2:6].scatter_add_(
                    1,
                    positions[matched].expand(4, -1),
                    moments[:, matched],
                )
                continue

            if (
                not legacy_ordinary
                and not torch.isfinite(loss_values[selected_mask]).all()
            ):
                raise ValueError(
                    "Online Control selector loss must be finite on valid rows."
                )
            if mode_for_domain in PAIRED_SIGNAL_SELECTION_MODES:
                if student_entropy_batches is None:
                    raise RuntimeError(
                        "Validated paired selection is missing Student entropy."
                    )
                score_values = paired_selection_bonus(
                    selection_mode=mode_for_domain,
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
            if not torch.isfinite(score_values[selected_mask]).all():
                raise ValueError(
                    "Online Control selector score must be finite on valid rows."
                )
            score_sum_channel = 0 if legacy_ordinary else 10
            score_count_channel = 1 if legacy_ordinary else 11
            packed[domain_index, score_sum_channel, -1].add_(
                score_values[selected_mask].sum()
            )
            packed[domain_index, score_count_channel, -1].add_(selected_mask.sum())
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
    if q_mode and (
        not torch.isfinite(packed).all() or bool((packed[:, 6, -1] > 0).any())
    ):
        diagnostics = {
            domain: {
                "invalid_occurrences": float(packed[domain_index, 6, -1]),
                "nonfinite_loss": float(packed[domain_index, 7, -1]),
                "nonfinite_entropy": float(packed[domain_index, 8, -1]),
                "negative_finite_entropy": float(packed[domain_index, 9, -1]),
            }
            for domain_index, domain in enumerate(normalized_domains)
        }
        raise ValueError(
            "Q requires finite loss and non-negative finite student entropy and "
            f"moments. Global input counts: {diagnostics}; "
            f"nonfinite_reduced_entries={int((~torch.isfinite(packed)).sum())}."
        )

    q_normalization_stats: dict[str, dict[str, float]] = {}
    by_domain: dict[str, dict[int, tuple[float, int]]] = {}
    valid_token_counts: dict[str, int] = {}
    valid_score_sums: dict[str, float] = {}
    packed_cpu = packed.cpu()
    for domain_index, domain in enumerate(normalized_domains):
        if effective_modes[domain] == TOP_LOSS_TEACHER_CONFIDENCE_SELECTION_MODE:
            candidate_moments = packed_cpu[domain_index, 2:5, :-1]
            counts = candidate_moments[2]
            observed = counts > 0
            if normalization_strict_occurrence_gate:
                normalization_mask = counts > normalization_min_occurrences
            else:
                normalization_mask = counts >= normalization_min_occurrences
            normalization_mask &= observed
            scores, normalization = loss_teacher_confidence_type_scores(
                loss_abs_sums=candidate_moments[0],
                teacher_logp_sums=candidate_moments[1],
                counts=counts,
                normalization_mask=normalization_mask,
            )
            score_sums = scores * counts
            q_normalization_stats[domain] = normalization
            domain_candidate_set = set(domain_candidates[domain])
            by_domain[domain] = {
                token_id: (
                    float(score_sums[candidate_index]),
                    int(counts[candidate_index]),
                )
                for candidate_index, token_id in enumerate(candidates)
                if token_id in domain_candidate_set and observed[candidate_index]
            }
            valid_token_counts[domain] = int(packed_cpu[domain_index, 14, -1])
            valid_score_sums[domain] = float(score_sums.sum())
            continue
        if effective_modes[domain] == TOP_Q_LOSS_ENTROPY_SELECTION_MODE:
            valid_moments = packed_cpu[domain_index, 12:16, -1]
            valid_count = valid_moments[3]
            if valid_count <= 0:
                raise ValueError(
                    "Q requires non-empty all-valid occurrences in each Q domain."
                )
            mean_loss = valid_moments[0] / valid_count
            mean_entropy = valid_moments[1] / valid_count
            if (
                not torch.isfinite(torch.stack((mean_loss, mean_entropy))).all()
                or mean_loss <= 0
                or mean_entropy <= 0
            ):
                raise ValueError(
                    "Q requires positive finite all-valid mean loss and mean "
                    "entropy."
                )
            candidate_moments = packed_cpu[domain_index, 2:6, :-1]
            scores = (
                candidate_moments[0] / mean_loss
                + candidate_moments[1] / mean_entropy
                + candidate_moments[2] / mean_loss / mean_entropy
            )
            counts = candidate_moments[3]
            valid_score = (
                valid_moments[0] / mean_loss
                + valid_moments[1] / mean_entropy
                + valid_moments[2] / mean_loss / mean_entropy
            )
            if not torch.isfinite(scores).all() or not torch.isfinite(valid_score):
                raise ValueError("Q scores must remain finite after normalization.")
            q_normalization_stats[domain] = {
                "q_all_valid_mean_abs_loss": float(mean_loss),
                "q_all_valid_mean_student_entropy": float(mean_entropy),
                "q_all_valid_mean_score": float(valid_score / valid_count),
            }
            by_domain[domain] = {
                token_id: (float(scores[candidate_index]), int(counts[candidate_index]))
                for candidate_index, token_id in enumerate(candidates)
                if token_id in set(domain_candidates[domain])
                and counts[candidate_index] > 0
            }
            valid_token_counts[domain] = int(valid_count)
            valid_score_sums[domain] = float(valid_score)
            continue
        scores = packed_cpu[domain_index, 0, :-1]
        counts = packed_cpu[domain_index, 1, :-1]
        by_domain[domain] = {
            token_id: (float(scores[candidate_index]), int(counts[candidate_index]))
            for candidate_index, token_id in enumerate(candidates)
            if token_id in set(domain_candidates[domain])
            and counts[candidate_index] > 0
        }
        score_sum_channel = 0 if legacy_ordinary else 10
        score_count_channel = 1 if legacy_ordinary else 11
        valid_token_counts[domain] = int(
            packed_cpu[domain_index, score_count_channel, -1]
        )
        valid_score_sums[domain] = float(
            packed_cpu[domain_index, score_sum_channel, -1]
        )
    return GlobalCandidateLossStatistics(
        by_domain=by_domain,
        valid_token_counts=valid_token_counts,
        valid_score_sums=valid_score_sums,
        q_normalization_stats=(q_normalization_stats or None),
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
