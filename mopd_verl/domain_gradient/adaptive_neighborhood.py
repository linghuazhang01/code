"""Same-batch per-token adaptive weighting around active Control tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class PerTokenAdaptiveNeighborhoodSpec:
    """Immutable production configuration for one per-token adaptive gate."""

    domains: tuple[str, ...]
    domain_token_ids: tuple[tuple[str, tuple[int, ...]], ...]
    max_distance: int
    epsilon: float
    clip_max: float
    threshold: float
    min_far_tokens: int
    control_weight: float
    normalize_per_response: bool

    def token_id_map(self) -> dict[str, tuple[int, ...]]:
        return dict(self.domain_token_ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": True,
            "decision_unit": "individual_center_neighbor_pair",
            "timing": "same_batch_detached_loss",
            "max_distance": self.max_distance,
            "epsilon": self.epsilon,
            "clip_max": self.clip_max,
            "relative_loss_threshold": self.threshold,
            "far_baseline": "response_local_lower_median_outside_neighborhood",
            "min_far_tokens": self.min_far_tokens,
            "overlap": "max",
            "neighbor_weight": "relative_score",
            "control_weight": self.control_weight,
            "normalize_per_response": self.normalize_per_response,
        }


@dataclass(frozen=True)
class PerTokenAdaptiveNeighborhoodResult:
    """Per-token proposal, multiplier, and reconstruction diagnostics."""

    multiplier: torch.Tensor
    raw_multiplier: torch.Tensor
    relative_scores: torch.Tensor
    center_mask: torch.Tensor
    candidate_neighbor_mask: torch.Tensor
    eligible_neighbor_mask: torch.Tensor
    selected_neighbor_mask: torch.Tensor
    far_baselines: torch.Tensor
    far_token_counts: torch.Tensor
    center_denominators: torch.Tensor
    valid_denominator_mask: torch.Tensor


def _shift_from_source(
    value: torch.Tensor,
    offset: int,
    *,
    fill: float | bool,
) -> torch.Tensor:
    """Move each source position to its signed target offset."""

    shifted = torch.full_like(value, fill)
    if offset > 0:
        shifted[..., offset:] = value[..., :-offset]
    elif offset < 0:
        shifted[..., :offset] = value[..., -offset:]
    else:
        shifted = value.clone()
    return shifted


def _control_mask(
    token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    labels: Sequence[str],
    spec: PerTokenAdaptiveNeighborhoodSpec,
) -> torch.Tensor:
    if len(labels) != int(token_ids.shape[0]):
        raise ValueError("Adaptive labels must align with response rows.")
    token_id_map = spec.token_id_map()
    if set(token_id_map) != set(spec.domains):
        raise ValueError("Adaptive domain token IDs must exactly match domains.")
    rows: list[torch.Tensor] = []
    for row_index, raw_label in enumerate(labels):
        domain = str(raw_label)
        if domain not in token_id_map:
            raise ValueError(f"Unknown adaptive-neighborhood domain: {domain!r}.")
        selected = torch.as_tensor(
            token_id_map[domain],
            device=token_ids.device,
            dtype=token_ids.dtype,
        )
        rows.append(torch.isin(token_ids[row_index], selected))
    return torch.stack(rows, dim=0) & response_mask.bool()


def _local_far_baseline(
    loss: torch.Tensor,
    valid_loss: torch.Tensor,
    center_mask: torch.Tensor,
    *,
    max_distance: int,
    min_far_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    near = center_mask.clone()
    for distance in range(1, max_distance + 1):
        near |= _shift_from_source(center_mask, distance, fill=False)
        near |= _shift_from_source(center_mask, -distance, fill=False)
    far_mask = valid_loss & ~near
    far_counts = far_mask.sum(dim=-1)
    sortable = torch.where(far_mask, loss, torch.full_like(loss, math.inf))
    sorted_loss = sortable.sort(dim=-1).values
    lower_median_index = ((far_counts - 1).clamp(min=0) // 2).unsqueeze(-1)
    baseline = sorted_loss.gather(-1, lower_median_index).squeeze(-1)
    baseline_valid = far_counts.ge(min_far_tokens) & torch.isfinite(baseline)
    baseline = torch.where(
        baseline_valid,
        baseline,
        torch.full_like(baseline, math.nan),
    )
    return baseline, far_counts, baseline_valid


def build_per_token_adaptive_neighborhood(
    configured_loss: torch.Tensor,
    configured_loss_mask: torch.Tensor,
    response_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    labels: Sequence[str],
    spec: PerTokenAdaptiveNeighborhoodSpec,
) -> PerTokenAdaptiveNeighborhoodResult:
    """Build same-batch multipliers from individual center-neighbor ratios."""

    shapes = {
        configured_loss.shape,
        configured_loss_mask.shape,
        response_token_ids.shape,
        response_mask.shape,
    }
    if len(shapes) != 1 or configured_loss.ndim != 2:
        raise ValueError("Adaptive loss, IDs, and masks must align as [batch, seq].")
    loss = configured_loss.detach().float()
    response_valid = response_mask.to(device=loss.device, dtype=torch.bool)
    valid_loss = (
        configured_loss_mask.to(device=loss.device, dtype=torch.bool)
        & response_valid
        & torch.isfinite(loss)
    )
    token_ids = response_token_ids.to(device=loss.device, dtype=torch.long)
    centers = _control_mask(token_ids, response_valid, labels, spec)
    valid_centers = centers & valid_loss
    far, far_counts, far_valid = _local_far_baseline(
        loss,
        valid_loss,
        centers,
        max_distance=spec.max_distance,
        min_far_tokens=spec.min_far_tokens,
    )
    denominators = loss - far.unsqueeze(-1) + spec.epsilon
    valid_denominator = (
        valid_centers
        & far_valid.unsqueeze(-1)
        & torch.isfinite(denominators)
        & denominators.ne(0.0)
    )
    scores = torch.zeros_like(loss)
    candidates = torch.zeros_like(valid_loss)
    eligible = torch.zeros_like(valid_loss)
    numerator = loss - far.unsqueeze(-1)
    for offset in range(-spec.max_distance, spec.max_distance + 1):
        if offset == 0:
            continue
        target_candidate = (
            _shift_from_source(valid_centers, offset, fill=False) & valid_loss
        )
        target_eligible = (
            _shift_from_source(valid_denominator, offset, fill=False) & valid_loss
        )
        target_denominator = _shift_from_source(denominators, offset, fill=math.nan)
        pair_score = torch.where(
            target_eligible,
            (numerator / target_denominator).clamp(min=0.0, max=spec.clip_max),
            torch.zeros_like(loss),
        )
        candidates |= target_candidate
        eligible |= target_eligible
        scores = torch.maximum(scores, pair_score)
    candidates &= ~centers
    eligible &= ~centers
    selected = eligible & scores.gt(0.0) & scores.ge(spec.threshold)
    kernel = torch.where(selected, scores, torch.zeros_like(scores))
    kernel = torch.where(centers, torch.ones_like(kernel), kernel)
    raw_multiplier = 1.0 + (spec.control_weight - 1.0) * kernel
    multiplier = raw_multiplier
    if spec.normalize_per_response:
        denominator = valid_loss.sum(dim=-1, keepdim=True).clamp(min=1)
        mean = (
            torch.where(valid_loss, raw_multiplier, 0.0).sum(
                dim=-1,
                keepdim=True,
            )
            / denominator
        )
        multiplier = torch.where(valid_loss, raw_multiplier / mean, 1.0)
    scores = torch.where(centers, torch.ones_like(scores), scores)
    return PerTokenAdaptiveNeighborhoodResult(
        multiplier=multiplier.detach(),
        raw_multiplier=raw_multiplier.detach(),
        relative_scores=scores.detach(),
        center_mask=centers.detach(),
        candidate_neighbor_mask=candidates.detach(),
        eligible_neighbor_mask=eligible.detach(),
        selected_neighbor_mask=selected.detach(),
        far_baselines=far.detach(),
        far_token_counts=far_counts.detach(),
        center_denominators=denominators.detach(),
        valid_denominator_mask=valid_denominator.detach(),
    )
