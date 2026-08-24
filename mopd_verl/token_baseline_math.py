"""Numerical helpers for detached OPD token-baseline scores."""

from __future__ import annotations

import math
from typing import Any

import torch


def finite_non_negative(value: Any, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative, got {parsed}.")
    return parsed


def checked_matrix(
    label: str,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    matrix = value.detach().float().to(device=mask.device)
    if matrix.shape != mask.shape:
        raise ValueError(
            f"{label} must match response_mask, got "
            f"{tuple(matrix.shape)} and {tuple(mask.shape)}."
        )
    if not torch.isfinite(matrix[mask.bool()]).all():
        raise ValueError(f"{label} contains non-finite values on valid tokens.")
    return torch.where(mask.bool(), matrix, torch.zeros_like(matrix))


def clip_masked_upper_quantile(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    quantile: float,
) -> torch.Tensor:
    valid = mask.bool()
    valid_values = value[valid]
    if valid_values.numel() == 0:
        return torch.zeros_like(value)
    upper = torch.quantile(valid_values.float(), float(quantile))
    return torch.where(valid, value.clamp(max=upper), torch.zeros_like(value))


def masked_batch_minmax(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask.bool()
    valid_values = value[valid]
    if valid_values.numel() == 0:
        return torch.zeros_like(value)
    batch_min = valid_values.amin()
    batch_max = valid_values.amax()
    denominator = (batch_max - batch_min).clamp(min=1e-12)
    normalized = (value - batch_min) / denominator
    return torch.where(valid, normalized, torch.zeros_like(normalized))


def normalize_by_batch_max(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask.bool()
    valid_values = value[valid]
    if valid_values.numel() == 0:
        return torch.zeros_like(value)
    batch_max = valid_values.amax().clamp(min=1e-12)
    normalized = value / batch_max
    return torch.where(valid, normalized, torch.zeros_like(normalized))


def normalize_per_sequence(
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    masked_weights = weights.detach().float() * mask
    valid_count = mask.sum(dim=-1, keepdim=True)
    weight_sum = masked_weights.sum(dim=-1, keepdim=True)
    scale = valid_count / weight_sum.clamp(min=1e-12)
    normalized = masked_weights * scale
    return torch.where(weight_sum > 0, normalized, torch.zeros_like(normalized))


def select_positions(
    *,
    score: torch.Tensor,
    mask: torch.Tensor,
    retention_ratio: float,
    selection_mode: str,
) -> torch.Tensor:
    selected = torch.zeros_like(mask)
    for row_index in range(int(mask.shape[0])):
        valid_indices = torch.nonzero(
            mask[row_index].bool(),
            as_tuple=False,
        ).squeeze(-1)
        valid_count = int(valid_indices.numel())
        if valid_count == 0:
            continue
        selected_count = max(1, int(math.floor(retention_ratio * valid_count)))
        selected_count = min(selected_count, valid_count)
        valid_scores = score[row_index].index_select(0, valid_indices).float()
        if selection_mode == "topk":
            local_indices = torch.argsort(
                valid_scores,
                descending=True,
                stable=True,
            )[:selected_count]
        elif selection_mode == "sample":
            probabilities = valid_scores.clamp(min=0.0)
            if float(probabilities.sum().item()) <= 0.0:
                probabilities = torch.ones_like(probabilities)
            local_indices = torch.multinomial(
                probabilities,
                num_samples=selected_count,
                replacement=False,
            )
        else:
            raise AssertionError(f"Unhandled token selection mode: {selection_mode}")
        selected[row_index, valid_indices.index_select(0, local_indices)] = 1.0
    return selected
