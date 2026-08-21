"""Ranking signals for online Control-token selection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


TOP_LOSS_SELECTION_MODE = "top_loss"
TOP_SPEED_SELECTION_MODE = "top_speed"
TOP_KL_STUDENT_ENTROPY_SELECTION_MODE = "top_kl_student_entropy"
TOP_TEACHER_CONFIDENCE_STUDENT_ENTROPY_SELECTION_MODE = (
    "top_teacher_confidence_student_entropy"
)
PAIRED_SIGNAL_SELECTION_MODES = frozenset(
    {
        TOP_KL_STUDENT_ENTROPY_SELECTION_MODE,
        TOP_TEACHER_CONFIDENCE_STUDENT_ENTROPY_SELECTION_MODE,
    }
)
ONLINE_CONTROL_SELECTION_MODES = frozenset(
    {
        TOP_LOSS_SELECTION_MODE,
        TOP_SPEED_SELECTION_MODE,
        *PAIRED_SIGNAL_SELECTION_MODES,
    }
)

FIXED_ONLINE_WEIGHT_MODE = "fixed"
PAIRED_ONLINE_WEIGHT_MODE = "paired"
ONLINE_CONTROL_WEIGHT_MODES = frozenset(
    {FIXED_ONLINE_WEIGHT_MODE, PAIRED_ONLINE_WEIGHT_MODE}
)


@dataclass(frozen=True)
class OptimizationSpeed:
    """Occurrence-weighted loss slope over distinct optimizer steps."""

    value: float
    observed_step_count: int


def normalize_selection_mode(value: object) -> str:
    """Return one supported online Control-token selection mode."""

    mode = str(value).strip().lower()
    if mode not in ONLINE_CONTROL_SELECTION_MODES:
        allowed = ", ".join(sorted(ONLINE_CONTROL_SELECTION_MODES))
        raise ValueError(
            "Online Control selection mode must be one of: " f"{allowed}."
        )
    return mode


def normalize_online_weight_mode(value: object) -> str:
    """Return one supported online Control-token weighting mode."""

    mode = str(value).strip().lower()
    if mode not in ONLINE_CONTROL_WEIGHT_MODES:
        allowed = ", ".join(sorted(ONLINE_CONTROL_WEIGHT_MODES))
        raise ValueError(
            "Online Control weight mode must be one of: " f"{allowed}."
        )
    return mode


def _masked_sequence_max_normalize(
    values: "torch.Tensor",
    mask: "torch.Tensor",
    *,
    name: str,
    absolute: bool,
) -> "torch.Tensor":
    """Normalize each response by its valid-position maximum."""

    import torch

    if values.ndim != 2 or values.shape != mask.shape:
        raise ValueError(
            f"{name} must be a rank-2 matrix matching response_mask, got "
            f"{tuple(values.shape)} and {tuple(mask.shape)}."
        )
    valid = mask.detach().to(device=values.device, dtype=torch.bool)
    detached = values.detach().float()
    if not torch.isfinite(detached[valid]).all():
        raise ValueError(f"{name} must be finite on valid response positions.")
    non_negative = detached.abs() if absolute else detached
    if not absolute and bool((non_negative[valid] < 0.0).any()):
        raise ValueError(f"{name} must be non-negative on valid positions.")
    masked = torch.where(valid, non_negative, torch.zeros_like(non_negative))
    maximum = masked.amax(dim=-1, keepdim=True)
    normalized = masked / maximum.clamp(min=1e-12)
    return torch.where(valid, normalized, torch.zeros_like(normalized))


def paired_selection_bonus(
    *,
    selection_mode: str,
    student_entropy: "torch.Tensor",
    response_mask: "torch.Tensor",
    configured_loss: "torch.Tensor | None" = None,
    teacher_entropy: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Return detached ``A + B + A*B`` scores for paired selectors."""

    import torch

    mode = normalize_selection_mode(selection_mode)
    if mode not in PAIRED_SIGNAL_SELECTION_MODES:
        raise ValueError(
            "Paired selection bonus requires a paired-signal selection mode."
        )
    valid = response_mask.detach().to(
        device=student_entropy.device,
        dtype=torch.bool,
    )
    student = _masked_sequence_max_normalize(
        student_entropy,
        valid,
        name="student_entropy",
        absolute=False,
    )
    if mode == TOP_KL_STUDENT_ENTROPY_SELECTION_MODE:
        if configured_loss is None:
            raise ValueError("KL + entropy selection requires configured_loss.")
        primary = _masked_sequence_max_normalize(
            configured_loss,
            valid,
            name="configured_loss",
            absolute=True,
        )
    else:
        if teacher_entropy is None:
            raise ValueError(
                "Teacher-confidence + entropy selection requires teacher_entropy."
            )
        normalized_teacher_entropy = _masked_sequence_max_normalize(
            teacher_entropy,
            valid,
            name="teacher_entropy",
            absolute=False,
        )
        primary = torch.where(
            valid,
            1.0 - normalized_teacher_entropy,
            torch.zeros_like(normalized_teacher_entropy),
        )
    return ((primary + student + primary * student) * valid).detach()


def occurrence_weighted_optimization_speed(
    observations: Sequence[tuple[int, float, int]],
) -> OptimizationSpeed | None:
    """Return negative weighted loss slope, or ``None`` when undefined."""

    valid = tuple(
        (int(step), float(mean_abs_loss), int(count))
        for step, mean_abs_loss, count in observations
        if count > 0 and math.isfinite(mean_abs_loss)
    )
    if len(valid) < 2 or len({step for step, _, _ in valid}) < 2:
        return None

    total_weight = float(sum(count for _, _, count in valid))
    mean_step = sum(step * count for step, _, count in valid) / total_weight
    mean_loss = (
        sum(mean_abs_loss * count for _, mean_abs_loss, count in valid)
        / total_weight
    )
    time_variance = sum(
        count * (step - mean_step) ** 2 for step, _, count in valid
    )
    if time_variance <= 1e-12:
        return None
    covariance = sum(
        count * (step - mean_step) * (mean_abs_loss - mean_loss)
        for step, mean_abs_loss, count in valid
    )
    speed = -covariance / time_variance
    if not math.isfinite(speed):
        return None
    return OptimizationSpeed(
        value=float(speed),
        observed_step_count=len(valid),
    )
