"""Ranking signals for online Control-token selection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


TOP_LOSS_SELECTION_MODE = "top_loss"
TOP_SPEED_SELECTION_MODE = "top_speed"
ONLINE_CONTROL_SELECTION_MODES = frozenset(
    {TOP_LOSS_SELECTION_MODE, TOP_SPEED_SELECTION_MODE}
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
