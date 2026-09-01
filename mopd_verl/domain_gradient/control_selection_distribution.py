"""Finite score-distribution summaries for online Control selection."""

from __future__ import annotations

import math
from collections.abc import Sequence

from mopd_verl.domain_gradient.control_selection_types import (
    SelectionScoreDistribution,
)


def _linear_quantile(sorted_values: Sequence[float], quantile: float) -> float:
    """Return the linearly interpolated quantile of sorted finite values."""

    if not sorted_values:
        raise ValueError("Selection-score quantiles require at least one value.")
    position = (len(sorted_values) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def selection_score_distribution(
    values: Sequence[float],
) -> SelectionScoreDistribution | None:
    """Summarize one domain's token-type ranking-score distribution."""

    if not values:
        return None
    ordered = tuple(sorted(float(value) for value in values))
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("Selection-score distributions require finite values.")
    mean = sum(ordered) / len(ordered)
    variance = sum((value - mean) ** 2 for value in ordered) / len(ordered)
    return SelectionScoreDistribution(
        count=len(ordered),
        mean=float(mean),
        std=math.sqrt(variance),
        minimum=ordered[0],
        p10=_linear_quantile(ordered, 0.10),
        p50=_linear_quantile(ordered, 0.50),
        p90=_linear_quantile(ordered, 0.90),
        maximum=ordered[-1],
    )
