"""Numerically safe math helpers for dynamic domain budgeting."""

from __future__ import annotations

import math
from collections.abc import Mapping

_FLOAT32_MIN_NORMAL = 2.0**-126


def normalize_domain_weights(values: Mapping[str, float]) -> dict[str, float]:
    """Normalize finite positive domain weights to a probability mapping."""

    numeric = {domain: float(value) for domain, value in values.items()}
    if any(not math.isfinite(value) or value <= 0.0 for value in numeric.values()):
        raise ValueError("Domain weights must be positive and finite.")
    total = sum(numeric.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Domain weights must have a positive finite sum.")
    return {domain: value / total for domain, value in numeric.items()}


def exponential_moving_average(
    previous: float | None,
    current: float,
    beta: float,
) -> float:
    """Return one EMA update, initializing directly from the first observation."""

    return current if previous is None else beta * previous + (1.0 - beta) * current


def apply_probability_floor(
    weights: Mapping[str, float],
    floor: float,
) -> dict[str, float]:
    """Project weights onto ``p_d >= floor`` without moving inactive domains.

    This active-set solution preserves the original distribution exactly when
    every domain already satisfies the floor. Only domains that violate the
    constraint are clamped; the remaining mass is redistributed proportionally.
    """

    base = normalize_domain_weights(weights)
    if floor == 0.0:
        return base
    if floor < 0.0 or floor * len(base) >= 1.0:
        raise ValueError("Probability floor must be in [0, 1 / domain_count).")

    clamped: set[str] = set()
    while True:
        free = [domain for domain in base if domain not in clamped]
        free_weight = sum(base[domain] for domain in free)
        remaining_mass = 1.0 - floor * len(clamped)
        scale = remaining_mass / free_weight
        newly_clamped = {domain for domain in free if scale * base[domain] < floor}
        if not newly_clamped:
            return {
                domain: floor if domain in clamped else scale * base[domain]
                for domain in base
            }
        clamped.update(newly_clamped)


def power_weighted_distribution(
    priors: Mapping[str, float],
    signals: Mapping[str, float],
    alpha: float,
) -> dict[str, float]:
    """Compute ``prior * signal**alpha`` using a stable log-space softmax."""

    if set(priors) != set(signals):
        raise ValueError("Priors and signals must cover the same domains.")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("Power exponent must be finite and non-negative.")
    log_weights = {
        domain: math.log(float(priors[domain]))
        + alpha * math.log(float(signals[domain]))
        for domain in priors
    }
    offset = max(log_weights.values())
    # Loss scales are transported as float32; keep every contribution
    # representable after that cast even under an extreme alpha.
    minimum_log_weight = math.log(_FLOAT32_MIN_NORMAL)
    return normalize_domain_weights(
        {
            domain: math.exp(max(value - offset, minimum_log_weight))
            for domain, value in log_weights.items()
        }
    )
