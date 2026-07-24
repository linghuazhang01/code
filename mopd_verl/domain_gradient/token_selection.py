"""Deterministic domain-level token selection by absolute loss mass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RankedToken:
    """Globally unique token score used by distributed audit selection."""

    owner_rank: int
    owner_index: int
    loss_abs: float


def _descending_key(token: RankedToken) -> tuple[float, int, int]:
    return (-token.loss_abs, token.owner_rank, token.owner_index)


def _ascending_key(token: RankedToken) -> tuple[float, int, int]:
    return (token.loss_abs, token.owner_rank, token.owner_index)


def total_loss_abs_mass(tokens: Iterable[RankedToken]) -> float:
    """Return finite non-negative score mass."""

    return sum(max(float(token.loss_abs), 0.0) for token in tokens)


def select_top_k(
    tokens: Iterable[RankedToken],
    top_k: int,
) -> tuple[RankedToken, ...]:
    """Select the globally largest ``top_k`` absolute token losses."""

    if top_k <= 0:
        return tuple()
    return tuple(sorted(tokens, key=_descending_key)[:top_k])


def select_top_loss_mass(
    tokens: Iterable[RankedToken],
    fraction: float,
) -> tuple[RankedToken, ...]:
    """Select the smallest high-loss prefix reaching ``fraction`` mass."""

    ordered = tuple(sorted(tokens, key=_descending_key))
    return _select_until_mass(ordered, fraction, minimum_tokens=0)


def select_tail_loss_mass(
    tokens: Iterable[RankedToken],
    fraction: float,
    *,
    minimum_tokens: int,
) -> tuple[RankedToken, ...]:
    """Select low-loss tokens until their cumulative mass reaches a fraction."""

    ordered = tuple(sorted(tokens, key=_ascending_key))
    return _select_until_mass(
        ordered,
        fraction,
        minimum_tokens=max(0, minimum_tokens),
    )


def _select_until_mass(
    ordered: tuple[RankedToken, ...],
    fraction: float,
    *,
    minimum_tokens: int,
) -> tuple[RankedToken, ...]:
    if not ordered:
        return tuple()
    bounded_fraction = min(1.0, max(0.0, float(fraction)))
    minimum = min(max(0, minimum_tokens), len(ordered))
    if bounded_fraction >= 1.0:
        return ordered
    if bounded_fraction <= 0.0 and minimum == 0:
        return tuple()

    total_mass = total_loss_abs_mass(ordered)
    if total_mass <= 0.0:
        fallback_count = max(minimum, 1 if bounded_fraction > 0.0 else 0)
        return ordered[:fallback_count]

    target_mass = total_mass * bounded_fraction
    selected: list[RankedToken] = []
    selected_mass = 0.0
    for token in ordered:
        selected.append(token)
        selected_mass += max(float(token.loss_abs), 0.0)
        if len(selected) >= minimum and selected_mass >= target_mass:
            break
    return tuple(selected)
