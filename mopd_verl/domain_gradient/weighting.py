"""Pure helpers for gradient-norm-driven domain loss weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DomainWeightState:
    """Persistent immutable state for one actor's domain-weight controller."""

    domains: tuple[str, ...]
    ema_norms: tuple[float, ...]
    weights: tuple[float, ...]
    target_weights: tuple[float, ...] = ()
    update_count: int = 0
    last_updated_step: int | None = None

    def weight_map(self) -> dict[str, float]:
        return dict(zip(self.domains, self.weights, strict=True))

    def target_weight_map(self) -> dict[str, float]:
        target_weights = (
            self.target_weights
            if len(self.target_weights) == len(self.domains)
            else self.weights
        )
        return dict(zip(self.domains, target_weights, strict=True))

    def ema_norm_map(self) -> dict[str, float]:
        return dict(zip(self.domains, self.ema_norms, strict=True))

    def as_dict(self) -> dict[str, object]:
        target_weights = (
            self.target_weights
            if len(self.target_weights) == len(self.domains)
            else self.weights
        )
        return {
            "domains": self.domains,
            "ema_norms": self.ema_norms,
            "weights": self.weights,
            "target_weights": target_weights,
            "update_count": self.update_count,
            "last_updated_step": self.last_updated_step,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "DomainWeightState":
        domains = tuple(str(item) for item in value.get("domains", ()))
        ema_norms = tuple(float(item) for item in value.get("ema_norms", ()))
        weights = tuple(float(item) for item in value.get("weights", ()))
        target_weights = tuple(
            float(item) for item in value.get("target_weights", weights)
        )
        if len(domains) != len(ema_norms) or len(domains) != len(weights):
            raise ValueError(
                "Serialized domain weight state has mismatched lengths."
            )
        if len(domains) != len(target_weights):
            raise ValueError(
                "Serialized domain target weights have mismatched lengths."
            )
        raw_step = value.get("last_updated_step")
        return cls(
            domains=domains,
            ema_norms=ema_norms,
            weights=weights,
            target_weights=target_weights,
            update_count=int(value.get("update_count", 0)),
            last_updated_step=None if raw_step is None else int(raw_step),
        )


def initial_domain_weight_state(domains: tuple[str, ...]) -> DomainWeightState:
    """Create unit weights before the first gradient observation."""

    return DomainWeightState(
        domains=domains,
        ema_norms=tuple(0.0 for _ in domains),
        weights=tuple(1.0 for _ in domains),
        target_weights=tuple(1.0 for _ in domains),
    )


def _bounded_mean_one(
    values: tuple[float, ...],
    *,
    minimum: float,
    maximum: float,
) -> tuple[float, ...]:
    """Scale positive values to mean one while respecting hard bounds."""

    if not values:
        return tuple()
    if minimum > 1.0 or maximum < 1.0:
        raise ValueError("Domain weight bounds must contain 1.0.")

    def bounded_mean(scale: float) -> float:
        return sum(
            min(maximum, max(minimum, scale * value))
            for value in values
        ) / len(values)

    lower = 0.0
    upper = 1.0
    while bounded_mean(upper) < 1.0:
        upper *= 2.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if bounded_mean(midpoint) < 1.0:
            lower = midpoint
        else:
            upper = midpoint
    scale = (lower + upper) / 2.0
    return tuple(
        min(maximum, max(minimum, scale * value))
        for value in values
    )


def update_domain_weight_state(
    state: DomainWeightState,
    gradient_norms: Mapping[str, float],
    *,
    ema_beta: float,
    weight_ema_beta: float,
    alpha: float,
    minimum: float,
    maximum: float,
    step: int | None = None,
    epsilon: float = 1e-12,
) -> DomainWeightState:
    """EMA-smooth bounded inverse-norm targets from globally reduced norms."""

    if not 0.0 <= ema_beta < 1.0:
        raise ValueError("ema_beta must be in [0, 1).")
    if not 0.0 <= weight_ema_beta < 1.0:
        raise ValueError("weight_ema_beta must be in [0, 1).")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    if minimum <= 0.0 or maximum < minimum:
        raise ValueError("Domain weight bounds must be positive and ordered.")

    observed = tuple(
        max(0.0, float(gradient_norms.get(domain, 0.0)))
        for domain in state.domains
    )
    if state.update_count == 0:
        ema_norms = observed
    else:
        ema_norms = tuple(
            ema_beta * previous + (1.0 - ema_beta) * current
            for previous, current in zip(
                state.ema_norms,
                observed,
                strict=True,
            )
        )

    positive = tuple(value for value in ema_norms if value > epsilon)
    reference = sum(positive) / len(positive) if positive else 1.0
    raw_weights = tuple(
        (reference / max(value, epsilon)) ** alpha
        if value > epsilon
        else previous
        for value, previous in zip(ema_norms, state.weights, strict=True)
    )
    target_weights = _bounded_mean_one(
        raw_weights,
        minimum=minimum,
        maximum=maximum,
    )
    smoothed_weights = tuple(
        weight_ema_beta * previous
        + (1.0 - weight_ema_beta) * target
        for previous, target in zip(
            state.weights,
            target_weights,
            strict=True,
        )
    )
    weights = _bounded_mean_one(
        smoothed_weights,
        minimum=minimum,
        maximum=maximum,
    )
    return DomainWeightState(
        domains=state.domains,
        ema_norms=ema_norms,
        weights=weights,
        target_weights=target_weights,
        update_count=state.update_count + 1,
        last_updated_step=step,
    )
