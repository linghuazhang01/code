"""Domain-specific Control-token weights driven by gap-reduction speed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from mopd_verl.domain_gradient.phase_control import domain_control_masks


SpeedWeightKnot = tuple[float, float]


@dataclass(frozen=True)
class ControlGapObservation:
    """Control gap and applied-weight statistics for one domain and step."""

    gap: float
    count: int
    applied_normalized_weight: float


@dataclass(frozen=True)
class ControlSpeedState:
    """Checkpointable state for the per-domain speed controller."""

    domains: tuple[str, ...]
    weights: tuple[float, ...]
    gap_emas: tuple[float, ...]
    gap_histories: tuple[tuple[tuple[int, float], ...], ...]
    observation_counts: tuple[int, ...]
    speeds: tuple[float, ...]
    reference_steps: tuple[int | None, ...]
    last_weight_update_steps: tuple[int | None, ...]
    weight_knots: tuple[SpeedWeightKnot, ...]
    last_observed_step: int | None = None

    def weight_map(self) -> dict[str, float]:
        """Return the raw Control weight currently assigned to each domain."""

        return dict(zip(self.domains, self.weights, strict=True))

    def as_dict(self) -> dict[str, object]:
        """Return a serialization-safe checkpoint payload."""

        return {
            "domains": self.domains,
            "weights": self.weights,
            "gap_emas": self.gap_emas,
            "gap_histories": self.gap_histories,
            "observation_counts": self.observation_counts,
            "speeds": self.speeds,
            "reference_steps": self.reference_steps,
            "last_weight_update_steps": self.last_weight_update_steps,
            "weight_knots": self.weight_knots,
            "last_observed_step": self.last_observed_step,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ControlSpeedState":
        """Restore controller state from an optimizer checkpoint mapping."""

        domains = tuple(str(item) for item in value.get("domains", ()))
        weights = tuple(float(item) for item in value.get("weights", ()))
        gap_emas = tuple(float(item) for item in value.get("gap_emas", ()))
        histories = tuple(
            tuple((int(step), float(gap)) for step, gap in history)
            for history in value.get("gap_histories", ())
        )
        observation_counts = tuple(
            int(item) for item in value.get("observation_counts", ())
        )
        speeds = tuple(float(item) for item in value.get("speeds", ()))
        reference_steps = tuple(
            None if item is None else int(item)
            for item in value.get("reference_steps", ())
        )
        update_steps = tuple(
            None if item is None else int(item)
            for item in value.get("last_weight_update_steps", ())
        )
        knots = tuple(
            (float(speed), float(weight))
            for speed, weight in value.get("weight_knots", ())
        )
        lengths = {
            len(domains),
            len(weights),
            len(gap_emas),
            len(histories),
            len(observation_counts),
            len(speeds),
            len(reference_steps),
            len(update_steps),
        }
        if len(lengths) != 1:
            raise ValueError(
                "Serialized Control-speed state has mismatched domain lengths."
            )
        raw_step = value.get("last_observed_step")
        return cls(
            domains=domains,
            weights=weights,
            gap_emas=gap_emas,
            gap_histories=histories,
            observation_counts=observation_counts,
            speeds=speeds,
            reference_steps=reference_steps,
            last_weight_update_steps=update_steps,
            weight_knots=knots,
            last_observed_step=None if raw_step is None else int(raw_step),
        )


def initial_control_speed_state(
    domains: Sequence[str],
    *,
    initial_weight: float,
    weight_knots: Sequence[SpeedWeightKnot],
) -> ControlSpeedState:
    """Create independent controller state for every configured domain."""

    normalized_domains = tuple(str(domain) for domain in domains)
    return ControlSpeedState(
        domains=normalized_domains,
        weights=tuple(float(initial_weight) for _ in normalized_domains),
        gap_emas=tuple(0.0 for _ in normalized_domains),
        gap_histories=tuple(() for _ in normalized_domains),
        observation_counts=tuple(0 for _ in normalized_domains),
        speeds=tuple(0.0 for _ in normalized_domains),
        reference_steps=tuple(None for _ in normalized_domains),
        last_weight_update_steps=tuple(None for _ in normalized_domains),
        weight_knots=tuple(
            (float(speed), float(weight)) for speed, weight in weight_knots
        ),
    )


def piecewise_linear_weight(
    speed: float,
    knots: Sequence[SpeedWeightKnot],
) -> float:
    """Map speed to weight by clamped piecewise-linear interpolation."""

    normalized = tuple(
        (float(knot_speed), float(weight))
        for knot_speed, weight in knots
    )
    if not normalized:
        raise ValueError("At least one speed-weight knot is required.")
    if speed <= normalized[0][0]:
        return normalized[0][1]
    if speed >= normalized[-1][0]:
        return normalized[-1][1]
    for (left_speed, left_weight), (right_speed, right_weight) in zip(
        normalized,
        normalized[1:],
    ):
        if speed > right_speed:
            continue
        fraction = (float(speed) - left_speed) / (right_speed - left_speed)
        return left_weight + fraction * (right_weight - left_weight)
    raise RuntimeError("Speed-weight interpolation did not find a segment.")


def domain_control_token_weights(
    token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    labels: Sequence[str],
    *,
    domain_token_ids: Mapping[str, Sequence[int] | torch.Tensor],
    domain_weights: Mapping[str, float],
    normalize_per_domain: bool,
) -> torch.Tensor:
    """Build domain-specific Control weights with optional mean-one scaling."""

    marker_mask, _ = domain_control_masks(
        token_ids,
        response_mask,
        labels,
        domain_token_ids,
    )
    row_weights = torch.tensor(
        [float(domain_weights.get(str(label), 1.0)) for label in labels],
        device=token_ids.device,
        dtype=torch.float32,
    ).unsqueeze(-1)
    weights = torch.where(marker_mask, row_weights, 1.0)
    if not normalize_per_domain:
        return weights

    valid = response_mask.bool()
    for domain in domain_token_ids:
        domain_rows = torch.tensor(
            [str(label) == domain for label in labels],
            device=token_ids.device,
            dtype=torch.bool,
        ).unsqueeze(-1)
        domain_valid = domain_rows & valid
        if not bool(domain_valid.any()):
            continue
        mean_weight = weights[domain_valid].mean()
        if float(mean_weight.item()) <= 1e-12:
            weights = torch.where(domain_valid, 1.0, weights)
            continue
        weights = torch.where(domain_valid, weights / mean_weight, weights)
    return weights


def control_gap_and_weight_totals(
    token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    labels: Sequence[str],
    absolute_gap: torch.Tensor,
    *,
    domain_token_ids: Mapping[str, Sequence[int] | torch.Tensor],
    applied_domain_weights: Mapping[str, float],
    normalize_per_domain: bool,
) -> dict[str, tuple[float, int, float]]:
    """Return additive gap/count/effective-weight totals for reduction."""

    marker_mask, _ = domain_control_masks(
        token_ids,
        response_mask,
        labels,
        domain_token_ids,
    )
    applied_weights = domain_control_token_weights(
        token_ids,
        response_mask,
        labels,
        domain_token_ids=domain_token_ids,
        domain_weights=applied_domain_weights,
        normalize_per_domain=normalize_per_domain,
    )
    totals: dict[str, tuple[float, int, float]] = {}
    for domain in domain_token_ids:
        domain_rows = torch.tensor(
            [str(label) == domain for label in labels],
            device=token_ids.device,
            dtype=torch.bool,
        ).unsqueeze(-1)
        selected = marker_mask & domain_rows
        totals[domain] = (
            float(absolute_gap[selected].sum().item()),
            int(selected.sum().item()),
            float(applied_weights[selected].sum().item()),
        )
    return totals


def update_control_speed_state(
    state: ControlSpeedState,
    observations: Mapping[str, ControlGapObservation],
    *,
    window_steps: int,
    ema_beta: float,
    update_interval_steps: int,
    minimum_occurrences: int,
    step: int,
) -> ControlSpeedState:
    """Update EMA/speed every step and raw weights at the configured cadence."""

    if state.last_observed_step is not None and step < state.last_observed_step:
        raise ValueError("Control-speed state cannot move backward in step.")
    if state.last_observed_step == step:
        return state

    weights = list(state.weights)
    gap_emas = list(state.gap_emas)
    histories = [list(history) for history in state.gap_histories]
    observation_counts = list(state.observation_counts)
    speeds = list(state.speeds)
    reference_steps = list(state.reference_steps)
    update_steps = list(state.last_weight_update_steps)
    history_limit = max(window_steps * 3, window_steps + update_interval_steps + 2)

    for index, domain in enumerate(state.domains):
        observation = observations.get(domain)
        if observation is None or observation.count < minimum_occurrences:
            continue
        raw_gap = float(observation.gap)
        if observation_counts[index] == 0:
            gap_ema = raw_gap
        else:
            gap_ema = (
                float(ema_beta) * gap_emas[index]
                + (1.0 - float(ema_beta)) * raw_gap
            )
        gap_emas[index] = gap_ema
        observation_counts[index] += 1
        histories[index].append((int(step), gap_ema))
        histories[index] = histories[index][-history_limit:]

        target_reference_step = int(step) - int(window_steps)
        reference = next(
            (
                (history_step, history_gap)
                for history_step, history_gap in histories[index]
                if history_step == target_reference_step
            ),
            None,
        )
        if reference is None:
            continue
        reference_step, reference_gap = reference
        speed = (reference_gap - gap_ema) / float(window_steps)
        speeds[index] = speed
        reference_steps[index] = reference_step
        last_update = update_steps[index]
        if (
            last_update is not None
            and int(step) - last_update < int(update_interval_steps)
        ):
            continue
        weights[index] = piecewise_linear_weight(speed, state.weight_knots)
        update_steps[index] = int(step)

    return ControlSpeedState(
        domains=state.domains,
        weights=tuple(weights),
        gap_emas=tuple(gap_emas),
        gap_histories=tuple(tuple(history) for history in histories),
        observation_counts=tuple(observation_counts),
        speeds=tuple(speeds),
        reference_steps=tuple(reference_steps),
        last_weight_update_steps=tuple(update_steps),
        weight_knots=state.weight_knots,
        last_observed_step=int(step),
    )
