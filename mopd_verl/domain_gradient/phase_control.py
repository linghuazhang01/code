"""Simple domain-specific control-token phase weighting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PhaseGapObservation:
    """Occurrence-mean gaps for one domain at one optimizer step."""

    control_gap: float
    span_gap: float
    control_count: int
    span_count: int


@dataclass(frozen=True)
class PhaseControlState:
    """Checkpointable per-domain phase gates and short gap histories."""

    domains: tuple[str, ...]
    gates: tuple[float, ...]
    control_histories: tuple[tuple[float, ...], ...]
    span_histories: tuple[tuple[float, ...], ...]
    control_gap_ema: tuple[float, ...]
    span_gap_ema: tuple[float, ...]
    gap_update_counts: tuple[int, ...]
    last_updated_step: int | None = None

    def gate_map(self) -> dict[str, float]:
        return dict(zip(self.domains, self.gates, strict=True))

    def as_dict(self) -> dict[str, object]:
        return {
            "domains": self.domains,
            "gates": self.gates,
            "control_histories": self.control_histories,
            "span_histories": self.span_histories,
            "control_gap_ema": self.control_gap_ema,
            "span_gap_ema": self.span_gap_ema,
            "gap_update_counts": self.gap_update_counts,
            "last_updated_step": self.last_updated_step,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PhaseControlState:
        domains = tuple(str(item) for item in value.get("domains", ()))
        gates = tuple(float(item) for item in value.get("gates", ()))
        control_histories = tuple(
            tuple(float(item) for item in history)
            for history in value.get("control_histories", ())
        )
        span_histories = tuple(
            tuple(float(item) for item in history)
            for history in value.get("span_histories", ())
        )
        control_gap_ema = tuple(
            float(item) for item in value.get("control_gap_ema", ())
        )
        span_gap_ema = tuple(
            float(item) for item in value.get("span_gap_ema", ())
        )
        gap_update_counts = tuple(
            int(item) for item in value.get("gap_update_counts", ())
        )
        lengths = {
            len(domains),
            len(gates),
            len(control_histories),
            len(span_histories),
            len(control_gap_ema),
            len(span_gap_ema),
            len(gap_update_counts),
        }
        if len(lengths) != 1:
            raise ValueError("Serialized phase-control state has mismatched lengths.")
        raw_step = value.get("last_updated_step")
        return cls(
            domains=domains,
            gates=gates,
            control_histories=control_histories,
            span_histories=span_histories,
            control_gap_ema=control_gap_ema,
            span_gap_ema=span_gap_ema,
            gap_update_counts=gap_update_counts,
            last_updated_step=None if raw_step is None else int(raw_step),
        )


def initial_phase_control_state(
    domains: Sequence[str],
    *,
    initial_gate: float,
) -> PhaseControlState:
    """Create the phase controller before its first gap observation."""

    normalized_domains = tuple(str(domain) for domain in domains)
    return PhaseControlState(
        domains=normalized_domains,
        gates=tuple(float(initial_gate) for _ in normalized_domains),
        control_histories=tuple(() for _ in normalized_domains),
        span_histories=tuple(() for _ in normalized_domains),
        control_gap_ema=tuple(0.0 for _ in normalized_domains),
        span_gap_ema=tuple(0.0 for _ in normalized_domains),
        gap_update_counts=tuple(0 for _ in normalized_domains),
    )


def successor_span_scores(
    marker_mask: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    span_length: int,
    decay_tau: float,
) -> torch.Tensor:
    """Return max-decayed successor scores after marker occurrences."""

    scores = torch.zeros_like(marker_mask, dtype=torch.float32)
    marker_values = marker_mask.to(dtype=torch.float32)
    sequence_length = int(marker_mask.shape[-1])
    for distance in range(1, min(span_length, sequence_length - 1) + 1):
        decay = math.exp(-float(distance) / float(decay_tau))
        shifted = marker_values[..., :-distance] * decay
        scores[..., distance:] = torch.maximum(
            scores[..., distance:],
            shifted,
        )
    return scores * response_mask.to(dtype=torch.float32)


def domain_control_masks(
    token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    labels: Sequence[str],
    domain_token_ids: Mapping[str, Sequence[int] | torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build marker masks plus a row-domain mask for configured domains."""

    marker_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    configured_rows = torch.zeros_like(response_mask, dtype=torch.bool)
    for row_index, label in enumerate(labels):
        selected_ids = domain_token_ids.get(str(label), ())
        if isinstance(selected_ids, torch.Tensor):
            selected = selected_ids.to(
                device=token_ids.device,
                dtype=token_ids.dtype,
            )
        else:
            selected = torch.tensor(
                tuple(selected_ids),
                device=token_ids.device,
                dtype=token_ids.dtype,
            )
        if selected.numel() == 0:
            continue
        configured_rows[row_index] = response_mask[row_index].bool()
        marker_mask[row_index] = torch.isin(
            token_ids[row_index],
            selected,
        ) & response_mask[row_index].bool()
    return marker_mask, configured_rows


def phase_token_weights(
    token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    labels: Sequence[str],
    *,
    domain_token_ids: Mapping[str, Sequence[int] | torch.Tensor],
    control_weight: float,
    phase_enabled: bool,
    span_enabled: bool,
    phase_gates: Mapping[str, float],
    span_length: int,
    span_decay_tau: float,
    normalize_per_domain: bool,
) -> torch.Tensor:
    """Build fixed domain-token or phase-shifted marker/span weights."""

    marker_mask, _ = domain_control_masks(
        token_ids,
        response_mask,
        labels,
        domain_token_ids,
    )
    if phase_enabled:
        gate_rows = torch.tensor(
            [float(phase_gates.get(str(label), 1.0)) for label in labels],
            device=token_ids.device,
            dtype=torch.float32,
        ).unsqueeze(-1)
        targeting = marker_mask.float() * gate_rows
        if span_enabled:
            span_scores = successor_span_scores(
                marker_mask,
                response_mask,
                span_length=span_length,
                decay_tau=span_decay_tau,
            )
            targeting = torch.maximum(
                targeting,
                span_scores * (1.0 - gate_rows),
            )
    else:
        targeting = marker_mask.float()

    weights = 1.0 + (float(control_weight) - 1.0) * targeting
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
        count = int(domain_valid.sum().item())
        if count == 0:
            continue
        mean_weight = weights[domain_valid].mean()
        weights = torch.where(
            domain_valid,
            weights / mean_weight.clamp(min=1e-12),
            weights,
        )
    return weights


def gap_observations(
    token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    labels: Sequence[str],
    absolute_gap: torch.Tensor,
    *,
    domain_token_ids: Mapping[str, Sequence[int] | torch.Tensor],
    span_length: int,
    span_decay_tau: float,
) -> dict[str, tuple[float, float, int, int]]:
    """Return additive control/span gap totals for distributed reduction."""

    marker_mask, _ = domain_control_masks(
        token_ids,
        response_mask,
        labels,
        domain_token_ids,
    )
    span_mask = successor_span_scores(
        marker_mask,
        response_mask,
        span_length=span_length,
        decay_tau=span_decay_tau,
    ).gt(0) & ~marker_mask
    totals: dict[str, tuple[float, float, int, int]] = {}
    for domain in domain_token_ids:
        domain_rows = torch.tensor(
            [str(label) == domain for label in labels],
            device=token_ids.device,
            dtype=torch.bool,
        ).unsqueeze(-1)
        control = marker_mask & domain_rows
        span = span_mask & domain_rows
        totals[domain] = (
            float(absolute_gap[control].sum().item()),
            float(absolute_gap[span].sum().item()),
            int(control.sum().item()),
            int(span.sum().item()),
        )
    return totals


def update_phase_control_state(
    state: PhaseControlState,
    observations: Mapping[str, PhaseGapObservation],
    *,
    window_steps: int,
    ema_beta: float,
    temperature: float,
    step: int,
) -> PhaseControlState:
    """Update independent smooth phase gates from recent residual gaps."""

    if state.last_updated_step == step:
        return state

    gates = list(state.gates)
    control_histories = [list(values) for values in state.control_histories]
    span_histories = [list(values) for values in state.span_histories]
    control_emas = list(state.control_gap_ema)
    span_emas = list(state.span_gap_ema)
    update_counts = list(state.gap_update_counts)
    history_limit = int(window_steps)

    for index, domain in enumerate(state.domains):
        observation = observations.get(domain)
        if (
            observation is None
            or observation.control_count <= 0
            or observation.span_count <= 0
        ):
            continue
        control_histories[index].append(float(observation.control_gap))
        span_histories[index].append(float(observation.span_gap))
        control_histories[index] = control_histories[index][-history_limit:]
        span_histories[index] = span_histories[index][-history_limit:]
        if len(control_histories[index]) < history_limit:
            continue

        control_gap = math.fsum(control_histories[index]) / history_limit
        span_gap = math.fsum(span_histories[index]) / history_limit
        if update_counts[index] == 0:
            control_emas[index] = control_gap
            span_emas[index] = span_gap
        else:
            control_emas[index] = (
                ema_beta * control_emas[index]
                + (1.0 - ema_beta) * control_gap
            )
            span_emas[index] = (
                ema_beta * span_emas[index]
                + (1.0 - ema_beta) * span_gap
            )
        target = 1.0 / (
            1.0
            + math.exp(
                -max(
                    -60.0,
                    min(
                        60.0,
                        (control_emas[index] - span_emas[index])
                        / temperature,
                    ),
                )
            )
        )
        gates[index] = target
        update_counts[index] += 1

    return PhaseControlState(
        domains=state.domains,
        gates=tuple(gates),
        control_histories=tuple(tuple(values) for values in control_histories),
        span_histories=tuple(tuple(values) for values in span_histories),
        control_gap_ema=tuple(control_emas),
        span_gap_ema=tuple(span_emas),
        gap_update_counts=tuple(update_counts),
        last_updated_step=int(step),
    )
