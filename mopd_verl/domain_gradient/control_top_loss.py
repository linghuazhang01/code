"""Rolling online selection for Control-token weighting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from mopd_verl.domain_gradient.control_selection_scoring import (
    FIXED_ONLINE_WEIGHT_MODE,
    PAIRED_ONLINE_WEIGHT_MODE,
    PAIRED_SIGNAL_SELECTION_MODES,
    TOP_LOSS_SELECTION_MODE,
    TOP_SPEED_SELECTION_MODE,
    normalize_online_weight_mode,
    normalize_selection_mode,
    occurrence_weighted_optimization_speed,
)
from mopd_verl.domain_gradient.control_selection_types import (
    DomainSelectionResult,
    DomainStatistics,
    OnlineControlSelectionOutcome,
    SelectedControlToken,
    SelectionScoreDistribution,
    StepStatistics,
    TokenStatistic,
)


@dataclass(frozen=True)
class OnlineControlSelectionState:
    """Checkpointable rolling statistics and lagged active token IDs."""

    domains: tuple[str, ...]
    domain_candidate_token_ids: tuple[tuple[str, tuple[int, ...]], ...]
    audit_interval_steps: int
    window_steps: int
    min_mean_occurrences_per_step: float
    top_k: int
    selection_mode: str
    weight_mode: str
    history: tuple[StepStatistics, ...]
    active_token_ids: tuple[tuple[str, tuple[int, ...]], ...]
    active_token_weights: tuple[
        tuple[str, tuple[tuple[int, float], ...]], ...
    ]
    last_observed_step: int | None = None
    last_audit_step: int | None = None
    update_count: int = 0

    def active_map(self) -> dict[str, tuple[int, ...]]:
        return dict(self.active_token_ids)

    def candidate_map(self) -> dict[str, tuple[int, ...]]:
        return dict(self.domain_candidate_token_ids)

    def active_weight_map(self) -> dict[str, dict[int, float]]:
        return {
            domain: dict(token_weights)
            for domain, token_weights in self.active_token_weights
        }

    @property
    def candidate_token_ids(self) -> tuple[int, ...]:
        """Return the cross-domain union for legacy metrics and callers."""

        return tuple(
            sorted(
                {
                    token_id
                    for _, token_ids in self.domain_candidate_token_ids
                    for token_id in token_ids
                }
            )
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 4,
            "domains": self.domains,
            "domain_candidate_token_ids": self.domain_candidate_token_ids,
            "audit_interval_steps": self.audit_interval_steps,
            "window_steps": self.window_steps,
            "min_mean_occurrences_per_step": (self.min_mean_occurrences_per_step),
            "top_k": self.top_k,
            "selection_mode": self.selection_mode,
            "weight_mode": self.weight_mode,
            "history": self.history,
            "active_token_ids": self.active_token_ids,
            "active_token_weights": self.active_token_weights,
            "last_observed_step": self.last_observed_step,
            "last_audit_step": self.last_audit_step,
            "update_count": self.update_count,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> OnlineControlSelectionState:
        domains = tuple(str(item) for item in value.get("domains", ()))
        raw_domain_candidates = value.get("domain_candidate_token_ids")
        if raw_domain_candidates is None:
            legacy_candidates = tuple(
                int(item) for item in value.get("candidate_token_ids", ())
            )
            domain_candidates = tuple((domain, legacy_candidates) for domain in domains)
        else:
            domain_candidates = tuple(
                (
                    str(domain),
                    tuple(int(token_id) for token_id in token_ids),
                )
                for domain, token_ids in raw_domain_candidates
            )
        history = tuple(
            (
                int(step),
                tuple(
                    (
                        str(domain),
                        tuple(
                            (int(token_id), float(loss_sum), int(count))
                            for token_id, loss_sum, count in statistics
                        ),
                    )
                    for domain, statistics in domain_rows
                ),
            )
            for step, domain_rows in value.get("history", ())
        )
        active = tuple(
            (
                str(domain),
                tuple(int(token_id) for token_id in token_ids),
            )
            for domain, token_ids in value.get("active_token_ids", ())
        )
        raw_active_weights = value.get("active_token_weights")
        active_weights = (
            tuple((domain, ()) for domain in domains)
            if raw_active_weights is None
            else tuple(
                (
                    str(domain),
                    tuple(
                        (int(token_id), float(weight))
                        for token_id, weight in token_weights
                    ),
                )
                for domain, token_weights in raw_active_weights
            )
        )
        raw_observed = value.get("last_observed_step")
        raw_audit = value.get("last_audit_step")
        state = cls(
            domains=domains,
            domain_candidate_token_ids=domain_candidates,
            audit_interval_steps=int(value.get("audit_interval_steps", 0)),
            window_steps=int(value.get("window_steps", 0)),
            min_mean_occurrences_per_step=float(
                value.get("min_mean_occurrences_per_step", 0.0)
            ),
            top_k=int(value.get("top_k", 0)),
            selection_mode=normalize_selection_mode(
                value.get("selection_mode", TOP_LOSS_SELECTION_MODE)
            ),
            weight_mode=normalize_online_weight_mode(
                value.get("weight_mode", FIXED_ONLINE_WEIGHT_MODE)
            ),
            history=history,
            active_token_ids=active,
            active_token_weights=active_weights,
            last_observed_step=(None if raw_observed is None else int(raw_observed)),
            last_audit_step=None if raw_audit is None else int(raw_audit),
            update_count=int(value.get("update_count", 0)),
        )
        if state.weight_mode == PAIRED_ONLINE_WEIGHT_MODE:
            active_map = state.active_map()
            weight_map = state.active_weight_map()
            for domain in state.domains:
                if set(active_map.get(domain, ())) != set(
                    weight_map.get(domain, {})
                ):
                    raise ValueError(
                        "Checkpointed paired online Control weights must "
                        "exactly match active token IDs."
                    )
                if any(
                    not math.isfinite(weight) or not 1.0 <= weight <= 4.0
                    for weight in weight_map.get(domain, {}).values()
                ):
                    raise ValueError(
                        "Checkpointed paired online Control weights must be "
                        "finite and in [1, 4]."
                    )
        return state


def initial_online_control_selection_state(
    domains: Sequence[str],
    candidate_token_ids: Mapping[str, Sequence[int]] | Sequence[int],
    *,
    audit_interval_steps: int,
    window_steps: int,
    min_mean_occurrences_per_step: float,
    top_k: int,
    selection_mode: str = TOP_LOSS_SELECTION_MODE,
    weight_mode: str = FIXED_ONLINE_WEIGHT_MODE,
) -> OnlineControlSelectionState:
    """Create an empty selector state with a frozen configuration signature."""

    normalized_domains = tuple(dict.fromkeys(str(domain) for domain in domains))
    if not normalized_domains:
        raise ValueError("Online Control selection requires at least one domain.")
    if isinstance(candidate_token_ids, Mapping):
        candidate_domains = {str(domain) for domain in candidate_token_ids}
        if candidate_domains != set(normalized_domains):
            raise ValueError(
                "Online Control domain candidates must exactly match domains."
            )
        raw_candidates = {
            str(domain): token_ids for domain, token_ids in candidate_token_ids.items()
        }
    else:
        raw_candidates = {domain: candidate_token_ids for domain in normalized_domains}
    domain_candidates = tuple(
        (
            domain,
            tuple(sorted({int(token_id) for token_id in raw_candidates[domain]})),
        )
        for domain in normalized_domains
    )
    if any(
        not token_ids or any(token_id < 0 for token_id in token_ids)
        for _, token_ids in domain_candidates
    ):
        raise ValueError(
            "Online Control candidate token IDs must be non-empty and non-negative."
        )
    if audit_interval_steps < 1 or window_steps < 1 or top_k < 1:
        raise ValueError(
            "Online Control audit interval, window, and Top-K must be positive."
        )
    normalized_selection_mode = normalize_selection_mode(selection_mode)
    normalized_weight_mode = normalize_online_weight_mode(weight_mode)
    if normalized_selection_mode == TOP_SPEED_SELECTION_MODE and window_steps < 2:
        raise ValueError(
            "Online top-speed selection requires window_steps to be at least 2."
        )
    if (
        normalized_weight_mode == PAIRED_ONLINE_WEIGHT_MODE
        and normalized_selection_mode not in PAIRED_SIGNAL_SELECTION_MODES
    ):
        raise ValueError(
            "Online paired weight mode requires a paired-signal selection mode."
        )
    if (
        not math.isfinite(min_mean_occurrences_per_step)
        or min_mean_occurrences_per_step < 0.0
    ):
        raise ValueError(
            "Online Control minimum mean occurrences must be finite and non-negative."
        )
    return OnlineControlSelectionState(
        domains=normalized_domains,
        domain_candidate_token_ids=domain_candidates,
        audit_interval_steps=int(audit_interval_steps),
        window_steps=int(window_steps),
        min_mean_occurrences_per_step=float(min_mean_occurrences_per_step),
        top_k=int(top_k),
        selection_mode=normalized_selection_mode,
        weight_mode=normalized_weight_mode,
        history=(),
        active_token_ids=tuple((domain, ()) for domain in normalized_domains),
        active_token_weights=tuple((domain, ()) for domain in normalized_domains),
    )


def _normalize_statistics(
    state: OnlineControlSelectionState,
    statistics: Mapping[str, Mapping[int, tuple[float, int]]],
) -> DomainStatistics:
    unknown_domains = set(statistics) - set(state.domains)
    if unknown_domains:
        raise ValueError(
            "Online Control statistics contain unknown domains: "
            + ", ".join(sorted(unknown_domains))
        )
    candidates_by_domain = {
        domain: set(token_ids) for domain, token_ids in state.domain_candidate_token_ids
    }
    rows: list[tuple[str, tuple[TokenStatistic, ...]]] = []
    for domain in state.domains:
        normalized: list[TokenStatistic] = []
        for token_id, (loss_sum, count) in statistics.get(domain, {}).items():
            token_id = int(token_id)
            loss_sum = float(loss_sum)
            count = int(count)
            if token_id not in candidates_by_domain[domain] or count == 0:
                continue
            if count < 0 or not math.isfinite(loss_sum) or loss_sum < 0.0:
                raise ValueError(
                    "Online Control statistics require finite non-negative "
                    "selection-score sums and counts."
                )
            normalized.append((token_id, loss_sum, count))
        rows.append((domain, tuple(sorted(normalized))))
    return tuple(rows)


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


def _selection_score_distribution(
    values: Sequence[float],
) -> SelectionScoreDistribution | None:
    """Summarize one domain's token-ID ranking-score distribution."""

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


def _select_from_history(
    state: OnlineControlSelectionState,
    history: Sequence[StepStatistics],
) -> tuple[
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[tuple[str, tuple[tuple[int, float], ...]], ...],
    tuple[DomainSelectionResult, ...],
]:
    totals: dict[str, dict[int, tuple[float, int]]] = {
        domain: {} for domain in state.domains
    }
    observations: dict[str, dict[int, list[tuple[int, float, int]]]] = {
        domain: {} for domain in state.domains
    }
    for step, domain_rows in history:
        for domain, statistics in domain_rows:
            for token_id, loss_sum, count in statistics:
                prior_loss, prior_count = totals[domain].get(token_id, (0.0, 0))
                totals[domain][token_id] = (
                    prior_loss + loss_sum,
                    prior_count + count,
                )
                observations[domain].setdefault(token_id, []).append(
                    (step, loss_sum / count, count)
                )

    active: list[tuple[str, tuple[int, ...]]] = []
    active_weights: list[tuple[str, tuple[tuple[int, float], ...]]] = []
    results: list[DomainSelectionResult] = []
    for domain in state.domains:
        eligible: list[SelectedControlToken] = []
        for token_id, (loss_sum, count) in totals[domain].items():
            frequency = count / float(state.window_steps)
            if frequency < state.min_mean_occurrences_per_step:
                continue
            speed = (
                occurrence_weighted_optimization_speed(
                    observations[domain].get(token_id, ())
                )
                if state.selection_mode == TOP_SPEED_SELECTION_MODE
                else None
            )
            if state.selection_mode == TOP_SPEED_SELECTION_MODE and speed is None:
                continue
            eligible.append(
                SelectedControlToken(
                    token_id=token_id,
                    occurrence_count=count,
                    mean_occurrences_per_step=frequency,
                    mean_abs_loss=(
                        loss_sum / count
                        if state.selection_mode
                        in {TOP_LOSS_SELECTION_MODE, TOP_SPEED_SELECTION_MODE}
                        else None
                    ),
                    mean_selection_score=loss_sum / count,
                    optimization_speed=(None if speed is None else speed.value),
                    observed_step_count=(
                        0 if speed is None else speed.observed_step_count
                    ),
                )
            )
        if state.selection_mode == TOP_SPEED_SELECTION_MODE:
            ranked = sorted(
                eligible,
                key=lambda item: (
                    -cast(float, item.optimization_speed),
                    item.token_id,
                ),
            )
        else:
            ranked = sorted(
                eligible,
                key=lambda item: (-item.mean_selection_score, item.token_id),
            )
        selected = tuple(ranked[: state.top_k])
        eligible_ranking_scores = tuple(
            cast(float, item.optimization_speed)
            if state.selection_mode == TOP_SPEED_SELECTION_MODE
            else item.mean_selection_score
            for item in eligible
        )
        selected_ranking_scores = tuple(
            cast(float, item.optimization_speed)
            if state.selection_mode == TOP_SPEED_SELECTION_MODE
            else item.mean_selection_score
            for item in selected
        )
        active.append((domain, tuple(item.token_id for item in selected)))
        active_weights.append(
            (
                domain,
                (
                    tuple(
                        (item.token_id, 1.0 + item.mean_selection_score)
                        for item in selected
                    )
                    if state.weight_mode == PAIRED_ONLINE_WEIGHT_MODE
                    else ()
                ),
            )
        )
        results.append(
            DomainSelectionResult(
                domain=domain,
                eligible_token_count=len(eligible),
                selected_tokens=selected,
                eligible_score_distribution=_selection_score_distribution(
                    eligible_ranking_scores
                ),
                selected_score_distribution=_selection_score_distribution(
                    selected_ranking_scores
                ),
            )
        )
    return tuple(active), tuple(active_weights), tuple(results)


def update_online_control_selection(
    state: OnlineControlSelectionState,
    statistics: Mapping[str, Mapping[int, tuple[float, int]]],
    *,
    step: int,
) -> tuple[OnlineControlSelectionOutcome, OnlineControlSelectionState]:
    """Ingest one complete step and update active IDs only at audit boundaries."""

    step = int(step)
    prior_step = state.last_observed_step
    if prior_step is not None and step < prior_step:
        raise ValueError("Online Control selection cannot move backward in step.")
    if prior_step == step:
        return (
            OnlineControlSelectionOutcome(
                observed_step=step,
                audit_triggered=False,
                duplicate_step=True,
                history_reset=False,
                window_fill_steps=len(state.history),
                domain_results=(),
            ),
            state,
        )

    history_reset = prior_step is not None and step != prior_step + 1
    history = () if history_reset else state.history
    active = (
        tuple((domain, ()) for domain in state.domains)
        if history_reset
        else state.active_token_ids
    )
    active_weights = (
        tuple((domain, ()) for domain in state.domains)
        if history_reset
        else state.active_token_weights
    )
    history = (
        *history,
        (step, _normalize_statistics(state, statistics)),
    )[-state.window_steps :]
    audit_triggered = (
        len(history) == state.window_steps and step % state.audit_interval_steps == 0
    )
    domain_results: tuple[DomainSelectionResult, ...] = ()
    last_audit_step = state.last_audit_step
    update_count = state.update_count
    if audit_triggered:
        active, active_weights, domain_results = _select_from_history(
            state,
            history,
        )
        last_audit_step = step
        update_count += 1

    next_state = OnlineControlSelectionState(
        domains=state.domains,
        domain_candidate_token_ids=state.domain_candidate_token_ids,
        audit_interval_steps=state.audit_interval_steps,
        window_steps=state.window_steps,
        min_mean_occurrences_per_step=(state.min_mean_occurrences_per_step),
        top_k=state.top_k,
        selection_mode=state.selection_mode,
        weight_mode=state.weight_mode,
        history=tuple(history),
        active_token_ids=active,
        active_token_weights=active_weights,
        last_observed_step=step,
        last_audit_step=last_audit_step,
        update_count=update_count,
    )
    return (
        OnlineControlSelectionOutcome(
            observed_step=step,
            audit_triggered=audit_triggered,
            duplicate_step=False,
            history_reset=history_reset,
            window_fill_steps=len(history),
            domain_results=domain_results,
        ),
        next_state,
    )
