"""Rolling online selection for Control-token weighting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from mopd_verl.domain_gradient.control_selection_budget import (
    normalize_candidate_statistics,
    normalize_valid_token_counts,
    select_ranked_tokens,
    top_p_target_occurrence_count,
)
from mopd_verl.domain_gradient.control_selection_distribution import (
    selection_score_distribution,
)
from mopd_verl.domain_gradient.control_selection_scoring import (
    FIXED_ONLINE_WEIGHT_MODE,
    LOSS_RATIO_ONLINE_WEIGHT_MODE,
    PAIRED_ONLINE_WEIGHT_MODE,
    PAIRED_SIGNAL_SELECTION_MODES,
    TOP_K_BUDGET_MODE,
    TOP_LOSS_SELECTION_MODE,
    TOP_P_BUDGET_MODE,
    TOP_SPEED_SELECTION_MODE,
    normalize_online_budget_mode,
    normalize_online_weight_mode,
    normalize_selection_mode,
    occurrence_weighted_optimization_speed,
    selected_to_other_loss_ratio_weight,
    validate_loss_ratio_alpha,
)
from mopd_verl.domain_gradient.control_selection_types import (
    DomainSelectionResult,
    OnlineControlSelectionOutcome,
    SelectedControlToken,
    StepStatistics,
    StepValidScoreSums,
    StepValidTokenCounts,
)


@dataclass(frozen=True)
class OnlineControlSelectionState:
    """Checkpointable rolling statistics and lagged active token IDs."""

    domains: tuple[str, ...]
    domain_candidate_token_ids: tuple[tuple[str, tuple[int, ...]], ...]
    audit_interval_steps: int
    window_steps: int
    min_mean_occurrences_per_step: float
    strict_occurrence_gate: bool
    top_k: int
    budget_mode: str
    top_p: float
    selection_mode: str
    weight_mode: str
    history: tuple[StepStatistics, ...]
    valid_token_count_history: tuple[StepValidTokenCounts, ...]
    valid_score_sum_history: tuple[StepValidScoreSums, ...]
    active_token_ids: tuple[tuple[str, tuple[int, ...]], ...]
    active_token_weights: tuple[
        tuple[str, tuple[tuple[int, float], ...]], ...
    ]
    last_observed_step: int | None = None
    last_audit_step: int | None = None
    update_count: int = 0
    domain_candidate_token_groups: tuple[
        tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...
    ] = ()
    top_k_per_group: int | None = None
    loss_ratio_alpha: float = 1.0

    def active_map(self) -> dict[str, tuple[int, ...]]:
        return dict(self.active_token_ids)

    def candidate_map(self) -> dict[str, tuple[int, ...]]:
        return dict(self.domain_candidate_token_ids)

    def candidate_group_map(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return {
            domain: dict(groups)
            for domain, groups in self.domain_candidate_token_groups
        }

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
            "schema_version": 10,
            "domains": self.domains,
            "domain_candidate_token_ids": self.domain_candidate_token_ids,
            "audit_interval_steps": self.audit_interval_steps,
            "window_steps": self.window_steps,
            "min_mean_occurrences_per_step": (self.min_mean_occurrences_per_step),
            "strict_occurrence_gate": self.strict_occurrence_gate,
            "top_k": self.top_k,
            "budget_mode": self.budget_mode,
            "top_p": self.top_p,
            "domain_candidate_token_groups": self.domain_candidate_token_groups,
            "top_k_per_group": self.top_k_per_group,
            "selection_mode": self.selection_mode,
            "weight_mode": self.weight_mode,
            "loss_ratio_alpha": self.loss_ratio_alpha,
            "history": self.history,
            "valid_token_count_history": self.valid_token_count_history,
            "valid_score_sum_history": self.valid_score_sum_history,
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
        from mopd_verl.domain_gradient.control_top_loss_checkpoint import (
            restore_online_control_selection_state,
        )

        return restore_online_control_selection_state(cls, value)


def initial_online_control_selection_state(
    domains: Sequence[str],
    candidate_token_ids: Mapping[str, Sequence[int]] | Sequence[int],
    *,
    audit_interval_steps: int,
    window_steps: int,
    min_mean_occurrences_per_step: float,
    strict_occurrence_gate: bool = False,
    top_k: int,
    budget_mode: str = TOP_K_BUDGET_MODE,
    top_p: float = 1.0,
    candidate_token_groups: (
        Mapping[str, Mapping[str, Sequence[int]]] | None
    ) = None,
    top_k_per_group: int | None = None,
    selection_mode: str = TOP_LOSS_SELECTION_MODE,
    weight_mode: str = FIXED_ONLINE_WEIGHT_MODE,
    loss_ratio_alpha: float = 1.0,
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
    domain_candidate_groups: tuple[
        tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...
    ] = ()
    if candidate_token_groups is not None:
        if set(candidate_token_groups) != set(normalized_domains):
            raise ValueError(
                "Online Control domain candidate groups must exactly match domains."
            )
        normalized_groups = []
        candidate_map = dict(domain_candidates)
        for domain in normalized_domains:
            raw_groups = candidate_token_groups[domain]
            if not raw_groups:
                raise ValueError(
                    "Online Control domain candidate groups must be non-empty."
                )
            groups = tuple(
                (
                    str(group),
                    tuple(sorted({int(token_id) for token_id in token_ids})),
                )
                for group, token_ids in raw_groups.items()
            )
            if any(
                not token_ids or any(token_id < 0 for token_id in token_ids)
                for _, token_ids in groups
            ):
                raise ValueError(
                    "Online Control candidate group IDs must be non-empty and "
                    "non-negative."
                )
            flattened = [
                token_id for _, token_ids in groups for token_id in token_ids
            ]
            if len(flattened) != len(set(flattened)):
                raise ValueError(
                    "Online Control candidate groups must be disjoint within "
                    f"domain {domain!r}."
                )
            if set(flattened) != set(candidate_map[domain]):
                raise ValueError(
                    "Online Control candidate groups must exactly partition "
                    f"the candidate IDs for domain {domain!r}."
                )
            normalized_groups.append((domain, groups))
        domain_candidate_groups = tuple(normalized_groups)
    if audit_interval_steps < 1 or window_steps < 1 or top_k < 1:
        raise ValueError(
            "Online Control audit interval, window, and Top-K must be positive."
        )
    normalized_budget_mode = normalize_online_budget_mode(budget_mode)
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("Online Control top_p must be finite and in (0, 1].")
    if domain_candidate_groups and normalized_budget_mode == TOP_K_BUDGET_MODE:
        if top_k_per_group is None or top_k_per_group < 1:
            raise ValueError(
                "Grouped online Control selection requires a positive "
                "top_k_per_group."
            )
        group_counts = {len(groups) for _, groups in domain_candidate_groups}
        if len(group_counts) != 1 or top_k != top_k_per_group * next(
            iter(group_counts)
        ):
            raise ValueError(
                "Grouped online Control top_k must equal top_k_per_group times "
                "the number of groups in every domain."
            )
    elif top_k_per_group is not None:
        raise ValueError(
            "Online Control top_k_per_group requires grouped Top-K selection."
        )
    normalized_selection_mode = normalize_selection_mode(selection_mode)
    normalized_weight_mode = normalize_online_weight_mode(weight_mode)
    validate_loss_ratio_alpha(loss_ratio_alpha, weight_mode=normalized_weight_mode)
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
        normalized_weight_mode == LOSS_RATIO_ONLINE_WEIGHT_MODE
        and normalized_selection_mode != TOP_LOSS_SELECTION_MODE
    ):
        raise ValueError(
            "Online loss-ratio weight mode requires top-loss selection mode."
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
        strict_occurrence_gate=bool(strict_occurrence_gate),
        top_k=int(top_k),
        budget_mode=normalized_budget_mode,
        top_p=float(top_p),
        selection_mode=normalized_selection_mode,
        weight_mode=normalized_weight_mode,
        loss_ratio_alpha=float(loss_ratio_alpha),
        history=(),
        valid_token_count_history=(),
        valid_score_sum_history=(),
        active_token_ids=tuple((domain, ()) for domain in normalized_domains),
        active_token_weights=tuple((domain, ()) for domain in normalized_domains),
        domain_candidate_token_groups=domain_candidate_groups,
        top_k_per_group=(None if top_k_per_group is None else int(top_k_per_group)),
    )


def _ranking_score(
    state: OnlineControlSelectionState,
    item: SelectedControlToken,
) -> float:
    """Return the scalar used to rank one token type."""

    if state.selection_mode == TOP_SPEED_SELECTION_MODE:
        return cast(float, item.optimization_speed)
    return item.mean_selection_score


def _select_from_history(
    state: OnlineControlSelectionState,
    history: Sequence[StepStatistics],
    valid_token_count_history: Sequence[StepValidTokenCounts],
    valid_score_sum_history: Sequence[StepValidScoreSums],
    *,
    loss_ratio_max_weight: float,
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
    valid_token_totals = {domain: 0 for domain in state.domains}
    valid_score_totals = {domain: 0.0 for domain in state.domains}
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
    for _, domain_counts in valid_token_count_history:
        for domain, count in domain_counts:
            valid_token_totals[domain] += count
    for _, domain_sums in valid_score_sum_history:
        for domain, score_sum in domain_sums:
            valid_score_totals[domain] += score_sum

    active: list[tuple[str, tuple[int, ...]]] = []
    active_weights: list[tuple[str, tuple[tuple[int, float], ...]]] = []
    results: list[DomainSelectionResult] = []
    for domain in state.domains:
        eligible: list[SelectedControlToken] = []
        for token_id, (loss_sum, count) in totals[domain].items():
            frequency = count / float(state.window_steps)
            token_observations = observations[domain].get(token_id, ())
            if state.strict_occurrence_gate:
                threshold = state.min_mean_occurrences_per_step
                if len(token_observations) != state.window_steps or any(
                    step_count <= threshold
                    for _, _, step_count in token_observations
                ):
                    continue
            elif frequency < state.min_mean_occurrences_per_step:
                continue
            speed = (
                occurrence_weighted_optimization_speed(
                    observations[domain].get(token_id, ())
                )
                if state.selection_mode
                in {TOP_LOSS_SELECTION_MODE, TOP_SPEED_SELECTION_MODE}
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
        candidate_groups = state.candidate_group_map().get(domain, {})
        if candidate_groups and state.budget_mode == TOP_K_BUDGET_MODE:
            top_k_per_group = state.top_k_per_group
            if top_k_per_group is None:
                raise RuntimeError(
                    "Grouped online Control state is missing top_k_per_group."
                )
            grouped_selected: list[SelectedControlToken] = []
            for token_ids in candidate_groups.values():
                group_ids = set(token_ids)
                grouped_selected.extend(
                    select_ranked_tokens(
                        tuple(item for item in ranked if item.token_id in group_ids),
                        budget_mode=state.budget_mode,
                        top_k=top_k_per_group,
                        top_p=state.top_p,
                    )
                )
            selected = tuple(grouped_selected)
        else:
            selected = select_ranked_tokens(
                ranked,
                budget_mode=state.budget_mode,
                top_k=state.top_k,
                top_p=state.top_p,
                valid_token_count=valid_token_totals[domain],
            )
        eligible_ranking_scores = tuple(
            _ranking_score(state, item)
            for item in eligible
        )
        selected_ranking_scores = tuple(
            _ranking_score(state, item)
            for item in selected
        )
        selected_occurrence_count = sum(
            item.occurrence_count for item in selected
        )
        loss_ratio = None
        if state.weight_mode == LOSS_RATIO_ONLINE_WEIGHT_MODE:
            selected_loss_abs_sum = sum(
                totals[domain][item.token_id][0]
                for item in selected
            )
            loss_ratio = selected_to_other_loss_ratio_weight(
                selected_loss_abs_sum=selected_loss_abs_sum,
                selected_occurrence_count=selected_occurrence_count,
                valid_loss_abs_sum=valid_score_totals[domain],
                valid_occurrence_count=valid_token_totals[domain],
                max_weight=loss_ratio_max_weight,
                alpha=state.loss_ratio_alpha,
            )
        top_p_target = (
            top_p_target_occurrence_count(
                state.top_p,
                valid_token_totals[domain],
            )
            if state.budget_mode == TOP_P_BUDGET_MODE
            else None
        )
        active.append((domain, tuple(item.token_id for item in selected)))
        if state.weight_mode == PAIRED_ONLINE_WEIGHT_MODE:
            domain_active_weights = tuple(
                (item.token_id, 1.0 + item.mean_selection_score)
                for item in selected
            )
        elif loss_ratio is not None:
            domain_active_weights = tuple(
                (item.token_id, loss_ratio.scaled_weight)
                for item in selected
            )
        else:
            domain_active_weights = ()
        active_weights.append(
            (
                domain,
                domain_active_weights,
            )
        )
        results.append(
            DomainSelectionResult(
                domain=domain,
                valid_token_count=valid_token_totals[domain],
                eligible_token_count=len(eligible),
                selected_occurrence_count=selected_occurrence_count,
                selected_occurrence_fraction=(
                    selected_occurrence_count / valid_token_totals[domain]
                    if valid_token_totals[domain] > 0
                    else 0.0
                ),
                target_occurrence_count=top_p_target,
                top_p_target_reached=(
                    selected_occurrence_count >= top_p_target
                    if top_p_target is not None
                    and valid_token_totals[domain] > 0
                    else None
                ),
                top_p_occurrence_shortfall=(
                    max(0, top_p_target - selected_occurrence_count)
                    if top_p_target is not None
                    and valid_token_totals[domain] > 0
                    else None
                ),
                selected_tokens=selected,
                eligible_score_distribution=selection_score_distribution(
                    eligible_ranking_scores
                ),
                selected_score_distribution=selection_score_distribution(
                    selected_ranking_scores
                ),
                selected_occurrence_mean_abs_loss=(
                    None
                    if loss_ratio is None
                    else loss_ratio.selected_mean_abs_loss
                ),
                other_occurrence_count=(
                    None if loss_ratio is None else loss_ratio.other_occurrence_count
                ),
                other_occurrence_mean_abs_loss=(
                    None if loss_ratio is None else loss_ratio.other_mean_abs_loss
                ),
                raw_selected_to_other_loss_ratio=(
                    None if loss_ratio is None else loss_ratio.raw_ratio
                ),
                selected_raw_loss_ratio_weight=(
                    None if loss_ratio is None else loss_ratio.scaled_weight
                ),
                selected_unscaled_loss_ratio_weight=(
                    None if loss_ratio is None else loss_ratio.clipped_weight
                ),
            )
        )
    return tuple(active), tuple(active_weights), tuple(results)


def update_online_control_selection(
    state: OnlineControlSelectionState,
    statistics: Mapping[str, Mapping[int, tuple[float, int]]],
    *,
    step: int,
    valid_token_counts: Mapping[str, int] | None = None,
    valid_score_sums: Mapping[str, float] | None = None,
    loss_ratio_max_weight: float = 4.0,
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
    valid_token_count_history = (
        () if history_reset else state.valid_token_count_history
    )
    valid_score_sum_history = (
        () if history_reset else state.valid_score_sum_history
    )
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
    normalized_statistics = normalize_candidate_statistics(
        domains=state.domains,
        domain_candidate_token_ids=state.candidate_map(),
        statistics=statistics,
    )
    if (
        state.weight_mode == LOSS_RATIO_ONLINE_WEIGHT_MODE
        and valid_token_counts is None
    ):
        raise ValueError(
            "Online loss-ratio weighting requires valid_token_counts."
        )
    normalized_valid_token_counts = normalize_valid_token_counts(
        domains=state.domains,
        budget_mode=state.budget_mode,
        valid_token_counts=valid_token_counts,
        statistics=normalized_statistics,
    )
    if valid_score_sums is None:
        if state.weight_mode == LOSS_RATIO_ONLINE_WEIGHT_MODE:
            raise ValueError(
                "Online loss-ratio weighting requires valid_score_sums."
            )
        normalized_valid_score_sums = tuple(
            (domain, 0.0) for domain in state.domains
        )
    else:
        if set(valid_score_sums) != set(state.domains):
            raise ValueError(
                "Online Control valid score sums must exactly match domains."
            )
        normalized_valid_score_sums = tuple(
            (domain, float(valid_score_sums[domain]))
            for domain in state.domains
        )
        if any(
            not math.isfinite(score_sum) or score_sum < 0.0
            for _, score_sum in normalized_valid_score_sums
        ):
            raise ValueError(
                "Online Control valid score sums must be finite and non-negative."
            )
    if (
        state.weight_mode == LOSS_RATIO_ONLINE_WEIGHT_MODE
        and (
            not math.isfinite(loss_ratio_max_weight)
            or loss_ratio_max_weight < 1.0
        )
    ):
        raise ValueError(
            "Online loss-ratio maximum weight must be finite and at least 1."
        )
    history = (
        *history,
        (step, normalized_statistics),
    )[-state.window_steps :]
    valid_token_count_history = (
        *valid_token_count_history,
        (step, normalized_valid_token_counts),
    )[-state.window_steps :]
    valid_score_sum_history = (
        *valid_score_sum_history,
        (step, normalized_valid_score_sums),
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
            valid_token_count_history,
            valid_score_sum_history,
            loss_ratio_max_weight=loss_ratio_max_weight,
        )
        last_audit_step = step
        update_count += 1

    next_state = OnlineControlSelectionState(
        domains=state.domains,
        domain_candidate_token_ids=state.domain_candidate_token_ids,
        audit_interval_steps=state.audit_interval_steps,
        window_steps=state.window_steps,
        min_mean_occurrences_per_step=(state.min_mean_occurrences_per_step),
        strict_occurrence_gate=state.strict_occurrence_gate,
        top_k=state.top_k,
        budget_mode=state.budget_mode,
        top_p=state.top_p,
        selection_mode=state.selection_mode,
        weight_mode=state.weight_mode,
        loss_ratio_alpha=state.loss_ratio_alpha,
        history=tuple(history),
        valid_token_count_history=tuple(valid_token_count_history),
        valid_score_sum_history=tuple(valid_score_sum_history),
        active_token_ids=active,
        active_token_weights=active_weights,
        last_observed_step=step,
        last_audit_step=last_audit_step,
        update_count=update_count,
        domain_candidate_token_groups=state.domain_candidate_token_groups,
        top_k_per_group=state.top_k_per_group,
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
