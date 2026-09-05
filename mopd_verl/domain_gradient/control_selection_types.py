"""Immutable records emitted by online Control-token selection."""

from __future__ import annotations

from dataclasses import dataclass


TokenStatistic = tuple[int, float, int]
DomainStatistics = tuple[tuple[str, tuple[TokenStatistic, ...]], ...]
StepStatistics = tuple[int, DomainStatistics]
DomainValidTokenCounts = tuple[tuple[str, int], ...]
StepValidTokenCounts = tuple[int, DomainValidTokenCounts]
DomainValidScoreSums = tuple[tuple[str, float], ...]
StepValidScoreSums = tuple[int, DomainValidScoreSums]


@dataclass(frozen=True)
class SelectedControlToken:
    """One selected token and its rolling-window ranking statistics."""

    token_id: int
    occurrence_count: int
    mean_occurrences_per_step: float
    mean_abs_loss: float | None
    mean_selection_score: float
    optimization_speed: float | None
    observed_step_count: int


@dataclass(frozen=True)
class SelectionScoreDistribution:
    """Finite summary of token-ID ranking scores at one audit boundary."""

    count: int
    mean: float
    std: float
    minimum: float
    p10: float
    p50: float
    p90: float
    maximum: float


@dataclass(frozen=True)
class DomainSelectionResult:
    """Selection result for one domain at an audit boundary."""

    domain: str
    valid_token_count: int
    eligible_token_count: int
    selected_occurrence_count: int
    selected_occurrence_fraction: float
    target_occurrence_count: int | None
    top_p_target_reached: bool | None
    top_p_occurrence_shortfall: int | None
    selected_tokens: tuple[SelectedControlToken, ...]
    eligible_score_distribution: SelectionScoreDistribution | None
    selected_score_distribution: SelectionScoreDistribution | None
    selected_occurrence_mean_abs_loss: float | None
    other_occurrence_count: int | None
    other_occurrence_mean_abs_loss: float | None
    raw_selected_to_other_loss_ratio: float | None
    selected_raw_loss_ratio_weight: float | None
    selected_unscaled_loss_ratio_weight: float | None = None


@dataclass(frozen=True)
class OnlineControlSelectionOutcome:
    """Observable result of ingesting one completed optimizer step."""

    observed_step: int
    audit_triggered: bool
    duplicate_step: bool
    history_reset: bool
    window_fill_steps: int
    domain_results: tuple[DomainSelectionResult, ...]
