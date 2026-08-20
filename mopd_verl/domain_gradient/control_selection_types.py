"""Immutable records emitted by online Control-token selection."""

from __future__ import annotations

from dataclasses import dataclass


TokenStatistic = tuple[int, float, int]
DomainStatistics = tuple[tuple[str, tuple[TokenStatistic, ...]], ...]
StepStatistics = tuple[int, DomainStatistics]


@dataclass(frozen=True)
class SelectedControlToken:
    """One selected token and its rolling-window ranking statistics."""

    token_id: int
    occurrence_count: int
    mean_occurrences_per_step: float
    mean_abs_loss: float
    optimization_speed: float | None
    observed_step_count: int


@dataclass(frozen=True)
class DomainSelectionResult:
    """Selection result for one domain at an audit boundary."""

    domain: str
    eligible_token_count: int
    selected_tokens: tuple[SelectedControlToken, ...]


@dataclass(frozen=True)
class OnlineControlSelectionOutcome:
    """Observable result of ingesting one completed optimizer step."""

    observed_step: int
    audit_triggered: bool
    duplicate_step: bool
    history_reset: bool
    window_fill_steps: int
    domain_results: tuple[DomainSelectionResult, ...]
