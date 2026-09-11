"""Validation and budget application for online Control-token selection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import ROUND_CEILING, Decimal

from mopd_verl.domain_gradient.control_selection_scoring import (
    TOP_P_BUDGET_MODE,
)
from mopd_verl.domain_gradient.control_selection_types import (
    DomainStatistics,
    DomainValidTokenCounts,
    SelectedControlToken,
    TokenStatistic,
)


def normalize_candidate_statistics(
    *,
    domains: Sequence[str],
    domain_candidate_token_ids: Mapping[str, Sequence[int]],
    statistics: Mapping[str, Mapping[int, tuple[float, int]]],
) -> DomainStatistics:
    """Return finite, positive-count statistics for configured candidates."""

    unknown_domains = set(statistics) - set(domains)
    if unknown_domains:
        raise ValueError(
            "Online Control statistics contain unknown domains: "
            + ", ".join(sorted(unknown_domains))
        )
    candidates_by_domain = {
        domain: set(token_ids)
        for domain, token_ids in domain_candidate_token_ids.items()
    }
    rows: list[tuple[str, tuple[TokenStatistic, ...]]] = []
    for domain in domains:
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


def normalize_valid_token_counts(
    *,
    domains: Sequence[str],
    budget_mode: str,
    valid_token_counts: Mapping[str, int] | None,
    statistics: DomainStatistics,
) -> DomainValidTokenCounts:
    """Validate the per-domain denominator paired with one observed step."""

    if valid_token_counts is None:
        if budget_mode == TOP_P_BUDGET_MODE:
            raise ValueError(
                "Online Control Top-P selection requires valid_token_counts."
            )
        return tuple((domain, 0) for domain in domains)
    if set(valid_token_counts) != set(domains):
        raise ValueError(
            "Online Control valid-token counts must exactly match domains."
        )
    candidate_occurrences = {
        domain: sum(count for _, _, count in domain_statistics)
        for domain, domain_statistics in statistics
    }
    normalized: list[tuple[str, int]] = []
    for domain in domains:
        count = int(valid_token_counts[domain])
        if count < candidate_occurrences.get(domain, 0):
            raise ValueError(
                "Online Control valid-token count cannot be smaller than "
                f"candidate occurrences for domain {domain!r}."
            )
        normalized.append((domain, count))
    return tuple(normalized)


def normalize_top_p_by_domain(
    domains: Sequence[str],
    value: Mapping[str, float]
    | Sequence[tuple[str, float]]
    | None,
) -> tuple[tuple[str, float], ...]:
    """Validate optional per-domain Top-P overrides."""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        items = tuple(value.items())
    elif isinstance(value, (str, bytes)):
        raise TypeError(
            "control_token_online_top_p_by_domain must be a mapping."
        )
    else:
        try:
            items = tuple(value)
        except TypeError as exc:
            raise TypeError(
                "control_token_online_top_p_by_domain must be a mapping."
            ) from exc
    if not items:
        return ()

    normalized_domains = tuple(dict.fromkeys(str(domain) for domain in domains))
    normalized: dict[str, float] = {}
    for item in items:
        try:
            raw_domain, raw_top_p = item
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "control_token_online_top_p_by_domain entries must be "
                "(domain, value) pairs."
            ) from exc
        domain = str(raw_domain)
        if domain in normalized:
            raise ValueError(
                "control_token_online_top_p_by_domain must not contain "
                f"duplicate domain {domain!r}."
            )
        try:
            top_p = float(raw_top_p)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "control_token_online_top_p_by_domain values must be "
                "finite and in (0, 1]."
            ) from exc
        if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
            raise ValueError(
                "control_token_online_top_p_by_domain values must be "
                "finite and in (0, 1]."
            )
        normalized[domain] = top_p
    if set(normalized) != set(normalized_domains):
        raise ValueError(
            "control_token_online_top_p_by_domain keys must exactly match "
            "domains."
        )
    return tuple((domain, normalized[domain]) for domain in normalized_domains)


def select_ranked_tokens(
    ranked: Sequence[SelectedControlToken],
    *,
    budget_mode: str,
    top_k: int,
    top_p: float,
    valid_token_count: int = 0,
) -> tuple[SelectedControlToken, ...]:
    """Apply a fixed-count or valid-token-occurrence budget."""

    ordered = tuple(ranked)
    if budget_mode != TOP_P_BUDGET_MODE:
        return ordered[:top_k]
    if not ordered or valid_token_count <= 0:
        return ()
    target_occurrences = top_p_target_occurrence_count(
        top_p,
        valid_token_count,
    )
    selected: list[SelectedControlToken] = []
    selected_occurrences = 0
    for item in ordered:
        selected.append(item)
        selected_occurrences += item.occurrence_count
        if selected_occurrences >= target_occurrences:
            break
    return tuple(selected)


def top_p_target_occurrence_count(top_p: float, valid_token_count: int) -> int:
    """Return the minimum integer occurrence count satisfying ``top_p``."""

    if valid_token_count <= 0:
        return 0
    return int(
        (Decimal(str(top_p)) * valid_token_count).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
