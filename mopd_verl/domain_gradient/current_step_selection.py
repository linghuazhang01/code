"""Detached, global head/tail ranking for the current optimizer batch."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from mopd_verl.domain_gradient.control_loss_teacher_confidence import (
    loss_teacher_confidence_type_scores,
)
from mopd_verl.domain_gradient.control_selection_budget import (
    top_p_target_occurrence_count,
)

if TYPE_CHECKING:
    from mopd_verl.domain_gradient.occurrence import Position


@dataclass(frozen=True)
class SelectionUnit:
    key: tuple[int, ...]
    count: int
    loss: float
    teacher_logp: float | None


@dataclass(frozen=True)
class CurrentStepSelection:
    head: dict[str, frozenset[tuple[int, ...]]]
    tail: dict[str, frozenset[tuple[int, ...]]]
    metrics: dict[str, float]


def compact_token_positions(positions: Sequence[Position]) -> list[Position]:
    """Gather sufficient statistics, rather than all positions, for ID ranking."""
    from mopd_verl.domain_gradient.occurrence import Position

    groups: dict[tuple[str, int], list[Position]] = defaultdict(list)
    for position in positions:
        groups[position.domain, position.token_id].append(position)
    compact = []
    for (domain, token_id), group in sorted(groups.items()):
        count = sum(p.occurrence_count for p in group)
        confidence = None
        if all(p.teacher_logp is not None for p in group):
            confidence = (
                math.fsum(p.teacher_logp * p.occurrence_count for p in group) / count
            )
        compact.append(
            Position(
                domain,
                0,
                0,
                0,
                token_id,
                math.fsum(abs(p.score) * p.occurrence_count for p in group) / count,
                confidence,
                count,
            )
        )
    return compact


def _units(
    pool: Sequence[tuple[int, Position]],
    unit: str,
    minimum: float,
    strict: bool,
) -> list[SelectionUnit]:
    groups: dict[int, list[tuple[int, Position]]] = defaultdict(list)
    for rank, position in pool:
        groups[position.token_id].append((rank, position))
    result = []
    for token_id, group in sorted(groups.items()):
        count = sum(p.occurrence_count for _, p in group)
        if not (count > minimum if strict else count >= minimum):
            continue
        if unit == "token_id":
            confidence = None
            if all(p.teacher_logp is not None for _, p in group):
                confidence = (
                    math.fsum(p.teacher_logp * p.occurrence_count for _, p in group)
                    / count
                )
            result.append(
                SelectionUnit(
                    (token_id,),
                    count,
                    math.fsum(abs(p.score) * p.occurrence_count for _, p in group)
                    / count,
                    confidence,
                )
            )
        else:
            result.extend(
                SelectionUnit(
                    (rank, p.batch, p.row, p.column),
                    1,
                    abs(p.score),
                    p.teacher_logp,
                )
                for rank, p in group
            )
    return result


def _scores(
    units: Sequence[SelectionUnit],
    mode: str,
) -> tuple[list[float], dict[str, float]]:
    signal = mode.partition("_")[2]
    if signal == "loss":
        return [item.loss for item in units], {}
    if any(
        item.teacher_logp is None
        or not math.isfinite(item.teacher_logp)
        or item.teacher_logp > 1e-6
        for item in units
    ):
        raise ValueError("Teacher confidence requires finite chosen-token logp <= 0")
    if signal == "teacher_confidence":
        return [float(item.teacher_logp) for item in units], {}
    if signal != "loss_teacher_confidence":
        raise ValueError(f"Unsupported current-step score: {mode}")
    loss = torch.tensor([item.loss for item in units], dtype=torch.float64)
    confidence = torch.tensor(
        [item.teacher_logp for item in units],
        dtype=torch.float64,
    )
    # Each row represents a unit; its means have already been computed globally.
    scores, metrics = loss_teacher_confidence_type_scores(
        loss_abs_sums=loss,
        teacher_logp_sums=confidence,
        counts=torch.ones_like(loss),
        normalization_mask=torch.ones_like(loss).bool(),
    )
    return scores.tolist(), metrics


def _take(
    units: Sequence[SelectionUnit],
    scores: Sequence[float],
    budget: int,
    *,
    descending: bool,
    exclude: frozenset[tuple[int, ...]] = frozenset(),
) -> tuple[frozenset[tuple[int, ...]], int]:
    ordered = sorted(
        zip(units, scores),
        key=lambda pair: (-pair[1] if descending else pair[1], pair[0].key),
    )
    chosen: set[tuple[int, ...]] = set()
    count = 0
    for item, _ in ordered:
        if count >= budget:
            break
        if item.key in exclude:
            continue
        chosen.add(item.key)
        count += item.count
    return frozenset(chosen), count


def current_step_selection_record(
    ranks: Sequence[tuple[dict[str, int], Sequence[Position], str | None]],
    candidates: dict[str, Sequence[int]],
    selection: CurrentStepSelection,
    *,
    step: int,
    unit: str,
    head_fractions: dict[str, float],
    tail_fractions: dict[str, float],
    minimum: float,
    strict: bool,
) -> dict[str, object]:
    """Describe one current-step selection, including every candidate's TopLoss.

    Read-only: it summarizes the gathered positions and never alters selection.
    """
    domains = {}
    for domain, ids in candidates.items():
        allowed = set(ids)
        sums: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        for _, positions, _ in ranks:
            for p in positions:
                if p.domain == domain and p.token_id in allowed:
                    # Compacted token-ID positions carry a mean |loss|; raw
                    # occurrence positions carry one signed loss with count 1.
                    sums[p.token_id] += abs(p.score) * p.occurrence_count
                    counts[p.token_id] += p.occurrence_count
        token_ids = sorted(counts)
        entry: dict[str, object] = {
            "valid_token_count": int(
                sum(rank_counts.get(domain, 0) for rank_counts, _, _ in ranks)
            ),
            "top_p": float(head_fractions[domain]),
            "tail_top_p": float(tail_fractions[domain]),
            "token_ids": token_ids,
            "abs_loss_sums": [float(sums[token_id]) for token_id in token_ids],
            "counts": [int(counts[token_id]) for token_id in token_ids],
        }
        for name, selected in (
            ("head", selection.head.get(domain, frozenset())),
            ("tail", selection.tail.get(domain, frozenset())),
        ):
            if unit == "token_id":
                entry[name + "_token_ids"] = sorted(key[0] for key in selected)
            else:
                entry[name + "_position_count"] = len(selected)
            prefix = f"{domain}/current_step/{name}/"
            entry[name + "_budget_count"] = selection.metrics.get(prefix + "budget_count")
            entry[name + "_selected_count"] = selection.metrics.get(
                prefix + "selected_count"
            )
        domains[str(domain)] = entry
    return {
        "step": int(step),
        "selection_timing": "current_step",
        "unit": unit,
        "score": "abs_selector_rkl",
        "min_occurrences": float(minimum),
        "strict_occurrence_gate": bool(strict),
        "domains": domains,
    }


def select_current_step(
    ranks: Sequence[tuple[dict[str, int], Sequence[Position], str | None]],
    candidates: dict[str, Sequence[int]],
    head_fractions: dict[str, float],
    tail_fractions: dict[str, float],
    minimum: float,
    strict: bool,
    *,
    unit: str,
    head_modes: dict[str, str],
    tail_mode: str,
) -> CurrentStepSelection:
    """Select disjoint sets; whole IDs can overshoot an occurrence budget."""
    if unit not in {"token_id", "occurrence"}:
        raise ValueError("current-step unit must be token_id or occurrence")
    errors = [error for _, _, error in ranks if error]
    if errors:
        raise ValueError("current-step scoring failed: " + "; ".join(errors))
    heads, tails, metrics = {}, {}, {}
    for domain, ids in candidates.items():
        count = sum(counts.get(domain, 0) for counts, _, _ in ranks)
        allowed = set(ids)
        pool = [
            (rank, p)
            for rank, (_, positions, _) in enumerate(ranks)
            for p in positions
            if p.domain == domain and p.token_id in allowed
        ]
        if any(not math.isfinite(p.score) or p.occurrence_count < 1 for _, p in pool):
            raise ValueError(
                "current-step requires finite raw loss and positive counts"
            )
        if unit == "occurrence" and any(p.occurrence_count != 1 for _, p in pool):
            raise ValueError("Occurrence ranking requires unaggregated positions")
        eligible = _units(pool, unit, minimum, strict)
        head_scores, head_norm = _scores(eligible, head_modes[domain])
        head_budget = top_p_target_occurrence_count(head_fractions[domain], count)
        tail_budget = top_p_target_occurrence_count(tail_fractions[domain], count)
        head, head_count = _take(eligible, head_scores, head_budget, descending=True)
        # Normalize composite scores over the same eligible pool before excluding head.
        tail_scores, tail_norm = (
            _scores(eligible, tail_mode) if tail_budget else ([], {})
        )
        tail, tail_count = _take(
            eligible,
            tail_scores,
            tail_budget,
            descending=False,
            exclude=head,
        )
        heads[domain], tails[domain] = head, tail
        prefix = f"{domain}/current_step/"
        metrics.update(
            {
                prefix + "valid_count": float(count),
                prefix + "eligible_count": float(sum(item.count for item in eligible)),
                prefix + "eligible_unit_count": float(len(eligible)),
            }
        )
        for name, selected, selected_count, budget, normalization in (
            ("head", head, head_count, head_budget, head_norm),
            ("tail", tail, tail_count, tail_budget, tail_norm),
        ):
            metrics.update(
                {
                    prefix + name + "/budget_count": float(budget),
                    prefix + name + "/selected_count": float(selected_count),
                    prefix + name + "/selected_unit_count": float(len(selected)),
                    prefix
                    + name
                    + "/coverage": selected_count / count if count else 0.0,
                    prefix
                    + name
                    + "/shortfall": float(max(0, budget - selected_count)),
                    prefix
                    + name
                    + "/overshoot": float(max(0, selected_count - budget)),
                }
            )
            metrics.update(
                {prefix + name + "/" + k: v for k, v in normalization.items()}
            )
    return CurrentStepSelection(heads, tails, metrics)
