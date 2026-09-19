"""Current-batch positional selection; no persistent token-ID weights."""

from collections import Counter
from dataclasses import dataclass
import math
import time
from typing import Any, Sequence

import torch

from mopd_verl.domain_gradient.occurrence_config import (
    validate_occurrence_actor,
    validate_occurrence_config,
)
from mopd_verl.domain_gradient.control_selection_budget import (
    top_p_target_occurrence_count,
)


@dataclass(frozen=True)
class Position:
    domain: str
    batch: int
    row: int
    column: int
    token_id: int
    score: float
    teacher_logp: float | None = None


def select_positions(
    ranks: Sequence[tuple[dict[str, int], Sequence[Position], str | None]],
    candidates: dict[str, Sequence[int]],
    fractions: dict[str, float],
    minimum: float,
    strict: bool,
    selection_modes: dict[str, str] | None = None,
) -> tuple[set[tuple[int, int, int, int]], dict[str, float], dict[str, float]]:
    """Rank raw losses, tie-break by rank then local batch/row/position."""
    errors = [error for _, _, error in ranks if error]
    if errors:
        raise ValueError("occurrence scoring failed: " + "; ".join(errors))
    selected: set[tuple[int, int, int, int]] = set()
    means: dict[str, float] = {}
    metrics: dict[str, float] = {}
    for domain, ids in candidates.items():
        count = sum(counts.get(domain, 0) for counts, _, _ in ranks)
        pool = [
            (rank, p)
            for rank, (_, positions, _) in enumerate(ranks)
            for p in positions
            if p.domain == domain
        ]
        if any(not math.isfinite(p.score) for _, p in pool):
            raise ValueError("occurrence requires finite raw RKL")
        counts = Counter(p.token_id for _, p in pool)
        allowed = set(ids)
        eligible = [
            (rank, p)
            for rank, p in pool
            if p.token_id in allowed
            and (
                counts[p.token_id] > minimum
                if strict
                else counts[p.token_id] >= minimum
            )
        ]
        mode = (selection_modes or {}).get(domain, "top_loss")
        if mode == "top_loss_teacher_confidence":
            from mopd_verl.domain_gradient.control_loss_teacher_confidence import (
                loss_teacher_confidence_type_scores,
            )

            if any(
                p.teacher_logp is None or not math.isfinite(p.teacher_logp)
                for _, p in pool
            ):
                raise ValueError("Loss+TC requires finite teacher chosen-token logp")
            losses = torch.tensor(
                [abs(p.score) for _, p in eligible], dtype=torch.float64
            )
            confidence = torch.tensor(
                [p.teacher_logp for _, p in eligible], dtype=torch.float64
            )
            # Each row is ONE occurrence: unit counts deliberately avoid ID aggregation.
            scores, normalization = loss_teacher_confidence_type_scores(
                loss_abs_sums=losses,
                teacher_logp_sums=confidence,
                counts=torch.ones_like(losses),
                normalization_mask=torch.ones_like(losses, dtype=torch.bool),
            )
            metrics.update(
                {
                    f"{domain}/occurrence/{key}": value
                    for key, value in normalization.items()
                }
            )
            ranked = sorted(
                zip(scores.tolist(), eligible),
                key=lambda item: (
                    -item[0],
                    item[1][0],
                    item[1][1].batch,
                    item[1][1].row,
                    item[1][1].column,
                ),
            )
            eligible = [position for _, position in ranked]
        elif mode == "top_loss":
            eligible.sort(
                key=lambda item: (
                    -item[1].score,
                    item[0],
                    item[1].batch,
                    item[1].row,
                    item[1].column,
                )
            )
        else:
            raise ValueError(f"Unsupported occurrence selection mode: {mode}")
        budget = top_p_target_occurrence_count(fractions[domain], count)
        chosen = eligible[:budget]
        selected.update((rank, p.batch, p.row, p.column) for rank, p in chosen)
        means[domain] = 1.0 + 3.0 * len(chosen) / count if count else 1.0
        prefix = f"{domain}/occurrence/"
        metrics.update(
            {
                prefix + "valid_count": float(count),
                prefix + "budget_count": float(budget),
                prefix + "eligible_count": float(len(eligible)),
                prefix + "selected_count": float(len(chosen)),
                prefix + "raw_weight_mean": means[domain],
            }
        )
    return selected, means, metrics


def prepare_occurrence_masks(
    audit: Any,
    micro_batches: Sequence[Any],
    loss_scales: Sequence[float],
    *,
    on_policy: bool,
    temperature: float,
) -> dict[str, float]:
    """One no-grad actor forward per microbatch, one actor-world gather."""
    from mopd_verl.domain_gradient.state import AuditState
    from mopd_verl.domain_gradient.token_weighting import aligned_response_token_ids
    from mopd_verl.full_gradient.actor_loss import build_actor_micro_batch_loss
    from mopd_verl.full_gradient.labels import _labels_from_mapping

    config = audit.config
    audit._occurrence_masks = {}
    if len(micro_batches) != len(loss_scales):
        raise ValueError("Each occurrence microbatch requires one loss scale")
    actor_config = audit.actor.config
    validate_occurrence_actor(actor_config)
    policy = (
        actor_config.get("policy_loss", {})
        if hasattr(actor_config, "get")
        else actor_config.policy_loss
    )
    validate_occurrence_config(config, policy)
    if getattr(audit.actor, "ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError("occurrence requires sequence parallel size 1")
    started = time.perf_counter()
    state = AuditState.capture(audit.actor)
    counts = {domain: 0 for domain in config.domains}
    positions: list[Position] = []
    templates: list[tuple[Any, torch.Tensor, list[str]]] = []
    error = None
    candidates = config.effective_domain_candidate_map()
    selection_modes = config.online_selection_mode_map()
    try:
        for index, (batch, scale) in enumerate(
            zip(micro_batches, loss_scales, strict=True)
        ):
            with torch.no_grad():
                result = build_actor_micro_batch_loss(
                    audit.actor,
                    batch,
                    loss_scale_factor=float(scale),
                    on_policy=on_policy,
                    include_metrics=False,
                    return_configured_token_loss=True,
                    temperature=temperature,
                )
            inputs = {**batch.batch, **batch.non_tensor_batch}
            mask = inputs["response_mask"].detach().bool().cpu()
            labels = _labels_from_mapping(inputs, mask.shape[0])
            raw, valid = result.selector_token_loss, result.selector_token_loss_mask
            ids = aligned_response_token_ids(inputs, inputs["response_mask"])
            if raw is None or valid is None or ids is None:
                error = "raw selector RKL, validity, and response IDs are required"
                continue
            raw, valid, ids = (
                raw.detach().float().cpu(),
                valid.detach().bool().cpu(),
                ids.cpu(),
            )
            if (
                raw.shape != mask.shape
                or valid.shape != mask.shape
                or not torch.equal(valid, mask)
            ):
                error = "selector validity must equal the full response mask"
                continue
            if not torch.isfinite(raw[mask]).all():
                error = "nonfinite raw RKL on a valid response position"
            teacher_logp = None
            tc_rows = torch.tensor(
                [
                    selection_modes.get(domain) == "top_loss_teacher_confidence"
                    for domain in labels
                ],
                dtype=torch.bool,
            )
            if tc_rows.any():
                from mopd_verl.full_gradient.loss_support import (
                    selected_teacher_log_prob,
                )

                try:
                    teacher_logp = (
                        selected_teacher_log_prob(inputs, policy)
                        .detach()
                        .double()
                        .cpu()
                    )
                    if teacher_logp.shape != mask.shape:
                        error = "teacher chosen-token logp must align with response positions"
                        continue
                    if not torch.isfinite(
                        teacher_logp[mask & tc_rows.unsqueeze(1)]
                    ).all():
                        error = "nonfinite teacher chosen-token logp on a valid Loss+TC position"
                except (KeyError, ValueError, RuntimeError) as exc:
                    error = f"teacher chosen-token logp unavailable: {exc}"
                    continue
            templates.append((batch, mask, labels))
            for row, domain in enumerate(labels):
                if domain not in counts:
                    error = f"unknown response domain {domain!r}"
                    continue
                counts[domain] += int(mask[row].sum())
                allowed = torch.tensor(candidates[domain], dtype=ids.dtype)
                eligible_columns = (
                    (mask[row] & torch.isin(ids[row], allowed)).nonzero().flatten()
                )
                for column in eligible_columns.tolist():
                    token_id = int(ids[row, column])
                    positions.append(
                        Position(
                            domain,
                            index,
                            row,
                            column,
                            token_id,
                            float(raw[row, column]),
                            float(teacher_logp[row, column])
                            if teacher_logp is not None
                            and selection_modes.get(domain)
                            == "top_loss_teacher_confidence"
                            else None,
                        )
                    )
    finally:
        state.restore()
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    ranks = (
        [None] * torch.distributed.get_world_size()
        if distributed
        else [(counts, positions, error)]
    )
    if distributed:
        torch.distributed.all_gather_object(ranks, (counts, positions, error))
    fractions = dict(config.control_token_online_top_p_by_domain)
    selected, means, metrics = select_positions(
        ranks,
        candidates,
        {
            d: fractions.get(d, config.control_token_online_top_p)
            for d in config.domains
        },
        config.control_token_online_min_mean_occurrences_per_step,
        config.control_token_online_strict_occurrence_gate,
        selection_modes=selection_modes,
    )
    rank = torch.distributed.get_rank() if distributed else 0
    for index, (batch, mask, labels) in enumerate(templates):
        weights = mask.float()
        denominators = torch.tensor([means[domain] for domain in labels]).unsqueeze(1)
        weights /= denominators
        local = [
            (row, column)
            for owner, batch_index, row, column in selected
            if owner == rank and batch_index == index
        ]
        if local:
            rows, columns = zip(*local)
            weights[list(rows), list(columns)] *= 4.0
        # Strong references prevent Python object-ID reuse until explicit retirement.
        audit._occurrence_masks[id(batch)] = (
            batch,
            weights.to(batch.batch["response_mask"].device),
        )
    metrics["global/occurrence/prepass_seconds"] = time.perf_counter() - started
    metrics["global/occurrence/prepass_forward_count"] = float(len(micro_batches))
    metrics["global/occurrence/gathered_candidates"] = float(
        sum(len(p) for _, p, _ in ranks)
    )
    return metrics


def occurrence_mask(audit: Any, batch: Any) -> torch.Tensor:
    entry = getattr(audit, "_occurrence_masks", {}).get(id(batch))
    if entry is None or entry[0] is not batch:
        raise RuntimeError("occurrence mask unavailable for this current-batch object")
    return entry[1]
