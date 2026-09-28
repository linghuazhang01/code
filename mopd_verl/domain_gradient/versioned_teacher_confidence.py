"""Lagged shared C+S Teacher Confidence from production batches."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from mopd_verl.domain_gradient.control_top_loss_runtime import (
    GlobalCandidateLossStatistics,
)
from mopd_verl.domain_gradient.versioned_position_eligibility import (
    versioned_position_eligibility,
)
from mopd_verl.domain_gradient.token_taxonomy_registry import (
    normalize_token_taxonomy_version,
    token_taxonomy,
)
from mopd_verl.domain_gradient.token_weighting import aligned_response_token_ids
from mopd_verl.full_gradient.labels import _labels_from_mapping


def _collective_device(micro_batches: Sequence[Any]) -> torch.device:
    if micro_batches:
        return micro_batches[0].batch["response_mask"].device
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_backend() == "nccl"
    ):
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


@torch.no_grad()
def next_step_teacher_confidence_statistics(
    config: Any,
    micro_batches: Sequence[Any],
    teacher_log_prob_batches: Sequence[torch.Tensor] | None,
) -> GlobalCandidateLossStatistics:
    """Aggregate frozen C+S by chosen-token confidence across ranks.

    C and S use the same position eligibility as production TopLoss and weight
    application. The shared Top-P denominator is all valid response tokens.
    ``1 / (1 - mean_logp)`` preserves ranking with a non-negative score.
    """

    domains = tuple(str(domain) for domain in config.domains)
    modes = config.versioned_cs_selection_mode_map()
    confidence_domains = tuple(
        domain for domain in domains
        if modes[domain] == "top_teacher_confidence"
    )
    statistics: dict[str, dict[int, tuple[float, int]]] = {
        domain: {} for domain in domains
    }
    counts = {domain: 0 for domain in domains}
    score_sums = {domain: 0.0 for domain in domains}
    if not confidence_domains:
        return GlobalCandidateLossStatistics(statistics, counts, score_sums)

    version = normalize_token_taxonomy_version(config.token_taxonomy_version)
    frozen = token_taxonomy(version)
    structures = config.effective_domain_structure_map()
    controls = config.effective_domain_candidate_map()
    for domain in confidence_domains:
        configured_structure = tuple(int(token_id) for token_id in structures[domain])
        configured_control = tuple(int(token_id) for token_id in controls[domain])
        if len(configured_structure) != len(set(configured_structure)) or set(
            configured_structure
        ) != set(
            frozen[domain].structure
        ):
            raise ValueError(
                f"{version}/{domain} Structure IDs must match the frozen taxonomy."
            )
        if len(configured_control) != len(set(configured_control)) or set(
            configured_control
        ) != set(frozen[domain].control):
            raise ValueError(
                f"{version}/{domain} Control IDs must match the frozen taxonomy."
            )

    candidate_ids = tuple(sorted({
        token_id for domain in confidence_domains
        for token_id in (*controls[domain], *structures[domain])
    }))
    device = _collective_device(micro_batches)
    candidates = torch.tensor(candidate_ids, device=device, dtype=torch.long)
    allowed = {
        domain: torch.tensor(
            (*controls[domain], *structures[domain]),
            device=device, dtype=torch.long,
        )
        for domain in confidence_domains
    }
    control_tensors = {
        domain: torch.tensor(controls[domain], device=device, dtype=torch.long)
        for domain in confidence_domains
    }
    # Keep candidate logp sums and counts separate from synchronized errors.
    packed = torch.zeros(
        (len(confidence_domains), 2, len(candidate_ids)),
        device=device,
        dtype=torch.float64,
    )
    valid_counts = torch.zeros(len(confidence_domains), device=device, dtype=torch.float64)
    errors = torch.zeros(4, device=device, dtype=torch.float64)
    if (
        teacher_log_prob_batches is None
        or len(teacher_log_prob_batches) != len(micro_batches)
        or not micro_batches
    ):
        errors[0] = 1
    else:
        for batch, source_logp in zip(
            micro_batches, teacher_log_prob_batches, strict=True
        ):
            try:
                inputs = {**batch.batch, **batch.non_tensor_batch}
                valid = inputs["response_mask"].detach().bool().to(device)
                ids = aligned_response_token_ids(inputs, inputs["response_mask"])
                labels = _labels_from_mapping(inputs, int(valid.shape[0]))
                if (
                    valid.ndim != 2
                    or ids is None
                    or ids.shape != valid.shape
                    or source_logp.shape != valid.shape
                    or set(labels) - set(domains)
                ):
                    errors[1] += 1
                    continue
                ids = ids.detach().to(device=device, dtype=torch.long)
                logp = source_logp.detach().to(device=device, dtype=torch.float64)
                control_eligible, structure_eligible = (
                    versioned_position_eligibility(
                        batch, config, valid, labels,
                        active_domains=confidence_domains,
                        include_fixed_structure=False,
                    )
                )
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                errors[1] += 1
                continue

            for index, domain in enumerate(confidence_domains):
                rows = torch.tensor(
                    [label == domain for label in labels],
                    device=device,
                    dtype=torch.bool,
                ).unsqueeze(-1)
                domain_valid = valid & rows
                valid_counts[index] += domain_valid.sum()
                errors[2] += (~torch.isfinite(logp) & domain_valid).sum()
                errors[3] += ((logp > 1e-6) & domain_valid).sum()
                eligible = (
                    domain_valid & torch.isin(ids, allowed[domain])
                    & torch.where(
                        torch.isin(ids, control_tensors[domain]),
                        control_eligible,
                        structure_eligible,
                    )
                )
                if not eligible.any():
                    continue
                positions = torch.searchsorted(candidates, ids[eligible])
                values = logp[eligible].clamp(max=0.0)
                packed[index, 0].scatter_add_(0, positions, values)
                packed[index, 1].scatter_add_(
                    0, positions, torch.ones_like(values)
                )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(errors, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(valid_counts, op=torch.distributed.ReduceOp.SUM)
    if errors[0] > 0:
        raise ValueError(
            "C+S teacher-confidence selection requires one aligned "
            "teacher chosen-token logp batch per production microbatch."
        )
    if errors[1] > 0:
        raise ValueError(
            "C+S teacher-confidence source IDs, labels, logp, and "
            "response masks must align."
        )
    if errors[2] > 0 or errors[3] > 0:
        raise ValueError(
            "C+S teacher-confidence requires finite chosen-token "
            "logp <= 1e-6 on valid response positions."
        )

    reduced = packed.cpu()
    for index, domain in enumerate(confidence_domains):
        domain_ids = set(int(token_id) for token_id in (
            *controls[domain], *structures[domain],
        ))
        counts[domain] = int(valid_counts[index])
        for candidate_index, token_id in enumerate(candidate_ids):
            count = int(reduced[index, 1, candidate_index])
            if token_id not in domain_ids or count == 0:
                continue
            mean_logp = float(reduced[index, 0, candidate_index]) / count
            score = 1.0 / (1.0 - mean_logp)
            if not math.isfinite(score) or score <= 0.0:
                raise ValueError(
                    "C+S teacher-confidence aggregate must remain "
                    "finite and positive."
                )
            statistics[domain][token_id] = (score * count, count)
        score_sums[domain] = sum(
            score_sum for score_sum, _ in statistics[domain].values()
        )
    return GlobalCandidateLossStatistics(statistics, counts, score_sums)
