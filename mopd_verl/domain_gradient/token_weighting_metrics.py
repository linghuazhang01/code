"""Pure metric helpers for token-level backward amplification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from verl.utils.device import get_device_id

from mopd_verl.domain_gradient.token_logging import LocalTokenCandidate

AMPLIFICATION_SUM_NAMES = (
    "occurrence_count",
    "amplified_occurrence_count",
    "control_occurrence_count",
    "shared_occurrence_count",
    "control_shared_overlap_occurrence_count",
    "token_multiplier_sum",
    "effective_multiplier_sum",
    "gradient_multiplier_abs_error_sum",
    "raw_configured_loss_mass",
    "effective_configured_loss_mass",
    "raw_configured_loss_abs_mass",
    "token_weighted_configured_loss_abs_mass",
    "effective_configured_loss_abs_mass",
)


def local_loss_amplification_statistics(
    candidates_by_domain: Mapping[
        str,
        Sequence[LocalTokenCandidate],
    ],
    domains: Sequence[str],
    actual_masks: Sequence[torch.Tensor],
    *,
    domain_weights: Mapping[str, float],
    dynamic_weighting_enabled: bool,
    control_token_ids: Sequence[int],
    control_weighting_enabled: bool,
    control_weight: float,
    shared_token_ids: Sequence[int],
    shared_weighting_enabled: bool,
    shared_weight: float,
) -> dict[str, dict[str, float]]:
    """Summarize raw loss mass and the exact production gradient masks."""

    control_ids = set(control_token_ids)
    shared_ids = set(shared_token_ids)
    local_by_domain = {
        domain: {name: 0.0 for name in AMPLIFICATION_SUM_NAMES}
        for domain in domains
    }
    for domain in domains:
        domain_factor = (
            domain_weights.get(domain, 1.0)
            if dynamic_weighting_enabled
            else 1.0
        )
        for candidate in candidates_by_domain.get(domain, ()):
            token_id = candidate.token_id
            control_match = (
                control_weighting_enabled and token_id in control_ids
            )
            shared_match = (
                shared_weighting_enabled and token_id in shared_ids
            )
            token_factor = (
                control_weight if control_match else 1.0
            ) * (
                shared_weight if shared_match else 1.0
            )
            expected_effective_factor = domain_factor * token_factor
            effective_factor = float(
                actual_masks[candidate.micro_batch_index][
                    candidate.sample_index,
                    candidate.token_index,
                ].item()
            )
            stats = local_by_domain[domain]
            stats["occurrence_count"] += 1.0
            stats["amplified_occurrence_count"] += float(
                token_factor > 1.0 + 1e-12
            )
            stats["control_occurrence_count"] += float(control_match)
            stats["shared_occurrence_count"] += float(shared_match)
            stats["control_shared_overlap_occurrence_count"] += float(
                control_match and shared_match
            )
            stats["token_multiplier_sum"] += token_factor
            stats["effective_multiplier_sum"] += effective_factor
            stats["gradient_multiplier_abs_error_sum"] += abs(
                effective_factor - expected_effective_factor
            )
            stats["raw_configured_loss_mass"] += float(
                candidate.configured_loss
            )
            stats["effective_configured_loss_mass"] += (
                float(candidate.configured_loss) * effective_factor
            )
            stats["raw_configured_loss_abs_mass"] += float(
                candidate.loss_abs
            )
            stats["token_weighted_configured_loss_abs_mass"] += (
                float(candidate.loss_abs) * token_factor
            )
            stats["effective_configured_loss_abs_mass"] += (
                float(candidate.loss_abs) * effective_factor
            )
    return local_by_domain


def format_loss_amplification_metrics(
    reduced_by_domain: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Format globally reduced additive statistics as monitoring metrics."""

    global_stats = {name: 0.0 for name in AMPLIFICATION_SUM_NAMES}
    metrics: dict[str, float] = {}
    for domain, stats in reduced_by_domain.items():
        for name, value in stats.items():
            global_stats[name] += float(value)
        count = stats["occurrence_count"]
        raw_abs = stats["raw_configured_loss_abs_mass"]
        prefix = f"{domain}/token_weight"
        metrics[f"{prefix}/configured_token_occurrence_count"] = count
        metrics[f"{prefix}/amplified_token_occurrence_fraction"] = (
            stats["amplified_occurrence_count"] / count
            if count > 0.0
            else 0.0
        )
        metrics[f"{prefix}/raw_configured_loss_abs_mass"] = raw_abs
        metrics[
            f"{prefix}/effective_configured_loss_abs_mass"
        ] = stats["effective_configured_loss_abs_mass"]
        metrics[
            f"{prefix}/effective_to_raw_abs_loss_mass_ratio"
        ] = (
            stats["effective_configured_loss_abs_mass"] / raw_abs
            if raw_abs > 0.0
            else 1.0
        )

    count = global_stats["occurrence_count"]
    raw_abs = global_stats["raw_configured_loss_abs_mass"]
    prefix = "global/token_weight"
    for name in (
        "occurrence_count",
        "amplified_occurrence_count",
        "control_occurrence_count",
        "shared_occurrence_count",
        "control_shared_overlap_occurrence_count",
        "raw_configured_loss_mass",
        "effective_configured_loss_mass",
        "raw_configured_loss_abs_mass",
        "token_weighted_configured_loss_abs_mass",
        "effective_configured_loss_abs_mass",
        "gradient_multiplier_abs_error_sum",
    ):
        metrics[f"{prefix}/{name}"] = global_stats[name]
    metrics[f"{prefix}/amplified_token_occurrence_fraction"] = (
        global_stats["amplified_occurrence_count"] / count
        if count > 0.0
        else 0.0
    )
    metrics[f"{prefix}/mean_token_gradient_multiplier"] = (
        global_stats["token_multiplier_sum"] / count
        if count > 0.0
        else 1.0
    )
    metrics[f"{prefix}/mean_effective_gradient_multiplier"] = (
        global_stats["effective_multiplier_sum"] / count
        if count > 0.0
        else 1.0
    )
    metrics[f"{prefix}/gradient_multiplier_mean_abs_error"] = (
        global_stats["gradient_multiplier_abs_error_sum"] / count
        if count > 0.0
        else 0.0
    )
    metrics[f"{prefix}/token_weighted_to_raw_abs_loss_mass_ratio"] = (
        global_stats["token_weighted_configured_loss_abs_mass"] / raw_abs
        if raw_abs > 0.0
        else 1.0
    )
    metrics[f"{prefix}/effective_to_raw_abs_loss_mass_ratio"] = (
        global_stats["effective_configured_loss_abs_mass"] / raw_abs
        if raw_abs > 0.0
        else 1.0
    )
    return metrics


def reduce_loss_amplification_statistics(
    local_by_domain: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """All-reduce every additive statistic in one collective."""

    domains = tuple(local_by_domain)
    values = [
        float(local_by_domain[domain][name])
        for domain in domains
        for name in AMPLIFICATION_SUM_NAMES
    ]
    if not values:
        return {}
    tensor = torch.tensor(
        values,
        device=get_device_id(),
        dtype=torch.float64,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.SUM,
        )
    reduced_values = iter(float(value) for value in tensor.tolist())
    return {
        domain: {
            name: next(reduced_values)
            for name in AMPLIFICATION_SUM_NAMES
        }
        for domain in domains
    }
