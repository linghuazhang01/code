"""Typed configuration for control-anchored Region-DPO rerollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any


REGION_DPO_SELECTION_STRATEGIES = {"first", "random", "uniform"}


@dataclass(frozen=True)
class RegionDPOConfig:
    """Configuration shared by rerollout acquisition and actor loss."""

    enabled: bool = False
    points_per_rollout: int = 1
    branches_per_point: int = 4
    max_new_tokens: int = 256
    beta: float = 0.1
    loss_weight: float = 0.1
    min_reward_margin: float = 0.0
    selection_strategy: str = "random"
    seed: int = 42
    control_token_ids: list[int] = field(default_factory=list)
    domain_control_token_ids: dict[str, list[int]] = field(
        default_factory=dict
    )


def _token_ids(value: Any, key: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Expected '{key}' to be a list of token IDs.")
    output = [int(token_id) for token_id in value]
    if any(token_id < 0 for token_id in output):
        raise ValueError(f"Expected '{key}' token IDs to be non-negative.")
    if len(output) != len(set(output)):
        raise ValueError(f"Expected '{key}' token IDs to be unique.")
    return output


def _domain_token_ids(value: Any) -> dict[str, list[int]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            "Expected 'region_dpo.domain_control_token_ids' to be a mapping."
        )
    output: dict[str, list[int]] = {}
    for domain, token_ids in value.items():
        key = f"region_dpo.domain_control_token_ids.{domain}"
        parsed = _token_ids(token_ids, key)
        if not parsed:
            raise ValueError(f"Expected '{key}' to be non-empty.")
        output[str(domain)] = parsed
    return output


def parse_region_dpo_config(value: Any) -> RegionDPOConfig:
    """Parse the optional top-level ``region_dpo`` mapping."""

    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("Expected 'region_dpo' to be a mapping.")
    return RegionDPOConfig(
        enabled=bool(value.get("enabled", RegionDPOConfig.enabled)),
        points_per_rollout=int(
            value.get(
                "points_per_rollout",
                RegionDPOConfig.points_per_rollout,
            )
        ),
        branches_per_point=int(
            value.get(
                "branches_per_point",
                RegionDPOConfig.branches_per_point,
            )
        ),
        max_new_tokens=int(
            value.get("max_new_tokens", RegionDPOConfig.max_new_tokens)
        ),
        beta=float(value.get("beta", RegionDPOConfig.beta)),
        loss_weight=float(
            value.get("loss_weight", RegionDPOConfig.loss_weight)
        ),
        min_reward_margin=float(
            value.get(
                "min_reward_margin",
                RegionDPOConfig.min_reward_margin,
            )
        ),
        selection_strategy=str(
            value.get(
                "selection_strategy",
                RegionDPOConfig.selection_strategy,
            )
        ).strip().lower(),
        seed=int(value.get("seed", RegionDPOConfig.seed)),
        control_token_ids=_token_ids(
            value.get("control_token_ids"),
            "region_dpo.control_token_ids",
        ),
        domain_control_token_ids=_domain_token_ids(
            value.get("domain_control_token_ids")
        ),
    )


def with_control_token_fallback(
    config: RegionDPOConfig,
    *,
    control_token_ids: list[int],
    domain_control_token_ids: dict[str, list[int]],
) -> RegionDPOConfig:
    """Reuse the frozen audit taxonomy when Region-DPO omits its own IDs."""

    if config.control_token_ids or config.domain_control_token_ids:
        return config
    return replace(
        config,
        control_token_ids=list(control_token_ids),
        domain_control_token_ids={
            domain: list(token_ids)
            for domain, token_ids in domain_control_token_ids.items()
        },
    )


def validate_region_dpo_config(
    config: RegionDPOConfig,
    *,
    max_response_length: int,
) -> None:
    """Validate acquisition counts, DPO coefficients, and anchor support."""

    if not config.enabled:
        return
    if config.points_per_rollout < 1:
        raise ValueError("region_dpo.points_per_rollout must be positive.")
    if config.branches_per_point < 2:
        raise ValueError(
            "region_dpo.branches_per_point must be at least 2."
        )
    if config.max_new_tokens < 1:
        raise ValueError("region_dpo.max_new_tokens must be positive.")
    if config.max_new_tokens > int(max_response_length):
        raise ValueError(
            "region_dpo.max_new_tokens must not exceed "
            "data.max_response_length."
        )
    if not math.isfinite(config.beta) or config.beta <= 0.0:
        raise ValueError("region_dpo.beta must be finite and positive.")
    if not math.isfinite(config.loss_weight) or config.loss_weight <= 0.0:
        raise ValueError(
            "region_dpo.loss_weight must be finite and positive."
        )
    if (
        not math.isfinite(config.min_reward_margin)
        or config.min_reward_margin < 0.0
    ):
        raise ValueError(
            "region_dpo.min_reward_margin must be finite and non-negative."
        )
    if config.selection_strategy not in REGION_DPO_SELECTION_STRATEGIES:
        supported = ", ".join(sorted(REGION_DPO_SELECTION_STRATEGIES))
        raise ValueError(
            "region_dpo.selection_strategy must be one of: " + supported
        )
    if config.seed < 0:
        raise ValueError("region_dpo.seed must be non-negative.")
    if config.enabled and not (
        config.control_token_ids or config.domain_control_token_ids
    ):
        raise ValueError(
            "Enabled Region-DPO requires region_dpo control-token IDs or "
            "the frozen audit control-token taxonomy."
        )
