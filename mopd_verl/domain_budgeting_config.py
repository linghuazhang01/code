"""Typed configuration and fail-fast contracts for domain budgeting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class DomainBudgetingConfig:
    """Capability-need objective weights with variance-aware sampling."""

    enabled: bool = False
    domains: list[str] = field(default_factory=list)
    domain_priors: dict[str, float] = field(default_factory=dict)
    teacher_scores: dict[str, float] = field(default_factory=dict)
    teacher_scores_calibrated: bool = False
    validation_metric_keys: dict[str, list[str]] = field(default_factory=dict)
    validation_reducer: str = "mean"
    gap_ema_beta: float = 0.9
    gap_alpha: float = 1.0
    gap_epsilon: float = 1e-6
    gap_normalization_floor: float = 0.05
    max_normalized_gap: float = 2.0
    exploration_mass: float = 0.05
    variance_log_ema_beta: float = 0.9
    variance_epsilon: float = 1e-8
    variance_min_samples: int = 2
    variance_update_freq_steps: int = 1
    min_samples_per_domain: int = 2
    min_sampling_probability: float = 0.02
    output_dir: str = "domain_budgeting"


def _string_list(value: Any, key: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(f"Expected '{key}' to be a string or a list of strings.")


def _positive_float_mapping(value: Any, key: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    output: dict[str, float] = {}
    for item_key, item_value in value.items():
        numeric = float(item_value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"Expected '{key}.{item_key}' to be positive and finite.")
        output[str(item_key)] = numeric
    return output


def _finite_float_mapping(value: Any, key: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    output: dict[str, float] = {}
    for item_key, item_value in value.items():
        numeric = float(item_value)
        if not math.isfinite(numeric):
            raise ValueError(f"Expected '{key}.{item_key}' to be finite.")
        output[str(item_key)] = numeric
    return output


def _string_list_mapping(value: Any, key: str) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    return {
        str(item_key): _string_list(item_value, f"{key}.{item_key}")
        for item_key, item_value in value.items()
    }


def parse_domain_budgeting_config(raw: dict[str, Any]) -> DomainBudgetingConfig:
    """Parse the optional top-level domain_budgeting mapping."""

    allowed_keys = {item.name for item in fields(DomainBudgetingConfig)}
    unknown_keys = sorted(set(raw) - allowed_keys)
    if unknown_keys:
        raise ValueError("Unknown domain_budgeting config keys: " f"{unknown_keys}.")

    return DomainBudgetingConfig(
        enabled=bool(raw.get("enabled", DomainBudgetingConfig.enabled)),
        domains=_string_list(raw.get("domains", []), "domain_budgeting.domains"),
        domain_priors=_positive_float_mapping(
            raw.get("domain_priors"), "domain_budgeting.domain_priors"
        ),
        teacher_scores=_finite_float_mapping(
            raw.get("teacher_scores"), "domain_budgeting.teacher_scores"
        ),
        teacher_scores_calibrated=bool(
            raw.get(
                "teacher_scores_calibrated",
                DomainBudgetingConfig.teacher_scores_calibrated,
            )
        ),
        validation_metric_keys=_string_list_mapping(
            raw.get("validation_metric_keys"),
            "domain_budgeting.validation_metric_keys",
        ),
        validation_reducer=str(
            raw.get("validation_reducer", DomainBudgetingConfig.validation_reducer)
        ),
        gap_ema_beta=float(raw.get("gap_ema_beta", DomainBudgetingConfig.gap_ema_beta)),
        gap_alpha=float(raw.get("gap_alpha", DomainBudgetingConfig.gap_alpha)),
        gap_epsilon=float(raw.get("gap_epsilon", DomainBudgetingConfig.gap_epsilon)),
        gap_normalization_floor=float(
            raw.get(
                "gap_normalization_floor",
                DomainBudgetingConfig.gap_normalization_floor,
            )
        ),
        max_normalized_gap=float(
            raw.get("max_normalized_gap", DomainBudgetingConfig.max_normalized_gap)
        ),
        exploration_mass=float(
            raw.get("exploration_mass", DomainBudgetingConfig.exploration_mass)
        ),
        variance_log_ema_beta=float(
            raw.get(
                "variance_log_ema_beta",
                DomainBudgetingConfig.variance_log_ema_beta,
            )
        ),
        variance_epsilon=float(
            raw.get("variance_epsilon", DomainBudgetingConfig.variance_epsilon)
        ),
        variance_min_samples=int(
            raw.get("variance_min_samples", DomainBudgetingConfig.variance_min_samples)
        ),
        variance_update_freq_steps=int(
            raw.get(
                "variance_update_freq_steps",
                DomainBudgetingConfig.variance_update_freq_steps,
            )
        ),
        min_samples_per_domain=int(
            raw.get(
                "min_samples_per_domain",
                DomainBudgetingConfig.min_samples_per_domain,
            )
        ),
        min_sampling_probability=float(
            raw.get(
                "min_sampling_probability",
                DomainBudgetingConfig.min_sampling_probability,
            )
        ),
        output_dir=str(raw.get("output_dir", DomainBudgetingConfig.output_dir)),
    )


def validate_domain_budgeting_config(
    config: DomainBudgetingConfig,
    *,
    data: Any,
    model: Any,
    actor: Any,
    rollout: Any,
    audit: Any,
    trainer: Any,
) -> None:
    """Validate the sample-level objective and causal runtime assumptions."""

    if not config.enabled:
        return
    configured_domains = set(config.domains)
    data_domains = set(data.domain_train_files)
    if (
        not configured_domains
        or len(configured_domains) != len(config.domains)
        or configured_domains != data_domains
    ):
        raise ValueError(
            "domain_budgeting.domains must exactly match data.domain_train_files."
        )
    for key, values in (
        ("domain_priors", config.domain_priors),
        ("teacher_scores", config.teacher_scores),
        ("validation_metric_keys", config.validation_metric_keys),
    ):
        if values and set(values) != configured_domains:
            raise ValueError(
                f"domain_budgeting.{key} must exactly cover all configured domains."
            )
    if not config.teacher_scores or not config.validation_metric_keys:
        raise ValueError(
            "domain_budgeting teacher_scores and validation_metric_keys are required."
        )
    missing_teacher_domains = configured_domains - set(model.domain_teacher_paths)
    if missing_teacher_domains:
        raise ValueError(
            "model.domain_teacher_paths must cover every dynamic domain; missing "
            f"{sorted(missing_teacher_domains)}."
        )
    if any(not keys for keys in config.validation_metric_keys.values()):
        raise ValueError(
            "Every domain_budgeting.validation_metric_keys entry must be non-empty."
        )
    if (
        data.domain_sampling_weights
        and set(data.domain_sampling_weights) != configured_domains
    ):
        raise ValueError(
            "data.domain_sampling_weights must exactly cover dynamic domains."
        )
    if not data.domain_sampling_replacement:
        raise ValueError("Dynamic domain budgeting requires replacement sampling.")
    if data.dataloader_num_workers != 0:
        raise ValueError(
            "Dynamic domain budgeting requires data.dataloader_num_workers=0 "
            "to avoid stale prefetched allocations."
        )
    if actor.loss_agg_mode != "seq-mean-token-mean":
        raise ValueError(
            "Dynamic domain budgeting requires actor.loss_agg_mode=seq-mean-token-mean."
        )
    topk_objective_enabled = (
        actor.topk_distill_enabled
        or actor.distill_loss_builder == "topk_kl"
        or actor.distill_mode.startswith("topk_")
    )
    if (
        not topk_objective_enabled
        or not math.isfinite(actor.topk_distill_loss_weight)
        or actor.topk_distill_loss_weight <= 0.0
    ):
        raise ValueError(
            "Dynamic domain budgeting requires a positive top-k OPD objective."
        )
    if actor.teacher_prefix_enabled or rollout.teacher_prefix_sampling_enabled:
        raise ValueError(
            "Dynamic domain budgeting does not support teacher-prefix training."
        )
    if actor.ppo_epochs != 1 or actor.ppo_mini_batch_size != data.train_batch_size:
        raise ValueError(
            "Dynamic domain budgeting requires one PPO epoch and one optimizer "
            "mini-batch per actor batch."
        )
    if rollout.n != 1:
        raise ValueError("Dynamic domain budgeting currently requires rollout.n=1.")
    if actor.entropy_coeff != 0 or actor.kl_loss_coef != 0:
        raise ValueError(
            "Dynamic domain budgeting currently requires zero actor entropy "
            "and KL auxiliary coefficients."
        )
    if audit.dynamic_domain_loss_weighting_enabled:
        raise ValueError(
            "domain_budgeting and audit.dynamic_domain_loss_weighting_enabled "
            "cannot be enabled together."
        )
    if not trainer.val_before_train or trainer.test_freq <= 0:
        raise ValueError(
            "Dynamic capability gaps require trainer.val_before_train=true "
            "and trainer.test_freq>0."
        )
    if config.validation_reducer != "mean":
        raise ValueError("Only domain_budgeting.validation_reducer=mean is supported.")
    for key, beta in (
        ("gap_ema_beta", config.gap_ema_beta),
        ("variance_log_ema_beta", config.variance_log_ema_beta),
    ):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"domain_budgeting.{key} must be in [0, 1).")
    if config.gap_alpha < 0.0:
        raise ValueError("domain_budgeting.gap_alpha must be non-negative.")
    finite_scalars = {
        "gap_ema_beta": config.gap_ema_beta,
        "gap_alpha": config.gap_alpha,
        "gap_epsilon": config.gap_epsilon,
        "gap_normalization_floor": config.gap_normalization_floor,
        "max_normalized_gap": config.max_normalized_gap,
        "exploration_mass": config.exploration_mass,
        "variance_log_ema_beta": config.variance_log_ema_beta,
        "variance_epsilon": config.variance_epsilon,
        "min_sampling_probability": config.min_sampling_probability,
    }
    non_finite = [
        key for key, value in finite_scalars.items() if not math.isfinite(value)
    ]
    if non_finite:
        raise ValueError(f"Non-finite domain_budgeting values: {sorted(non_finite)}.")
    if (
        config.gap_epsilon <= 0.0
        or config.gap_normalization_floor <= 0.0
        or config.variance_epsilon <= 0.0
        or config.max_normalized_gap <= 0.0
    ):
        raise ValueError("Domain budgeting epsilons and gap cap must be positive.")
    if not 0.0 <= config.exploration_mass < 1.0:
        raise ValueError("domain_budgeting.exploration_mass must be in [0, 1).")
    domain_count = len(config.domains)
    if not 0.0 <= config.min_sampling_probability < 1.0 / domain_count:
        raise ValueError(
            "domain_budgeting.min_sampling_probability must be in "
            "[0, 1 / domain_count)."
        )
    if (
        config.variance_min_samples < 2
        or config.min_samples_per_domain < 1
        or config.min_samples_per_domain < config.variance_min_samples
        or config.variance_update_freq_steps < 1
    ):
        raise ValueError(
            "Domain budgeting sample counts and update frequency are invalid."
        )
    if data.train_batch_size < domain_count * config.min_samples_per_domain:
        raise ValueError(
            "data.train_batch_size must be at least domain_count * "
            "domain_budgeting.min_samples_per_domain."
        )
