"""Metrics and checkpoint support for the dynamic domain budget controller."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from mopd_verl.domain_budgeting_state import (
    DomainBudgetState,
    validate_domain_budget_state,
)


def config_get(config: Any, key: str, default: Any = None) -> Any:
    """Read one value from a mapping, OmegaConf object, or dataclass."""

    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def plain_mapping(value: Any) -> dict[str, Any]:
    """Convert mapping-like config values into a plain string-keyed dict."""

    if value is None:
        return {}
    if not hasattr(value, "items"):
        raise ValueError("Expected a domain mapping.")
    return {str(key): item for key, item in value.items()}


def validate_runtime_controller(controller: Any) -> None:
    """Recheck Hydra-overridable controller settings inside the trainer."""

    if not controller.enabled:
        return
    if not controller.domains or len(set(controller.domains)) != len(
        controller.domains
    ):
        raise ValueError("Domain budgeting requires unique configured domains.")
    if not controller.teacher_scores_calibrated:
        raise ValueError(
            "Dynamic domain budgeting requires calibrated teacher scores; "
            "replace the placeholders and set teacher_scores_calibrated=true."
        )
    for mapping_name, mapping in (
        ("teacher_scores", controller.teacher_scores),
        ("validation_metric_keys", controller.validation_metric_keys),
    ):
        if set(mapping) != set(controller.domains):
            raise ValueError(
                f"domain_budgeting.{mapping_name} must exactly cover "
                f"{controller.domains}."
            )
    if any(not keys for keys in controller.validation_metric_keys.values()):
        raise ValueError("Each domain must configure a validation metric key.")
    if any(not math.isfinite(value) for value in controller.teacher_scores.values()):
        raise ValueError("Teacher scores must be finite.")
    if controller.metric_reducer != "mean":
        raise ValueError("Only domain_budgeting.validation_reducer=mean is supported.")
    for label, beta in (
        ("gap_ema_beta", controller.gap_ema_beta),
        ("variance_log_ema_beta", controller.variance_log_ema_beta),
    ):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"{label} must be in [0, 1).")
    finite_values = (
        controller.gap_alpha,
        controller.gap_epsilon,
        controller.gap_normalization_floor,
        controller.variance_epsilon,
        controller.max_normalized_gap,
        controller.exploration_mass,
        controller.min_sampling_probability,
    )
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("Domain budgeting scalar values must be finite.")
    if (
        controller.gap_alpha < 0.0
        or controller.gap_epsilon <= 0.0
        or controller.gap_normalization_floor <= 0.0
        or controller.variance_epsilon <= 0.0
        or controller.max_normalized_gap <= 0.0
    ):
        raise ValueError("Domain budgeting exponents, epsilons, or caps are invalid.")
    if not 0.0 <= controller.exploration_mass < 1.0:
        raise ValueError("exploration_mass must be in [0, 1).")
    if not (0.0 <= controller.min_sampling_probability < 1.0 / len(controller.domains)):
        raise ValueError("min_sampling_probability must be in [0, 1 / domain_count).")
    if (
        controller.variance_min_samples < 2
        or controller.min_samples_per_domain < controller.variance_min_samples
        or controller.variance_update_freq_steps < 1
    ):
        raise ValueError(
            "Domain sample minimums must be positive and variance needs at "
            "least two samples."
        )


class DomainBudgetControllerSupport:
    """Shared non-update methods kept separate from the controller hot path."""

    enabled: bool
    domains: tuple[str, ...]
    priors: dict[str, float]
    teacher_scores: dict[str, float]
    teacher_scores_calibrated: bool
    validation_metric_keys: dict[str, list[str]]
    metric_reducer: str
    gap_ema_beta: float
    gap_alpha: float
    gap_epsilon: float
    gap_normalization_floor: float
    max_normalized_gap: float
    exploration_mass: float
    variance_log_ema_beta: float
    variance_epsilon: float
    variance_min_samples: int
    variance_update_freq_steps: int
    min_samples_per_domain: int
    min_sampling_probability: float
    state: DomainBudgetState

    def metrics(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        output: dict[str, float] = {}
        for domain in self.domains:
            prefix = f"domain_budgeting/{domain}"
            q_value = self.state.target_contributions.get(domain, self.priors[domain])
            observed = self.state.observed_sampling.get(domain)
            effective = self.state.effective_sampling.get(domain)
            valid_fraction = self.state.valid_fractions.get(domain)
            scale = self.state.loss_scales.get(domain)
            output[f"{prefix}/q"] = q_value
            output[f"{prefix}/desired_p"] = self.state.desired_sampling.get(
                domain, q_value
            )
            if domain in self.state.student_scores:
                output[f"{prefix}/student_score"] = self.state.student_scores[domain]
                output[f"{prefix}/capability_gap"] = self.state.gap_ema[domain]
                output[f"{prefix}/normalized_gap"] = self.state.normalized_gaps[domain]
            if domain in self.state.raw_variances:
                output[f"{prefix}/sequence_loss_variance"] = self.state.raw_variances[
                    domain
                ]
                output[f"{prefix}/smoothed_variance"] = math.exp(
                    self.state.log_variance_ema[domain]
                )
            if (
                observed is not None
                and effective is not None
                and valid_fraction is not None
                and scale is not None
            ):
                output[f"{prefix}/observed_p"] = observed
                output[f"{prefix}/active_p"] = effective
                output[f"{prefix}/empty_response_rate"] = 1.0 - float(valid_fraction)
                output[f"{prefix}/lambda"] = scale
                output[f"{prefix}/closure_error"] = abs(effective * scale - q_value)
        return output

    def next_allocation_metrics(self) -> dict[str, float]:
        """Report a validation update without overwriting applied-batch metrics."""

        output = self.metrics()
        for domain in self.domains:
            prefix = f"domain_budgeting/{domain}"
            output[f"{prefix}/next_q"] = output.pop(f"{prefix}/q")
            output[f"{prefix}/next_desired_p"] = output.pop(f"{prefix}/desired_p")
            for metric in (
                "observed_p",
                "active_p",
                "empty_response_rate",
                "lambda",
                "closure_error",
            ):
                output.pop(f"{prefix}/{metric}", None)
        return output

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller_schema_version": 1,
            "controller_spec": self._controller_spec(),
            "state": asdict(self.state),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if int(payload.get("controller_schema_version", -1)) != 1:
            raise ValueError(
                "Unsupported domain budgeting controller checkpoint schema."
            )
        restored_spec = payload.get("controller_spec")
        expected_spec = self._controller_spec()
        if restored_spec != expected_spec:
            restored = restored_spec if isinstance(restored_spec, Mapping) else {}
            mismatches = sorted(
                key
                for key in set(expected_spec) | set(restored)
                if restored.get(key) != expected_spec.get(key)
            )
            raise ValueError(
                "Domain budgeting checkpoint config does not match current "
                f"controller semantics: {mismatches}."
            )
        raw_state = payload.get("state")
        if not isinstance(raw_state, Mapping):
            raise ValueError("Domain budgeting checkpoint is missing controller state.")
        state_payload = dict(raw_state)
        if int(state_payload.get("schema_version", -1)) != 2:
            raise ValueError("Unsupported domain budgeting state schema.")
        restored_state = DomainBudgetState(**state_payload)
        validate_domain_budget_state(restored_state, self.domains)
        self.state = restored_state

    def _controller_spec(self) -> dict[str, Any]:
        return {
            "domains": list(self.domains),
            "domain_priors": dict(self.priors),
            "teacher_scores": dict(self.teacher_scores),
            "teacher_scores_calibrated": self.teacher_scores_calibrated,
            "validation_metric_keys": dict(self.validation_metric_keys),
            "validation_reducer": self.metric_reducer,
            "gap_ema_beta": self.gap_ema_beta,
            "gap_alpha": self.gap_alpha,
            "gap_epsilon": self.gap_epsilon,
            "gap_normalization_floor": self.gap_normalization_floor,
            "max_normalized_gap": self.max_normalized_gap,
            "exploration_mass": self.exploration_mass,
            "variance_log_ema_beta": self.variance_log_ema_beta,
            "variance_epsilon": self.variance_epsilon,
            "variance_min_samples": self.variance_min_samples,
            "variance_update_freq_steps": self.variance_update_freq_steps,
            "min_samples_per_domain": self.min_samples_per_domain,
            "min_sampling_probability": self.min_sampling_probability,
        }
