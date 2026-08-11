"""Driver-side capability-gap and variance-aware domain budgeting."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mopd_verl.domain_budgeting_math import (
    apply_probability_floor,
    exponential_moving_average,
    normalize_domain_weights,
    power_weighted_distribution,
)
from mopd_verl.domain_budgeting_state import (
    DomainBudgetState,
    persist_controller_payload,
)
from mopd_verl.domain_budgeting_support import (
    DomainBudgetControllerSupport,
    config_get,
    plain_mapping,
    validate_runtime_controller,
)


class DynamicDomainBudgetController(DomainBudgetControllerSupport):
    """Maintain q, desired p, and exact per-batch lambda=q/p_active."""

    def __init__(self, config: Any) -> None:
        self.enabled = bool(config_get(config, "enabled", False))
        self.domains = tuple(str(item) for item in config_get(config, "domains", []))
        priors = plain_mapping(config_get(config, "domain_priors", {}))
        self.priors = (
            normalize_domain_weights(
                {domain: float(priors.get(domain, 1.0)) for domain in self.domains}
            )
            if self.domains
            else {}
        )
        self.teacher_scores = {
            domain: float(value)
            for domain, value in plain_mapping(
                config_get(config, "teacher_scores", {})
            ).items()
        }
        self.teacher_scores_calibrated = bool(
            config_get(config, "teacher_scores_calibrated", False)
        )
        raw_metric_keys = plain_mapping(
            config_get(config, "validation_metric_keys", {})
        )
        self.validation_metric_keys = {
            domain: (
                [str(value)]
                if isinstance(value, str)
                else [str(item) for item in value]
            )
            for domain, value in raw_metric_keys.items()
        }
        self.metric_reducer = str(config_get(config, "validation_reducer", "mean"))
        self.gap_ema_beta = float(config_get(config, "gap_ema_beta", 0.9))
        self.gap_alpha = float(config_get(config, "gap_alpha", 1.0))
        self.gap_epsilon = float(config_get(config, "gap_epsilon", 1e-6))
        self.gap_normalization_floor = float(
            config_get(config, "gap_normalization_floor", 0.05)
        )
        self.max_normalized_gap = float(config_get(config, "max_normalized_gap", 2.0))
        self.exploration_mass = float(config_get(config, "exploration_mass", 0.05))
        self.variance_log_ema_beta = float(
            config_get(config, "variance_log_ema_beta", 0.9)
        )
        self.variance_epsilon = float(config_get(config, "variance_epsilon", 1e-8))
        self.variance_min_samples = int(config_get(config, "variance_min_samples", 2))
        self.min_samples_per_domain = int(
            config_get(config, "min_samples_per_domain", 1)
        )
        self.variance_update_freq_steps = int(
            config_get(config, "variance_update_freq_steps", 1)
        )
        self.min_sampling_probability = float(
            config_get(config, "min_sampling_probability", 0.0)
        )
        self.output_dir = Path(
            str(config_get(config, "output_dir", "domain_budgeting"))
        )
        validate_runtime_controller(self)

        initial_q = dict(self.priors)
        self.state = DomainBudgetState(
            target_contributions=initial_q,
            desired_sampling=dict(initial_q),
        )

    @property
    def desired_sampling(self) -> dict[str, float]:
        return dict(self.state.desired_sampling)

    def observe_validation(
        self, metrics: Mapping[str, Any], step: int
    ) -> dict[str, float]:
        if not self.enabled:
            return {}
        if self.state.last_validation_step == int(step):
            return self.next_allocation_metrics()
        if (
            self.state.last_validation_step is not None
            and self.state.last_validation_step > int(step)
        ):
            raise ValueError("Validation steps must be monotonically increasing.")
        student_scores: dict[str, float] = {}
        for domain in self.domains:
            missing_keys = [
                key
                for key in self.validation_metric_keys[domain]
                if key not in metrics or metrics[key] is None
            ]
            if missing_keys:
                raise KeyError(
                    f"Missing fixed-probe validation metrics for domain {domain!r}: "
                    f"{missing_keys}."
                )
            values = [
                float(metrics[key]) for key in self.validation_metric_keys[domain]
            ]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite validation metric for domain {domain!r}.")
            student_scores[domain] = statistics.fmean(values)

        raw_gaps = {
            domain: max(0.0, self.teacher_scores[domain] - student_scores[domain])
            for domain in self.domains
        }
        initial_gaps = dict(self.state.initial_gaps)
        gap_ema = dict(self.state.gap_ema)
        normalized_gaps: dict[str, float] = {}
        for domain in self.domains:
            initial_gaps.setdefault(domain, raw_gaps[domain])
            gap_ema[domain] = exponential_moving_average(
                gap_ema.get(domain), raw_gaps[domain], self.gap_ema_beta
            )
            denominator = max(
                initial_gaps[domain],
                self.gap_normalization_floor,
                self.gap_epsilon,
            )
            normalized_gaps[domain] = min(
                self.max_normalized_gap,
                max(0.0, gap_ema[domain] / denominator),
            )

        gap_q = power_weighted_distribution(
            self.priors,
            {
                domain: normalized_gaps[domain] + self.gap_epsilon
                for domain in self.domains
            },
            self.gap_alpha,
        )
        uniform = 1.0 / len(self.domains)
        q_values = normalize_domain_weights(
            {
                domain: (1.0 - self.exploration_mass) * gap_q[domain]
                + self.exploration_mass * uniform
                for domain in self.domains
            }
        )
        self.state = DomainBudgetState(
            **{
                **asdict(self.state),
                "validation_updates": self.state.validation_updates + 1,
                "last_validation_step": int(step),
                "initial_gaps": initial_gaps,
                "gap_ema": gap_ema,
                "normalized_gaps": normalized_gaps,
                "student_scores": student_scores,
                "target_contributions": q_values,
                "observed_sampling": {},
                "effective_sampling": {},
                "valid_fractions": {},
                "loss_scales": {},
            }
        )
        self._refresh_desired_sampling()
        self._persist("validation", step)
        return self.next_allocation_metrics()

    def observe_sequence_losses(
        self,
        domains: Sequence[str],
        sequence_losses: Sequence[float],
        step: int,
    ) -> dict[str, float]:
        if not self.enabled:
            return {}
        if self.state.last_variance_step == int(step):
            return self.metrics()
        if (
            self.state.last_variance_step is not None
            and self.state.last_variance_step > int(step)
        ):
            raise ValueError("Variance steps must be monotonically increasing.")
        if step % self.variance_update_freq_steps != 0:
            return {}
        if len(domains) != len(sequence_losses):
            raise ValueError(
                "Domain labels and sequence losses must have equal length."
            )
        unknown = sorted(set(str(domain) for domain in domains) - set(self.domains))
        if unknown:
            raise ValueError(f"Unknown domains in sequence losses: {unknown}.")
        grouped: dict[str, list[float]] = {domain: [] for domain in self.domains}
        for domain, loss in zip(domains, sequence_losses, strict=True):
            if domain in grouped:
                value = float(loss)
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite sequence loss for domain {domain!r}.")
                grouped[domain].append(value)

        raw_variances = dict(self.state.raw_variances)
        log_variance_ema = dict(self.state.log_variance_ema)
        updated = False
        for domain, values in grouped.items():
            if len(values) < self.variance_min_samples:
                continue
            variance = statistics.variance(values)
            if not math.isfinite(variance) or variance < 0.0:
                raise ValueError(
                    f"Invalid sequence-loss variance for domain {domain!r}."
                )
            log_variance = math.log(variance + self.variance_epsilon)
            raw_variances[domain] = variance
            log_variance_ema[domain] = exponential_moving_average(
                log_variance_ema.get(domain),
                log_variance,
                self.variance_log_ema_beta,
            )
            updated = True
        if not updated:
            return {}
        self.state = DomainBudgetState(
            **{
                **asdict(self.state),
                "variance_updates": self.state.variance_updates + 1,
                "last_variance_step": int(step),
                "raw_variances": raw_variances,
                "log_variance_ema": log_variance_ema,
            }
        )
        self._refresh_desired_sampling()
        self._persist("variance", step)
        return self.metrics()

    def _refresh_desired_sampling(self) -> None:
        q_values = self.state.target_contributions
        allocation_weights = {}
        for domain in self.domains:
            smoothed_variance = math.exp(self.state.log_variance_ema.get(domain, 0.0))
            allocation_weights[domain] = q_values[domain] * math.sqrt(
                smoothed_variance + self.variance_epsilon
            )
        desired = apply_probability_floor(
            allocation_weights,
            self.min_sampling_probability,
        )
        self.state = DomainBudgetState(
            **{**asdict(self.state), "desired_sampling": desired}
        )

    def loss_scales_for_batch(
        self,
        domains: Sequence[str],
        step: int,
        active_mask: Sequence[bool] | None = None,
    ) -> tuple[list[float], dict[str, float]]:
        if not self.enabled:
            return [1.0] * len(domains), {}
        counts = Counter(str(domain) for domain in domains)
        unknown = sorted(set(counts) - set(self.domains))
        if unknown:
            raise ValueError(f"Unknown domains in actor batch: {unknown}.")
        missing = [domain for domain in self.domains if counts[domain] == 0]
        if missing:
            raise ValueError(
                "Exact q=p*lambda scaling requires every domain in each actor "
                f"batch; missing {missing}."
            )
        total = len(domains)
        if total == 0:
            raise ValueError("Actor batch must contain at least one sample.")
        active = (
            [True] * total
            if active_mask is None
            else [bool(item) for item in active_mask]
        )
        if len(active) != total:
            raise ValueError("active_mask must match the actor batch size.")
        active_counts = Counter(
            str(domain)
            for domain, is_active in zip(domains, active, strict=True)
            if is_active
        )
        inactive_domains = [
            domain for domain in self.domains if active_counts[domain] == 0
        ]
        if inactive_domains:
            raise ValueError(
                "Exact q=p*lambda scaling requires at least one valid response "
                f"per domain; missing active responses for {inactive_domains}."
            )
        active_total = sum(active_counts.values())
        observed = {domain: counts[domain] / total for domain in self.domains}
        effective = {
            domain: active_counts[domain] / active_total for domain in self.domains
        }
        valid_fractions = {
            domain: active_counts[domain] / counts[domain] for domain in self.domains
        }
        scales = {
            domain: self.state.target_contributions[domain] / effective[domain]
            for domain in self.domains
        }
        self.state = DomainBudgetState(
            **{
                **asdict(self.state),
                "observed_sampling": observed,
                "effective_sampling": effective,
                "valid_fractions": valid_fractions,
                "loss_scales": scales,
            }
        )
        self._persist("batch", step, counts=dict(counts))
        return [scales[str(domain)] for domain in domains], self.metrics()

    def _persist(self, event: str, step: int, **extra: Any) -> None:
        if not self.enabled:
            return
        persist_controller_payload(
            self.output_dir,
            self.state_dict(),
            event,
            step,
            extra,
        )
