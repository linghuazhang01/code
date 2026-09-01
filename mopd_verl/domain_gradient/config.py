"""Configuration contract for the domain-gradient sidecar."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mopd_verl.domain_gradient.adaptive_neighborhood import (
    PerTokenAdaptiveNeighborhoodSpec,
)
from mopd_verl.domain_gradient.control_selection_scoring import (
    FIXED_ONLINE_WEIGHT_MODE,
    ONLINE_CONTROL_BUDGET_MODES,
    ONLINE_CONTROL_SELECTION_MODES,
    ONLINE_CONTROL_WEIGHT_MODES,
    PAIRED_ONLINE_WEIGHT_MODE,
    PAIRED_SIGNAL_SELECTION_MODES,
    TOP_K_BUDGET_MODE,
    TOP_LOSS_SELECTION_MODE,
    TOP_SPEED_SELECTION_MODE,
)
from mopd_verl.domain_gradient.token_weighting_state import (
    PER_STEP_MEAN_ABS_LOSS_SELECTION,
    SHARED_TOKEN_SELECTION_MODES,
)
from mopd_verl.domain_gradient.weighting import DYNAMIC_WEIGHT_SIGNALS


DEFAULT_CONTROL_SPEED_WEIGHT_KNOTS = (
    (-0.0025, 0.0),
    (0.0, 0.2),
    (0.005, 2.0),
    (0.010, 3.0),
    (0.015, 4.0),
)


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _domain_token_ids(
    value: Any,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError("domain_control_token_ids must be a mapping.")
    return tuple(
        (
            str(domain),
            tuple(dict.fromkeys(int(token_id) for token_id in token_ids)),
        )
        for domain, token_ids in value.items()
    )


def _domain_candidate_token_ids(
    value: Any,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError("domain_control_token_candidate_ids must be a mapping.")
    return tuple(
        (
            str(domain),
            tuple(sorted({int(token_id) for token_id in token_ids})),
        )
        for domain, token_ids in value.items()
    )


def _domain_candidate_token_groups(
    value: Any,
) -> tuple[tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError("domain_control_token_candidate_groups must be a mapping.")
    normalized: list[
        tuple[str, tuple[tuple[str, tuple[int, ...]], ...]]
    ] = []
    for domain, groups in value.items():
        if not isinstance(groups, Mapping):
            raise TypeError(
                "Each domain_control_token_candidate_groups entry must be a mapping."
            )
        normalized.append(
            (
                str(domain),
                tuple(
                    (
                        str(group),
                        tuple(sorted({int(token_id) for token_id in token_ids})),
                    )
                    for group, token_ids in groups.items()
                ),
            )
        )
    return tuple(normalized)


def _speed_weight_knots(value: Any) -> tuple[tuple[float, float], ...]:
    if value is None:
        return DEFAULT_CONTROL_SPEED_WEIGHT_KNOTS
    if isinstance(value, (str, bytes)):
        raise TypeError("control_token_speed_weight_knots must be a sequence.")
    knots: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, (str, bytes)) or len(item) != 2:
            raise ValueError(
                "Each control_token_speed_weight_knots entry must contain "
                "exactly [speed, weight]."
            )
        knots.append((float(item[0]), float(item[1])))
    return tuple(knots)


@dataclass(frozen=True)
class DomainGradientConfig:
    """The small subset of audit config that changes gradient replay."""

    enabled: bool
    step: int
    output_dir: str
    domains: tuple[str, ...]
    storage_dtype: str
    parity_enabled: bool
    parity_rel_l2_threshold: float
    closure_rel_l2_threshold: float
    token_gradient_enabled: bool
    token_gradient_tail_enabled: bool
    token_gradient_tail_fraction: float
    token_gradient_tail_min_tokens: int
    token_gradient_top_p_enabled: bool
    token_gradient_top_k: int | None
    token_gradient_top_p: float
    token_gradient_loss_abs_selection_enabled: bool
    token_gradient_log_tokens_jsonl_enabled: bool
    token_gradient_vocab_size: int | None
    dynamic_weighting_enabled: bool
    dynamic_weighting_update_enabled: bool
    dynamic_weighting_signal_source: str
    dynamic_weighting_ema_beta: float
    dynamic_weighting_weight_ema_beta: float
    dynamic_weighting_alpha: float
    dynamic_weighting_min: float
    dynamic_weighting_max: float
    control_token_weighting_enabled: bool
    control_token_weight: float
    control_token_ids: tuple[int, ...]
    domain_control_token_ids: tuple[tuple[str, tuple[int, ...]], ...]
    control_token_candidate_ids: tuple[int, ...]
    domain_control_token_candidate_ids: tuple[tuple[str, tuple[int, ...]], ...]
    domain_control_token_candidate_groups: tuple[
        tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...
    ]
    control_token_normalize_per_domain: bool
    control_token_online_selection_enabled: bool
    control_token_online_audit_interval_steps: int
    control_token_online_window_steps: int
    control_token_online_min_mean_occurrences_per_step: float
    control_token_online_strict_occurrence_gate: bool
    control_token_online_top_k: int
    control_token_online_top_k_per_group: int | None
    control_token_online_budget_mode: str
    control_token_online_top_p: float
    control_token_online_selection_mode: str
    control_token_online_weight_mode: str
    control_token_adaptive_neighborhood_enabled: bool
    control_token_adaptive_neighborhood_max_distance: int
    control_token_adaptive_neighborhood_epsilon: float
    control_token_adaptive_neighborhood_relative_loss_clip_max: float
    control_token_adaptive_neighborhood_relative_loss_threshold: float
    control_token_adaptive_neighborhood_min_far_tokens: int
    control_token_phase_gate_enabled: bool
    control_token_span_weighting_enabled: bool
    control_token_phase_gate_window_steps: int
    control_token_phase_gate_ema_beta: float
    control_token_phase_gate_temperature: float
    control_token_phase_gate_initial: float
    control_token_span_length: int
    control_token_span_decay_tau: float
    control_token_speed_weighting_enabled: bool
    control_token_speed_window_steps: int
    control_token_speed_ema_beta: float
    control_token_speed_update_interval_steps: int
    control_token_speed_initial_weight: float
    control_token_speed_min_occurrences: int
    control_token_speed_weight_knots: tuple[tuple[float, float], ...]
    all_domain_shared_token_weighting_enabled: bool
    all_domain_shared_token_weight: float
    all_domain_shared_token_selection_mode: str
    all_domain_shared_token_top_k: int | None
    unsupported_modes: tuple[str, ...]

    def effective_domain_candidate_map(self) -> dict[str, tuple[int, ...]]:
        """Return one canonical candidate whitelist for every domain."""

        domain_candidate_groups = self.effective_domain_candidate_group_map()
        if domain_candidate_groups:
            return {
                domain: tuple(
                    sorted(
                        {
                            token_id
                            for token_ids in groups.values()
                            for token_id in token_ids
                        }
                    )
                )
                for domain, groups in domain_candidate_groups.items()
            }
        domain_candidates = dict(self.domain_control_token_candidate_ids)
        if domain_candidates:
            return domain_candidates
        return {domain: self.control_token_candidate_ids for domain in self.domains}

    def effective_domain_candidate_group_map(
        self,
    ) -> dict[str, dict[str, tuple[int, ...]]]:
        """Return per-domain selection groups, or an empty map for unified mode."""

        return {
            domain: dict(groups)
            for domain, groups in self.domain_control_token_candidate_groups
        }

    def adaptive_neighborhood_spec(
        self,
        *,
        domain_token_ids: Mapping[str, Any] | None = None,
    ) -> PerTokenAdaptiveNeighborhoodSpec | None:
        """Return the stateless same-batch gate specification when enabled."""

        if not self.control_token_adaptive_neighborhood_enabled:
            return None
        if domain_token_ids is None:
            effective_domain_token_ids = self.domain_control_token_ids
        else:
            token_id_map = dict(_domain_token_ids(domain_token_ids))
            if set(token_id_map) != set(self.domains):
                raise ValueError(
                    "Adaptive domain token ID overrides must exactly match "
                    "domains."
                )
            effective_domain_token_ids = tuple(
                (domain, token_id_map[domain]) for domain in self.domains
            )
        return PerTokenAdaptiveNeighborhoodSpec(
            domains=self.domains,
            domain_token_ids=effective_domain_token_ids,
            max_distance=self.control_token_adaptive_neighborhood_max_distance,
            epsilon=self.control_token_adaptive_neighborhood_epsilon,
            clip_max=(
                self.control_token_adaptive_neighborhood_relative_loss_clip_max
            ),
            threshold=(
                self.control_token_adaptive_neighborhood_relative_loss_threshold
            ),
            min_far_tokens=(
                self.control_token_adaptive_neighborhood_min_far_tokens
            ),
            control_weight=self.control_token_weight,
            normalize_per_response=self.control_token_normalize_per_domain,
        )

    @classmethod
    def from_meta(cls, meta: Any) -> "DomainGradientConfig":
        domains = tuple(
            dict.fromkeys(str(value) for value in _get(meta, "domains", ()))
        )
        parity_frequency = int(_get(meta, "full_grad_training_parity_freq_steps", 1))
        step = int(_get(meta, "step", 0))
        raw_top_k = _get(meta, "token_gradient_top_k", 100)
        raw_shared_top_k = _get(
            meta,
            "all_domain_shared_token_top_k",
            100,
        )
        config = cls(
            enabled=bool(_get(meta, "enabled", False))
            and bool(_get(meta, "domain_gradient_enabled", True)),
            step=step,
            output_dir=str(_get(meta, "output_dir", "mopd_audit")),
            domains=domains,
            storage_dtype=str(_get(meta, "storage_dtype", "float32")),
            parity_enabled=(
                parity_frequency >= 0 and step % max(1, parity_frequency) == 0
            ),
            parity_rel_l2_threshold=float(
                _get(meta, "full_grad_training_parity_rel_l2_threshold", 1e-5)
            ),
            closure_rel_l2_threshold=float(
                _get(meta, "sequence_masked_target_closure_rel_l2_threshold", 0.02)
            ),
            token_gradient_enabled=bool(_get(meta, "token_gradient_enabled", False)),
            token_gradient_tail_enabled=bool(
                _get(meta, "token_gradient_tail_enabled", True)
            ),
            token_gradient_tail_fraction=float(
                _get(meta, "token_gradient_tail_fraction", 0.10)
            ),
            token_gradient_tail_min_tokens=max(
                1,
                int(_get(meta, "token_gradient_tail_min_tokens", 1)),
            ),
            token_gradient_top_p_enabled=bool(
                _get(meta, "token_gradient_top_p_enabled", False)
            ),
            token_gradient_top_k=(
                None if raw_top_k is None else max(1, int(raw_top_k))
            ),
            token_gradient_top_p=min(
                1.0,
                max(
                    0.0,
                    float(_get(meta, "token_gradient_top_p", 0.10)),
                ),
            ),
            token_gradient_loss_abs_selection_enabled=bool(
                _get(meta, "token_gradient_loss_abs_selection_enabled", True)
            ),
            token_gradient_log_tokens_jsonl_enabled=bool(
                _get(
                    meta,
                    "token_gradient_log_tokens_jsonl_enabled",
                    True,
                )
            ),
            token_gradient_vocab_size=(
                None
                if _get(meta, "token_gradient_vocab_size", None) is None
                else max(
                    1,
                    int(_get(meta, "token_gradient_vocab_size")),
                )
            ),
            dynamic_weighting_enabled=bool(
                _get(meta, "dynamic_domain_loss_weighting_enabled", False)
            ),
            dynamic_weighting_update_enabled=bool(
                _get(
                    meta,
                    "dynamic_domain_loss_weighting_update_enabled",
                    False,
                )
            ),
            dynamic_weighting_signal_source=str(
                _get(
                    meta,
                    "dynamic_domain_loss_weighting_signal_source",
                    "gradient_norm",
                )
            )
            .strip()
            .lower(),
            dynamic_weighting_ema_beta=float(
                _get(meta, "dynamic_domain_loss_weighting_ema_beta", 0.90)
            ),
            dynamic_weighting_weight_ema_beta=float(
                _get(
                    meta,
                    "dynamic_domain_loss_weighting_weight_ema_beta",
                    0.90,
                )
            ),
            dynamic_weighting_alpha=float(
                _get(meta, "dynamic_domain_loss_weighting_alpha", 0.50)
            ),
            dynamic_weighting_min=float(
                _get(
                    meta,
                    "dynamic_domain_loss_weighting_min",
                    1.0 / 3.0,
                )
            ),
            dynamic_weighting_max=float(
                _get(meta, "dynamic_domain_loss_weighting_max", 3.0)
            ),
            control_token_weighting_enabled=bool(
                _get(meta, "control_token_loss_weighting_enabled", False)
            ),
            control_token_weight=float(_get(meta, "control_token_loss_weight", 1.0)),
            control_token_ids=tuple(
                dict.fromkeys(
                    int(token_id) for token_id in _get(meta, "control_token_ids", ())
                )
            ),
            domain_control_token_ids=_domain_token_ids(
                _get(meta, "domain_control_token_ids", {})
            ),
            control_token_candidate_ids=tuple(
                sorted(
                    {
                        int(token_id)
                        for token_id in _get(
                            meta,
                            "control_token_candidate_ids",
                            (),
                        )
                    }
                )
            ),
            domain_control_token_candidate_ids=_domain_candidate_token_ids(
                _get(meta, "domain_control_token_candidate_ids", {})
            ),
            domain_control_token_candidate_groups=_domain_candidate_token_groups(
                _get(meta, "domain_control_token_candidate_groups", {})
            ),
            control_token_normalize_per_domain=bool(
                _get(meta, "control_token_normalize_per_domain", False)
            ),
            control_token_online_selection_enabled=bool(
                _get(
                    meta,
                    "control_token_online_selection_enabled",
                    False,
                )
            ),
            control_token_online_audit_interval_steps=int(
                _get(
                    meta,
                    "control_token_online_audit_interval_steps",
                    3,
                )
            ),
            control_token_online_window_steps=int(
                _get(meta, "control_token_online_window_steps", 3)
            ),
            control_token_online_min_mean_occurrences_per_step=float(
                _get(
                    meta,
                    ("control_token_online_" "min_mean_occurrences_per_step"),
                    20.0,
                )
            ),
            control_token_online_strict_occurrence_gate=bool(
                _get(
                    meta,
                    "control_token_online_strict_occurrence_gate",
                    False,
                )
            ),
            control_token_online_top_k=int(
                _get(meta, "control_token_online_top_k", 30)
            ),
            control_token_online_top_k_per_group=(
                None
                if _get(meta, "control_token_online_top_k_per_group", None) is None
                else int(_get(meta, "control_token_online_top_k_per_group", None))
            ),
            control_token_online_budget_mode=str(
                _get(
                    meta,
                    "control_token_online_budget_mode",
                    TOP_K_BUDGET_MODE,
                )
            )
            .strip()
            .lower(),
            control_token_online_top_p=float(
                _get(meta, "control_token_online_top_p", 1.0)
            ),
            control_token_online_selection_mode=str(
                _get(
                    meta,
                    "control_token_online_selection_mode",
                    TOP_LOSS_SELECTION_MODE,
                )
            )
            .strip()
            .lower(),
            control_token_online_weight_mode=str(
                _get(
                    meta,
                    "control_token_online_weight_mode",
                    FIXED_ONLINE_WEIGHT_MODE,
                )
            )
            .strip()
            .lower(),
            control_token_adaptive_neighborhood_enabled=bool(
                _get(meta, "control_token_adaptive_neighborhood_enabled", False)
            ),
            control_token_adaptive_neighborhood_max_distance=int(
                _get(meta, "control_token_adaptive_neighborhood_max_distance", 8)
            ),
            control_token_adaptive_neighborhood_epsilon=float(
                _get(meta, "control_token_adaptive_neighborhood_epsilon", 1e-8)
            ),
            control_token_adaptive_neighborhood_relative_loss_clip_max=float(
                _get(
                    meta,
                    "control_token_adaptive_neighborhood_relative_loss_clip_max",
                    1.5,
                )
            ),
            control_token_adaptive_neighborhood_relative_loss_threshold=float(
                _get(
                    meta,
                    "control_token_adaptive_neighborhood_relative_loss_threshold",
                    0.3,
                )
            ),
            control_token_adaptive_neighborhood_min_far_tokens=int(
                _get(
                    meta,
                    "control_token_adaptive_neighborhood_min_far_tokens",
                    1,
                )
            ),
            control_token_phase_gate_enabled=bool(
                _get(meta, "control_token_phase_gate_enabled", False)
            ),
            control_token_span_weighting_enabled=bool(
                _get(meta, "control_token_span_weighting_enabled", False)
            ),
            control_token_phase_gate_window_steps=max(
                1,
                int(_get(meta, "control_token_phase_gate_window_steps", 5)),
            ),
            control_token_phase_gate_ema_beta=float(
                _get(meta, "control_token_phase_gate_ema_beta", 0.90)
            ),
            control_token_phase_gate_temperature=float(
                _get(meta, "control_token_phase_gate_temperature", 0.10)
            ),
            control_token_phase_gate_initial=float(
                _get(meta, "control_token_phase_gate_initial", 0.80)
            ),
            control_token_span_length=max(
                0,
                int(_get(meta, "control_token_span_length", 16)),
            ),
            control_token_span_decay_tau=float(
                _get(meta, "control_token_span_decay_tau", 8.0)
            ),
            control_token_speed_weighting_enabled=bool(
                _get(meta, "control_token_speed_weighting_enabled", False)
            ),
            control_token_speed_window_steps=max(
                1,
                int(_get(meta, "control_token_speed_window_steps", 5)),
            ),
            control_token_speed_ema_beta=float(
                _get(meta, "control_token_speed_ema_beta", 0.8)
            ),
            control_token_speed_update_interval_steps=max(
                1,
                int(
                    _get(
                        meta,
                        "control_token_speed_update_interval_steps",
                        2,
                    )
                ),
            ),
            control_token_speed_initial_weight=float(
                _get(meta, "control_token_speed_initial_weight", 3.0)
            ),
            control_token_speed_min_occurrences=max(
                1,
                int(_get(meta, "control_token_speed_min_occurrences", 128)),
            ),
            control_token_speed_weight_knots=_speed_weight_knots(
                _get(meta, "control_token_speed_weight_knots", None)
            ),
            all_domain_shared_token_weighting_enabled=bool(
                _get(
                    meta,
                    "all_domain_shared_token_loss_weighting_enabled",
                    False,
                )
            ),
            all_domain_shared_token_weight=float(
                _get(meta, "all_domain_shared_token_loss_weight", 1.0)
            ),
            all_domain_shared_token_selection_mode=str(
                _get(
                    meta,
                    "all_domain_shared_token_selection_mode",
                    PER_STEP_MEAN_ABS_LOSS_SELECTION,
                )
            )
            .strip()
            .lower(),
            all_domain_shared_token_top_k=(
                None if raw_shared_top_k is None else int(raw_shared_top_k)
            ),
            unsupported_modes=tuple(
                name
                for name in ("sample_gradient_enabled",)
                if bool(_get(meta, name, False))
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.unsupported_modes:
            names = ", ".join(self.unsupported_modes)
            raise ValueError(
                "The rebuilt audit does not support sample-gradient replay. "
                f"Disable: {names}."
            )
        if not 0.0 < self.token_gradient_tail_fraction <= 1.0:
            raise ValueError("token_gradient_tail_fraction must be in (0, 1].")
        if not 0.0 <= self.token_gradient_top_p <= 1.0:
            raise ValueError("token_gradient_top_p must be in [0, 1].")
        if (
            self.token_gradient_enabled
            and (self.token_gradient_tail_enabled or self.token_gradient_top_p_enabled)
            and not self.token_gradient_loss_abs_selection_enabled
        ):
            raise ValueError(
                "Loss-ranked token-gradient statistics require "
                "token_gradient_loss_abs_selection_enabled=true."
            )
        if not 0.0 <= self.dynamic_weighting_ema_beta < 1.0:
            raise ValueError(
                "dynamic_domain_loss_weighting_ema_beta must be in [0, 1)."
            )
        if not 0.0 <= self.dynamic_weighting_weight_ema_beta < 1.0:
            raise ValueError(
                "dynamic_domain_loss_weighting_weight_ema_beta must be " "in [0, 1)."
            )
        if self.dynamic_weighting_signal_source not in DYNAMIC_WEIGHT_SIGNALS:
            allowed = ", ".join(sorted(DYNAMIC_WEIGHT_SIGNALS))
            raise ValueError(
                "dynamic_domain_loss_weighting_signal_source must be one "
                f"of: {allowed}."
            )
        if self.dynamic_weighting_alpha < 0.0:
            raise ValueError(
                "dynamic_domain_loss_weighting_alpha must be non-negative."
            )
        if (
            self.dynamic_weighting_min <= 0.0
            or self.dynamic_weighting_min > 1.0
            or self.dynamic_weighting_max < 1.0
        ):
            raise ValueError(
                "Dynamic domain loss weight bounds must be positive and " "contain 1.0."
            )
        if (
            not math.isfinite(self.control_token_weight)
            or self.control_token_weight < 0.0
        ):
            raise ValueError(
                "control_token_loss_weight must be finite and non-negative."
            )
        if (
            self.control_token_weighting_enabled
            and not self.control_token_ids
            and not self.domain_control_token_ids
            and not (
                self.control_token_online_selection_enabled
                and (
                    self.control_token_candidate_ids
                    or self.domain_control_token_candidate_ids
                    or self.domain_control_token_candidate_groups
                )
            )
        ):
            raise ValueError(
                "Control-token loss weighting requires fixed token IDs or "
                "online candidate token IDs."
            )
        if self.control_token_ids and self.domain_control_token_ids:
            raise ValueError(
                "Configure either control_token_ids or "
                "domain_control_token_ids, not both."
            )
        unknown_domains = {domain for domain, _ in self.domain_control_token_ids} - set(
            self.domains
        )
        if unknown_domains:
            raise ValueError(
                "domain_control_token_ids contains unknown domains: "
                + ", ".join(sorted(unknown_domains))
            )
        empty_domains = [
            domain
            for domain, token_ids in self.domain_control_token_ids
            if not token_ids
        ]
        if empty_domains:
            raise ValueError(
                "domain_control_token_ids entries must be non-empty: "
                + ", ".join(empty_domains)
            )
        if any(token_id < 0 for token_id in self.control_token_candidate_ids):
            raise ValueError("control_token_candidate_ids must be non-negative.")
        domain_candidates = dict(self.domain_control_token_candidate_ids)
        domain_candidate_groups = self.effective_domain_candidate_group_map()
        candidate_source_count = sum(
            bool(value)
            for value in (
                self.control_token_candidate_ids,
                domain_candidates,
                domain_candidate_groups,
            )
        )
        if candidate_source_count > 1:
            raise ValueError(
                "Configure either global, domain, or grouped domain "
                "Control-token candidate IDs, not more than one."
            )
        if domain_candidates and set(domain_candidates) != set(self.domains):
            raise ValueError(
                "domain_control_token_candidate_ids keys must exactly match " "domains."
            )
        if any(not token_ids for token_ids in domain_candidates.values()):
            raise ValueError(
                "domain_control_token_candidate_ids entries must be non-empty."
            )
        if any(
            token_id < 0
            for token_ids in domain_candidates.values()
            for token_id in token_ids
        ):
            raise ValueError("domain_control_token_candidate_ids must be non-negative.")
        if domain_candidate_groups and set(domain_candidate_groups) != set(
            self.domains
        ):
            raise ValueError(
                "domain_control_token_candidate_groups keys must exactly match "
                "domains."
            )
        if any(not groups for groups in domain_candidate_groups.values()):
            raise ValueError(
                "domain_control_token_candidate_groups entries must contain "
                "at least one group."
            )
        for domain, groups in domain_candidate_groups.items():
            if any(not token_ids for token_ids in groups.values()):
                raise ValueError(
                    "domain_control_token_candidate_groups group entries must "
                    "be non-empty."
                )
            flattened = [
                token_id
                for token_ids in groups.values()
                for token_id in token_ids
            ]
            if any(token_id < 0 for token_id in flattened):
                raise ValueError(
                    "domain_control_token_candidate_groups must be non-negative."
                )
            if len(flattened) != len(set(flattened)):
                raise ValueError(
                    "domain_control_token_candidate_groups groups must be "
                    f"disjoint within domain {domain!r}."
                )
        if (
            self.control_token_online_audit_interval_steps < 1
            or self.control_token_online_window_steps < 1
            or self.control_token_online_top_k < 1
        ):
            raise ValueError(
                "Online Control audit interval, window, and Top-K must be " "positive."
            )
        if self.control_token_online_budget_mode not in ONLINE_CONTROL_BUDGET_MODES:
            allowed = ", ".join(sorted(ONLINE_CONTROL_BUDGET_MODES))
            raise ValueError(
                "Online Control budget mode must be one of: " f"{allowed}."
            )
        if (
            not math.isfinite(self.control_token_online_top_p)
            or not 0.0 < self.control_token_online_top_p <= 1.0
        ):
            raise ValueError(
                "control_token_online_top_p must be finite and in (0, 1]."
            )
        if (
            domain_candidate_groups
            and self.control_token_online_budget_mode == TOP_K_BUDGET_MODE
        ):
            if (
                self.control_token_online_top_k_per_group is None
                or self.control_token_online_top_k_per_group < 1
            ):
                raise ValueError(
                    "Grouped online Control selection requires a positive "
                    "control_token_online_top_k_per_group."
                )
            expected_top_ks = {
                self.control_token_online_top_k_per_group * len(groups)
                for groups in domain_candidate_groups.values()
            }
            if expected_top_ks != {self.control_token_online_top_k}:
                raise ValueError(
                    "control_token_online_top_k must equal per-group K times "
                    "the number of groups in every domain."
                )
        elif self.control_token_online_top_k_per_group is not None:
            raise ValueError(
                "control_token_online_top_k_per_group requires grouped "
                "Top-K selection."
            )
        if (
            self.control_token_online_selection_mode
            not in ONLINE_CONTROL_SELECTION_MODES
        ):
            allowed = ", ".join(sorted(ONLINE_CONTROL_SELECTION_MODES))
            raise ValueError(
                "Online Control selection mode must be one of: " f"{allowed}."
            )
        if self.control_token_online_weight_mode not in ONLINE_CONTROL_WEIGHT_MODES:
            allowed = ", ".join(sorted(ONLINE_CONTROL_WEIGHT_MODES))
            raise ValueError(
                "Online Control weight mode must be one of: " f"{allowed}."
            )
        if (
            self.control_token_online_weight_mode == PAIRED_ONLINE_WEIGHT_MODE
            and self.control_token_online_selection_mode
            not in PAIRED_SIGNAL_SELECTION_MODES
        ):
            raise ValueError(
                "Online paired weight mode requires a paired-signal "
                "selection mode."
            )
        if (
            self.control_token_online_selection_mode == TOP_SPEED_SELECTION_MODE
            and self.control_token_online_window_steps < 2
        ):
            raise ValueError(
                "Online top-speed selection requires a window of at least 2 steps."
            )
        if (
            not math.isfinite(self.control_token_online_min_mean_occurrences_per_step)
            or self.control_token_online_min_mean_occurrences_per_step < 0.0
        ):
            raise ValueError(
                "Online Control minimum mean occurrences must be finite and "
                "non-negative."
            )
        if self.control_token_online_selection_enabled:
            if not self.control_token_weighting_enabled:
                raise ValueError(
                    "Online Control selection requires Control-token loss "
                    "weighting to be enabled."
                )
            if not (
                self.control_token_candidate_ids
                or self.domain_control_token_candidate_ids
                or self.domain_control_token_candidate_groups
            ):
                raise ValueError(
                    "Online Control selection requires "
                    "global or domain candidate token IDs."
                )
            if self.control_token_ids or self.domain_control_token_ids:
                raise ValueError(
                    "Online Control selection cannot be combined with fixed "
                    "Control-token IDs."
                )
            if (
                self.control_token_speed_weighting_enabled
                or self.control_token_phase_gate_enabled
                or self.control_token_span_weighting_enabled
            ):
                raise ValueError(
                    "Online Control selection is mutually exclusive with "
                    "speed, phase-gate, and successor-span weighting."
                )
        if (
            self.control_token_adaptive_neighborhood_max_distance < 1
            or self.control_token_adaptive_neighborhood_min_far_tokens < 1
        ):
            raise ValueError(
                "Adaptive-neighborhood distance and min_far_tokens must be positive."
            )
        if (
            not math.isfinite(self.control_token_adaptive_neighborhood_epsilon)
            or self.control_token_adaptive_neighborhood_epsilon <= 0.0
        ):
            raise ValueError(
                "control_token_adaptive_neighborhood_epsilon must be positive."
            )
        clip_max = self.control_token_adaptive_neighborhood_relative_loss_clip_max
        threshold = self.control_token_adaptive_neighborhood_relative_loss_threshold
        if not all(math.isfinite(value) for value in (clip_max, threshold)):
            raise ValueError(
                "Adaptive-neighborhood relative-loss bounds must be finite."
            )
        if not 0.0 <= threshold <= clip_max:
            raise ValueError(
                "Adaptive-neighborhood threshold must be in [0, clip_max]."
            )
        if self.control_token_adaptive_neighborhood_enabled:
            if self.token_gradient_enabled:
                raise ValueError(
                    "Adaptive-neighborhood control cannot be combined with "
                    "token-gradient replay because the diagnostic ranking "
                    "does not reconstruct the per-token adaptive multiplier."
                )
            fixed_domains = dict(self.domain_control_token_ids)
            if (
                not self.control_token_online_selection_enabled
                and set(fixed_domains) != set(self.domains)
            ):
                raise ValueError(
                    "Adaptive-neighborhood control requires fixed "
                    "domain_control_token_ids that exactly match domains or "
                    "an enabled online selector."
                )
            if (
                not self.control_token_weighting_enabled
                or self.control_token_weight < 1.0
            ):
                raise ValueError(
                    "Adaptive-neighborhood control requires enabled fixed "
                    "Control-token weighting with weight at least 1."
                )
            if (
                self.control_token_ids
                or self.control_token_speed_weighting_enabled
                or self.control_token_phase_gate_enabled
                or self.control_token_span_weighting_enabled
                or self.all_domain_shared_token_weighting_enabled
            ):
                raise ValueError(
                    "Adaptive-neighborhood control is mutually exclusive with "
                    "global IDs, speed, phase, span, and shared-token modes."
                )
            if (
                self.control_token_online_selection_enabled
                and self.control_token_online_weight_mode
                != FIXED_ONLINE_WEIGHT_MODE
            ):
                raise ValueError(
                    "Adaptive-neighborhood online selection requires fixed "
                    "online Control-token weights."
                )
        if not 0.0 <= self.control_token_phase_gate_ema_beta < 1.0:
            raise ValueError("control_token_phase_gate_ema_beta must be in [0, 1).")
        if self.control_token_phase_gate_temperature <= 0.0:
            raise ValueError("control_token_phase_gate_temperature must be positive.")
        if not 0.0 <= self.control_token_phase_gate_initial <= 1.0:
            raise ValueError("control_token_phase_gate_initial must be in [0, 1].")
        if self.control_token_span_decay_tau <= 0.0:
            raise ValueError("control_token_span_decay_tau must be positive.")
        if not 0.0 <= self.control_token_speed_ema_beta < 1.0:
            raise ValueError("control_token_speed_ema_beta must be in [0, 1).")
        if self.control_token_speed_initial_weight < 0.0:
            raise ValueError("control_token_speed_initial_weight must be non-negative.")
        if len(self.control_token_speed_weight_knots) < 2:
            raise ValueError(
                "control_token_speed_weight_knots requires at least two knots."
            )
        for index, (speed, weight) in enumerate(self.control_token_speed_weight_knots):
            if not math.isfinite(speed) or not math.isfinite(weight):
                raise ValueError("control_token_speed_weight_knots must be finite.")
            if weight < 0.0:
                raise ValueError(
                    "control_token_speed_weight_knots weights must be " "non-negative."
                )
            if (
                index > 0
                and speed <= self.control_token_speed_weight_knots[index - 1][0]
            ):
                raise ValueError(
                    "control_token_speed_weight_knots speeds must be strictly "
                    "increasing."
                )
        if self.control_token_speed_weighting_enabled and (
            not self.control_token_weighting_enabled
            or not self.domain_control_token_ids
        ):
            raise ValueError(
                "Control-token speed weighting requires enabled "
                "domain-specific control-token loss weighting."
            )
        if (
            self.control_token_speed_weighting_enabled
            and self.control_token_phase_gate_enabled
        ):
            raise ValueError(
                "Control-token speed weighting and phase gating are mutually "
                "exclusive."
            )
        if self.control_token_phase_gate_enabled and (
            not self.control_token_weighting_enabled
            or not self.domain_control_token_ids
            or self.control_token_span_length < 1
        ):
            raise ValueError(
                "Control-token phase gating requires enabled domain-specific "
                "control weighting and a positive span length."
            )
        if (
            self.control_token_span_weighting_enabled
            and not self.control_token_phase_gate_enabled
        ):
            raise ValueError(
                "Successor-span weighting requires control-token phase " "gating."
            )
        if self.all_domain_shared_token_weight < 1.0:
            raise ValueError(
                "all_domain_shared_token_loss_weight must be at least 1.0."
            )
        if (
            self.all_domain_shared_token_selection_mode
            not in SHARED_TOKEN_SELECTION_MODES
        ):
            allowed = ", ".join(sorted(SHARED_TOKEN_SELECTION_MODES))
            raise ValueError(
                "all_domain_shared_token_selection_mode must be one of: " f"{allowed}."
            )
        if (
            self.all_domain_shared_token_top_k is not None
            and self.all_domain_shared_token_top_k < 1
        ):
            raise ValueError(
                "all_domain_shared_token_top_k must be null or at least 1."
            )
        if self.all_domain_shared_token_weighting_enabled and len(self.domains) < 2:
            raise ValueError(
                "All-domain shared-token weighting requires at least two "
                "configured domains."
            )
        if not self.enabled:
            return
        if not self.domains:
            raise ValueError(
                "Domain-gradient audit requires at least one configured domain."
            )
        if self.storage_dtype.lower() not in {
            "float32",
            "fp32",
            "float16",
            "fp16",
            "half",
            "bfloat16",
            "bf16",
        }:
            raise ValueError(
                f"Unsupported domain-gradient storage dtype: {self.storage_dtype!r}"
            )
