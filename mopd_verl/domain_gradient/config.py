"""Configuration contract for the domain-gradient sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mopd_verl.domain_gradient.token_weighting_state import (
    PER_STEP_MEAN_ABS_LOSS_SELECTION,
    SHARED_TOKEN_SELECTION_MODES,
)
from mopd_verl.domain_gradient.weighting import DYNAMIC_WEIGHT_SIGNALS


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


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
    all_domain_shared_token_weighting_enabled: bool
    all_domain_shared_token_weight: float
    all_domain_shared_token_selection_mode: str
    all_domain_shared_token_top_k: int | None
    unsupported_modes: tuple[str, ...]

    @classmethod
    def from_meta(cls, meta: Any) -> "DomainGradientConfig":
        domains = tuple(dict.fromkeys(str(value) for value in _get(meta, "domains", ())))
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
                parity_frequency >= 0
                and step % max(1, parity_frequency) == 0
            ),
            parity_rel_l2_threshold=float(
                _get(meta, "full_grad_training_parity_rel_l2_threshold", 1e-5)
            ),
            closure_rel_l2_threshold=float(
                _get(meta, "sequence_masked_target_closure_rel_l2_threshold", 0.02)
            ),
            token_gradient_enabled=bool(
                _get(meta, "token_gradient_enabled", False)
            ),
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
            ).strip().lower(),
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
            control_token_weight=float(
                _get(meta, "control_token_loss_weight", 1.0)
            ),
            control_token_ids=tuple(
                dict.fromkeys(
                    int(token_id)
                    for token_id in _get(meta, "control_token_ids", ())
                )
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
            ).strip().lower(),
            all_domain_shared_token_top_k=(
                None
                if raw_shared_top_k is None
                else int(raw_shared_top_k)
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
        if self.token_gradient_enabled and (
            self.token_gradient_tail_enabled
            or self.token_gradient_top_p_enabled
        ) and not self.token_gradient_loss_abs_selection_enabled:
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
                "dynamic_domain_loss_weighting_weight_ema_beta must be "
                "in [0, 1)."
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
                "Dynamic domain loss weight bounds must be positive and "
                "contain 1.0."
            )
        if self.control_token_weight < 1.0:
            raise ValueError(
                "control_token_loss_weight must be at least 1.0."
            )
        if (
            self.control_token_weighting_enabled
            and not self.control_token_ids
        ):
            raise ValueError(
                "Control-token loss weighting requires control_token_ids."
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
                "all_domain_shared_token_selection_mode must be one of: "
                f"{allowed}."
            )
        if (
            self.all_domain_shared_token_top_k is not None
            and self.all_domain_shared_token_top_k < 1
        ):
            raise ValueError(
                "all_domain_shared_token_top_k must be null or at least 1."
            )
        if (
            self.all_domain_shared_token_weighting_enabled
            and len(self.domains) < 2
        ):
            raise ValueError(
                "All-domain shared-token weighting requires at least two "
                "configured domains."
            )
        if not self.enabled:
            return
        if not self.domains:
            raise ValueError("Domain-gradient audit requires at least one configured domain.")
        if self.storage_dtype.lower() not in {
            "float32",
            "fp32",
            "float16",
            "fp16",
            "half",
            "bfloat16",
            "bf16",
        }:
            raise ValueError(f"Unsupported domain-gradient storage dtype: {self.storage_dtype!r}")
