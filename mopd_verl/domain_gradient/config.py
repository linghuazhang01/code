"""Configuration contract for the domain-gradient sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    domains: tuple[str, ...]
    storage_dtype: str
    parity_enabled: bool
    parity_rel_l2_threshold: float
    closure_rel_l2_threshold: float
    unsupported_modes: tuple[str, ...]

    @classmethod
    def from_meta(cls, meta: Any) -> "DomainGradientConfig":
        domains = tuple(dict.fromkeys(str(value) for value in _get(meta, "domains", ())))
        parity_frequency = int(_get(meta, "full_grad_training_parity_freq_steps", 1))
        step = int(_get(meta, "step", 0))
        config = cls(
            enabled=bool(_get(meta, "enabled", False))
            and bool(_get(meta, "domain_gradient_enabled", True)),
            step=step,
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
            unsupported_modes=tuple(
                name
                for name in ("sample_gradient_enabled", "token_gradient_enabled")
                if bool(_get(meta, name, False))
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.unsupported_modes:
            names = ", ".join(self.unsupported_modes)
            raise ValueError(
                "The rebuilt audit intentionally supports domain gradients only. "
                f"Disable retired nested replay modes: {names}."
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
