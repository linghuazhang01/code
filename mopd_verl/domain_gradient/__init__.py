"""Read-only domain-gradient audit sidecar."""

from typing import Any

__all__ = ["DomainGradientAudit"]


def __getattr__(name: str) -> Any:
    if name != "DomainGradientAudit":
        raise AttributeError(name)
    from mopd_verl.domain_gradient.audit import DomainGradientAudit

    return DomainGradientAudit
