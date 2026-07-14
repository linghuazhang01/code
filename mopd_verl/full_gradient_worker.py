"""Compatibility entry point for domain-gradient audit."""

from mopd_verl.domain_gradient import DomainGradientAudit

SequentialBackwardDomainGradientTracker = DomainGradientAudit

__all__ = ["DomainGradientAudit", "SequentialBackwardDomainGradientTracker"]
