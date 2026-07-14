"""Compatibility import for the rebuilt domain-gradient sidecar.

The historical tracker reached into FSDP private state and mixed domain,
sample, and token replay. New code should import :class:`DomainGradientAudit`
from ``mopd_verl.domain_gradient`` directly.
"""

from mopd_verl.domain_gradient import DomainGradientAudit

SequentialBackwardDomainGradientTracker = DomainGradientAudit

__all__ = ["DomainGradientAudit", "SequentialBackwardDomainGradientTracker"]
