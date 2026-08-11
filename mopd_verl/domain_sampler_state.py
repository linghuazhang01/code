"""Checkpoint support for the exact domain batch sampler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class DomainSamplerStateMixin:
    """Serialize dynamic allocation and RNG state for dataloader resume."""

    target_weights: dict[str, float]
    batch_counts: dict[str, int]
    batch_size: int
    replacement: bool
    seed: int | None
    allocation_version: int
    batches_yielded: int
    _labels_fingerprint: str
    _generator: Any

    def _normalize_checkpoint_weights(self, raw_weights: Any) -> dict[str, float]:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "domains": list(self.target_weights),
            "batch_size": self.batch_size,
            "replacement": self.replacement,
            "seed": self.seed,
            "labels_fingerprint": self._labels_fingerprint,
            "target_weights": dict(self.target_weights),
            "batch_counts": dict(self.batch_counts),
            "allocation_version": self.allocation_version,
            "batches_yielded": self.batches_yielded,
            "generator_state": self._generator.get_state().clone(),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        """Restore a compatible sampler without replaying prior batches."""

        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported domain sampler checkpoint schema.")
        static_values = {
            "domains": list(self.target_weights),
            "batch_size": self.batch_size,
            "replacement": self.replacement,
            "seed": self.seed,
            "labels_fingerprint": self._labels_fingerprint,
        }
        mismatches = [
            key
            for key, expected in static_values.items()
            if payload.get(key) != expected
        ]
        if mismatches:
            raise ValueError(
                f"Domain sampler checkpoint is incompatible: {sorted(mismatches)}."
            )
        restored_weights = self._normalize_checkpoint_weights(
            payload.get("target_weights")
        )
        raw_counts = payload.get("batch_counts")
        if not isinstance(raw_counts, Mapping):
            raise ValueError("Domain sampler checkpoint is missing batch counts.")
        restored_counts = {
            str(domain): int(value) for domain, value in raw_counts.items()
        }
        if (
            set(restored_weights) != set(self.target_weights)
            or set(restored_counts) != set(self.target_weights)
            or any(value < 0 for value in restored_counts.values())
            or sum(restored_counts.values()) != self.batch_size
        ):
            raise ValueError("Invalid domain sampler allocation in checkpoint.")
        allocation_version = int(payload.get("allocation_version", -1))
        batches_yielded = int(payload.get("batches_yielded", -1))
        if allocation_version < 0 or batches_yielded < 0:
            raise ValueError("Invalid domain sampler counters in checkpoint.")
        generator_state = payload.get("generator_state")
        if generator_state is None:
            raise ValueError("Domain sampler checkpoint is missing RNG state.")
        self._generator.set_state(generator_state)
        self.target_weights = restored_weights
        self.batch_counts = restored_counts
        self.allocation_version = allocation_version
        self.batches_yielded = batches_yielded
