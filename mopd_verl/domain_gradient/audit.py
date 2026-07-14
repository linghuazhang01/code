"""Independent domain-gradient replay built on the production actor loss."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.geometry import (
    GradientVector,
    actor_group_sum,
    domain_metrics_from_gram,
    snapshot_gradients,
    training_parity_metrics,
    vector_dot,
    vector_nbytes,
    vector_squared_norm,
)
from mopd_verl.domain_gradient.state import AuditState
from mopd_verl.full_gradient.actor_loss import build_actor_micro_batch_loss
from mopd_verl.full_gradient.labels import _labels_from_mapping
from verl.utils.device import get_device_id


class DomainGradientAudit:
    """Read-only sidecar for one optimizer mini-batch."""

    def __init__(self, actor: Any, meta: Any):
        self.actor = actor
        self.config = DomainGradientConfig.from_meta(meta)
        self._audit_total: GradientVector = tuple()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @staticmethod
    def _domain_gradient_mask(
        micro_batch: Any,
        domain: str,
    ) -> torch.Tensor:
        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
        response_mask = model_inputs["response_mask"]
        labels = _labels_from_mapping(model_inputs, int(response_mask.shape[0]))
        rows = torch.tensor(
            [float(label == domain) for label in labels],
            device=response_mask.device,
            dtype=torch.float32,
        )
        weights = rows.unsqueeze(-1).expand(response_mask.shape)
        return weights

    def _backward_replay(
        self,
        state: AuditState,
        micro_batches: Sequence[Any],
        loss_scales: Sequence[float],
        *,
        on_policy: bool,
        temperature: float,
        domain: str | None,
    ) -> None:
        state.restore_runtime()
        state.clear_gradients()
        for micro_batch, loss_scale in zip(micro_batches, loss_scales, strict=True):
            gradient_mask = (
                self._domain_gradient_mask(micro_batch, domain)
                if domain is not None
                else None
            )
            result = build_actor_micro_batch_loss(
                self.actor,
                micro_batch,
                loss_scale_factor=float(loss_scale),
                on_policy=on_policy,
                gradient_mask_override=gradient_mask,
                include_metrics=False,
                temperature=temperature,
            )
            if self.actor.scaler is not None:
                self.actor.scaler.scale(result.loss).backward()
            else:
                result.loss.backward()

    def _coverage_metrics(self, micro_batches: Sequence[Any]) -> dict[str, float]:
        counts = [0.0 for _ in self.config.domains]
        total = 0.0
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            batch_size = int(model_inputs["response_mask"].shape[0])
            labels = _labels_from_mapping(model_inputs, batch_size)
            total += float(batch_size)
            for index, domain in enumerate(self.config.domains):
                counts[index] += float(sum(label == domain for label in labels))
        values = torch.tensor([total, *counts], device=get_device_id(), dtype=torch.float64)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        total_count, *domain_counts = (float(value) for value in values.tolist())
        metrics = {
            f"{domain}/full_grad/sample_count": domain_counts[index]
            for index, domain in enumerate(self.config.domains)
        }
        metrics["global/audit/domain_gradient_coverage_fraction"] = (
            sum(domain_counts) / total_count if total_count > 0.0 else 0.0
        )
        return metrics

    def run_before_training(
        self,
        micro_batches: Sequence[Any],
        loss_scales: Sequence[float],
        *,
        on_policy: bool,
        temperature: float,
    ) -> dict[str, float]:
        if not self.enabled:
            return {}
        if len(micro_batches) != len(loss_scales):
            raise ValueError("Each audit micro-batch must have exactly one training loss scale.")

        state = AuditState.capture(self.actor)
        try:
            self._backward_replay(
                state,
                micro_batches,
                loss_scales,
                on_policy=on_policy,
                temperature=temperature,
                domain=None,
            )
            audit_total = snapshot_gradients(
                self.actor,
                self.config.storage_dtype,
            )
            domain_vectors: dict[str, GradientVector] = {}
            for domain in self.config.domains:
                self._backward_replay(
                    state,
                    micro_batches,
                    loss_scales,
                    on_policy=on_policy,
                    temperature=temperature,
                    domain=domain,
                )
                domain_vectors[domain] = snapshot_gradients(
                    self.actor,
                    self.config.storage_dtype,
                )

            total_sq = vector_squared_norm(self.actor, audit_total)
            domain_sq = {
                domain: vector_squared_norm(self.actor, vector)
                for domain, vector in domain_vectors.items()
            }
            domain_total_dot = {
                domain: vector_dot(self.actor, vector, audit_total)
                for domain, vector in domain_vectors.items()
            }
            pair_dot = {
                (left_domain, right_domain): vector_dot(
                    self.actor,
                    domain_vectors[left_domain],
                    domain_vectors[right_domain],
                )
                for left_index, left_domain in enumerate(self.config.domains)
                for right_domain in self.config.domains[left_index + 1 :]
            }
            metrics = domain_metrics_from_gram(
                self.actor,
                self.config.domains,
                total_sq=total_sq,
                domain_sq=domain_sq,
                domain_total_dot=domain_total_dot,
                pair_dot=pair_dot,
                closure_threshold=self.config.closure_rel_l2_threshold,
                all_vectors_fp32=(
                    self.config.storage_dtype.lower() in {"float32", "fp32"}
                ),
                storage_dtype=self.config.storage_dtype,
            )
            domain_count = len(self.config.domains)
            metrics["global/audit/domain_gradient_backward_replay_count"] = float(
                1 + domain_count
            )
            metrics["global/audit/domain_gradient_source_step"] = float(
                self.config.step
            )
            peak_vector_bytes = vector_nbytes(audit_total) + sum(
                vector_nbytes(vector) for vector in domain_vectors.values()
            )
            metrics["global/audit/domain_gradient_peak_cpu_vector_bytes"] = float(
                peak_vector_bytes
            )
            metrics[
                "global/audit/domain_gradient_peak_cpu_vector_bytes_per_rank"
            ] = float(peak_vector_bytes)
            metrics[
                "global/audit/"
                "domain_gradient_peak_cpu_vector_bytes_actor_group_total"
            ] = actor_group_sum(self.actor, float(peak_vector_bytes))
            retained_vector_bytes = (
                vector_nbytes(audit_total) if self.config.parity_enabled else 0
            )
            metrics[
                "global/audit/domain_gradient_post_audit_retained_cpu_vector_bytes"
            ] = float(retained_vector_bytes)
            metrics[
                "global/audit/"
                "domain_gradient_post_audit_retained_cpu_vector_bytes_per_rank"
            ] = float(retained_vector_bytes)
            metrics[
                "global/audit/"
                "domain_gradient_post_audit_retained_cpu_vector_bytes_actor_group_total"
            ] = actor_group_sum(self.actor, float(retained_vector_bytes))
            metrics.update(self._coverage_metrics(micro_batches))
            if self.config.parity_enabled:
                self._audit_total = audit_total
            return metrics
        finally:
            state.restore()

    def compare_training_gradient(self) -> dict[str, float]:
        audit_total = self._audit_total
        self._audit_total = tuple()
        if not audit_total:
            return {}
        return training_parity_metrics(
            self.actor,
            audit_total,
            self.config.parity_rel_l2_threshold,
        )
