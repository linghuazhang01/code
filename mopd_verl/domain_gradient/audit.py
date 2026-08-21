"""Independent domain-gradient replay built on the production actor loss."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from verl.utils.device import get_device_id

from mopd_verl.domain_gradient.control_speed import (
    ControlGapObservation,
    ControlSpeedState,
    control_gap_and_weight_totals,
    domain_control_token_weights,
    initial_control_speed_state,
    piecewise_linear_weight,
    update_control_speed_state,
)
from mopd_verl.domain_gradient.control_selection_scoring import (
    PAIRED_ONLINE_WEIGHT_MODE,
    PAIRED_SIGNAL_SELECTION_MODES,
    TOP_KL_STUDENT_ENTROPY_SELECTION_MODE,
    TOP_TEACHER_CONFIDENCE_STUDENT_ENTROPY_SELECTION_MODE,
    TOP_SPEED_SELECTION_MODE,
)
from mopd_verl.domain_gradient.control_top_loss import (
    OnlineControlSelectionState,
    initial_online_control_selection_state,
    update_online_control_selection,
)
from mopd_verl.domain_gradient.control_top_loss_runtime import (
    append_online_control_selection_jsonl,
    global_candidate_loss_statistics,
)
from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.geometry import (
    GradientVector,
    actor_group_sum,
    domain_metrics_from_gram,
    gradient_partition_metrics_from_gram,
    snapshot_gradients,
    training_parity_metrics,
    vector_dot,
    vector_nbytes,
    vector_squared_norm,
)
from mopd_verl.domain_gradient.phase_control import (
    PhaseControlState,
    PhaseGapObservation,
    gap_observations,
    initial_phase_control_state,
    online_token_score_weights,
    phase_token_weights,
    update_phase_control_state,
)
from mopd_verl.domain_gradient.state import AuditState
from mopd_verl.domain_gradient.token_logging import (
    LocalTokenCandidate,
    append_token_vocab_vectors_jsonl,
    response_token_ids,
)
from mopd_verl.domain_gradient.token_selection import (
    RankedToken,
    select_tail_loss_mass,
    select_top_k,
    select_top_loss_mass,
    total_loss_abs_mass,
)
from mopd_verl.domain_gradient.token_weighting import (
    SharedTokenSelection,
    aligned_response_token_ids,
    append_shared_token_selection_jsonl,
    select_all_domain_shared_tokens,
    token_gradient_weights,
    update_cumulative_shared_token_selection,
)
from mopd_verl.domain_gradient.token_weighting_metrics import (
    format_loss_amplification_metrics,
    local_loss_amplification_statistics,
    reduce_loss_amplification_statistics,
)
from mopd_verl.domain_gradient.token_weighting_state import (
    CUMULATIVE_ABS_LOSS_SELECTION,
    CumulativeTokenLossState,
    initial_cumulative_token_loss_state,
)
from mopd_verl.domain_gradient.weighting import (
    DOMAIN_GRADIENT_PROJECTION_SHARE_SIGNAL,
    GRADIENT_NORM_SIGNAL,
    DomainWeightState,
    initial_domain_weight_state,
    update_domain_weight_state,
)
from mopd_verl.full_gradient.actor_loss import build_actor_micro_batch_loss
from mopd_verl.full_gradient.config import _cfg_get
from mopd_verl.full_gradient.labels import _labels_from_mapping
from mopd_verl.full_gradient.loss_support import (
    selected_teacher_entropy,
    selected_teacher_log_prob,
)


@dataclass(frozen=True)
class _TokenSelection:
    masks: tuple[torch.Tensor, ...]
    selected_tokens: tuple[RankedToken, ...]
    candidate_token_count: int
    candidate_loss_abs_mass: float
    selected_token_count: int
    selected_loss_abs_mass: float


_CandidateData = tuple[
    dict[str, tuple[LocalTokenCandidate, ...]],
    tuple[torch.Tensor, ...],
]


class DomainGradientAudit:
    """Read-only sidecar for one optimizer mini-batch."""

    def __init__(self, actor: Any, meta: Any):
        self.actor = actor
        self.config = DomainGradientConfig.from_meta(meta)
        self._audit_total: GradientVector = tuple()
        weight_state = getattr(
            actor,
            "_mopd_domain_weight_state",
            None,
        )
        optimizer_groups = getattr(
            getattr(actor, "actor_optimizer", None),
            "param_groups",
            (),
        )
        serialized_state = (
            optimizer_groups[0].get("mopd_domain_weight_state")
            if optimizer_groups
            else None
        )
        if not isinstance(weight_state, DomainWeightState) and isinstance(
            serialized_state,
            dict,
        ):
            weight_state = DomainWeightState.from_mapping(serialized_state)
        if (
            not isinstance(weight_state, DomainWeightState)
            or weight_state.domains != self.config.domains
            or (
                weight_state.signal_source
                != self.config.dynamic_weighting_signal_source
            )
        ):
            weight_state = initial_domain_weight_state(
                self.config.domains,
                signal_source=self.config.dynamic_weighting_signal_source,
            )
        self._weight_state = weight_state
        phase_state = getattr(actor, "_mopd_phase_control_state", None)
        serialized_phase_state = (
            optimizer_groups[0].get("mopd_phase_control_state")
            if optimizer_groups
            else None
        )
        if not isinstance(phase_state, PhaseControlState) and isinstance(
            serialized_phase_state,
            dict,
        ):
            phase_state = PhaseControlState.from_mapping(serialized_phase_state)
        if (
            not isinstance(phase_state, PhaseControlState)
            or phase_state.domains != self.config.domains
        ):
            phase_state = initial_phase_control_state(
                self.config.domains,
                initial_gate=self.config.control_token_phase_gate_initial,
            )
        self._phase_control_state = phase_state
        self._phase_gap_observations: dict[str, PhaseGapObservation] = {}
        speed_state = getattr(actor, "_mopd_control_speed_state", None)
        serialized_speed_state = (
            optimizer_groups[0].get("mopd_control_speed_state")
            if optimizer_groups
            else None
        )
        if not isinstance(speed_state, ControlSpeedState) and isinstance(
            serialized_speed_state,
            dict,
        ):
            speed_state = ControlSpeedState.from_mapping(serialized_speed_state)
        if (
            not isinstance(speed_state, ControlSpeedState)
            or speed_state.domains != self.config.domains
            or speed_state.weight_knots != self.config.control_token_speed_weight_knots
        ):
            speed_state = initial_control_speed_state(
                self.config.domains,
                initial_weight=(self.config.control_token_speed_initial_weight),
                weight_knots=(self.config.control_token_speed_weight_knots),
            )
        self._control_speed_state = speed_state
        self._applied_control_speed_weights = speed_state.weight_map()
        self._control_speed_observations: dict[
            str,
            ControlGapObservation,
        ] = {}
        self._online_control_selection_state: OnlineControlSelectionState | None = None
        self._applied_online_control_token_ids: dict[
            str,
            tuple[int, ...],
        ] = {}
        self._applied_online_control_token_weights: dict[
            str,
            dict[int, float],
        ] = {}
        if self.config.control_token_online_selection_enabled:
            online_state = getattr(
                actor,
                "_mopd_online_control_selection_state",
                None,
            )
            serialized_online_state = (
                optimizer_groups[0].get("mopd_online_control_selection_state")
                if optimizer_groups
                else None
            )
            if not isinstance(online_state, OnlineControlSelectionState) and isinstance(
                serialized_online_state, dict
            ):
                online_state = OnlineControlSelectionState.from_mapping(
                    serialized_online_state
                )
            expected_state = initial_online_control_selection_state(
                self.config.domains,
                self.config.effective_domain_candidate_map(),
                audit_interval_steps=(
                    self.config.control_token_online_audit_interval_steps
                ),
                window_steps=(self.config.control_token_online_window_steps),
                min_mean_occurrences_per_step=(
                    self.config.control_token_online_min_mean_occurrences_per_step
                ),
                top_k=self.config.control_token_online_top_k,
                selection_mode=(
                    self.config.control_token_online_selection_mode
                ),
                weight_mode=self.config.control_token_online_weight_mode,
            )
            if online_state is None:
                online_state = expected_state
            elif (
                online_state.domains != expected_state.domains
                or online_state.domain_candidate_token_ids
                != expected_state.domain_candidate_token_ids
                or online_state.audit_interval_steps
                != expected_state.audit_interval_steps
                or online_state.window_steps != expected_state.window_steps
                or online_state.min_mean_occurrences_per_step
                != expected_state.min_mean_occurrences_per_step
                or online_state.top_k != expected_state.top_k
                or online_state.selection_mode != expected_state.selection_mode
                or online_state.weight_mode != expected_state.weight_mode
            ):
                raise ValueError(
                    "Checkpointed online Control selection state does not "
                    "match the current configuration."
                )
            self._online_control_selection_state = online_state
            has_step_gap = (
                online_state.last_observed_step is not None
                and self.config.step > online_state.last_observed_step + 1
            )
            self._applied_online_control_token_ids = (
                {domain: () for domain in self.config.domains}
                if has_step_gap
                else online_state.active_map()
            )
            self._applied_online_control_token_weights = (
                {domain: {} for domain in self.config.domains}
                if has_step_gap
                else online_state.active_weight_map()
            )
        self._shared_token_selection = SharedTokenSelection(
            token_ids=tuple(),
            domain_top_token_ids=tuple(
                (domain, tuple()) for domain in self.config.domains
            ),
        )
        self._token_id_tensor_cache: dict[
            tuple[str, torch.device, torch.dtype],
            torch.Tensor,
        ] = {}
        self._cumulative_token_loss_state: CumulativeTokenLossState | None = None
        if (
            self.config.all_domain_shared_token_weighting_enabled
            and self.config.all_domain_shared_token_selection_mode
            == CUMULATIVE_ABS_LOSS_SELECTION
        ):
            cumulative_state = getattr(
                actor,
                "_mopd_cumulative_token_loss_state",
                None,
            )
            serialized_cumulative_state = (
                optimizer_groups[0].get("mopd_cumulative_token_loss_state")
                if optimizer_groups
                else None
            )
            if not isinstance(
                cumulative_state,
                CumulativeTokenLossState,
            ) and isinstance(serialized_cumulative_state, dict):
                cumulative_state = CumulativeTokenLossState.from_mapping(
                    serialized_cumulative_state
                )
            if cumulative_state is None:
                cumulative_state = initial_cumulative_token_loss_state(
                    self.config.domains
                )
            if not isinstance(
                cumulative_state,
                CumulativeTokenLossState,
            ):
                raise ValueError("Cumulative token-loss state has an unsupported type.")
            if cumulative_state.domains != self.config.domains:
                raise ValueError(
                    "Cumulative token-loss state domains do not match "
                    "the current shared-token configuration."
                )
            if (
                cumulative_state.selection_mode
                != self.config.all_domain_shared_token_selection_mode
            ):
                raise ValueError(
                    "Cumulative token-loss state selection mode does not "
                    "match the current configuration."
                )
            self._cumulative_token_loss_state = cumulative_state

    def _persist_weight_state(self) -> None:
        self.actor._mopd_domain_weight_state = self._weight_state
        optimizer_groups = getattr(
            getattr(self.actor, "actor_optimizer", None),
            "param_groups",
            (),
        )
        if optimizer_groups:
            optimizer_groups[0][
                "mopd_domain_weight_state"
            ] = self._weight_state.as_dict()

    def _persist_phase_control_state(self) -> None:
        self.actor._mopd_phase_control_state = self._phase_control_state
        optimizer_groups = getattr(
            getattr(self.actor, "actor_optimizer", None),
            "param_groups",
            (),
        )
        if optimizer_groups:
            optimizer_groups[0][
                "mopd_phase_control_state"
            ] = self._phase_control_state.as_dict()

    def _persist_control_speed_state(self) -> None:
        self.actor._mopd_control_speed_state = self._control_speed_state
        optimizer_groups = getattr(
            getattr(self.actor, "actor_optimizer", None),
            "param_groups",
            (),
        )
        if optimizer_groups:
            optimizer_groups[0][
                "mopd_control_speed_state"
            ] = self._control_speed_state.as_dict()

    def _persist_cumulative_token_loss_state(self) -> None:
        state = self._cumulative_token_loss_state
        if state is None:
            return
        self.actor._mopd_cumulative_token_loss_state = state
        optimizer_groups = getattr(
            getattr(self.actor, "actor_optimizer", None),
            "param_groups",
            (),
        )
        if optimizer_groups:
            optimizer_groups[0]["mopd_cumulative_token_loss_state"] = state.as_dict()

    def _persist_online_control_selection_state(self) -> None:
        state = self._online_control_selection_state
        if state is None:
            return
        self.actor._mopd_online_control_selection_state = state
        optimizer_groups = getattr(
            getattr(self.actor, "actor_optimizer", None),
            "param_groups",
            (),
        )
        if optimizer_groups:
            optimizer_groups[0]["mopd_online_control_selection_state"] = state.as_dict()

    def _should_update_dynamic_weighting(self) -> bool:
        return (
            self.config.dynamic_weighting_enabled
            and self.config.dynamic_weighting_update_enabled
            and self._weight_state.last_updated_step != self.config.step
        )

    def _production_weighting_enabled(self) -> bool:
        return (
            self.config.dynamic_weighting_enabled
            or self.config.control_token_weighting_enabled
            or self.config.all_domain_shared_token_weighting_enabled
        )

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

    def _cached_token_id_tensor(
        self,
        name: str,
        token_ids: Sequence[int],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Reuse immutable selected-token tensors across micro-batches."""

        key = (name, reference.device, reference.dtype)
        cached = self._token_id_tensor_cache.get(key)
        if cached is None:
            cached = torch.tensor(
                tuple(token_ids),
                device=reference.device,
                dtype=reference.dtype,
            )
            self._token_id_tensor_cache[key] = cached
        return cached

    def _active_domain_control_token_ids(
        self,
    ) -> dict[str, tuple[int, ...]]:
        if self.config.control_token_online_selection_enabled:
            return dict(self._applied_online_control_token_ids)
        return dict(self.config.domain_control_token_ids)

    def _domain_control_token_tensor_map(
        self,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self._cached_token_id_tensor(
                f"domain_control:{domain}",
                token_ids,
                reference,
            )
            for domain, token_ids in (self._active_domain_control_token_ids().items())
        }

    def training_gradient_mask(
        self,
        micro_batch: Any,
    ) -> torch.Tensor | None:
        """Return current domain and token production gradient multipliers."""

        token_weighting_enabled = (
            self.config.control_token_weighting_enabled
            or self.config.all_domain_shared_token_weighting_enabled
        )
        if not self.config.dynamic_weighting_enabled and not token_weighting_enabled:
            return None
        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
        response_mask = model_inputs["response_mask"]
        gradient_weights = torch.ones_like(
            response_mask,
            dtype=torch.float32,
        )
        if self.config.dynamic_weighting_enabled:
            labels = _labels_from_mapping(
                model_inputs,
                int(response_mask.shape[0]),
            )
            domain_weights = self._weight_state.weight_map()
            rows = torch.tensor(
                [domain_weights.get(label, 1.0) for label in labels],
                device=response_mask.device,
                dtype=torch.float32,
            )
            gradient_weights = gradient_weights * rows.unsqueeze(-1)
        if token_weighting_enabled:
            token_ids = aligned_response_token_ids(
                model_inputs,
                response_mask,
            )
            if token_ids is None:
                raise ValueError(
                    "Token loss weighting requires response-aligned token "
                    "IDs in responses, response_ids, or input_ids."
                )
            labels = _labels_from_mapping(
                model_inputs,
                int(response_mask.shape[0]),
            )
            if self.config.control_token_weighting_enabled and (
                self.config.domain_control_token_ids
                or self._applied_online_control_token_ids
            ):
                domain_token_ids = self._domain_control_token_tensor_map(token_ids)
                if (
                    self.config.control_token_online_selection_enabled
                    and self.config.control_token_online_weight_mode
                    == PAIRED_ONLINE_WEIGHT_MODE
                ):
                    token_weights = online_token_score_weights(
                        token_ids,
                        response_mask,
                        labels,
                        domain_token_weights=(
                            self._applied_online_control_token_weights
                        ),
                        normalize_per_domain=(
                            self.config.control_token_normalize_per_domain
                        ),
                    )
                elif self.config.control_token_speed_weighting_enabled:
                    token_weights = domain_control_token_weights(
                        token_ids,
                        response_mask,
                        labels,
                        domain_token_ids=domain_token_ids,
                        domain_weights=(self._applied_control_speed_weights),
                        normalize_per_domain=(
                            self.config.control_token_normalize_per_domain
                        ),
                    )
                else:
                    token_weights = phase_token_weights(
                        token_ids,
                        response_mask,
                        labels,
                        domain_token_ids=domain_token_ids,
                        control_weight=self.config.control_token_weight,
                        phase_enabled=(self.config.control_token_phase_gate_enabled),
                        span_enabled=(self.config.control_token_span_weighting_enabled),
                        phase_gates=self._phase_control_state.gate_map(),
                        span_length=self.config.control_token_span_length,
                        span_decay_tau=(self.config.control_token_span_decay_tau),
                        normalize_per_domain=(
                            self.config.control_token_normalize_per_domain
                        ),
                    )
                gradient_weights = gradient_weights * token_weights
            gradient_weights = gradient_weights * token_gradient_weights(
                token_ids,
                control_token_ids=(
                    self._cached_token_id_tensor(
                        "control",
                        self.config.control_token_ids,
                        token_ids,
                    )
                    if (
                        self.config.control_token_weighting_enabled
                        and self.config.control_token_ids
                    )
                    else ()
                ),
                control_token_weight=self.config.control_token_weight,
                shared_token_ids=(
                    self._cached_token_id_tensor(
                        "shared",
                        self._shared_token_selection.token_ids,
                        token_ids,
                    )
                    if self.config.all_domain_shared_token_weighting_enabled
                    else ()
                ),
                shared_token_weight=(self.config.all_domain_shared_token_weight),
            )
        return gradient_weights

    def _production_gradient_masks(
        self,
        micro_batches: Sequence[Any],
    ) -> tuple[torch.Tensor, ...]:
        """Snapshot the exact token multipliers used by production backward."""

        optional_masks = tuple(
            self.training_gradient_mask(micro_batch) for micro_batch in micro_batches
        )
        if any(mask is None for mask in optional_masks):
            raise RuntimeError(
                "Production-weighted token gradients require training "
                "gradient masks."
            )
        return tuple(mask for mask in optional_masks if mask is not None)

    def _domain_production_gradient_masks(
        self,
        micro_batches: Sequence[Any],
        production_masks: Sequence[torch.Tensor],
        domain: str,
    ) -> tuple[torch.Tensor, ...]:
        """Restrict production token multipliers to one configured domain."""

        if len(micro_batches) != len(production_masks):
            raise ValueError(
                "Production masks must align one-to-one with micro-batches."
            )
        return tuple(
            production_mask
            * self._domain_gradient_mask(micro_batch, domain).to(
                device=production_mask.device,
                dtype=production_mask.dtype,
            )
            for micro_batch, production_mask in zip(
                micro_batches,
                production_masks,
                strict=True,
            )
        )

    @staticmethod
    def _apply_production_gradient_masks(
        selection_masks: Sequence[torch.Tensor],
        production_masks: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Compose a binary audit selection with production reweighting."""

        if len(selection_masks) != len(production_masks):
            raise ValueError(
                "Selection and production masks must have the same length."
            )
        return tuple(
            selection_mask
            * production_mask.to(
                device=selection_mask.device,
                dtype=selection_mask.dtype,
            )
            for selection_mask, production_mask in zip(
                selection_masks,
                production_masks,
                strict=True,
            )
        )

    def _reweighted_candidate_data(
        self,
        candidate_data: _CandidateData,
        production_masks: Sequence[torch.Tensor],
    ) -> _CandidateData:
        """Rank tokens by post-reweight absolute gradient-loss mass."""

        candidates_by_domain, mask_templates = candidate_data
        if len(mask_templates) != len(production_masks):
            raise ValueError("Candidate templates and production masks must align.")
        masks_cpu = tuple(mask.detach().float().cpu() for mask in production_masks)
        reweighted: dict[str, tuple[LocalTokenCandidate, ...]] = {}
        for domain, candidates in candidates_by_domain.items():
            weighted_candidates: list[LocalTokenCandidate] = []
            for candidate in candidates:
                multiplier = float(
                    masks_cpu[candidate.micro_batch_index][
                        candidate.sample_index,
                        candidate.token_index,
                    ].item()
                )
                if not math.isfinite(multiplier):
                    raise ValueError(
                        "Production gradient masks must contain finite values."
                    )
                weighted_candidates.append(
                    LocalTokenCandidate(
                        micro_batch_index=candidate.micro_batch_index,
                        sample_index=candidate.sample_index,
                        token_index=candidate.token_index,
                        token_id=candidate.token_id,
                        configured_loss=(candidate.configured_loss * multiplier),
                        loss_abs=(candidate.loss_abs * abs(multiplier)),
                    )
                )
            reweighted[domain] = tuple(weighted_candidates)
        return reweighted, mask_templates

    def _backward_replay(
        self,
        state: AuditState,
        micro_batches: Sequence[Any],
        loss_scales: Sequence[float],
        *,
        on_policy: bool,
        temperature: float,
        domain: str | None,
        gradient_masks: Sequence[torch.Tensor] | None = None,
        collect_loss_abs_candidates: bool = False,
    ) -> _CandidateData | None:
        if gradient_masks is not None and len(gradient_masks) != len(micro_batches):
            raise ValueError(
                "Token selection must provide one gradient mask per micro-batch."
            )
        candidates: dict[str, list[LocalTokenCandidate]] = {
            domain_name: [] for domain_name in self.config.domains
        }
        mask_templates: list[torch.Tensor] = []
        state.restore_runtime()
        state.clear_gradients()
        for index, (micro_batch, loss_scale) in enumerate(
            zip(micro_batches, loss_scales, strict=True)
        ):
            gradient_mask = (
                gradient_masks[index]
                if gradient_masks is not None
                else (
                    self._domain_gradient_mask(micro_batch, domain)
                    if domain is not None
                    else None
                )
            )
            result = build_actor_micro_batch_loss(
                self.actor,
                micro_batch,
                loss_scale_factor=float(loss_scale),
                on_policy=on_policy,
                gradient_mask_override=gradient_mask,
                include_metrics=False,
                return_configured_token_loss=collect_loss_abs_candidates,
                temperature=temperature,
            )
            if self.actor.scaler is not None:
                self.actor.scaler.scale(result.loss).backward()
            else:
                result.loss.backward()
            if collect_loss_abs_candidates:
                micro_candidates, mask_template = (
                    self._loss_abs_candidates_for_micro_batch(
                        result,
                        micro_batch,
                        micro_batch_index=index,
                    )
                )
                for domain_name in self.config.domains:
                    candidates[domain_name].extend(micro_candidates[domain_name])
                mask_templates.append(mask_template)
        if not collect_loss_abs_candidates:
            return None
        return (
            {
                domain_name: tuple(domain_candidates)
                for domain_name, domain_candidates in candidates.items()
            },
            tuple(mask_templates),
        )

    def _snapshot_training_gradient_reference(
        self,
        state: AuditState,
        micro_batches: Sequence[Any],
        loss_scales: Sequence[float],
        *,
        on_policy: bool,
        temperature: float,
    ) -> GradientVector:
        """Replay the exact dynamic-weighted production gradient for parity."""

        gradient_masks = self._production_gradient_masks(micro_batches)
        self._backward_replay(
            state,
            micro_batches,
            loss_scales,
            on_policy=on_policy,
            temperature=temperature,
            domain=None,
            gradient_masks=gradient_masks,
        )
        return snapshot_gradients(
            self.actor,
            self.config.storage_dtype,
        )

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
        values = torch.tensor(
            [total, *counts], device=get_device_id(), dtype=torch.float64
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        total_count, *domain_counts = (float(value) for value in values.tolist())
        metrics = {
            f"{domain}/pre_reweight_full_grad/sample_count": (domain_counts[index])
            for index, domain in enumerate(self.config.domains)
        }
        metrics["global/audit/domain_gradient_coverage_fraction"] = (
            sum(domain_counts) / total_count if total_count > 0.0 else 0.0
        )
        return metrics

    def _loss_abs_candidates_for_micro_batch(
        self,
        result: Any,
        micro_batch: Any,
        *,
        micro_batch_index: int,
    ) -> tuple[
        dict[str, tuple[LocalTokenCandidate, ...]],
        torch.Tensor,
    ]:
        configured_loss = result.configured_token_loss
        configured_mask = result.configured_token_loss_mask
        if configured_loss is None or configured_mask is None:
            raise RuntimeError(
                "Loss-ranked token gradients require configured token loss."
            )
        if configured_loss.shape != configured_mask.shape:
            raise ValueError("Configured token loss and mask must have the same shape.")
        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
        labels = _labels_from_mapping(
            model_inputs,
            int(configured_loss.shape[0]),
        )
        token_ids_cpu = response_token_ids(model_inputs, configured_loss)
        loss_cpu = configured_loss.detach().float().cpu()
        loss_abs_cpu = loss_cpu.abs()
        active_cpu = configured_mask.detach().bool().cpu()
        candidates: dict[str, list[LocalTokenCandidate]] = {
            domain: [] for domain in self.config.domains
        }
        for sample_index, domain in enumerate(labels):
            if domain not in candidates:
                continue
            positions = torch.nonzero(
                active_cpu[sample_index],
                as_tuple=False,
            ).flatten()
            for token_index in positions.tolist():
                loss_abs = float(loss_abs_cpu[sample_index, token_index].item())
                if not math.isfinite(loss_abs):
                    continue
                configured_token_loss = float(
                    loss_cpu[sample_index, token_index].item()
                )
                candidates[domain].append(
                    LocalTokenCandidate(
                        micro_batch_index=micro_batch_index,
                        sample_index=sample_index,
                        token_index=int(token_index),
                        token_id=(
                            int(
                                token_ids_cpu[
                                    sample_index,
                                    token_index,
                                ].item()
                            )
                            if token_ids_cpu is not None
                            else None
                        ),
                        configured_loss=configured_token_loss,
                        loss_abs=loss_abs,
                    )
                )
        return (
            {
                domain: tuple(domain_candidates)
                for domain, domain_candidates in candidates.items()
            },
            torch.zeros_like(configured_mask, dtype=torch.float32),
        )

    def _collect_loss_abs_candidates(
        self,
        micro_batches: Sequence[Any],
        loss_scales: Sequence[float],
        *,
        on_policy: bool,
        temperature: float,
    ) -> _CandidateData:
        candidates: dict[str, list[LocalTokenCandidate]] = {
            domain: [] for domain in self.config.domains
        }
        mask_templates: list[torch.Tensor] = []
        for micro_batch_index, (micro_batch, loss_scale) in enumerate(
            zip(micro_batches, loss_scales, strict=True)
        ):
            with torch.no_grad():
                result = build_actor_micro_batch_loss(
                    self.actor,
                    micro_batch,
                    loss_scale_factor=float(loss_scale),
                    on_policy=on_policy,
                    include_metrics=False,
                    return_configured_token_loss=True,
                    temperature=temperature,
                )
            micro_candidates, mask_template = self._loss_abs_candidates_for_micro_batch(
                result,
                micro_batch,
                micro_batch_index=micro_batch_index,
            )
            for domain in self.config.domains:
                candidates[domain].extend(micro_candidates[domain])
            mask_templates.append(mask_template)
        return (
            {
                domain: tuple(domain_candidates)
                for domain, domain_candidates in candidates.items()
            },
            tuple(mask_templates),
        )

    @staticmethod
    def _global_ranked_tokens(
        local_candidates: Sequence[LocalTokenCandidate],
    ) -> tuple[RankedToken, ...]:
        local_scores = [float(candidate.loss_abs) for candidate in local_candidates]
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        gathered_scores: list[list[float] | None]
        if not distributed:
            gathered_scores = [local_scores]
        else:
            gathered_scores = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered_scores, local_scores)
        return tuple(
            RankedToken(
                owner_rank=owner_rank,
                owner_index=owner_index,
                loss_abs=float(loss_abs),
            )
            for owner_rank, rank_scores in enumerate(gathered_scores)
            for owner_index, loss_abs in enumerate(rank_scores or ())
        )

    @staticmethod
    def _selection_masks(
        selected: Sequence[RankedToken],
        local_candidates: Sequence[LocalTokenCandidate],
        mask_templates: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        local_rank = torch.distributed.get_rank() if distributed else 0
        masks = [torch.zeros_like(template) for template in mask_templates]
        positions_by_micro_batch: list[list[tuple[int, int]]] = [
            [] for _ in mask_templates
        ]
        for token in selected:
            if token.owner_rank != local_rank:
                continue
            candidate = local_candidates[token.owner_index]
            positions_by_micro_batch[candidate.micro_batch_index].append(
                (candidate.sample_index, candidate.token_index)
            )
        for micro_batch_index, positions in enumerate(positions_by_micro_batch):
            if not positions:
                continue
            sample_indices, token_indices = zip(*positions, strict=True)
            device = masks[micro_batch_index].device
            masks[micro_batch_index][
                torch.tensor(sample_indices, device=device),
                torch.tensor(token_indices, device=device),
            ] = 1.0
        return tuple(masks)

    def _make_token_selection(
        self,
        selected: Sequence[RankedToken],
        global_candidates: Sequence[RankedToken],
        local_candidates: Sequence[LocalTokenCandidate],
        mask_templates: Sequence[torch.Tensor],
    ) -> _TokenSelection:
        return _TokenSelection(
            masks=self._selection_masks(
                selected,
                local_candidates,
                mask_templates,
            ),
            selected_tokens=tuple(selected),
            candidate_token_count=len(global_candidates),
            candidate_loss_abs_mass=total_loss_abs_mass(global_candidates),
            selected_token_count=len(selected),
            selected_loss_abs_mass=total_loss_abs_mass(selected),
        )

    def _loss_ranked_token_selections(
        self,
        micro_batches: Sequence[Any],
        loss_scales: Sequence[float],
        *,
        on_policy: bool,
        temperature: float,
        candidate_data: (
            tuple[
                dict[str, tuple[LocalTokenCandidate, ...]],
                tuple[torch.Tensor, ...],
            ]
            | None
        ) = None,
        candidate_loss_basis: str = "configured_loss_abs",
    ) -> dict[str, dict[str, _TokenSelection]]:
        local_by_domain, mask_templates = (
            candidate_data
            if candidate_data is not None
            else self._collect_loss_abs_candidates(
                micro_batches,
                loss_scales,
                on_policy=on_policy,
                temperature=temperature,
            )
        )
        selections: dict[str, dict[str, _TokenSelection]] = {}
        for domain in self.config.domains:
            local_candidates = local_by_domain[domain]
            global_candidates = self._global_ranked_tokens(local_candidates)
            domain_selections: dict[str, _TokenSelection] = {}
            if self.config.token_gradient_tail_enabled:
                selected_tail = select_tail_loss_mass(
                    global_candidates,
                    self.config.token_gradient_tail_fraction,
                    minimum_tokens=self.config.token_gradient_tail_min_tokens,
                )
                domain_selections["tail"] = self._make_token_selection(
                    selected_tail,
                    global_candidates,
                    local_candidates,
                    mask_templates,
                )
            if self.config.token_gradient_top_p_enabled:
                if self.config.token_gradient_top_k is not None:
                    selected_top_k = select_top_k(
                        global_candidates,
                        self.config.token_gradient_top_k,
                    )
                    domain_selections["top_k"] = self._make_token_selection(
                        selected_top_k,
                        global_candidates,
                        local_candidates,
                        mask_templates,
                    )
                selected_top_p = select_top_loss_mass(
                    global_candidates,
                    self.config.token_gradient_top_p,
                )
                domain_selections["top_p"] = self._make_token_selection(
                    selected_top_p,
                    global_candidates,
                    local_candidates,
                    mask_templates,
                )
            selections[domain] = domain_selections
        if self.config.token_gradient_log_tokens_jsonl_enabled:
            append_token_vocab_vectors_jsonl(
                output_dir=self.config.output_dir,
                step=self.config.step,
                configured_vocab_size=self.config.token_gradient_vocab_size,
                candidates_by_domain=local_by_domain,
                selections_by_domain={
                    domain: {
                        selection_name: selection.selected_tokens
                        for selection_name, selection in domain_selections.items()
                    }
                    for domain, domain_selections in selections.items()
                },
                loss_mass_basis=candidate_loss_basis,
            )
        return selections

    def _refresh_shared_token_selection(
        self,
        candidates_by_domain: dict[
            str,
            tuple[LocalTokenCandidate, ...],
        ],
    ) -> dict[str, float]:
        if (
            self.config.all_domain_shared_token_selection_mode
            == CUMULATIVE_ABS_LOSS_SELECTION
        ):
            if self._cumulative_token_loss_state is None:
                raise RuntimeError("Cumulative shared-token selection requires state.")
            (
                self._shared_token_selection,
                self._cumulative_token_loss_state,
            ) = update_cumulative_shared_token_selection(
                candidates_by_domain,
                self.config.domains,
                top_k=self.config.all_domain_shared_token_top_k,
                step=self.config.step,
                state=self._cumulative_token_loss_state,
            )
            self._persist_cumulative_token_loss_state()
        else:
            self._shared_token_selection = select_all_domain_shared_tokens(
                candidates_by_domain,
                self.config.domains,
                top_k=self.config.all_domain_shared_token_top_k,
            )
        self._token_id_tensor_cache = {
            key: value
            for key, value in self._token_id_tensor_cache.items()
            if key[0] != "shared"
        }
        append_shared_token_selection_jsonl(
            output_dir=self.config.output_dir,
            step=self.config.step,
            top_k=self.config.all_domain_shared_token_top_k,
            selection=self._shared_token_selection,
            selection_mode=(self.config.all_domain_shared_token_selection_mode),
            cumulative_state=self._cumulative_token_loss_state,
        )
        metrics: dict[str, float] = {
            "global/token_weight/all_domain_shared_token_type_count": float(
                len(self._shared_token_selection.token_ids)
            ),
            "global/token_weight/all_domain_shared_token_weight": (
                self.config.all_domain_shared_token_weight
            ),
            "global/token_weight/shared_selection_is_cumulative": float(
                self.config.all_domain_shared_token_selection_mode
                == CUMULATIVE_ABS_LOSS_SELECTION
            ),
        }
        for domain, token_ids in self._shared_token_selection.domain_top_token_ids:
            metrics[f"{domain}/token_weight/high_loss_token_type_count"] = float(
                len(token_ids)
            )
        if self._cumulative_token_loss_state is not None:
            for (
                domain,
                summary,
            ) in self._cumulative_token_loss_state.domain_summaries().items():
                for name, value in summary.items():
                    metrics[f"{domain}/token_weight/{name}"] = value
        return metrics

    def _token_weighting_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self.config.control_token_weighting_enabled:
            metrics.update(
                {
                    "global/token_weight/control_token_id_count": float(
                        len(self.config.control_token_ids)
                    ),
                    "global/token_weight/control_token_weight": (
                        self.config.control_token_weight
                    ),
                }
            )
            domain_control_token_ids = self._active_domain_control_token_ids()
            if domain_control_token_ids:
                metrics["global/token_weight/domain_control_token_id_count"] = float(
                    sum(
                        len(token_ids)
                        for token_ids in domain_control_token_ids.values()
                    )
                )
                for domain, token_ids in domain_control_token_ids.items():
                    metrics[f"{domain}/token_weight/control_token_id_count"] = float(
                        len(token_ids)
                    )
            if self.config.control_token_online_selection_enabled:
                metrics.update(
                    {
                        "global/token_weight/candidate_token_count": float(
                            len(
                                {
                                    token_id
                                    for token_ids in self.config.effective_domain_candidate_map().values()
                                    for token_id in token_ids
                                }
                            )
                        ),
                        "global/token_weight/window_steps": float(
                            self.config.control_token_online_window_steps
                        ),
                    }
                )
        if self.config.all_domain_shared_token_weighting_enabled:
            metrics.update(
                {
                    "global/token_weight/"
                    "all_domain_shared_token_type_count": float(
                        len(self._shared_token_selection.token_ids)
                    ),
                    "global/token_weight/all_domain_shared_token_weight": (
                        self.config.all_domain_shared_token_weight
                    ),
                }
            )
        return metrics

    def _collect_control_speed_observations(
        self,
        micro_batches: Sequence[Any],
    ) -> dict[str, ControlGapObservation]:
        domains = tuple(domain for domain, _ in self.config.domain_control_token_ids)
        totals = {domain: [0.0, 0.0, 0.0] for domain in domains}
        policy_loss_cfg = _cfg_get(self.actor.config, "policy_loss", {})
        for micro_batch in micro_batches:
            model_inputs = {
                **micro_batch.batch,
                **micro_batch.non_tensor_batch,
            }
            response_mask = model_inputs["response_mask"]
            token_ids = aligned_response_token_ids(
                model_inputs,
                response_mask,
            )
            if token_ids is None:
                raise ValueError(
                    "Control-speed weighting requires response-aligned token " "IDs."
                )
            teacher_log_prob = selected_teacher_log_prob(
                model_inputs,
                policy_loss_cfg,
            )
            student_log_prob = model_inputs["old_log_probs"]
            if teacher_log_prob.shape != student_log_prob.shape:
                raise ValueError(
                    "Control-speed teacher and student log-probability "
                    "shapes must match."
                )
            labels = _labels_from_mapping(
                model_inputs,
                int(response_mask.shape[0]),
            )
            local = control_gap_and_weight_totals(
                token_ids,
                response_mask,
                labels,
                (teacher_log_prob - student_log_prob).detach().float().abs(),
                domain_token_ids=(self._domain_control_token_tensor_map(token_ids)),
                applied_domain_weights=(self._applied_control_speed_weights),
                normalize_per_domain=(self.config.control_token_normalize_per_domain),
            )
            for domain, values in local.items():
                for index, value in enumerate(values):
                    totals[domain][index] += float(value)

        flattened = [value for domain in domains for value in totals[domain]]
        if not flattened:
            return {}
        reduced = torch.tensor(
            flattened,
            device=get_device_id(),
            dtype=torch.float64,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                reduced,
                op=torch.distributed.ReduceOp.SUM,
            )
        values = iter(float(value) for value in reduced.tolist())
        observations: dict[str, ControlGapObservation] = {}
        for domain in domains:
            gap_sum = next(values)
            count = round(next(values))
            normalized_weight_sum = next(values)
            if count <= 0:
                continue
            observations[domain] = ControlGapObservation(
                gap=gap_sum / count,
                count=count,
                applied_normalized_weight=(normalized_weight_sum / count),
            )
        return observations

    def _control_speed_metrics(self) -> dict[str, float]:
        if not self.config.control_token_speed_weighting_enabled:
            return {}
        metrics = {
            "global/control_speed/enabled": 1.0,
            "global/control_speed/window_steps": float(
                self.config.control_token_speed_window_steps
            ),
            "global/control_speed/update_interval_steps": float(
                self.config.control_token_speed_update_interval_steps
            ),
        }
        for index, domain in enumerate(self._control_speed_state.domains):
            prefix = f"{domain}/control_speed"
            observation = self._control_speed_observations.get(domain)
            reference_step = self._control_speed_state.reference_steps[index]
            update_step = self._control_speed_state.last_weight_update_steps[index]
            metrics[f"{prefix}/control_gap_ema"] = self._control_speed_state.gap_emas[
                index
            ]
            metrics[f"{prefix}/optimization_speed"] = self._control_speed_state.speeds[
                index
            ]
            if reference_step is not None:
                metrics[f"{prefix}/control_weight_mapped_from_speed"] = (
                    piecewise_linear_weight(
                        self._control_speed_state.speeds[index],
                        self._control_speed_state.weight_knots,
                    )
                )
            metrics[f"{prefix}/speed_reference_step"] = float(
                -1 if reference_step is None else reference_step
            )
            metrics[f"{prefix}/speed_computed_this_step"] = float(
                reference_step
                == self.config.step - self.config.control_token_speed_window_steps
            )
            metrics[f"{prefix}/weight_update_triggered"] = float(
                update_step == self.config.step
            )
            metrics[f"{prefix}/state_observation_count"] = float(
                self._control_speed_state.observation_counts[index]
            )
            metrics[f"{prefix}/control_weight_applied_raw"] = float(
                self._applied_control_speed_weights.get(domain, 1.0)
            )
            metrics[f"{prefix}/control_weight_next"] = (
                self._control_speed_state.weights[index]
            )
            if observation is not None:
                metrics[f"{prefix}/observation_available"] = 1.0
                metrics[f"{prefix}/minimum_occurrences_met"] = float(
                    observation.count >= self.config.control_token_speed_min_occurrences
                )
                metrics[f"{prefix}/control_gap_raw"] = observation.gap
                metrics[f"{prefix}/control_occurrence_count"] = float(observation.count)
                metrics[f"{prefix}/control_weight_applied_normalized"] = (
                    observation.applied_normalized_weight
                )
            else:
                metrics[f"{prefix}/observation_available"] = 0.0
                metrics[f"{prefix}/minimum_occurrences_met"] = 0.0
                metrics[f"{prefix}/control_occurrence_count"] = 0.0
        return metrics

    def _update_control_speed(
        self,
        micro_batches: Sequence[Any],
    ) -> dict[str, float]:
        if not self.config.control_token_speed_weighting_enabled:
            return {}
        self._control_speed_observations = self._collect_control_speed_observations(
            micro_batches
        )
        self._control_speed_state = update_control_speed_state(
            self._control_speed_state,
            self._control_speed_observations,
            window_steps=self.config.control_token_speed_window_steps,
            ema_beta=self.config.control_token_speed_ema_beta,
            update_interval_steps=(
                self.config.control_token_speed_update_interval_steps
            ),
            minimum_occurrences=(self.config.control_token_speed_min_occurrences),
            step=self.config.step,
        )
        self._persist_control_speed_state()
        return self._control_speed_metrics()

    def _collect_phase_gap_observations(
        self,
        micro_batches: Sequence[Any],
    ) -> dict[str, PhaseGapObservation]:
        domains = tuple(domain for domain, _ in self.config.domain_control_token_ids)
        totals = {domain: [0.0, 0.0, 0.0, 0.0] for domain in domains}
        policy_loss_cfg = _cfg_get(self.actor.config, "policy_loss", {})
        for micro_batch in micro_batches:
            model_inputs = {
                **micro_batch.batch,
                **micro_batch.non_tensor_batch,
            }
            response_mask = model_inputs["response_mask"]
            token_ids = aligned_response_token_ids(
                model_inputs,
                response_mask,
            )
            if token_ids is None:
                raise ValueError("Phase control requires response-aligned token IDs.")
            teacher_log_prob = selected_teacher_log_prob(
                model_inputs,
                policy_loss_cfg,
            )
            old_log_prob = model_inputs["old_log_probs"]
            if teacher_log_prob.shape != old_log_prob.shape:
                raise ValueError(
                    "Phase-control teacher and student log-probability "
                    "shapes must match."
                )
            labels = _labels_from_mapping(
                model_inputs,
                int(response_mask.shape[0]),
            )
            local = gap_observations(
                token_ids,
                response_mask,
                labels,
                (teacher_log_prob - old_log_prob).detach().float().abs(),
                domain_token_ids=(self._domain_control_token_tensor_map(token_ids)),
                span_length=self.config.control_token_span_length,
                span_decay_tau=self.config.control_token_span_decay_tau,
            )
            for domain, values in local.items():
                for index, value in enumerate(values):
                    totals[domain][index] += float(value)

        flattened = [value for domain in domains for value in totals[domain]]
        if not flattened:
            return {}
        reduced = torch.tensor(
            flattened,
            device=get_device_id(),
            dtype=torch.float64,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                reduced,
                op=torch.distributed.ReduceOp.SUM,
            )
        values = iter(float(value) for value in reduced.tolist())
        observations: dict[str, PhaseGapObservation] = {}
        for domain in domains:
            control_sum = next(values)
            span_sum = next(values)
            control_count = round(next(values))
            span_count = round(next(values))
            if control_count <= 0 or span_count <= 0:
                continue
            observations[domain] = PhaseGapObservation(
                control_gap=control_sum / control_count,
                span_gap=span_sum / span_count,
                control_count=control_count,
                span_count=span_count,
            )
        return observations

    def _phase_control_metrics(self) -> dict[str, float]:
        if not self.config.control_token_phase_gate_enabled:
            return {}
        metrics = {
            "global/phase_control/enabled": 1.0,
            "global/phase_control/span_weighting_enabled": float(
                self.config.control_token_span_weighting_enabled
            ),
        }
        for index, domain in enumerate(self._phase_control_state.domains):
            prefix = f"{domain}/phase_control"
            observation = self._phase_gap_observations.get(domain)
            metrics[f"{prefix}/gate"] = self._phase_control_state.gates[index]
            metrics[f"{prefix}/control_gap_ema"] = (
                self._phase_control_state.control_gap_ema[index]
            )
            metrics[f"{prefix}/span_gap_ema"] = self._phase_control_state.span_gap_ema[
                index
            ]
            if observation is not None:
                metrics[f"{prefix}/control_gap"] = observation.control_gap
                metrics[f"{prefix}/span_gap"] = observation.span_gap
                metrics[f"{prefix}/control_occurrence_count"] = float(
                    observation.control_count
                )
                metrics[f"{prefix}/span_occurrence_count"] = float(
                    observation.span_count
                )
        return metrics

    def _update_phase_control(
        self,
        micro_batches: Sequence[Any],
    ) -> dict[str, float]:
        if not self.config.control_token_phase_gate_enabled:
            return self._phase_control_metrics()
        self._phase_gap_observations = self._collect_phase_gap_observations(
            micro_batches
        )
        self._phase_control_state = update_phase_control_state(
            self._phase_control_state,
            self._phase_gap_observations,
            window_steps=self.config.control_token_phase_gate_window_steps,
            ema_beta=self.config.control_token_phase_gate_ema_beta,
            temperature=self.config.control_token_phase_gate_temperature,
            step=self.config.step,
        )
        self._persist_phase_control_state()
        return self._phase_control_metrics()

    def _loss_amplification_metrics(
        self,
        candidates_by_domain: dict[
            str,
            tuple[LocalTokenCandidate, ...],
        ],
        micro_batches: Sequence[Any],
    ) -> dict[str, float]:
        """Compare raw and effective configured-loss mass for production masks."""

        actual_masks = tuple(
            mask.detach().float().cpu()
            for micro_batch in micro_batches
            if (mask := self.training_gradient_mask(micro_batch)) is not None
        )
        if len(actual_masks) != len(micro_batches):
            raise RuntimeError(
                "Loss amplification metrics require production gradient masks."
            )
        local_by_domain = local_loss_amplification_statistics(
            candidates_by_domain,
            self.config.domains,
            actual_masks,
            domain_weights=self._weight_state.weight_map(),
            dynamic_weighting_enabled=self.config.dynamic_weighting_enabled,
            control_token_ids=self.config.control_token_ids,
            domain_control_token_ids=(self._active_domain_control_token_ids()),
            control_weighting_enabled=(self.config.control_token_weighting_enabled),
            control_weight=self.config.control_token_weight,
            shared_token_ids=self._shared_token_selection.token_ids,
            shared_weighting_enabled=(
                self.config.all_domain_shared_token_weighting_enabled
            ),
            shared_weight=self.config.all_domain_shared_token_weight,
        )
        return format_loss_amplification_metrics(
            reduce_loss_amplification_statistics(local_by_domain)
        )

    def observe_completed_step(
        self,
        micro_batches: Sequence[Any],
        configured_loss_batches: Sequence[torch.Tensor],
        configured_loss_mask_batches: Sequence[torch.Tensor],
        *,
        selector_token_loss_batches: Sequence[torch.Tensor] | None = None,
        selector_token_loss_mask_batches: Sequence[torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Update the online selector after a successful optimizer step."""

        if not self.config.control_token_online_selection_enabled:
            return {}
        state = self._online_control_selection_state
        if state is None:
            raise RuntimeError("Online Control selector state is unavailable.")
        if not (
            len(micro_batches)
            == len(configured_loss_batches)
            == len(configured_loss_mask_batches)
        ):
            raise ValueError(
                "Online Control configured-loss outputs must align with "
                "production micro-batches."
            )
        kl_entropy_mode = (
            self.config.control_token_online_selection_mode
            == TOP_KL_STUDENT_ENTROPY_SELECTION_MODE
        )
        if kl_entropy_mode and (
            selector_token_loss_batches is None
            or len(selector_token_loss_batches) != len(micro_batches)
            or selector_token_loss_mask_batches is None
            or len(selector_token_loss_mask_batches) != len(micro_batches)
        ):
            raise ValueError(
                "KL + Student-entropy selection requires one detached raw "
                "Top-K loss matrix per production micro-batch."
            )
        selection_loss_batches = (
            selector_token_loss_batches
            if kl_entropy_mode
            else configured_loss_batches
        )
        if selection_loss_batches is None:
            raise RuntimeError("Online Control selection loss batches are missing.")
        selection_loss_mask_batches = (
            selector_token_loss_mask_batches
            if kl_entropy_mode
            else configured_loss_mask_batches
        )
        if selection_loss_mask_batches is None:
            raise RuntimeError("Online Control selection masks are missing.")

        token_id_batches: list[torch.Tensor] = []
        label_batches: list[Sequence[str]] = []
        student_entropy_batches: list[torch.Tensor] = []
        teacher_entropy_batches: list[torch.Tensor] = []
        paired_mode = (
            self.config.control_token_online_selection_mode
            in PAIRED_SIGNAL_SELECTION_MODES
        )
        teacher_confidence_mode = (
            self.config.control_token_online_selection_mode
            == TOP_TEACHER_CONFIDENCE_STUDENT_ENTROPY_SELECTION_MODE
        )
        policy_loss_cfg = _cfg_get(self.actor.config, "policy_loss", {})
        for micro_batch, configured_loss in zip(
            micro_batches,
            selection_loss_batches,
            strict=True,
        ):
            model_inputs = {
                **micro_batch.batch,
                **micro_batch.non_tensor_batch,
            }
            token_ids = aligned_response_token_ids(
                model_inputs,
                configured_loss,
            )
            if token_ids is None:
                raise ValueError(
                    "Online Control selection requires response-aligned " "token IDs."
                )
            token_id_batches.append(token_ids)
            label_batches.append(
                _labels_from_mapping(
                    model_inputs,
                    int(configured_loss.shape[0]),
                )
            )
            if paired_mode:
                student_entropy = model_inputs.get("student_entropy")
                if student_entropy is None:
                    raise ValueError(
                        "Paired online Control selection requires "
                        "student_entropy in every training micro-batch."
                    )
                if student_entropy.shape != configured_loss.shape:
                    raise ValueError(
                        "Online Control Student entropy must align with the "
                        "configured token loss."
                    )
                student_entropy_batches.append(student_entropy.detach())
            if teacher_confidence_mode:
                teacher_entropy = selected_teacher_entropy(
                    model_inputs,
                    policy_loss_cfg,
                )
                if teacher_entropy.shape != configured_loss.shape:
                    raise ValueError(
                        "Online Control Teacher entropy must align with the "
                        "configured token loss."
                    )
                teacher_entropy_batches.append(teacher_entropy.detach())
        statistics = global_candidate_loss_statistics(
            token_id_batches,
            selection_loss_batches,
            selection_loss_mask_batches,
            label_batches,
            domains=self.config.domains,
            domain_candidate_token_ids=(self.config.effective_domain_candidate_map()),
            selection_mode=self.config.control_token_online_selection_mode,
            student_entropy_batches=(
                tuple(student_entropy_batches) if paired_mode else None
            ),
            teacher_entropy_batches=(
                tuple(teacher_entropy_batches)
                if teacher_confidence_mode
                else None
            ),
        )
        outcome, state = update_online_control_selection(
            state,
            statistics,
            step=self.config.step,
        )
        self._online_control_selection_state = state
        self._persist_online_control_selection_state()
        append_online_control_selection_jsonl(
            output_dir=self.config.output_dir,
            state=state,
            outcome=outcome,
            control_weight=self.config.control_token_weight,
        )

        metrics = {
            "global/token_weight/audit_triggered": float(outcome.audit_triggered),
            "global/token_weight/history_reset": float(outcome.history_reset),
            "global/token_weight/window_fill_steps": float(outcome.window_fill_steps),
            "global/token_weight/candidate_token_count": float(
                len(state.candidate_token_ids)
            ),
            "global/token_weight/top_speed_selection_enabled": float(
                state.selection_mode == TOP_SPEED_SELECTION_MODE
            ),
            "global/token_weight/paired_score_weighting_enabled": float(
                state.weight_mode == PAIRED_ONLINE_WEIGHT_MODE
            ),
        }
        result_map = {result.domain: result for result in outcome.domain_results}
        next_active = state.active_map()
        next_active_weights = state.active_weight_map()
        for domain in self.config.domains:
            result = result_map.get(domain)
            metrics[f"{domain}/token_weight/candidate_token_count"] = float(
                len(state.candidate_map()[domain])
            )
            metrics[f"{domain}/token_weight/active_token_count"] = float(
                len(self._applied_online_control_token_ids.get(domain, ()))
            )
            metrics[f"{domain}/token_weight/next_active_token_count"] = float(
                len(next_active.get(domain, ()))
            )
            next_weights = tuple(next_active_weights.get(domain, {}).values())
            if next_weights:
                metrics[
                    f"{domain}/token_weight/next_selected_raw_weight_mean"
                ] = sum(next_weights) / len(next_weights)
            if result is not None:
                metrics[f"{domain}/token_weight/eligible_token_count"] = float(
                    result.eligible_token_count
                )
                for population, distribution in (
                    ("eligible", result.eligible_score_distribution),
                    ("selected", result.selected_score_distribution),
                ):
                    if distribution is None:
                        continue
                    prefix = (
                        f"{domain}/token_weight/"
                        f"{population}_selection_score_"
                    )
                    metrics[f"{prefix}count"] = float(distribution.count)
                    metrics[f"{prefix}mean"] = distribution.mean
                    metrics[f"{prefix}std"] = distribution.std
                    metrics[f"{prefix}min"] = distribution.minimum
                    metrics[f"{prefix}p10"] = distribution.p10
                    metrics[f"{prefix}p50"] = distribution.p50
                    metrics[f"{prefix}p90"] = distribution.p90
                    metrics[f"{prefix}max"] = distribution.maximum
                selected_speeds = tuple(
                    item.optimization_speed
                    for item in result.selected_tokens
                    if item.optimization_speed is not None
                )
                if selected_speeds:
                    metrics[
                        f"{domain}/token_weight/selected_optimization_speed_mean"
                    ] = sum(selected_speeds) / len(selected_speeds)
                if result.selected_tokens:
                    metrics[
                        f"{domain}/token_weight/selected_score_mean"
                    ] = sum(
                        item.mean_selection_score
                        for item in result.selected_tokens
                    ) / len(result.selected_tokens)
        return metrics

    def _token_selection_metrics(
        self,
        selections: dict[str, dict[str, _TokenSelection]],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for domain, domain_selections in selections.items():
            if not domain_selections:
                continue
            first = next(iter(domain_selections.values()))
            candidate_count = first.candidate_token_count
            candidate_mass = first.candidate_loss_abs_mass
            metrics[f"{domain}/token_grad/domain_token_count"] = float(candidate_count)
            metrics[
                f"{domain}/token_grad/" "selection_matches_production_reweighting"
            ] = 1.0
            metrics[f"{domain}/token_grad/selection_reweighting_active"] = float(
                self._production_weighting_enabled()
            )
            metrics[f"{domain}/token_grad/global_candidate_loss_abs_mass"] = (
                candidate_mass
            )
            for selection_name, selection in domain_selections.items():
                prefix = selection_name
                metrics[f"{domain}/token_grad/{prefix}_token_count"] = float(
                    selection.selected_token_count
                )
                metrics[f"{domain}/token_grad/{prefix}_token_fraction"] = (
                    selection.selected_token_count / candidate_count
                    if candidate_count > 0
                    else 0.0
                )
                metrics[f"{domain}/token_grad/{prefix}_loss_abs_mass"] = (
                    selection.selected_loss_abs_mass
                )
                metrics[f"{domain}/token_grad/{prefix}_loss_abs_mass_frac"] = (
                    selection.selected_loss_abs_mass / candidate_mass
                    if candidate_mass > 0.0
                    else 0.0
                )
            if "tail" in domain_selections:
                metrics[f"{domain}/token_grad/tail_fraction_configured"] = (
                    self.config.token_gradient_tail_fraction
                )
            if "top_k" in domain_selections:
                if self.config.token_gradient_top_k is None:
                    raise RuntimeError("Top-k metrics require token_gradient_top_k.")
                metrics[f"{domain}/token_grad/top_k_configured"] = float(
                    self.config.token_gradient_top_k
                )
            if "top_p" in domain_selections:
                metrics[f"{domain}/token_grad/top_p_fraction_configured"] = (
                    self.config.token_gradient_top_p
                )
                if self.config.token_gradient_top_p >= 1.0 - 1e-12:
                    top_p = domain_selections["top_p"]
                    metrics[f"{domain}/token_grad/top_p1_token_count"] = float(
                        top_p.selected_token_count
                    )
                    metrics[f"{domain}/token_grad/top_p1_token_fraction"] = (
                        top_p.selected_token_count / candidate_count
                        if candidate_count > 0
                        else 0.0
                    )
        return metrics

    def _dynamic_weight_metrics(
        self,
        domain_sq: dict[str, float] | None = None,
        source_signals: dict[str, float] | None = None,
        signed_projection_shares: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if not self.config.dynamic_weighting_enabled:
            return {}
        weights = self._weight_state.weight_map()
        target_weights = self._weight_state.target_weight_map()
        ema_signals = self._weight_state.ema_signal_map()
        metrics: dict[str, float] = {}
        for domain in self.config.domains:
            weight = weights.get(domain, 1.0)
            metrics[f"{domain}/dynamic_weight/applied_gradient_weight"] = weight
            metrics[f"{domain}/dynamic_weight/bounded_target_gradient_weight"] = (
                target_weights.get(domain, weight)
            )
            metrics[f"{domain}/dynamic_weight/ema_source_signal"] = ema_signals.get(
                domain, 0.0
            )
            metrics[f"{domain}/dynamic_weight/" "source_is_projection_share"] = float(
                self.config.dynamic_weighting_signal_source
                == DOMAIN_GRADIENT_PROJECTION_SHARE_SIGNAL
            )
            if self.config.dynamic_weighting_signal_source == GRADIENT_NORM_SIGNAL:
                metrics[f"{domain}/dynamic_weight/ema_grad_norm"] = ema_signals.get(
                    domain, 0.0
                )
            if source_signals is not None:
                metrics[f"{domain}/dynamic_weight/source_signal"] = source_signals.get(
                    domain, 0.0
                )
            if signed_projection_shares is not None:
                metrics[f"{domain}/dynamic_weight/" "raw_signed_projection_share"] = (
                    signed_projection_shares.get(domain, 0.0)
                )
            if domain_sq is not None:
                metrics[f"{domain}/dynamic_weight/weighted_grad_norm"] = (
                    weight * max(domain_sq.get(domain, 0.0), 0.0) ** 0.5
                )
        return metrics

    def _dynamic_weight_signals(
        self,
        *,
        total_sq: float,
        domain_sq: dict[str, float],
        domain_total_dot: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float] | None]:
        if self.config.dynamic_weighting_signal_source == GRADIENT_NORM_SIGNAL:
            return (
                {domain: max(value, 0.0) ** 0.5 for domain, value in domain_sq.items()},
                None,
            )
        denominator = max(float(total_sq), 1e-12)
        signed_shares = {
            domain: float(domain_total_dot.get(domain, 0.0)) / denominator
            for domain in self.config.domains
        }
        return (
            {domain: abs(share) for domain, share in signed_shares.items()},
            signed_shares,
        )

    def run_before_training(
        self,
        micro_batches: Sequence[Any],
        loss_scales: Sequence[float],
        *,
        on_policy: bool,
        temperature: float,
    ) -> dict[str, float]:
        if len(micro_batches) != len(loss_scales):
            raise ValueError(
                "Each audit micro-batch must have exactly one training loss scale."
            )
        phase_control_metrics = self._update_phase_control(micro_batches)
        control_speed_metrics = self._update_control_speed(micro_batches)
        token_weighting_active = (
            self.config.control_token_weighting_enabled
            or self.config.all_domain_shared_token_weighting_enabled
        )
        shared_token_active = self.config.all_domain_shared_token_weighting_enabled
        token_gradient_active = self.config.token_gradient_enabled and (
            self.config.token_gradient_tail_enabled
            or self.config.token_gradient_top_p_enabled
        )
        if not self.enabled and not shared_token_active:
            return {
                **self._dynamic_weight_metrics(),
                **self._token_weighting_metrics(),
                **phase_control_metrics,
                **control_speed_metrics,
            }

        state = AuditState.capture(self.actor)
        try:
            metrics = {
                **self._token_weighting_metrics(),
                **phase_control_metrics,
                **control_speed_metrics,
            }
            candidate_data = None
            if shared_token_active and not self.enabled:
                candidate_data = self._collect_loss_abs_candidates(
                    micro_batches,
                    loss_scales,
                    on_policy=on_policy,
                    temperature=temperature,
                )
                metrics.update(self._refresh_shared_token_selection(candidate_data[0]))
            if not self.enabled:
                metrics.update(self._dynamic_weight_metrics())
                if candidate_data is not None:
                    metrics.update(
                        self._loss_amplification_metrics(
                            candidate_data[0],
                            micro_batches,
                        )
                    )
                return metrics

            candidate_data = self._backward_replay(
                state,
                micro_batches,
                loss_scales,
                on_policy=on_policy,
                temperature=temperature,
                domain=None,
                collect_loss_abs_candidates=(
                    token_weighting_active or token_gradient_active
                ),
            )
            if (
                token_weighting_active or token_gradient_active
            ) and candidate_data is None:
                raise RuntimeError(
                    "The total audit replay did not return requested "
                    "configured-loss candidates."
                )
            if shared_token_active:
                if candidate_data is None:
                    raise RuntimeError(
                        "Shared-token weighting requires loss candidates."
                    )
                metrics.update(self._refresh_shared_token_selection(candidate_data[0]))
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
            metrics.update(
                domain_metrics_from_gram(
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
            )
            source_signals, signed_projection_shares = self._dynamic_weight_signals(
                total_sq=total_sq,
                domain_sq=domain_sq,
                domain_total_dot=domain_total_dot,
            )
            if self._should_update_dynamic_weighting():
                self._weight_state = update_domain_weight_state(
                    self._weight_state,
                    source_signals,
                    ema_beta=self.config.dynamic_weighting_ema_beta,
                    weight_ema_beta=(self.config.dynamic_weighting_weight_ema_beta),
                    alpha=self.config.dynamic_weighting_alpha,
                    minimum=self.config.dynamic_weighting_min,
                    maximum=self.config.dynamic_weighting_max,
                    step=self.config.step,
                )
                self._persist_weight_state()

            token_vector_peak_bytes = 0
            token_reference_vector_bytes = 0
            effective_domain_replay_count = 0
            if token_gradient_active:
                if candidate_data is None:
                    raise RuntimeError(
                        "Token-gradient replay requires loss candidates."
                    )
                production_masks: tuple[torch.Tensor, ...] | None = None
                token_candidate_data = candidate_data
                candidate_loss_basis = "configured_loss_abs"
                token_domain_vectors = domain_vectors
                token_domain_sq = domain_sq
                if self._production_weighting_enabled():
                    production_masks = self._production_gradient_masks(micro_batches)
                    token_candidate_data = self._reweighted_candidate_data(
                        candidate_data,
                        production_masks,
                    )
                    candidate_loss_basis = "production_reweighted_configured_loss_abs"
                    token_domain_vectors = {}
                    for domain in self.config.domains:
                        domain_masks = self._domain_production_gradient_masks(
                            micro_batches,
                            production_masks,
                            domain,
                        )
                        self._backward_replay(
                            state,
                            micro_batches,
                            loss_scales,
                            on_policy=on_policy,
                            temperature=temperature,
                            domain=None,
                            gradient_masks=domain_masks,
                        )
                        token_domain_vectors[domain] = snapshot_gradients(
                            self.actor,
                            self.config.storage_dtype,
                        )
                    token_domain_sq = {
                        domain: vector_squared_norm(self.actor, vector)
                        for domain, vector in token_domain_vectors.items()
                    }
                    effective_domain_replay_count = len(self.config.domains)
                    token_reference_vector_bytes = sum(
                        vector_nbytes(vector)
                        for vector in token_domain_vectors.values()
                    )
                state.restore_runtime()
                state.clear_gradients()
                token_selections = self._loss_ranked_token_selections(
                    micro_batches,
                    loss_scales,
                    on_policy=on_policy,
                    temperature=temperature,
                    candidate_data=token_candidate_data,
                    candidate_loss_basis=candidate_loss_basis,
                )
                for domain, domain_selections in token_selections.items():
                    metrics[
                        f"{domain}/token_grad/" "domain_grad_norm_after_reweight"
                    ] = (max(token_domain_sq[domain], 0.0) ** 0.5)
                    for selection_name, selection in domain_selections.items():
                        selection_masks = (
                            self._apply_production_gradient_masks(
                                selection.masks,
                                production_masks,
                            )
                            if production_masks is not None
                            else selection.masks
                        )
                        self._backward_replay(
                            state,
                            micro_batches,
                            loss_scales,
                            on_policy=on_policy,
                            temperature=temperature,
                            domain=None,
                            gradient_masks=selection_masks,
                        )
                        selection_vector = snapshot_gradients(
                            self.actor,
                            self.config.storage_dtype,
                        )
                        token_vector_peak_bytes = max(
                            token_vector_peak_bytes,
                            vector_nbytes(selection_vector),
                        )
                        selection_sq = vector_squared_norm(
                            self.actor,
                            selection_vector,
                        )
                        selection_domain_dot = vector_dot(
                            self.actor,
                            selection_vector,
                            token_domain_vectors[domain],
                        )
                        metric_prefix = (
                            "top_p1"
                            if (
                                selection_name == "top_p"
                                and self.config.token_gradient_top_p >= 1.0 - 1e-12
                            )
                            else selection_name
                        )
                        selection_metrics = gradient_partition_metrics_from_gram(
                            prefix=metric_prefix,
                            domain_sq=token_domain_sq[domain],
                            subset_sq=selection_sq,
                            subset_domain_dot=selection_domain_dot,
                        )
                        metrics.update(
                            {
                                f"{domain}/token_grad/{key}": value
                                for key, value in selection_metrics.items()
                            }
                        )
                        del selection_vector
                metrics.update(self._token_selection_metrics(token_selections))
                metrics[
                    "global/audit/" "token_gradient_matches_production_reweighting"
                ] = 1.0
                metrics["global/audit/token_gradient_reweighting_active"] = float(
                    production_masks is not None
                )
            if candidate_data is not None:
                metrics.update(
                    self._loss_amplification_metrics(
                        candidate_data[0],
                        micro_batches,
                    )
                )
            parity_total = audit_total
            dynamic_parity_replay_count = 0
            if self._production_weighting_enabled() and self.config.parity_enabled:
                parity_total = self._snapshot_training_gradient_reference(
                    state,
                    micro_batches,
                    loss_scales,
                    on_policy=on_policy,
                    temperature=temperature,
                )
                dynamic_parity_replay_count = 1
            metrics.update(
                self._dynamic_weight_metrics(
                    domain_sq,
                    source_signals,
                    signed_projection_shares,
                )
            )
            domain_count = len(self.config.domains)
            token_selection_count = (
                int(self.config.token_gradient_tail_enabled)
                + int(self.config.token_gradient_top_p_enabled)
                * (1 + int(self.config.token_gradient_top_k is not None))
                if token_gradient_active
                else 0
            )
            metrics["global/audit/domain_gradient_backward_replay_count"] = float(
                1
                + domain_count
                + effective_domain_replay_count
                + domain_count * token_selection_count
                + dynamic_parity_replay_count
            )
            metrics["global/audit/domain_gradient_source_step"] = float(
                self.config.step
            )
            base_vector_bytes = (
                vector_nbytes(audit_total)
                + sum(vector_nbytes(vector) for vector in domain_vectors.values())
                + token_reference_vector_bytes
            )
            dynamic_vector_bytes = (
                vector_nbytes(parity_total) if dynamic_parity_replay_count else 0
            )
            peak_vector_bytes = base_vector_bytes + max(
                token_vector_peak_bytes,
                dynamic_vector_bytes,
            )
            metrics["global/audit/domain_gradient_peak_cpu_vector_bytes"] = float(
                peak_vector_bytes
            )
            metrics["global/audit/domain_gradient_peak_cpu_vector_bytes_per_rank"] = (
                float(peak_vector_bytes)
            )
            metrics[
                "global/audit/"
                "domain_gradient_peak_cpu_vector_bytes_actor_group_total"
            ] = actor_group_sum(self.actor, float(peak_vector_bytes))
            retained_vector_bytes = (
                vector_nbytes(parity_total) if self.config.parity_enabled else 0
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
                self._audit_total = parity_total
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
