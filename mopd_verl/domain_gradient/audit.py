"""Independent domain-gradient replay built on the production actor loss."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from verl.utils.device import get_device_id

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.geometry import (
    GradientVector,
    actor_group_sum,
    domain_metrics_from_gram,
    gradient_subset_metrics_from_gram,
    snapshot_gradients,
    training_parity_metrics,
    vector_dot,
    vector_nbytes,
    vector_squared_norm,
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
from mopd_verl.full_gradient.labels import _labels_from_mapping


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
        self._cumulative_token_loss_state: (
            CumulativeTokenLossState | None
        ) = None
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
                optimizer_groups[0].get(
                    "mopd_cumulative_token_loss_state"
                )
                if optimizer_groups
                else None
            )
            if (
                not isinstance(
                    cumulative_state,
                    CumulativeTokenLossState,
                )
                and isinstance(serialized_cumulative_state, dict)
            ):
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
                raise ValueError(
                    "Cumulative token-loss state has an unsupported type."
                )
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
            optimizer_groups[0]["mopd_domain_weight_state"] = (
                self._weight_state.as_dict()
            )

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
            optimizer_groups[0]["mopd_cumulative_token_loss_state"] = (
                state.as_dict()
            )

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

    def training_gradient_mask(
        self,
        micro_batch: Any,
    ) -> torch.Tensor | None:
        """Return current domain and token production gradient multipliers."""

        token_weighting_enabled = (
            self.config.control_token_weighting_enabled
            or self.config.all_domain_shared_token_weighting_enabled
        )
        if (
            not self.config.dynamic_weighting_enabled
            and not token_weighting_enabled
        ):
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
            gradient_weights = gradient_weights * token_gradient_weights(
                token_ids,
                control_token_ids=(
                    self._cached_token_id_tensor(
                        "control",
                        self.config.control_token_ids,
                        token_ids,
                    )
                    if self.config.control_token_weighting_enabled
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
                shared_token_weight=(
                    self.config.all_domain_shared_token_weight
                ),
            )
        return gradient_weights

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
                else self._domain_gradient_mask(micro_batch, domain)
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
                    candidates[domain_name].extend(
                        micro_candidates[domain_name]
                    )
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

        optional_masks = tuple(
            self.training_gradient_mask(micro_batch)
            for micro_batch in micro_batches
        )
        if any(mask is None for mask in optional_masks):
            raise RuntimeError(
                "Weighted parity replay requires a training gradient mask."
            )
        gradient_masks = tuple(
            mask for mask in optional_masks if mask is not None
        )
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
        values = torch.tensor([total, *counts], device=get_device_id(), dtype=torch.float64)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        total_count, *domain_counts = (float(value) for value in values.tolist())
        metrics = {
            f"{domain}/pre_reweight_full_grad/sample_count": (
                domain_counts[index]
            )
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
            raise ValueError(
                "Configured token loss and mask must have the same shape."
            )
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
                loss_abs = float(
                    loss_abs_cpu[sample_index, token_index].item()
                )
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
            micro_candidates, mask_template = (
                self._loss_abs_candidates_for_micro_batch(
                    result,
                    micro_batch,
                    micro_batch_index=micro_batch_index,
                )
            )
            for domain in self.config.domains:
                candidates[domain].extend(
                    micro_candidates[domain]
                )
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
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        gathered_scores: list[list[float] | None]
        if not distributed:
            gathered_scores = [local_scores]
        else:
            gathered_scores = [
                None for _ in range(torch.distributed.get_world_size())
            ]
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
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
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
        candidate_data: tuple[
            dict[str, tuple[LocalTokenCandidate, ...]],
            tuple[torch.Tensor, ...],
        ]
        | None = None,
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
                        for selection_name, selection
                        in domain_selections.items()
                    }
                    for domain, domain_selections in selections.items()
                },
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
                raise RuntimeError(
                    "Cumulative shared-token selection requires state."
                )
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
            selection_mode=(
                self.config.all_domain_shared_token_selection_mode
            ),
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
        for domain, token_ids in (
            self._shared_token_selection.domain_top_token_ids
        ):
            metrics[
                f"{domain}/token_weight/high_loss_token_type_count"
            ] = float(len(token_ids))
        if self._cumulative_token_loss_state is not None:
            for domain, summary in (
                self._cumulative_token_loss_state.domain_summaries().items()
            ):
                for name, value in summary.items():
                    metrics[
                        f"{domain}/token_weight/{name}"
                    ] = value
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
            control_weighting_enabled=(
                self.config.control_token_weighting_enabled
            ),
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
            metrics[f"{domain}/token_grad/domain_token_count"] = float(
                candidate_count
            )
            metrics[
                f"{domain}/token_grad/global_candidate_loss_abs_mass"
            ] = candidate_mass
            for selection_name, selection in domain_selections.items():
                prefix = selection_name
                metrics[
                    f"{domain}/token_grad/{prefix}_token_count"
                ] = float(selection.selected_token_count)
                metrics[
                    f"{domain}/token_grad/{prefix}_token_fraction"
                ] = (
                    selection.selected_token_count / candidate_count
                    if candidate_count > 0
                    else 0.0
                )
                metrics[
                    f"{domain}/token_grad/{prefix}_loss_abs_mass"
                ] = selection.selected_loss_abs_mass
                metrics[
                    f"{domain}/token_grad/{prefix}_loss_abs_mass_frac"
                ] = (
                    selection.selected_loss_abs_mass / candidate_mass
                    if candidate_mass > 0.0
                    else 0.0
                )
            if "tail" in domain_selections:
                metrics[
                    f"{domain}/token_grad/tail_fraction_configured"
                ] = self.config.token_gradient_tail_fraction
            if "top_k" in domain_selections:
                if self.config.token_gradient_top_k is None:
                    raise RuntimeError(
                        "Top-k metrics require token_gradient_top_k."
                    )
                metrics[
                    f"{domain}/token_grad/top_k_configured"
                ] = float(self.config.token_gradient_top_k)
            if "top_p" in domain_selections:
                metrics[
                    f"{domain}/token_grad/top_p_fraction_configured"
                ] = self.config.token_gradient_top_p
                if self.config.token_gradient_top_p >= 1.0 - 1e-12:
                    top_p = domain_selections["top_p"]
                    metrics[
                        f"{domain}/token_grad/top_p1_token_count"
                    ] = float(top_p.selected_token_count)
                    metrics[
                        f"{domain}/token_grad/top_p1_token_fraction"
                    ] = (
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
            metrics[
                f"{domain}/dynamic_weight/applied_gradient_weight"
            ] = weight
            metrics[
                f"{domain}/dynamic_weight/bounded_target_gradient_weight"
            ] = target_weights.get(domain, weight)
            metrics[f"{domain}/dynamic_weight/ema_source_signal"] = (
                ema_signals.get(domain, 0.0)
            )
            metrics[
                f"{domain}/dynamic_weight/"
                "source_is_projection_share"
            ] = float(
                self.config.dynamic_weighting_signal_source
                == DOMAIN_GRADIENT_PROJECTION_SHARE_SIGNAL
            )
            if (
                self.config.dynamic_weighting_signal_source
                == GRADIENT_NORM_SIGNAL
            ):
                metrics[f"{domain}/dynamic_weight/ema_grad_norm"] = (
                    ema_signals.get(domain, 0.0)
                )
            if source_signals is not None:
                metrics[f"{domain}/dynamic_weight/source_signal"] = (
                    source_signals.get(domain, 0.0)
                )
            if signed_projection_shares is not None:
                metrics[
                    f"{domain}/dynamic_weight/"
                    "raw_signed_projection_share"
                ] = signed_projection_shares.get(domain, 0.0)
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
        if (
            self.config.dynamic_weighting_signal_source
            == GRADIENT_NORM_SIGNAL
        ):
            return (
                {
                    domain: max(value, 0.0) ** 0.5
                    for domain, value in domain_sq.items()
                },
                None,
            )
        denominator = max(float(total_sq), 1e-12)
        signed_shares = {
            domain: float(domain_total_dot.get(domain, 0.0))
            / denominator
            for domain in self.config.domains
        }
        return (
            {
                domain: abs(share)
                for domain, share in signed_shares.items()
            },
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
        token_weighting_active = (
            self.config.control_token_weighting_enabled
            or self.config.all_domain_shared_token_weighting_enabled
        )
        shared_token_active = (
            self.config.all_domain_shared_token_weighting_enabled
        )
        token_gradient_active = (
            self.config.token_gradient_enabled
            and (
                self.config.token_gradient_tail_enabled
                or self.config.token_gradient_top_p_enabled
            )
        )
        if not self.enabled and not shared_token_active:
            return {
                **self._dynamic_weight_metrics(),
                **self._token_weighting_metrics(),
            }

        state = AuditState.capture(self.actor)
        try:
            metrics = self._token_weighting_metrics()
            candidate_data = None
            if shared_token_active and not self.enabled:
                candidate_data = self._collect_loss_abs_candidates(
                    micro_batches,
                    loss_scales,
                    on_policy=on_policy,
                    temperature=temperature,
                )
                metrics.update(
                    self._refresh_shared_token_selection(
                        candidate_data[0]
                    )
                )
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
                metrics.update(
                    self._refresh_shared_token_selection(
                        candidate_data[0]
                    )
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
                        self.config.storage_dtype.lower()
                        in {"float32", "fp32"}
                    ),
                    storage_dtype=self.config.storage_dtype,
                )
            )
            source_signals, signed_projection_shares = (
                self._dynamic_weight_signals(
                    total_sq=total_sq,
                    domain_sq=domain_sq,
                    domain_total_dot=domain_total_dot,
                )
            )
            token_vector_peak_bytes = 0
            if token_gradient_active:
                state.restore_runtime()
                state.clear_gradients()
                token_selections = self._loss_ranked_token_selections(
                    micro_batches,
                    loss_scales,
                    on_policy=on_policy,
                    temperature=temperature,
                    candidate_data=candidate_data,
                )
                for domain, domain_selections in token_selections.items():
                    for selection_name, selection in domain_selections.items():
                        self._backward_replay(
                            state,
                            micro_batches,
                            loss_scales,
                            on_policy=on_policy,
                            temperature=temperature,
                            domain=None,
                            gradient_masks=selection.masks,
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
                            domain_vectors[domain],
                        )
                        metric_prefix = (
                            "top_p1"
                            if (
                                selection_name == "top_p"
                                and self.config.token_gradient_top_p
                                >= 1.0 - 1e-12
                            )
                            else selection_name
                        )
                        selection_metrics = gradient_subset_metrics_from_gram(
                            prefix=metric_prefix,
                            domain_sq=domain_sq[domain],
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

            if self._should_update_dynamic_weighting():
                self._weight_state = update_domain_weight_state(
                    self._weight_state,
                    source_signals,
                    ema_beta=self.config.dynamic_weighting_ema_beta,
                    weight_ema_beta=(
                        self.config.dynamic_weighting_weight_ema_beta
                    ),
                    alpha=self.config.dynamic_weighting_alpha,
                    minimum=self.config.dynamic_weighting_min,
                    maximum=self.config.dynamic_weighting_max,
                    step=self.config.step,
                )
                self._persist_weight_state()
            if candidate_data is not None:
                metrics.update(
                    self._loss_amplification_metrics(
                        candidate_data[0],
                        micro_batches,
                    )
                )
            parity_total = audit_total
            dynamic_parity_replay_count = 0
            if (
                self._production_weighting_enabled()
                and self.config.parity_enabled
            ):
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
                * (
                    1
                    + int(self.config.token_gradient_top_k is not None)
                )
                if token_gradient_active
                else 0
            )
            metrics["global/audit/domain_gradient_backward_replay_count"] = float(
                1
                + domain_count
                + domain_count * token_selection_count
                + dynamic_parity_replay_count
            )
            metrics["global/audit/domain_gradient_source_step"] = float(
                self.config.step
            )
            base_vector_bytes = vector_nbytes(audit_total) + sum(
                vector_nbytes(vector) for vector in domain_vectors.values()
            )
            dynamic_vector_bytes = (
                vector_nbytes(parity_total)
                if dynamic_parity_replay_count
                else 0
            )
            peak_vector_bytes = base_vector_bytes + max(
                token_vector_peak_bytes,
                dynamic_vector_bytes,
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
                vector_nbytes(parity_total)
                if self.config.parity_enabled
                else 0
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
