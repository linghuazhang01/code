# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig
from mopd_verl.topk_distill import (
    TOPK_LOGPROB_MODE_SPARSE,
    TOPK_RENORMALIZED_FORWARD_KL,
    TOPK_SUPPORT_SOURCE_STUDENT,
    TOPK_SUPPORT_SOURCE_TEACHER,
    chosen_token_forward_kl_matrix,
    is_topk_distill_enabled,
    resolved_topk_distill_mode,
    select_teacher_log_prob_tensor,
    selected_logits_from_hidden_states,
    teacher_prefix_forward_weight,
    teacher_prefix_masks,
    topk_distill_bucket_metrics,
    topk_distill_include_tail,
    topk_distill_logprob_chunk_size,
    topk_distill_logprob_mode,
    topk_distill_loss_matrix,
    topk_distill_support_source,
    topk_distill_temperature,
    topk_distill_uses_renormalized_support,
    topk_distill_weight,
    topk_log_probs_from_logits,
    topk_teacher_student_cross_entropy_matrix,
)

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _teacher_type_at(opd_teacher: object, index: int) -> object:
    if isinstance(opd_teacher, np.ndarray):
        if opd_teacher.ndim == 0:
            return opd_teacher.item()
        return opd_teacher[index]
    if isinstance(opd_teacher, (list, tuple)):
        return opd_teacher[index]
    return opd_teacher


def _select_teacher_topk_tensors(
    model_inputs: dict[str, object],
    policy_loss_config: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "math_teacher_topk_ids" not in model_inputs or "math_teacher_topk_logprobs" not in model_inputs:
        raise ValueError(
            "Top-k distillation requires math_teacher_topk_ids and math_teacher_topk_logprobs in the batch."
        )
    math_ids = model_inputs["math_teacher_topk_ids"]
    math_log_probs = model_inputs["math_teacher_topk_logprobs"]
    code_ids = model_inputs.get("code_teacher_topk_ids", math_ids)
    code_log_probs = model_inputs.get("code_teacher_topk_logprobs", math_log_probs)
    if not bool(policy_loss_config.get("multi_teacher_distill", False)) or "opd_teacher" not in model_inputs:
        return math_ids, math_log_probs

    opd_teacher = model_inputs["opd_teacher"]
    selected_ids = torch.empty_like(math_ids)
    selected_log_probs = torch.empty_like(math_log_probs)
    for idx in range(int(math_log_probs.shape[0])):
        teacher_type = _teacher_type_at(opd_teacher, idx)
        if teacher_type == "code" and "code_teacher_topk_ids" in model_inputs:
            selected_ids[idx] = code_ids[idx]
            selected_log_probs[idx] = code_log_probs[idx]
        else:
            selected_ids[idx] = math_ids[idx]
            selected_log_probs[idx] = math_log_probs[idx]
    return selected_ids, selected_log_probs


def _select_student_topk_teacher_log_probs(
    model_inputs: dict[str, object],
    policy_loss_config: object,
) -> torch.Tensor:
    if "math_teacher_student_topk_logprobs" not in model_inputs:
        raise ValueError(
            "Student top-k distillation requires math_teacher_student_topk_logprobs in the batch."
        )
    math_log_probs = model_inputs["math_teacher_student_topk_logprobs"]
    code_log_probs = model_inputs.get("code_teacher_student_topk_logprobs", math_log_probs)
    if not bool(policy_loss_config.get("multi_teacher_distill", False)) or "opd_teacher" not in model_inputs:
        return math_log_probs

    opd_teacher = model_inputs["opd_teacher"]
    selected_log_probs = torch.empty_like(math_log_probs)
    for idx in range(int(math_log_probs.shape[0])):
        teacher_type = _teacher_type_at(opd_teacher, idx)
        if teacher_type == "code" and "code_teacher_student_topk_logprobs" in model_inputs:
            selected_log_probs[idx] = code_log_probs[idx]
        else:
            selected_log_probs[idx] = math_log_probs[idx]
    return selected_log_probs


def _select_topk_support_tensors(
    model_inputs: dict[str, object],
    policy_loss_config: object,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    support_source = topk_distill_support_source(policy_loss_config)
    if support_source == TOPK_SUPPORT_SOURCE_STUDENT:
        if "student_topk_ids" not in model_inputs:
            raise ValueError("Student top-k distillation requires student_topk_ids in the batch.")
        return (
            model_inputs["student_topk_ids"],
            _select_student_topk_teacher_log_probs(model_inputs, policy_loss_config),
            support_source,
        )
    if support_source != TOPK_SUPPORT_SOURCE_TEACHER:
        raise ValueError(f"Unsupported top-k support source: {support_source!r}.")
    support_ids, teacher_log_probs = _select_teacher_topk_tensors(model_inputs, policy_loss_config)
    return support_ids, teacher_log_probs, support_source


def _unwrap_module(module: nn.Module) -> nn.Module:
    for attr in ("_fsdp_wrapped_module", "module"):
        wrapped = getattr(module, attr, None)
        if isinstance(wrapped, nn.Module):
            return wrapped
    return module


def _causal_lm_body_and_head(module: nn.Module) -> tuple[nn.Module | None, nn.Module | None]:
    unwrapped = _unwrap_module(module)
    candidates = [unwrapped]
    base_model = getattr(unwrapped, "base_model", None)
    if isinstance(base_model, nn.Module):
        candidates.append(base_model)
        nested = getattr(base_model, "model", None)
        if isinstance(nested, nn.Module):
            candidates.append(nested)

    for candidate in candidates:
        body = getattr(candidate, "model", None)
        head = getattr(candidate, "lm_head", None)
        if isinstance(body, nn.Module) and isinstance(head, nn.Module) and hasattr(head, "weight"):
            return body, head
    return None, None


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _can_use_selected_topk_head(self, multi_modal_inputs: dict[str, object]) -> bool:
        if multi_modal_inputs:
            return False
        fsdp_config = self.config.fsdp_config
        raw_fsdp_size = fsdp_config.get("fsdp_size", -1) if hasattr(fsdp_config, "get") else -1
        try:
            fsdp_size = int(raw_fsdp_size)
        except (TypeError, ValueError):
            fsdp_size = -1
        if fsdp_size != 1:
            return False
        body, head = _causal_lm_body_and_head(self.actor_module)
        return body is not None and head is not None

    def _selected_topk_param_context(self):
        if isinstance(self.actor_module, FSDP):
            return FSDP.summon_full_params(
                self.actor_module,
                recurse=True,
                writeback=True,
                rank0_only=False,
                offload_to_cpu=False,
            )
        return nullcontext()

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        inplace_backward: bool | None = None,
        topk: int | None = None,
        gather_topk_ids: torch.Tensor | None = None,
        calculate_log_probs: bool = True,
        normalize_gathered_topk: bool = True,
        topk_logprob_chunk_size: int | None = None,
        topk_logprob_mode: str = "sparse",
        return_extra: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        needs_topk_extra = return_extra or topk is not None or gather_topk_ids is not None
        if needs_topk_extra and self.use_fused_kernels:
            raise ValueError("Top-k distillation requires non-fused logits; set actor/ref use_fused_kernels=False.")
        if needs_topk_extra and self.use_ulysses_sp:
            raise ValueError("Top-k distillation is not supported with Ulysses sequence parallelism yet.")
        topk_ids = None
        topk_log_probs = None
        gathered_topk_log_probs = None
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])
        use_selected_topk_head = (
            needs_topk_extra
            and gather_topk_ids is not None
            and topk is None
            and not normalize_gathered_topk
            and str(topk_logprob_mode).lower() == TOPK_LOGPROB_MODE_SPARSE
            and not calculate_log_probs
            and not calculate_entropy
            and self._can_use_selected_topk_head(multi_modal_inputs)
        )

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            # reset input_ids, attention_mask, position_ids to ref model inputs if ref model input_ids is different from actor input_ids
            if "ref_input_ids" in micro_batch.keys():
                input_ids = micro_batch["ref_input_ids"]
                attention_mask = micro_batch["ref_attention_mask"]
                position_ids = micro_batch["ref_position_ids"]
                batch_size, seqlen = input_ids.shape

            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                if use_selected_topk_head:
                    body, head = _causal_lm_body_and_head(self.actor_module)
                    if body is None or head is None:
                        raise RuntimeError("selected top-k head path requires a causal LM body and lm_head.")
                    output = body(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        use_cache=False,
                    )
                    hidden_states_rmpad = output[0]
                    if hidden_states_rmpad.dim() == 3 and int(hidden_states_rmpad.shape[0]) == 1:
                        hidden_states_rmpad = hidden_states_rmpad.squeeze(0)
                    log_probs = hidden_states_rmpad.new_zeros(input_ids_rmpad_rolled.shape)
                    gather_ids = gather_topk_ids.to(device=input_ids.device, dtype=torch.long)
                    full_gather_ids = torch.zeros(
                        (batch_size, seqlen, int(gather_ids.shape[-1])),
                        device=input_ids.device,
                        dtype=torch.long,
                    )
                    full_gather_ids[:, -response_length - 1 : -1, :] = gather_ids
                    gather_ids_rmpad = index_first_axis(
                        rearrange(full_gather_ids, "b s k -> (b s) k"),
                        indices,
                    )
                    gathered_log_probs_rmpad = selected_logits_from_hidden_states(
                        hidden_states_rmpad,
                        vocab_weights=head.weight,
                        token_ids=gather_ids_rmpad,
                        bias=getattr(head, "bias", None),
                        temperature=temperature,
                        chunk_size=topk_logprob_chunk_size or 16,
                    )
                else:
                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating

                if self.use_fused_kernels and not use_selected_topk_head:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                elif not use_selected_topk_head:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    if calculate_log_probs:
                        logprob_inplace_backward = True if inplace_backward is None else bool(inplace_backward)
                        if calculate_entropy or needs_topk_extra:
                            logprob_inplace_backward = False
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=logprob_inplace_backward,
                        )
                    else:
                        log_probs = logits_rmpad.new_zeros(input_ids_rmpad_rolled.shape)

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                    if needs_topk_extra:
                        gather_ids_rmpad = None
                        if gather_topk_ids is not None:
                            gather_ids = gather_topk_ids.to(device=input_ids.device, dtype=torch.long)
                            full_gather_ids = torch.zeros(
                                (batch_size, seqlen, int(gather_ids.shape[-1])),
                                device=input_ids.device,
                                dtype=torch.long,
                            )
                            full_gather_ids[:, -response_length - 1 : -1, :] = gather_ids
                            gather_ids_rmpad = index_first_axis(
                                rearrange(full_gather_ids, "b s k -> (b s) k"),
                                indices,
                            )
                        topk_ids_rmpad, topk_log_probs_rmpad, gathered_log_probs_rmpad = (
                            topk_log_probs_from_logits(
                                logits_rmpad,
                                topk=topk,
                                gather_topk_ids=gather_ids_rmpad,
                                normalize_gathered=normalize_gathered_topk,
                                chunk_size=topk_logprob_chunk_size or 16,
                                logprob_mode=topk_logprob_mode,
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if topk is not None:
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :].long()
                if gather_topk_ids is not None:
                    full_gathered_topk_log_probs = pad_input(
                        hidden_states=gathered_log_probs_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    gathered_topk_log_probs = full_gathered_topk_log_probs[:, -response_length - 1 : -1, :]

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                if use_selected_topk_head:
                    body, head = _causal_lm_body_and_head(self.actor_module)
                    if body is None or head is None:
                        raise RuntimeError("selected top-k head path requires a causal LM body and lm_head.")
                    output = body(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                    )
                    hidden_states = output[0][:, -response_length - 1 : -1, :]
                    log_probs = hidden_states.new_zeros(hidden_states.shape[:-1])
                    gathered_topk_log_probs = selected_logits_from_hidden_states(
                        hidden_states,
                        vocab_weights=head.weight,
                        token_ids=gather_topk_ids.to(device=input_ids.device, dtype=torch.long),
                        bias=getattr(head, "bias", None),
                        temperature=temperature,
                        chunk_size=topk_logprob_chunk_size or 16,
                    )
                else:
                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating

                if self.use_fused_kernels and not use_selected_topk_head:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                elif not use_selected_topk_head:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    if calculate_log_probs:
                        logprob_inplace_backward = True if inplace_backward is None else bool(inplace_backward)
                        if calculate_entropy or needs_topk_extra:
                            logprob_inplace_backward = False
                        log_probs = logprobs_from_logits(
                            logits,
                            micro_batch["responses"],
                            inplace_backward=logprob_inplace_backward,
                        )
                    else:
                        log_probs = logits.new_zeros(logits.shape[:-1])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    if needs_topk_extra:
                        topk_ids, topk_log_probs, gathered_topk_log_probs = topk_log_probs_from_logits(
                            logits,
                            topk=topk,
                            gather_topk_ids=gather_topk_ids,
                            normalize_gathered=normalize_gathered_topk,
                            chunk_size=topk_logprob_chunk_size or 16,
                            logprob_mode=topk_logprob_mode,
                        )

            if needs_topk_extra:
                return entropy, log_probs, topk_ids, topk_log_probs, gathered_topk_log_probs
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self,
        data: DataProto,
        calculate_entropy=False,
        topk: int | None = None,
        gather_topk_ids_key: str | None = None,
    ) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys() # handle when ref input_ids is different from actor input_ids
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if gather_topk_ids_key is not None:
            select_keys.append(gather_topk_ids_key)
        if has_ref_input_ids:
            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []
        gathered_topk_log_probs_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            gather_topk_ids = (
                model_inputs[gather_topk_ids_key]
                if gather_topk_ids_key is not None
                else None
            )
            with torch.no_grad():
                forward_output = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    topk=topk,
                    gather_topk_ids=gather_topk_ids,
                    return_extra=topk is not None or gather_topk_ids is not None,
                )
                if topk is None and gather_topk_ids is None:
                    entropy, log_probs = forward_output
                else:
                    entropy, log_probs, topk_ids, topk_log_probs, gathered_topk_log_probs = forward_output
                    if topk_ids is not None:
                        topk_ids_lst.append(topk_ids)
                    if topk_log_probs is not None:
                        topk_log_probs_lst.append(topk_log_probs)
                    if gathered_topk_log_probs is not None:
                        gathered_topk_log_probs_lst.append(gathered_topk_log_probs)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        topk_ids = torch.concat(topk_ids_lst, dim=0) if topk is not None else None
        topk_log_probs = torch.concat(topk_log_probs_lst, dim=0) if topk is not None else None
        gathered_topk_log_probs = (
            torch.concat(gathered_topk_log_probs_lst, dim=0)
            if gather_topk_ids_key is not None
            else None
        )

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if topk is not None:
                topk_ids = restore_dynamic_batch(topk_ids, batch_idx_list)
                topk_log_probs = restore_dynamic_batch(topk_log_probs, batch_idx_list)
            if gather_topk_ids_key is not None:
                gathered_topk_log_probs = restore_dynamic_batch(gathered_topk_log_probs, batch_idx_list)

        if topk is not None and gather_topk_ids_key is not None:
            return log_probs, entropys, topk_ids, topk_log_probs, gathered_topk_log_probs
        if topk is not None:
            return log_probs, entropys, topk_ids, topk_log_probs
        if gather_topk_ids_key is not None:
            return log_probs, entropys, gathered_topk_log_probs
        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor teacher-student cross entropy", logger=logger)
    def compute_teacher_student_cross_entropy(
        self,
        data: DataProto,
        *,
        teacher_topk_ids_key: str,
        teacher_topk_logprobs_key: str,
        include_tail: bool,
        distill_temperature: float,
    ) -> torch.Tensor:
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            teacher_topk_ids_key,
            teacher_topk_logprobs_key,
        ]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        cross_entropy_lst = []
        policy_loss_config = self.config.policy_loss
        use_renormalized_support = topk_distill_uses_renormalized_support(policy_loss_config)
        effective_topk_logprob_mode = topk_distill_logprob_mode(policy_loss_config)
        if use_renormalized_support:
            effective_topk_logprob_mode = TOPK_LOGPROB_MODE_SPARSE
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad(), self._selected_topk_param_context():
                _, _, _, _, student_topk_log_probs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    gather_topk_ids=model_inputs[teacher_topk_ids_key],
                    calculate_log_probs=False,
                    normalize_gathered_topk=not use_renormalized_support,
                    topk_logprob_chunk_size=topk_distill_logprob_chunk_size(policy_loss_config),
                    topk_logprob_mode=effective_topk_logprob_mode,
                    return_extra=True,
                )
                cross_entropy = topk_teacher_student_cross_entropy_matrix(
                    student_topk_log_probs=student_topk_log_probs,
                    teacher_topk_log_probs=model_inputs[teacher_topk_logprobs_key],
                    include_tail=include_tail,
                    temperature=distill_temperature,
                )
            cross_entropy_lst.append(cross_entropy)

        cross_entropy = torch.concat(cross_entropy_lst, dim=0)
        if use_dynamic_bsz:
            cross_entropy = restore_dynamic_batch(cross_entropy, batch_idx_list)
        return cross_entropy

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("math_teacher_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
         # Include base model log probs for corrected reward computation
        # These are computed when actor_rollout_ref.model.base_model_path and
        # actor_rollout_ref.ref.model.base_model_path are both specified
        if "base_log_prob" in data.batch.keys():
            select_keys.append("base_log_prob")
        if "code_teacher_log_prob" in data.batch.keys():
            select_keys.append("code_teacher_log_prob")
        for key in (
            "math_teacher_topk_ids",
            "math_teacher_topk_logprobs",
            "code_teacher_topk_ids",
            "code_teacher_topk_logprobs",
            "student_topk_ids",
            "math_teacher_student_topk_logprobs",
            "code_teacher_student_topk_logprobs",
            "teacher_prefix_mask",
            "student_suffix_mask",
        ):
            if key in data.batch.keys():
                select_keys.append(key)
        # Include math_teacher_log_prob for only_reverse_kl_advantages mode
        teacher_prefix_config_active = bool(self.config.policy_loss.get("teacher_prefix_enabled", False))
        if (
            (self.config.policy_loss.only_reverse_kl_advantages or teacher_prefix_config_active)
            and "math_teacher_log_prob" in data.batch.keys()
        ):
            if "math_teacher_log_prob" not in select_keys:
                select_keys.append("math_teacher_log_prob")
        if teacher_prefix_config_active and "code_teacher_log_prob" in data.batch.keys():
            if "code_teacher_log_prob" not in select_keys:
                select_keys.append("code_teacher_log_prob")
        
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # Include audit/domain metadata. Training consumes opd_teacher; the
        # remaining keys are used by MOPD sample-level gradient logging.
        for key in ("opd_teacher", "sample_id", "id", "domain", "source_domain", "ability", "data_source", "extra_info"):
            if key in data.non_tensor_batch.keys() and key not in non_tensor_select_keys:
                non_tensor_select_keys.append(key)

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        topk_distill_active = is_topk_distill_enabled(self.config.policy_loss)
        # MOPD audit: domain-gradient tracker begin
        mopd_gradient_tracker = None
        mopd_full_gradient_cfg = data.meta_info.get("mopd_full_gradient", {})
        if isinstance(mopd_full_gradient_cfg, dict) and mopd_full_gradient_cfg.get("enabled", False):
            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker

            mopd_gradient_tracker = SequentialBackwardDomainGradientTracker(self, mopd_full_gradient_cfg)
        # MOPD audit: domain-gradient tracker end
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                micro_batches = list(micro_batches)
                if mopd_gradient_tracker is not None:
                    tracked_micro_batches = mopd_gradient_tracker.prepare_micro_batches(micro_batches)
                else:
                    tracked_micro_batches = [(None, micro_batch) for micro_batch in micro_batches]

                self.actor_optimizer.zero_grad()
                # MOPD audit: domain-gradient tracker begin
                if mopd_gradient_tracker is not None:
                    mopd_gradient_tracker.start_mini_batch()
                # MOPD audit: domain-gradient tracker end

                for mopd_domain, micro_batch in tracked_micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    topk_support_ids = None
                    teacher_support_log_probs = None
                    topk_support_source_value = None
                    if topk_distill_active:
                        (
                            topk_support_ids,
                            teacher_support_log_probs,
                            topk_support_source_value,
                        ) = _select_topk_support_tensors(
                            model_inputs,
                            self.config.policy_loss,
                        )
                    use_renormalized_support = (
                        topk_distill_active
                        and topk_distill_uses_renormalized_support(self.config.policy_loss)
                    )
                    effective_topk_logprob_mode = topk_distill_logprob_mode(self.config.policy_loss)
                    if use_renormalized_support:
                        effective_topk_logprob_mode = TOPK_LOGPROB_MODE_SPARSE
                    kl_loss_coef = float(self.config.kl_loss_coef or 0.0)
                    calculate_log_probs = not topk_distill_active or (
                        self.config.use_kl_loss and kl_loss_coef != 0.0
                    )
                    selected_topk_param_context = nullcontext()
                    if (
                        topk_distill_active
                        and topk_support_ids is not None
                        and use_renormalized_support
                        and effective_topk_logprob_mode == TOPK_LOGPROB_MODE_SPARSE
                        and not calculate_log_probs
                        and not calculate_entropy
                    ):
                        selected_topk_param_context = self._selected_topk_param_context()
                    selected_topk_param_context.__enter__()
                    forward_output = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        gather_topk_ids=topk_support_ids,
                        calculate_log_probs=calculate_log_probs,
                        normalize_gathered_topk=not use_renormalized_support,
                        topk_logprob_chunk_size=topk_distill_logprob_chunk_size(self.config.policy_loss),
                        topk_logprob_mode=effective_topk_logprob_mode,
                        return_extra=topk_distill_active,
                    )
                    if topk_distill_active:
                        entropy, log_prob, _, _, student_topk_log_probs = forward_output
                    else:
                        entropy, log_prob = forward_output
                    prefix_loss_mask, suffix_loss_mask, teacher_prefix_active = teacher_prefix_masks(
                        model_inputs,
                        response_mask,
                        self.config.policy_loss,
                    )
                    distill_response_mask = suffix_loss_mask if teacher_prefix_active else response_mask
                    loss_token_mask = (
                        (prefix_loss_mask + suffix_loss_mask).clamp(max=1.0)
                        if teacher_prefix_active
                        else response_mask
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    if topk_distill_active:
                        pg_loss = log_prob.new_zeros(())
                    else:
                        # only use reverse KL for advantages if only_reverse_kl_advantages is True
                        if self.config.policy_loss.only_reverse_kl_advantages:
                            # Corrected reverse KL with base model normalization if base log probs are available
                            # Formula: (log_prob_actor - log_prob_ref) - (log_prob_actor_base - log_prob_ref_base)
                            # This removes the base model bias from both actor and ref models
                            if "base_log_prob" in model_inputs:
                                lambda_vals = self.config.policy_loss.lambda_vals

                                if self.config.policy_loss.multi_teacher_distill:
                                    #### multi-teacher distillation ####
                                    if "opd_teacher" in model_inputs:
                                        opd_teacher = model_inputs["opd_teacher"]
                                        batch_size = old_log_prob.shape[0]

                                        reverse_kl = torch.zeros_like(old_log_prob)

                                        for i in range(batch_size):
                                            teacher_type = _teacher_type_at(opd_teacher, i)
                                            if teacher_type == "code" and "code_teacher_log_prob" in model_inputs:
                                                teacher_log_prob = model_inputs["code_teacher_log_prob"][i]
                                            else:
                                                teacher_log_prob = model_inputs["math_teacher_log_prob"][i]
                                            if lambda_vals == 1.0:
                                                reverse_kl[i] = old_log_prob[i] - teacher_log_prob
                                            else:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_log_prob"][i] - (teacher_log_prob - model_inputs["base_log_prob"][i]) * lambda_vals
                                    else:
                                        reverse_kl = old_log_prob - model_inputs["math_teacher_log_prob"]
                                    #### multi-teacher distillation ####
                                else:
                                    #### single-teacher distillation ####
                                    reverse_kl = old_log_prob - model_inputs["base_log_prob"]
                                    reward_correction = model_inputs["math_teacher_log_prob"] - model_inputs["base_log_prob"]

                                    if lambda_vals == 1.0:
                                        reverse_kl = old_log_prob - model_inputs["math_teacher_log_prob"]
                                    else:
                                        reverse_kl = reverse_kl - reward_correction * lambda_vals
                                    #### single-teacher distillation ####
                            elif (
                                "code_teacher_log_prob" in model_inputs
                                and self.config.policy_loss.multi_teacher_distill
                                and "opd_teacher" in model_inputs
                            ):
                                opd_teacher = model_inputs["opd_teacher"]
                                batch_size = old_log_prob.shape[0]
                                reverse_kl = torch.zeros_like(old_log_prob)

                                for i in range(batch_size):
                                    teacher_type = _teacher_type_at(opd_teacher, i)
                                    teacher_log_prob = (
                                        model_inputs["code_teacher_log_prob"][i]
                                        if teacher_type == "code"
                                        else model_inputs["math_teacher_log_prob"][i]
                                    )
                                    reverse_kl[i] = old_log_prob[i] - teacher_log_prob
                            else:
                                # Standard reverse KL: log(π_actor / π_ref) = log_prob_actor - log_prob_ref
                                reverse_kl = old_log_prob - model_inputs["math_teacher_log_prob"]
                            advantages = (- (reverse_kl))

                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # Compute policy loss (any function is expected to return 2 values)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=distill_response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                        # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                        rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                        if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                            # Compute metrics using CURRENT policy π_θ vs π_rollout
                            # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                            rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=distill_response_mask,
                            )
                            micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=loss_token_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if topk_distill_active:
                        topk_loss_mat = topk_distill_loss_matrix(
                            student_topk_log_probs=student_topk_log_probs,
                            teacher_topk_log_probs=teacher_support_log_probs,
                            mode=resolved_topk_distill_mode(self.config.policy_loss),
                            include_tail=topk_distill_include_tail(self.config.policy_loss),
                            temperature=topk_distill_temperature(self.config.policy_loss),
                        )
                        topk_loss = agg_loss(
                            loss_mat=topk_loss_mat,
                            loss_mask=distill_response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        topk_weight = topk_distill_weight(self.config.policy_loss)
                        policy_loss = policy_loss + topk_loss * topk_weight
                        micro_batch_metrics["actor/topk_distill_loss"] = (
                            topk_loss.detach().item() * loss_scale_factor
                        )
                        micro_batch_metrics["actor/topk_distill_weight"] = topk_weight
                        micro_batch_metrics["actor/topk_distill_support_is_student"] = (
                            float(topk_support_source_value == TOPK_SUPPORT_SOURCE_STUDENT)
                        )
                        for key, value in topk_distill_bucket_metrics(
                            student_topk_log_probs=student_topk_log_probs,
                            teacher_topk_log_probs=teacher_support_log_probs,
                            response_mask=distill_response_mask,
                            student_values_are_log_probs=not use_renormalized_support,
                            support_source=topk_support_source_value,
                        ).items():
                            micro_batch_metrics[f"actor/{key}"] = value

                    if teacher_prefix_active:
                        prefix_weight = teacher_prefix_forward_weight(self.config.policy_loss)
                        if topk_distill_active:
                            prefix_loss_mat = topk_distill_loss_matrix(
                                student_topk_log_probs=student_topk_log_probs,
                                teacher_topk_log_probs=teacher_support_log_probs,
                                mode=TOPK_RENORMALIZED_FORWARD_KL,
                                include_tail=False,
                                temperature=topk_distill_temperature(self.config.policy_loss),
                            )
                        else:
                            teacher_log_prob = select_teacher_log_prob_tensor(
                                model_inputs,
                                self.config.policy_loss,
                            )
                            prefix_loss_mat = chosen_token_forward_kl_matrix(
                                student_log_probs=log_prob,
                                teacher_log_probs=teacher_log_prob,
                            )
                        prefix_loss = agg_loss(
                            loss_mat=prefix_loss_mat,
                            loss_mask=prefix_loss_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = policy_loss + prefix_loss * prefix_weight
                        micro_batch_metrics["actor/teacher_prefix_forward_kl_loss"] = (
                            prefix_loss.detach().item() * loss_scale_factor
                        )
                        micro_batch_metrics["actor/teacher_prefix_forward_kl_weight"] = prefix_weight
                        micro_batch_metrics["actor/teacher_prefix_token_count"] = (
                            prefix_loss_mask.detach().sum().item() * loss_scale_factor
                        )
                        micro_batch_metrics["actor/student_suffix_token_count"] = (
                            suffix_loss_mask.detach().sum().item() * loss_scale_factor
                        )

                    if self.config.use_kl_loss and kl_loss_coef != 0.0:
                        math_teacher_log_prob = model_inputs["math_teacher_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=math_teacher_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=distill_response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef


                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    # MOPD audit: sample-gradient tracker begin
                    if mopd_gradient_tracker is not None:
                        mopd_gradient_tracker.before_backward(
                            mopd_domain,
                            micro_batch,
                            loss_scale_factor=loss_scale_factor,
                            on_policy=on_policy,
                        )
                    # MOPD audit: sample-gradient tracker end
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    # MOPD audit: domain-gradient tracker begin
                    if mopd_gradient_tracker is not None:
                        mopd_gradient_tracker.after_backward(mopd_domain, len(micro_batch), micro_batch)
                    # MOPD audit: domain-gradient tracker end
                    selected_topk_param_context.__exit__(None, None, None)

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                # MOPD audit: domain-gradient tracker begin
                if mopd_gradient_tracker is not None:
                    append_to_dict(metrics, mopd_gradient_tracker.finish_mini_batch())
                # MOPD audit: domain-gradient tracker end
                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
