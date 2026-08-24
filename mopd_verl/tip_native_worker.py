"""Colocated, sequential teacher/student worker for paper-native TIP."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist

from mopd_verl.tip_native_math import (
    full_vocab_reverse_kl_per_token,
    normalized_entropy_per_token,
    select_tip_tokens,
)
from verl import DataProto
from verl.single_controller.base.decorator import (
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.fsdp_workers import ActorRolloutRefWorker

logger = logging.getLogger(__name__)


def _config_get(config: Any, key: str, default: Any) -> Any:
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


class TIPNativeWorker(ActorRolloutRefWorker):
    """Run frozen teacher and trainable student sequentially on each GPU.

    Full-vocabulary logits are cached in bf16 host memory between phases. This
    follows the executable OPSD topology used by TIP's base implementation and
    avoids transferring ``[batch, response, vocabulary]`` tensors through the
    Ray driver.
    """

    def __init__(self, config: Any, role: str, **kwargs: Any) -> None:
        if role == "actor_rollout":
            role = "actor_rollout_ref"
        super().__init__(config=config, role=role, **kwargs)

    @staticmethod
    def _response_logits(
        model: torch.nn.Module,
        micro_batch: DataProto,
    ) -> torch.Tensor:
        input_ids = micro_batch.batch["input_ids"]
        response_length = micro_batch.batch["responses"].shape[-1]
        with torch.autocast(
            device_type=get_device_name(),
            dtype=torch.bfloat16,
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=micro_batch.batch["attention_mask"],
                position_ids=micro_batch.batch["position_ids"],
                use_cache=False,
            )
            logits = outputs.logits[:, -response_length - 1 : -1, :]
        return logits

    @staticmethod
    def _gather_batch_tensor(local_tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return local_tensor
        gathered = [torch.empty_like(local_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local_tensor.contiguous())
        return torch.cat(gathered, dim=0)

    def _teacher_forward(
        self,
        micro_batches: list[DataProto],
        device: torch.device | int,
    ) -> list[torch.Tensor]:
        ref_offload = bool(self.config.ref.fsdp_config.get("param_offload", False))
        if ref_offload:
            load_fsdp_model_to_gpu(self.ref_module_fsdp)
        self.ref_module_fsdp.eval()
        cache: list[torch.Tensor] = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(device)
            mask = micro_batch.batch["response_mask"].bool()
            with torch.no_grad():
                logits = self._response_logits(self.ref_module_fsdp, micro_batch)
                cache.append(logits[mask].to(device="cpu", dtype=torch.bfloat16))
            del logits
        if ref_offload:
            offload_fsdp_model_to_cpu(self.ref_module_fsdp)
        aggressive_empty_cache(force_sync=True)
        return cache

    def _selector_forward(
        self,
        micro_batches: list[DataProto],
        teacher_cache: list[torch.Tensor],
        *,
        chunk_size: int,
        temperature: float,
        device: torch.device | int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entropy_rows: list[torch.Tensor] = []
        divergence_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        self.actor_module_fsdp.eval()
        for micro_batch, teacher_cpu in zip(
            micro_batches,
            teacher_cache,
            strict=True,
        ):
            micro_batch = micro_batch.to(device)
            mask = micro_batch.batch["response_mask"].bool()
            with torch.no_grad():
                student_logits = self._response_logits(
                    self.actor_module_fsdp,
                    micro_batch,
                )
                student_valid = student_logits[mask]
                teacher_valid = teacher_cpu.to(device)
                if student_valid.shape != teacher_valid.shape:
                    raise RuntimeError(
                        "Native TIP teacher/student response logits disagree: "
                        f"{tuple(teacher_valid.shape)} versus "
                        f"{tuple(student_valid.shape)}."
                    )
                entropy_valid = normalized_entropy_per_token(
                    student_valid,
                    chunk_size=chunk_size,
                    temperature=temperature,
                )
                divergence_valid = full_vocab_reverse_kl_per_token(
                    teacher_valid,
                    student_valid,
                    chunk_size=chunk_size,
                    temperature=temperature,
                )
            entropy_row = torch.zeros_like(mask, dtype=torch.float32)
            divergence_row = torch.zeros_like(mask, dtype=torch.float32)
            entropy_row[mask] = entropy_valid
            divergence_row[mask] = divergence_valid
            entropy_rows.append(entropy_row)
            divergence_rows.append(divergence_row)
            mask_rows.append(mask)
            del student_logits, student_valid, teacher_valid
        return (
            torch.cat(entropy_rows, dim=0),
            torch.cat(divergence_rows, dim=0),
            torch.cat(mask_rows, dim=0),
        )

    def _training_forward(
        self,
        micro_batches: list[DataProto],
        teacher_cache: list[torch.Tensor],
        selected: torch.Tensor,
        *,
        chunk_size: int,
        temperature: float,
        device: torch.device | int,
    ) -> tuple[float, int, torch.Tensor]:
        self.actor_module_fsdp.train()
        self.actor_optimizer.zero_grad()
        local_rows = len(micro_batches)
        total_loss = 0.0
        selected_tokens = 0
        for row, (micro_batch, teacher_cpu) in enumerate(
            zip(micro_batches, teacher_cache, strict=True)
        ):
            micro_batch = micro_batch.to(device)
            response_mask = micro_batch.batch["response_mask"].bool()
            selected_valid = selected[row : row + 1][response_mask]
            if not selected_valid.any():
                raise RuntimeError(
                    "Native TIP selected zero tokens for a rollout; increase "
                    "retention_ratio or require longer responses."
                )
            student_logits = self._response_logits(
                self.actor_module_fsdp,
                micro_batch,
            )
            student_selected = student_logits[response_mask][selected_valid]
            teacher_selected = teacher_cpu.to(device)[selected_valid]
            token_loss = full_vocab_reverse_kl_per_token(
                teacher_selected,
                student_selected,
                chunk_size=chunk_size,
                temperature=temperature,
            )
            row_loss = token_loss.mean()
            (row_loss / local_rows).backward()
            total_loss += float(row_loss.detach().item())
            selected_tokens += int(token_loss.numel())
            del student_logits, student_selected, teacher_selected, token_loss
        grad_norm = self.actor._optimizer_step()
        return total_loss / local_rows, selected_tokens, grad_norm

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_tip_native(self, data: DataProto) -> DataProto:
        """Run one exact TIP scoring and full-vocabulary optimization step."""

        if not self._is_actor or not self._is_ref:
            raise RuntimeError("Native TIP requires a colocated actor_rollout_ref worker.")
        data = data.to("cpu")
        policy = self.config.actor.policy_loss
        chunk_size = int(_config_get(policy, "tip_native_chunk_size", 512))
        temperature = float(_config_get(policy, "tip_native_temperature", 1.0))
        retention_ratio = float(
            _config_get(policy, "tip_native_retention_ratio", 0.5)
        )
        clip_quantile = float(
            _config_get(policy, "tip_native_entropy_clip_quantile", 0.98)
        )
        micro_batch_size = int(self.config.actor.ppo_micro_batch_size_per_gpu)
        if micro_batch_size != 1:
            raise RuntimeError("Native TIP worker requires micro_batch_size_per_gpu=1.")
        micro_batches = list(data.split(micro_batch_size))
        if not micro_batches:
            raise RuntimeError("Native TIP received an empty actor batch.")
        device = get_device_id()

        with self.ulysses_sharding_manager:
            teacher_cache = self._teacher_forward(micro_batches, device)
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                load_fsdp_optimizer(self.actor_optimizer, device_id=device)

            entropy, divergence, response_mask = self._selector_forward(
                micro_batches,
                teacher_cache,
                chunk_size=chunk_size,
                temperature=temperature,
                device=device,
            )
            global_entropy = self._gather_batch_tensor(entropy)
            global_divergence = self._gather_batch_tensor(divergence)
            global_mask = self._gather_batch_tensor(response_mask)
            selection = select_tip_tokens(
                global_entropy,
                global_divergence,
                global_mask,
                retention_ratio=retention_ratio,
                entropy_clip_quantile=clip_quantile,
            )
            if (selection.selected.sum(dim=-1) == 0).any():
                raise RuntimeError(
                    "Native TIP encountered a rollout shorter than "
                    "ceil(1 / retention_ratio), so floor(rho * m) selected "
                    "zero tokens. All ranks abort before backward."
                )
            rank = dist.get_rank() if dist.is_initialized() else 0
            local_batch_size = response_mask.shape[0]
            offset = rank * local_batch_size
            local_selected = selection.selected[
                offset : offset + local_batch_size
            ]
            loss, selected_tokens, grad_norm = self._training_forward(
                micro_batches,
                teacher_cache,
                local_selected,
                chunk_size=chunk_size,
                temperature=temperature,
                device=device,
            )
            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)
        aggressive_empty_cache(force_sync=True)

        valid_count = global_mask.sum().clamp(min=1)
        selected_count = selection.selected.sum()
        metrics = {
            "actor/tip_native_loss": loss,
            "actor/tip_native_selected_ratio": float(
                (selected_count.float() / valid_count.float()).item()
            ),
            "actor/tip_native_selected_tokens_local": selected_tokens,
            "actor/tip_native_entropy_mean": float(
                global_entropy[global_mask].mean().item()
            ),
            "actor/tip_native_reverse_kl_mean": float(
                global_divergence[global_mask].mean().item()
            ),
            "actor/grad_norm": float(grad_norm.detach().item()),
            "actor/lr": float(lr.item() if torch.is_tensor(lr) else lr),
            "perf/max_memory_allocated_gb": float(
                get_torch_device().max_memory_allocated() / (1024**3)
            ),
        }
        return DataProto(meta_info={"metrics": metrics}).to("cpu")
