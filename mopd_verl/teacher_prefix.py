"""Utilities for dataset-provided teacher-prefix roll-in."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def teacher_prefix_sampling_enabled(rollout_config: Any) -> bool:
    return bool(cfg_get(rollout_config, "teacher_prefix_sampling_enabled", False))


def teacher_prefix_length(rollout_config: Any, response_length: int) -> int:
    value = int(cfg_get(rollout_config, "teacher_prefix_length", 0) or 0)
    return max(0, min(value, int(response_length)))


def teacher_prefix_dataset_key(rollout_config: Any) -> str:
    return str(cfg_get(rollout_config, "teacher_prefix_dataset_key", "prefix") or "prefix")


def _valid_token_list(token_ids: torch.Tensor, mask: torch.Tensor) -> list[int]:
    return token_ids[mask.to(dtype=torch.bool)].detach().cpu().tolist()


def _prefix_value_to_token_list(value: Any, tokenizer: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if all(isinstance(item, (int, np.integer)) for item in value):
            return [int(item) for item in value]
        if all(isinstance(item, dict) for item in value) and hasattr(tokenizer, "apply_chat_template"):
            return list(tokenizer.apply_chat_template(list(value), add_generation_prompt=False))
        value = "".join(str(item) for item in value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        if not value:
            return []
        return list(tokenizer.encode(value, add_special_tokens=False))
    return list(tokenizer.encode(str(value), add_special_tokens=False))


def build_dataset_teacher_prefix(
    *,
    prompts: DataProto,
    tokenizer: Any,
    prefix_key: str,
    prefix_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(prompts.batch.batch_size[0])
    device = prompts.batch["input_ids"].device
    max_len = max(0, int(prefix_length))
    if max_len <= 0:
        empty_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        empty_mask = torch.empty((batch_size, 0), dtype=prompts.batch["attention_mask"].dtype, device=device)
        return empty_ids, empty_mask

    values = prompts.non_tensor_batch.get(prefix_key)
    if values is not None and len(values) != batch_size:
        raise ValueError(
            f"{prefix_key!r} must contain one entry per prompt, got "
            f"{len(values)} entries for batch size {batch_size}."
        )
    token_rows: list[list[int]] = []
    for idx in range(batch_size):
        raw_value = values[idx] if values is not None and idx < len(values) else None
        tokens = _prefix_value_to_token_list(raw_value, tokenizer)[:max_len]
        token_rows.append(tokens)

    prefix_ids = torch.full(
        (batch_size, max_len),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    prefix_mask = torch.zeros((batch_size, max_len), dtype=prompts.batch["attention_mask"].dtype, device=device)
    for idx, tokens in enumerate(token_rows):
        if not tokens:
            continue
        values_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        prefix_ids[idx, : len(tokens)] = values_tensor
        prefix_mask[idx, : len(tokens)] = 1
    return prefix_ids, prefix_mask


def _left_pad_token_lists(
    token_lists: list[list[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(tokens) for tokens in token_lists), default=0)
    max_len = max(1, max_len)
    input_ids = torch.full(
        (len(token_lists), max_len),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((len(token_lists), max_len), dtype=torch.long, device=device)
    for idx, tokens in enumerate(token_lists):
        if not tokens:
            continue
        values = torch.tensor(tokens, dtype=torch.long, device=device)
        input_ids[idx, -len(tokens) :] = values
        attention_mask[idx, -len(tokens) :] = 1
    return input_ids, attention_mask


def build_student_suffix_prompts(
    *,
    prompts: DataProto,
    teacher_prefix_ids: torch.Tensor,
    teacher_prefix_mask: torch.Tensor,
    pad_token_id: int,
) -> DataProto:
    prompt_ids = prompts.batch["input_ids"]
    prompt_attention = prompts.batch["attention_mask"]
    device = prompt_ids.device
    raw_prompt_ids = prompts.non_tensor_batch.get("raw_prompt_ids")
    batch_size = int(prompt_ids.shape[0])
    if raw_prompt_ids is not None and len(raw_prompt_ids) != batch_size:
        raise ValueError(
            "raw_prompt_ids must contain one entry per prompt, got "
            f"{len(raw_prompt_ids)} entries for batch size {batch_size}."
        )

    token_lists: list[list[int]] = []
    for idx in range(batch_size):
        if raw_prompt_ids is None:
            prompt_tokens = _valid_token_list(
                prompt_ids[idx],
                prompt_attention[idx],
            )
        else:
            prompt_tokens = [
                int(token_id) for token_id in raw_prompt_ids[idx]
            ]
        prefix_tokens = _valid_token_list(teacher_prefix_ids[idx], teacher_prefix_mask[idx])
        token_lists.append(prompt_tokens + prefix_tokens)

    input_ids, attention_mask = _left_pad_token_lists(token_lists, pad_token_id=pad_token_id, device=device)
    position_ids = compute_position_id_with_mask(attention_mask)
    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=input_ids.shape[0],
    )
    non_tensor_batch = dict(prompts.non_tensor_batch)
    conditioned_raw_prompt_ids = np.empty(batch_size, dtype=object)
    conditioned_raw_prompt_ids[:] = [
        list(token_ids) for token_ids in token_lists
    ]
    # vLLM/SGLang prefer raw_prompt_ids over the padded tensor input. Rebuild
    # this field so the actual rollout context matches input_ids.
    non_tensor_batch["raw_prompt_ids"] = conditioned_raw_prompt_ids
    return DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info=dict(prompts.meta_info),
    )


def merge_teacher_prefix_and_student_suffix(
    *,
    original_prompts: DataProto,
    teacher_prefix_ids: torch.Tensor,
    teacher_prefix_mask: torch.Tensor,
    student_suffix_output: DataProto | None,
    student_suffix_max_tokens: int | None = None,
    max_response_length: int,
    pad_token_id: int,
) -> DataProto:
    prompt_ids = original_prompts.batch["input_ids"]
    prompt_attention = original_prompts.batch["attention_mask"]
    device = prompt_ids.device
    batch_size = int(prompt_ids.shape[0])

    if student_suffix_output is None:
        suffix_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        suffix_mask = torch.empty((batch_size, 0), dtype=prompt_attention.dtype, device=device)
    else:
        suffix_ids = student_suffix_output.batch["responses"].to(device)
        suffix_attention = student_suffix_output.batch["attention_mask"].to(device)
        suffix_len = int(suffix_ids.shape[1])
        suffix_mask = suffix_attention[:, -suffix_len:] if suffix_len > 0 else suffix_attention[:, :0]
        if student_suffix_max_tokens is not None and int(student_suffix_max_tokens) < suffix_len:
            suffix_mask = suffix_mask.clone()
            suffix_mask[:, int(student_suffix_max_tokens) :] = 0

    response_rows: list[list[int]] = []
    prefix_mask_rows: list[list[int]] = []
    suffix_mask_rows: list[list[int]] = []
    suffix_rollout_log_probs = (
        student_suffix_output.batch.get("rollout_log_probs")
        if student_suffix_output is not None
        else None
    )
    rollout_log_prob_rows: list[torch.Tensor] = []
    for idx in range(batch_size):
        prefix_tokens = _valid_token_list(teacher_prefix_ids[idx], teacher_prefix_mask[idx])
        remaining = max(0, int(max_response_length) - len(prefix_tokens))
        valid_suffix = suffix_mask[idx].to(dtype=torch.bool)
        suffix_tokens = suffix_ids[idx][valid_suffix].detach().cpu().tolist()[
            :remaining
        ]
        response_tokens = (prefix_tokens + suffix_tokens)[: int(max_response_length)]
        prefix_count = min(len(prefix_tokens), int(max_response_length))
        suffix_count = max(0, len(response_tokens) - prefix_count)

        padding = int(max_response_length) - len(response_tokens)
        response_rows.append(response_tokens + [int(pad_token_id)] * padding)
        prefix_mask_rows.append([1] * prefix_count + [0] * (int(max_response_length) - prefix_count))
        suffix_mask_rows.append(
            [0] * prefix_count
            + [1] * suffix_count
            + [0] * (int(max_response_length) - prefix_count - suffix_count)
        )
        if isinstance(suffix_rollout_log_probs, torch.Tensor):
            valid_log_probs = suffix_rollout_log_probs[idx][valid_suffix][
                :suffix_count
            ]
            rollout_log_prob_rows.append(
                torch.cat(
                    [
                        torch.zeros(
                            prefix_count,
                            dtype=suffix_rollout_log_probs.dtype,
                            device=device,
                        ),
                        valid_log_probs.to(device),
                        torch.full(
                            (padding,),
                            -1.0,
                            dtype=suffix_rollout_log_probs.dtype,
                            device=device,
                        ),
                    ]
                )
            )

    responses = torch.tensor(response_rows, dtype=torch.long, device=device)
    final_prefix_mask = torch.tensor(prefix_mask_rows, dtype=prompt_attention.dtype, device=device)
    final_suffix_mask = torch.tensor(suffix_mask_rows, dtype=prompt_attention.dtype, device=device)
    response_mask = (final_prefix_mask + final_suffix_mask).clamp(max=1)

    input_ids = torch.cat([prompt_ids, responses], dim=-1)
    attention_mask = torch.cat([prompt_attention, response_mask], dim=-1)
    if "position_ids" in original_prompts.batch and original_prompts.batch["position_ids"].dim() == 3:
        prompt_position = original_prompts.batch["position_ids"]
        response_len = int(responses.shape[1])
        delta = torch.arange(1, response_len + 1, device=device)
        delta = delta.view(1, 1, -1).expand(batch_size, prompt_position.size(1), -1)
        response_position = prompt_position[..., -1:] + delta
        position_ids = torch.cat([prompt_position, response_position], dim=-1)
    else:
        position_ids = compute_position_id_with_mask(attention_mask)

    batch_values = {
        "prompts": prompt_ids,
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": response_mask,
        "teacher_prefix_mask": final_prefix_mask,
        "student_suffix_mask": final_suffix_mask,
    }
    if rollout_log_prob_rows:
        batch_values["rollout_log_probs"] = torch.stack(
            rollout_log_prob_rows,
            dim=0,
        )
    batch = TensorDict(batch_values, batch_size=batch_size)
    return DataProto(
        batch=batch,
        non_tensor_batch=dict(original_prompts.non_tensor_batch),
        meta_info=dict(original_prompts.meta_info),
    )


def fill_teacher_prefix_rollout_log_probs(
    *,
    rollout_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    teacher_prefix_mask: torch.Tensor,
) -> torch.Tensor:
    """Assign neutral IS ratios to dataset-provided teacher-prefix tokens."""

    prefix_mask = teacher_prefix_mask.to(
        device=rollout_log_probs.device,
        dtype=torch.bool,
    )
    return torch.where(
        prefix_mask,
        old_log_probs.detach().to(
            device=rollout_log_probs.device,
            dtype=rollout_log_probs.dtype,
        ),
        rollout_log_probs,
    )


def teacher_prefix_rollout_correction_masks(
    *,
    response_mask: torch.Tensor,
    teacher_prefix_mask: torch.Tensor,
    student_suffix_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate fixed prefix tokens from suffix-only rollout correction."""

    prefix_mask = teacher_prefix_mask.to(
        device=response_mask.device,
        dtype=response_mask.dtype,
    )
    suffix_mask = student_suffix_mask.to(
        device=response_mask.device,
        dtype=response_mask.dtype,
    )
    return (
        prefix_mask * response_mask,
        suffix_mask * response_mask,
    )


def restore_teacher_prefix_response_mask(
    *,
    prefix_mask: torch.Tensor,
    corrected_suffix_mask: torch.Tensor,
) -> torch.Tensor:
    """Restore prefix loss activity after suffix rejection sampling."""

    return (
        prefix_mask.to(
            device=corrected_suffix_mask.device,
            dtype=corrected_suffix_mask.dtype,
        )
        + corrected_suffix_mask
    ).clamp(max=1)


def teacher_prefix_rollin_metrics(
    *,
    teacher_prefix_mask: torch.Tensor,
    student_suffix_mask: torch.Tensor,
    selected: np.ndarray,
) -> dict[str, float]:
    prefix_lengths = teacher_prefix_mask.detach().float().sum(dim=-1)
    suffix_lengths = student_suffix_mask.detach().float().sum(dim=-1)
    selected_count = float(np.asarray(selected, dtype=bool).sum())
    batch_size = max(1, int(teacher_prefix_mask.shape[0]))
    return {
        "teacher_prefix/sample_frac": selected_count / float(batch_size),
        "teacher_prefix/mean_len": float(prefix_lengths.mean().cpu().item()),
        "teacher_prefix/max_len": float(prefix_lengths.max().cpu().item()) if batch_size > 0 else 0.0,
        "teacher_prefix/suffix_mean_len": float(suffix_lengths.mean().cpu().item()),
    }
