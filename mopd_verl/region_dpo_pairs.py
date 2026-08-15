"""Reward restoration and chosen/rejected packing for Region-DPO."""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from mopd_verl.region_dpo import (
    RegionRerolloutPlan,
    _response_mask,
    _valid_tokens,
)
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


def build_region_dpo_reward_batch(
    base_batch: DataProto,
    rerollout_output: DataProto,
    plan: RegionRerolloutPlan,
    *,
    pad_token_id: int,
) -> DataProto:
    """Restore the original problem prompt and pre-anchor response prefix."""

    if len(rerollout_output) != plan.candidate_count:
        raise ValueError(
            "Region-DPO rerollout count does not match the acquisition plan."
        )
    if rerollout_output.batch["position_ids"].ndim != 2:
        raise ValueError("Region-DPO currently supports text-only position IDs.")
    device = rerollout_output.batch["responses"].device
    prompt_width = int(base_batch.batch["prompts"].shape[-1])
    response_width = int(base_batch.batch["responses"].shape[-1])
    prompt_attention = base_batch.batch["attention_mask"][:, :prompt_width]
    suffix_mask = _response_mask(rerollout_output).to(dtype=torch.bool)
    full_responses = torch.full(
        (plan.candidate_count, response_width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    full_response_mask = torch.zeros(
        (plan.candidate_count, response_width),
        dtype=prompt_attention.dtype,
        device=device,
    )
    base_indices: list[int] = []
    for candidate_index in range(plan.candidate_count):
        anchor = plan.anchor_for_candidate(candidate_index)
        base_index = anchor.base_index
        prefix = _valid_tokens(
            base_batch.batch["responses"][
                base_index, : anchor.response_position
            ],
            _response_mask(base_batch)[
                base_index, : anchor.response_position
            ],
        )
        suffix = _valid_tokens(
            rerollout_output.batch["responses"][candidate_index],
            suffix_mask[candidate_index],
        )
        combined = (prefix + suffix)[:response_width]
        if combined:
            values = torch.tensor(
                combined,
                dtype=torch.long,
                device=device,
            )
            full_responses[candidate_index, : len(combined)] = values
            full_response_mask[candidate_index, : len(combined)] = 1
        base_indices.append(base_index)

    base_device = base_batch.batch["prompts"].device
    indices = torch.tensor(
        base_indices,
        dtype=torch.long,
        device=base_device,
    )
    prompts = base_batch.batch["prompts"][indices].to(device)
    selected_prompt_attention = prompt_attention[indices].to(device)
    input_ids = torch.cat([prompts, full_responses], dim=-1)
    attention_mask = torch.cat(
        [selected_prompt_attention, full_response_mask],
        dim=-1,
    )
    position_ids = compute_position_id_with_mask(attention_mask)
    selected_np = np.asarray(base_indices, dtype=np.int64)
    non_tensor = {
        key: values[selected_np]
        for key, values in base_batch.non_tensor_batch.items()
    }
    return DataProto(
        batch=TensorDict(
            {
                "prompts": prompts,
                "responses": full_responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": full_response_mask,
            },
            batch_size=(plan.candidate_count,),
        ),
        non_tensor_batch=non_tensor,
        meta_info=dict(base_batch.meta_info),
    )


def _first_divergence(
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
) -> int | None:
    for position in range(int(chosen_ids.shape[-1])):
        chosen_valid = bool(chosen_mask[position].item())
        rejected_valid = bool(rejected_mask[position].item())
        if not chosen_valid and not rejected_valid:
            break
        if chosen_valid != rejected_valid:
            return position
        if chosen_valid and int(chosen_ids[position].item()) != int(
            rejected_ids[position].item()
        ):
            return position
    return None


def attach_region_dpo_preference_pairs(
    base_batch: DataProto,
    rerollout_output: DataProto,
    plan: RegionRerolloutPlan,
    rewards: torch.Tensor,
    *,
    points_per_rollout: int,
    pad_token_id: int,
    min_reward_margin: float,
) -> dict[str, float]:
    """Choose best/worst siblings and pack them beside each base row."""

    if rewards.ndim != 1 or len(rewards) != plan.candidate_count:
        raise ValueError(
            "Region-DPO rewards must contain one scalar per rerollout."
        )
    if "rollout_log_probs" not in rerollout_output.batch:
        raise ValueError("Region-DPO requires rollout.calculate_log_probs=true.")
    responses = rerollout_output.batch["responses"]
    candidate_mask = _response_mask(rerollout_output).to(dtype=torch.bool)
    batch_size = int(base_batch.batch.batch_size[0])
    branch_shape = (batch_size, int(points_per_rollout), 2)

    def packed(source: torch.Tensor, fill: float | int) -> torch.Tensor:
        return torch.full(
            (*branch_shape, *source.shape[1:]),
            fill,
            dtype=source.dtype,
            device=source.device,
        )

    packed_responses = packed(responses, int(pad_token_id))
    packed_input_ids = packed(
        rerollout_output.batch["input_ids"], int(pad_token_id)
    )
    packed_attention = packed(
        rerollout_output.batch["attention_mask"], 0
    )
    packed_positions = packed(rerollout_output.batch["position_ids"], 0)
    packed_reference = packed(
        rerollout_output.batch["rollout_log_probs"], 0.0
    )
    packed_loss_mask = torch.zeros(
        (*branch_shape, responses.shape[-1]),
        dtype=torch.float32,
        device=responses.device,
    )
    pair_mask = torch.zeros(
        (batch_size, int(points_per_rollout)),
        dtype=torch.float32,
        device=responses.device,
    )
    pair_rewards = torch.zeros(
        branch_shape,
        dtype=torch.float32,
        device=responses.device,
    )
    reward_values = rewards.detach().cpu().float().tolist()
    margins: list[float] = []
    pair_token_count = 0.0
    for anchor_index, anchor in enumerate(plan.anchors):
        start = anchor_index * plan.branches_per_point
        candidate_indices = list(range(start, start + plan.branches_per_point))
        valid = [
            index
            for index in candidate_indices
            if bool(candidate_mask[index].any().item())
            and np.isfinite(reward_values[index])
        ]
        if len(valid) < 2:
            continue
        chosen_index = max(valid, key=lambda index: reward_values[index])
        rejected_index = min(valid, key=lambda index: reward_values[index])
        margin = reward_values[chosen_index] - reward_values[rejected_index]
        if margin <= float(min_reward_margin):
            continue
        divergence = _first_divergence(
            responses[chosen_index],
            candidate_mask[chosen_index],
            responses[rejected_index],
            candidate_mask[rejected_index],
        )
        if divergence is None:
            continue
        base_index = anchor.base_index
        slot_index = anchor.slot_index
        for branch_index, candidate_index in enumerate(
            (chosen_index, rejected_index)
        ):
            packed_responses[base_index, slot_index, branch_index] = responses[
                candidate_index
            ]
            packed_input_ids[base_index, slot_index, branch_index] = (
                rerollout_output.batch["input_ids"][candidate_index]
            )
            packed_attention[base_index, slot_index, branch_index] = (
                rerollout_output.batch["attention_mask"][candidate_index]
            )
            packed_positions[base_index, slot_index, branch_index] = (
                rerollout_output.batch["position_ids"][candidate_index]
            )
            packed_reference[base_index, slot_index, branch_index] = (
                rerollout_output.batch["rollout_log_probs"][candidate_index]
            )
            loss_mask = candidate_mask[candidate_index].clone()
            loss_mask[:divergence] = False
            packed_loss_mask[base_index, slot_index, branch_index] = (
                loss_mask.float()
            )
            pair_rewards[base_index, slot_index, branch_index] = reward_values[
                candidate_index
            ]
            pair_token_count += float(loss_mask.sum().item())
        pair_mask[base_index, slot_index] = 1.0
        margins.append(float(margin))

    pair_count = int(pair_mask.sum().item())
    if pair_count:
        base_batch.batch.update(
            {
                "region_dpo_responses": packed_responses,
                "region_dpo_input_ids": packed_input_ids,
                "region_dpo_attention_mask": packed_attention,
                "region_dpo_position_ids": packed_positions,
                "region_dpo_reference_log_probs": packed_reference,
                "region_dpo_loss_mask": packed_loss_mask,
                "region_dpo_pair_mask": pair_mask,
                "region_dpo_rewards": pair_rewards,
            }
        )
    return {
        "region_dpo/confirmed_pair_count": float(pair_count),
        "region_dpo/confirmed_pair_fraction": (
            float(pair_count) / max(1.0, float(len(plan.anchors)))
        ),
        "region_dpo/reward_margin_mean": (
            float(np.mean(margins)) if margins else 0.0
        ),
        "region_dpo/credit_token_count": pair_token_count,
    }
