"""Control-anchored sibling rerollouts for Region-DPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from mopd_verl.reproducibility import derive_seed
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


@dataclass(frozen=True)
class RegionAnchor:
    """One response occurrence selected as a rerollout branch point."""

    base_index: int
    slot_index: int
    response_position: int
    domain: str


@dataclass(frozen=True)
class RegionRerolloutPlan:
    """Stable anchor-to-candidate ordering for one training step."""

    anchors: tuple[RegionAnchor, ...]
    branches_per_point: int
    max_new_tokens: int

    @property
    def candidate_count(self) -> int:
        return len(self.anchors) * self.branches_per_point

    def anchor_for_candidate(self, candidate_index: int) -> RegionAnchor:
        return self.anchors[candidate_index // self.branches_per_point]


def _cfg_get(config: Any, key: str, default: Any) -> Any:
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


def _object_array(values: list[Any]) -> np.ndarray:
    output = np.empty(len(values), dtype=object)
    output[:] = values
    return output


def _valid_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> list[int]:
    return (
        tokens[mask.to(dtype=torch.bool)].detach().cpu().long().tolist()
    )


def _left_pad(
    token_rows: list[list[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(1, max((len(row) for row in token_rows), default=0))
    input_ids = torch.full(
        (len(token_rows), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(token_rows), width),
        dtype=torch.long,
        device=device,
    )
    for index, row in enumerate(token_rows):
        if not row:
            continue
        values = torch.tensor(row, dtype=torch.long, device=device)
        input_ids[index, -len(row) :] = values
        attention_mask[index, -len(row) :] = 1
    return input_ids, attention_mask


def _response_mask(data: DataProto) -> torch.Tensor:
    if "response_mask" in data.batch:
        return data.batch["response_mask"]
    response_length = int(data.batch["responses"].shape[-1])
    return data.batch["attention_mask"][:, -response_length:]


def _domain_values(batch: DataProto) -> list[str]:
    batch_size = int(batch.batch.batch_size[0])
    for key in (
        "domain",
        "source_domain",
        "ability",
        "data_source",
        "opd_teacher",
    ):
        values = batch.non_tensor_batch.get(key)
        if values is not None:
            return [str(value).strip().lower() for value in values]
    return ["default"] * batch_size


class RegionDPOController:
    """Select anchors, construct sibling prompts, and pack DPO pairs."""

    def __init__(self, config: Any, *, pad_token_id: int):
        self.enabled = bool(_cfg_get(config, "enabled", False))
        self.points_per_rollout = int(
            _cfg_get(config, "points_per_rollout", 1)
        )
        self.branches_per_point = int(
            _cfg_get(config, "branches_per_point", 4)
        )
        self.max_new_tokens = int(
            _cfg_get(config, "max_new_tokens", 256)
        )
        self.min_reward_margin = float(
            _cfg_get(config, "min_reward_margin", 0.0)
        )
        self.selection_strategy = str(
            _cfg_get(config, "selection_strategy", "random")
        ).lower()
        self.seed = int(_cfg_get(config, "seed", 42))
        self.pad_token_id = int(pad_token_id)
        self.control_token_ids = frozenset(
            int(token_id)
            for token_id in (_cfg_get(config, "control_token_ids", []) or [])
        )
        raw_domains = _cfg_get(config, "domain_control_token_ids", {}) or {}
        self.domain_control_token_ids = {
            str(domain).strip().lower(): frozenset(
                int(token_id) for token_id in token_ids
            )
            for domain, token_ids in raw_domains.items()
        }

    def _control_ids(self, domain: str) -> frozenset[int]:
        return self.domain_control_token_ids.get(
            domain,
            self.control_token_ids,
        )

    def _choose_positions(
        self,
        eligible: list[int],
        *,
        global_step: int,
        base_index: int,
        batch_size: int,
    ) -> list[int]:
        count = min(self.points_per_rollout, len(eligible))
        if count >= len(eligible):
            return eligible
        if self.selection_strategy == "first":
            return eligible[:count]
        if self.selection_strategy == "uniform":
            indices = np.linspace(
                0,
                len(eligible) - 1,
                num=count,
                dtype=int,
            )
            return [eligible[int(index)] for index in indices]
        offset = int(global_step) * max(1, int(batch_size)) + base_index
        generator = np.random.default_rng(derive_seed(self.seed, offset))
        selected = generator.choice(eligible, size=count, replace=False)
        return sorted(int(position) for position in selected.tolist())

    def select_anchors(
        self,
        batch: DataProto,
        *,
        global_step: int,
    ) -> tuple[tuple[RegionAnchor, ...], int]:
        """Select up to ``points_per_rollout`` control occurrences per row."""

        responses = batch.batch["responses"]
        response_mask = _response_mask(batch).to(dtype=torch.bool)
        domains = _domain_values(batch)
        batch_size, response_capacity = responses.shape
        anchors: list[RegionAnchor] = []
        eligible_count = 0
        for base_index in range(batch_size):
            control_ids = self._control_ids(domains[base_index])
            if not control_ids:
                continue
            eligible = [
                position
                for position in range(response_capacity)
                if bool(response_mask[base_index, position].item())
                and int(responses[base_index, position].item())
                in control_ids
                and position + self.max_new_tokens <= response_capacity
            ]
            eligible_count += len(eligible)
            selected = self._choose_positions(
                eligible,
                global_step=global_step,
                base_index=base_index,
                batch_size=batch_size,
            )
            anchors.extend(
                RegionAnchor(
                    base_index=base_index,
                    slot_index=slot_index,
                    response_position=position,
                    domain=domains[base_index],
                )
                for slot_index, position in enumerate(selected)
            )
        return tuple(anchors), eligible_count

    def build_rerollout_prompts(
        self,
        batch: DataProto,
        *,
        global_step: int,
    ) -> tuple[DataProto | None, RegionRerolloutPlan, dict[str, float]]:
        """Create anchor prefixes ending immediately before the control token."""

        if batch.batch["position_ids"].ndim != 2:
            raise ValueError(
                "Region-DPO currently supports text-only position IDs."
            )

        anchors, eligible_count = self.select_anchors(
            batch,
            global_step=global_step,
        )
        plan = RegionRerolloutPlan(
            anchors=anchors,
            branches_per_point=self.branches_per_point,
            max_new_tokens=self.max_new_tokens,
        )
        metrics = {
            "region_dpo/eligible_point_count": float(eligible_count),
            "region_dpo/selected_point_count": float(len(anchors)),
            "region_dpo/candidate_count": float(plan.candidate_count),
        }
        if not anchors:
            return None, plan, metrics

        prompt_width = int(batch.batch["prompts"].shape[-1])
        prompt_attention = batch.batch["attention_mask"][:, :prompt_width]
        token_rows: list[list[int]] = []
        base_indices: list[int] = []
        for anchor in anchors:
            base_index = anchor.base_index
            prompt_tokens = _valid_tokens(
                batch.batch["prompts"][base_index],
                prompt_attention[base_index],
            )
            response_prefix = _valid_tokens(
                batch.batch["responses"][
                    base_index, : anchor.response_position
                ],
                _response_mask(batch)[
                    base_index, : anchor.response_position
                ],
            )
            token_rows.append(prompt_tokens + response_prefix)
            base_indices.append(base_index)

        input_ids, attention_mask = _left_pad(
            token_rows,
            pad_token_id=self.pad_token_id,
            device=batch.batch["input_ids"].device,
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        selected = np.asarray(base_indices, dtype=np.int64)
        non_tensor = {
            key: values[selected]
            for key, values in batch.non_tensor_batch.items()
        }
        non_tensor["raw_prompt_ids"] = _object_array(
            [list(row) for row in token_rows]
        )
        anchor_prompts = DataProto(
            batch=TensorDict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                batch_size=(input_ids.shape[0],),
            ),
            non_tensor_batch=non_tensor,
            meta_info=dict(batch.meta_info),
        )
        prompts = anchor_prompts.repeat(
            repeat_times=self.branches_per_point,
            interleave=True,
        )
        prompts.meta_info.update(
            {
                "do_sample": True,
                "validate": False,
                "mopd_response_length": self.max_new_tokens,
            }
        )
        return prompts, plan, metrics
