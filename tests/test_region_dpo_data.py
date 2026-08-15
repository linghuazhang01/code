from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np


class RegionDPODataTests(unittest.TestCase):
    def _torch(self) -> Any:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is unavailable: {exc}")
        return torch

    @contextmanager
    def _stubbed_runtime(self, torch: Any) -> Iterator[type[Any]]:
        class TensorDict(dict):
            def __init__(
                self,
                values: dict[str, Any],
                *,
                batch_size: Any,
            ) -> None:
                super().__init__(values)
                self.batch_size = batch_size

        class DataProto:
            def __init__(
                self,
                *,
                batch: TensorDict,
                non_tensor_batch: dict[str, Any],
                meta_info: dict[str, Any],
            ) -> None:
                self.batch = batch
                self.non_tensor_batch = non_tensor_batch
                self.meta_info = meta_info

            def __len__(self) -> int:
                return int(self.batch.batch_size[0])

            def repeat(
                self,
                *,
                repeat_times: int,
                interleave: bool,
            ) -> Any:
                self_outer = self
                self_test = interleave
                if not self_test:
                    raise AssertionError("test stub expects interleaved repeat")
                indices = torch.arange(len(self_outer)).repeat_interleave(
                    repeat_times
                )
                return DataProto(
                    batch=TensorDict(
                        {
                            key: value[indices]
                            for key, value in self_outer.batch.items()
                        },
                        batch_size=(len(indices),),
                    ),
                    non_tensor_batch={
                        key: np.repeat(value, repeat_times, axis=0)
                        for key, value in self_outer.non_tensor_batch.items()
                    },
                    meta_info=dict(self_outer.meta_info),
                )

        tensordict_module = ModuleType("tensordict")
        tensordict_module.TensorDict = TensorDict
        verl_module = ModuleType("verl")
        verl_module.__path__ = []
        verl_module.DataProto = DataProto
        utils_module = ModuleType("verl.utils")
        utils_module.__path__ = []
        model_module = ModuleType("verl.utils.model")
        model_module.compute_position_id_with_mask = (
            lambda mask: (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        )
        module_names = (
            "mopd_verl.region_dpo",
            "mopd_verl.region_dpo_pairs",
        )
        saved = {name: sys.modules.pop(name, None) for name in module_names}
        try:
            with patch.dict(
                sys.modules,
                {
                    "tensordict": tensordict_module,
                    "verl": verl_module,
                    "verl.utils": utils_module,
                    "verl.utils.model": model_module,
                },
            ):
                yield DataProto
        finally:
            for name in module_names:
                sys.modules.pop(name, None)
                if saved[name] is not None:
                    sys.modules[name] = saved[name]

    def _base_batch(self, torch: Any, DataProto: type[Any]) -> Any:
        from tensordict import TensorDict

        prompts = torch.tensor([[0, 10, 11]])
        responses = torch.tensor([[50, 60, 70, 80, 0, 0]])
        prompt_mask = torch.tensor([[0, 1, 1]])
        response_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
        attention_mask = torch.cat([prompt_mask, response_mask], dim=-1)
        return DataProto(
            batch=TensorDict(
                {
                    "prompts": prompts,
                    "responses": responses,
                    "input_ids": torch.cat([prompts, responses], dim=-1),
                    "attention_mask": attention_mask,
                    "position_ids": (
                        attention_mask.cumsum(dim=-1) - 1
                    ).clamp(min=0),
                    "response_mask": response_mask,
                },
                batch_size=(1,),
            ),
            non_tensor_batch={
                "domain": np.asarray(["math"], dtype=object),
                "reward_model": np.asarray(
                    [{"ground_truth": "1"}], dtype=object
                ),
            },
            meta_info={"temperature": 1.0},
        )

    def test_points_and_branches_control_prompt_count(self) -> None:
        torch = self._torch()
        with self._stubbed_runtime(torch) as DataProto:
            from mopd_verl.region_dpo import RegionDPOController

            base = self._base_batch(torch, DataProto)
            controller = RegionDPOController(
                {
                    "enabled": True,
                    "points_per_rollout": 2,
                    "branches_per_point": 3,
                    "max_new_tokens": 2,
                    "selection_strategy": "first",
                    "domain_control_token_ids": {"math": [50, 70]},
                },
                pad_token_id=0,
            )
            prompts, plan, metrics = controller.build_rerollout_prompts(
                base,
                global_step=1,
            )

            self.assertIsNotNone(prompts)
            self.assertEqual(len(plan.anchors), 2)
            self.assertEqual(plan.candidate_count, 6)
            self.assertEqual(len(prompts), 6)
            self.assertEqual(metrics["region_dpo/selected_point_count"], 2.0)
            self.assertEqual(
                prompts.non_tensor_batch["raw_prompt_ids"].tolist(),
                [
                    [10, 11],
                    [10, 11],
                    [10, 11],
                    [10, 11, 50, 60],
                    [10, 11, 50, 60],
                    [10, 11, 50, 60],
                ],
            )

    def test_reward_restoration_and_pair_packing(self) -> None:
        torch = self._torch()
        with self._stubbed_runtime(torch) as DataProto:
            from tensordict import TensorDict

            from mopd_verl.region_dpo import (
                RegionAnchor,
                RegionRerolloutPlan,
            )
            from mopd_verl.region_dpo_pairs import (
                attach_region_dpo_preference_pairs,
                build_region_dpo_reward_batch,
            )

            base = self._base_batch(torch, DataProto)
            plan = RegionRerolloutPlan(
                anchors=(
                    RegionAnchor(
                        base_index=0,
                        slot_index=0,
                        response_position=2,
                        domain="math",
                    ),
                ),
                branches_per_point=3,
                max_new_tokens=2,
            )
            candidate_prompts = torch.tensor(
                [[10, 11, 50, 60]] * 3
            )
            candidate_responses = torch.tensor(
                [[101, 102], [101, 103], [104, 105]]
            )
            candidate_attention = torch.ones((3, 6), dtype=torch.long)
            rerollout = DataProto(
                batch=TensorDict(
                    {
                        "prompts": candidate_prompts,
                        "responses": candidate_responses,
                        "input_ids": torch.cat(
                            [candidate_prompts, candidate_responses], dim=-1
                        ),
                        "attention_mask": candidate_attention,
                        "position_ids": torch.arange(6).repeat(3, 1),
                        "rollout_log_probs": torch.tensor(
                            [
                                [-0.1, -0.2],
                                [-0.3, -0.4],
                                [-0.5, -0.6],
                            ]
                        ),
                    },
                    batch_size=(3,),
                ),
                non_tensor_batch={},
                meta_info={},
            )

            reward_batch = build_region_dpo_reward_batch(
                base,
                rerollout,
                plan,
                pad_token_id=0,
            )
            metrics = attach_region_dpo_preference_pairs(
                base,
                rerollout,
                plan,
                torch.tensor([0.5, 1.0, 0.0]),
                points_per_rollout=1,
                pad_token_id=0,
                min_reward_margin=0.0,
            )

            self.assertEqual(
                reward_batch.batch["responses"][1, :4].tolist(),
                [50, 60, 101, 103],
            )
            self.assertEqual(metrics["region_dpo/confirmed_pair_count"], 1.0)
            self.assertEqual(
                base.batch["region_dpo_responses"][0, 0, 0].tolist(),
                [101, 103],
            )
            self.assertEqual(
                base.batch["region_dpo_responses"][0, 0, 1].tolist(),
                [104, 105],
            )
            self.assertEqual(
                base.batch["region_dpo_loss_mask"][0, 0].tolist(),
                [[1.0, 1.0], [1.0, 1.0]],
            )


if __name__ == "__main__":
    unittest.main()
