from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Iterator
from unittest.mock import patch


class TeacherPrefixTests(unittest.TestCase):
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
        module_name = "mopd_verl.teacher_prefix"
        saved_module = sys.modules.pop(module_name, None)
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
            sys.modules.pop(module_name, None)
            if saved_module is not None:
                sys.modules[module_name] = saved_module

    def test_dataset_prefix_conditions_suffix_and_builds_active_masks(
        self,
    ) -> None:
        torch = self._torch()
        with self._stubbed_runtime(torch) as DataProto:
            from mopd_verl.teacher_prefix import (
                build_dataset_teacher_prefix,
                build_student_suffix_prompts,
                fill_teacher_prefix_rollout_log_probs,
                merge_teacher_prefix_and_student_suffix,
                restore_teacher_prefix_response_mask,
                teacher_prefix_rollout_correction_masks,
                teacher_prefix_rollin_metrics,
            )
            from tensordict import TensorDict

            prompts = DataProto(
                batch=TensorDict(
                    {
                        "input_ids": torch.tensor(
                            [[0, 10, 11], [0, 20, 21]]
                        ),
                        "attention_mask": torch.tensor(
                            [[0, 1, 1], [0, 1, 1]]
                        ),
                        "position_ids": torch.tensor(
                            [[0, 0, 1], [0, 0, 1]]
                        ),
                    },
                    batch_size=(2,),
                ),
                non_tensor_batch={
                    "data_source": ["ab", ""],
                    "raw_prompt_ids": [
                        [10, 11],
                        [20, 21],
                    ],
                },
                meta_info={"temperature": 1.0},
            )

            class Tokenizer:
                @staticmethod
                def encode(
                    value: str,
                    *,
                    add_special_tokens: bool,
                ) -> list[int]:
                    self.assertFalse(add_special_tokens)
                    return [ord(character) for character in value]

            prefix_ids, prefix_mask = build_dataset_teacher_prefix(
                prompts=prompts,
                tokenizer=Tokenizer(),
                prefix_key="data_source",
                prefix_length=2,
                pad_token_id=0,
            )
            misaligned_prompts = DataProto(
                batch=prompts.batch,
                non_tensor_batch={"data_source": ["only-one"]},
                meta_info={},
            )
            with self.assertRaisesRegex(
                ValueError,
                "one entry per prompt",
            ):
                build_dataset_teacher_prefix(
                    prompts=misaligned_prompts,
                    tokenizer=Tokenizer(),
                    prefix_key="data_source",
                    prefix_length=2,
                    pad_token_id=0,
                )
            suffix_prompts = build_student_suffix_prompts(
                prompts=prompts,
                teacher_prefix_ids=prefix_ids,
                teacher_prefix_mask=prefix_mask,
                pad_token_id=0,
            )
            suffix_output = DataProto(
                batch=TensorDict(
                    {
                        "responses": torch.tensor(
                            [[30, 31, 32, 0], [40, 41, 0, 0]]
                        ),
                        "attention_mask": torch.cat(
                            [
                                suffix_prompts.batch["attention_mask"],
                                torch.tensor(
                                    [[1, 1, 1, 0], [1, 1, 0, 0]]
                                ),
                            ],
                            dim=-1,
                        ),
                        "rollout_log_probs": torch.tensor(
                            [
                                [-0.1, -0.2, -0.3, -1.0],
                                [-0.4, -0.5, -1.0, -1.0],
                            ]
                        ),
                    },
                    batch_size=(2,),
                ),
                non_tensor_batch={},
                meta_info={"timing": {"generate": 1.0}},
            )
            merged = merge_teacher_prefix_and_student_suffix(
                original_prompts=prompts,
                teacher_prefix_ids=prefix_ids,
                teacher_prefix_mask=prefix_mask,
                student_suffix_output=suffix_output,
                max_response_length=4,
                pad_token_id=0,
            )
            metrics = teacher_prefix_rollin_metrics(
                teacher_prefix_mask=merged.batch["teacher_prefix_mask"],
                student_suffix_mask=merged.batch["student_suffix_mask"],
                selected=prefix_mask.bool().any(dim=-1).numpy(),
            )
            filled_rollout_log_probs = (
                fill_teacher_prefix_rollout_log_probs(
                    rollout_log_probs=merged.batch["rollout_log_probs"],
                    old_log_probs=torch.tensor(
                        [
                            [-9.0, -8.0, -7.0, -6.0],
                            [-5.0, -4.0, -3.0, -2.0],
                        ]
                    ),
                    teacher_prefix_mask=merged.batch[
                        "teacher_prefix_mask"
                    ],
                )
            )
            correction_prefix_mask, correction_suffix_mask = (
                teacher_prefix_rollout_correction_masks(
                    response_mask=merged.batch["response_mask"],
                    teacher_prefix_mask=merged.batch[
                        "teacher_prefix_mask"
                    ],
                    student_suffix_mask=merged.batch[
                        "student_suffix_mask"
                    ],
                )
            )
            restored_response_mask = restore_teacher_prefix_response_mask(
                prefix_mask=correction_prefix_mask,
                corrected_suffix_mask=correction_suffix_mask
                * torch.tensor([[1, 0, 1, 1], [1, 0, 1, 1]]),
            )

        torch.testing.assert_close(
            suffix_prompts.batch["input_ids"],
            torch.tensor([[10, 11, 97, 98], [0, 0, 20, 21]]),
        )
        self.assertEqual(
            suffix_prompts.non_tensor_batch[
                "raw_prompt_ids"
            ].tolist(),
            [[10, 11, 97, 98], [20, 21]],
        )
        self.assertEqual(
            prompts.non_tensor_batch["raw_prompt_ids"],
            [[10, 11], [20, 21]],
        )
        torch.testing.assert_close(
            merged.batch["responses"],
            torch.tensor([[97, 98, 30, 31], [40, 41, 0, 0]]),
        )
        torch.testing.assert_close(
            merged.batch["teacher_prefix_mask"],
            torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]]),
        )
        torch.testing.assert_close(
            merged.batch["student_suffix_mask"],
            torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]]),
        )
        torch.testing.assert_close(
            merged.batch["rollout_log_probs"],
            torch.tensor(
                [[0.0, 0.0, -0.1, -0.2], [-0.4, -0.5, -1.0, -1.0]]
            ),
        )
        torch.testing.assert_close(
            filled_rollout_log_probs,
            torch.tensor(
                [[-9.0, -8.0, -0.1, -0.2], [-0.4, -0.5, -1.0, -1.0]]
            ),
        )
        torch.testing.assert_close(
            correction_suffix_mask,
            torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]]),
        )
        torch.testing.assert_close(
            restored_response_mask,
            torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]]),
        )
        self.assertEqual(metrics["teacher_prefix/sample_frac"], 0.5)
        self.assertEqual(metrics["teacher_prefix/mean_len"], 1.0)
        self.assertEqual(metrics["teacher_prefix/max_len"], 2.0)
        self.assertEqual(metrics["teacher_prefix/suffix_mean_len"], 2.0)


if __name__ == "__main__":
    unittest.main()
