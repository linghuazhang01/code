from __future__ import annotations

import sys
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import torch

from mopd_verl.launch import build_command, format_command
from mopd_verl.full_gradient.loss_support import (
    selected_teacher_entropy,
    selected_teacher_topk_support,
)
from mopd_verl.settings import load_config
from mopd_verl.topk_distill import (
    TOPK_LOGPROB_MODE_FULL_VOCAB,
    distill_loss_builder,
    eopd_entropy_threshold,
    eopd_forward_kl_matrix,
    eopd_forward_kl_weight,
    eopd_teacher_student_cross_entropy_matrix,
    eopd_topk_k,
    topk_log_probs_from_logits,
    uses_eopd_loss,
    uses_teacher_topk_support,
)


class EOPDTests(unittest.TestCase):
    def test_truncated_forward_kl_renormalizes_only_teacher(self) -> None:
        teacher_full_log_probs = torch.log_softmax(
            torch.tensor([[[3.0, 1.0, 0.0, -1.0]]]),
            dim=-1,
        )
        student_full_log_probs = torch.log_softmax(
            torch.tensor([[[0.0, 2.0, 1.0, -2.0]]]),
            dim=-1,
        )
        teacher_topk_ids = torch.tensor([[[0, 1]]])
        teacher_topk_log_probs = teacher_full_log_probs.gather(
            -1,
            teacher_topk_ids,
        )
        student_at_teacher_ids = student_full_log_probs.gather(
            -1,
            teacher_topk_ids,
        )

        actual = eopd_forward_kl_matrix(
            student_full_vocab_log_probs=student_at_teacher_ids,
            teacher_topk_log_probs=teacher_topk_log_probs,
        )
        teacher_log_q = torch.log_softmax(teacher_topk_log_probs, dim=-1)
        expected = (
            teacher_log_q.exp()
            * (teacher_log_q - student_at_teacher_ids)
        ).sum(dim=-1)
        incorrectly_student_renormalized = (
            teacher_log_q.exp()
            * (
                teacher_log_q
                - torch.log_softmax(student_at_teacher_ids, dim=-1)
            )
        ).sum(dim=-1)

        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, incorrectly_student_renormalized))

    def test_full_vocab_gather_keeps_student_normalizer(self) -> None:
        logits = torch.tensor([[[0.0, 2.0, 1.0, -2.0]]])
        teacher_ids = torch.tensor([[[1, 2]]])

        _, _, gathered = topk_log_probs_from_logits(
            logits,
            gather_topk_ids=teacher_ids,
            normalize_gathered=True,
            logprob_mode=TOPK_LOGPROB_MODE_FULL_VOCAB,
        )

        expected = torch.log_softmax(logits, dim=-1).gather(-1, teacher_ids)
        torch.testing.assert_close(gathered, expected)
        self.assertLess(float(gathered.exp().sum()), 1.0)

    def test_builder_is_opt_in_and_paper_defaults_are_explicit(self) -> None:
        self.assertEqual(distill_loss_builder({}), "chosen_token_reverse_kl")
        self.assertFalse(uses_eopd_loss({}))
        self.assertFalse(uses_teacher_topk_support({}))
        self.assertFalse(
            uses_teacher_topk_support(
                {
                    "distill_loss_builder": "topk_kl",
                    "topk_distill_support_source": "student",
                }
            )
        )

        config = {"distill_loss_builder": "entropy_aware_opd"}
        self.assertEqual(distill_loss_builder(config), "eopd")
        self.assertTrue(uses_eopd_loss(config))
        self.assertTrue(uses_teacher_topk_support(config))
        self.assertEqual(eopd_entropy_threshold(config), 0.8)
        self.assertEqual(eopd_forward_kl_weight(config), 1.0)
        self.assertEqual(eopd_topk_k(config), 16)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            eopd_topk_k({"eopd_topk_k": 0})

    def test_multi_domain_teacher_entropy_and_support_are_routed_per_row(
        self,
    ) -> None:
        math_entropy = torch.tensor([[1.0], [1.1], [1.2]])
        code_entropy = torch.tensor([[2.0], [2.1], [2.2]])
        science_entropy = torch.tensor([[3.0], [3.1], [3.2]])
        math_ids = torch.tensor([[[10, 11]], [[12, 13]], [[14, 15]]])
        code_ids = torch.tensor([[[20, 21]], [[22, 23]], [[24, 25]]])
        science_ids = torch.tensor([[[30, 31]], [[32, 33]], [[34, 35]]])
        math_log_probs = -math_ids.float()
        code_log_probs = -code_ids.float()
        science_log_probs = -science_ids.float()
        model_inputs = {
            "opd_teacher": ["math", "science", "code"],
            "math_teacher_entropy": math_entropy,
            "code_teacher_entropy": code_entropy,
            "science_teacher_entropy": science_entropy,
            "math_teacher_topk_ids": math_ids,
            "code_teacher_topk_ids": code_ids,
            "science_teacher_topk_ids": science_ids,
            "math_teacher_topk_logprobs": math_log_probs,
            "code_teacher_topk_logprobs": code_log_probs,
            "science_teacher_topk_logprobs": science_log_probs,
        }
        policy_config = {"multi_teacher_distill": True}

        selected_entropy = selected_teacher_entropy(
            model_inputs,
            policy_config,
        )
        selected_ids, selected_log_probs = selected_teacher_topk_support(
            model_inputs,
            policy_config,
        )

        torch.testing.assert_close(
            selected_entropy,
            torch.tensor([[1.0], [3.1], [2.2]]),
        )
        torch.testing.assert_close(
            selected_ids,
            torch.tensor([[[10, 11]], [[32, 33]], [[24, 25]]]),
        )
        torch.testing.assert_close(
            selected_log_probs,
            -selected_ids.float(),
        )

        shared_teacher_inputs = {
            key: value
            for key, value in model_inputs.items()
            if not key.startswith("science_teacher_")
        }
        shared_entropy = selected_teacher_entropy(
            shared_teacher_inputs,
            policy_config,
        )
        shared_ids, _ = selected_teacher_topk_support(
            shared_teacher_inputs,
            policy_config,
        )
        torch.testing.assert_close(
            shared_entropy,
            torch.tensor([[1.0], [1.1], [2.2]]),
        )
        torch.testing.assert_close(
            shared_ids,
            torch.tensor([[[10, 11]], [[12, 13]], [[24, 25]]]),
        )

    def test_baseline_matrix_switches_only_the_objective(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "matrices"
            / "eopd_baseline_matrix.yaml"
        )
        opd = load_config(f"{config_path}::opd")
        eopd = load_config(f"{config_path}::eopd")
        rendered_opd = format_command(build_command(opd))
        rendered_eopd = format_command(build_command(eopd))

        self.assertEqual(opd.actor.distill_loss_builder, "policy_gradient")
        self.assertEqual(eopd.actor.distill_loss_builder, "eopd")
        self.assertEqual(opd.data, eopd.data)
        self.assertEqual(opd.model, eopd.model)
        self.assertEqual(opd.rollout, eopd.rollout)
        self.assertEqual(opd.rollout_correction.rollout_is, "token")
        self.assertIsNone(eopd.rollout_correction.rollout_is)
        self.assertEqual(
            replace(opd.rollout_correction, rollout_is=None),
            eopd.rollout_correction,
        )
        self.assertEqual(opd.worker_placement, eopd.worker_placement)
        self.assertEqual(
            opd.audit.output_dir,
            "audit/qwen4b-opd-baseline-domain-metrics",
        )
        self.assertEqual(
            eopd.audit.output_dir,
            "audit/qwen4b-eopd-tau0p8-alpha1-k16-domain-metrics",
        )
        self.assertEqual(
            opd.audit.loss_variance_signal,
            "policy_gradient_distillation_signal",
        )
        self.assertEqual(
            eopd.audit.loss_variance_signal,
            "policy_gradient+entropy_gated_topk_forward_kl",
        )
        self.assertEqual(
            replace(
                opd.audit,
                output_dir=eopd.audit.output_dir,
                loss_variance_signal=eopd.audit.loss_variance_signal,
            ),
            eopd.audit,
        )
        self.assertEqual(opd.domain_budgeting, eopd.domain_budgeting)
        self.assertEqual(opd.ray_kwargs, eopd.ray_kwargs)
        self.assertEqual(
            replace(opd.actor, distill_loss_builder="eopd"),
            eopd.actor,
        )
        self.assertEqual(
            replace(
                opd.trainer,
                experiment_name=eopd.trainer.experiment_name,
                default_local_dir=eopd.trainer.default_local_dir,
            ),
            eopd.trainer,
        )
        self.assertNotIn("policy_loss.eopd_entropy_threshold", rendered_opd)
        self.assertIn(
            "actor_rollout_ref.actor.policy_loss.eopd_entropy_threshold=0.8",
            rendered_eopd,
        )
        self.assertIn(
            "actor_rollout_ref.actor.policy_loss.eopd_forward_kl_weight=1.0",
            rendered_eopd,
        )
        self.assertIn(
            "actor_rollout_ref.actor.policy_loss.eopd_topk_k=16",
            rendered_eopd,
        )

    @contextmanager
    def _stubbed_verl(self) -> Iterator[None]:
        verl_module = ModuleType("verl")
        verl_module.__path__ = []
        verl_module.DataProto = object
        utils_module = ModuleType("verl.utils")
        utils_module.__path__ = []
        device_module = ModuleType("verl.utils.device")
        device_module.get_device_id = lambda: torch.device("cpu")
        trainer_module = ModuleType("verl.trainer")
        trainer_module.__path__ = []
        ppo_module = ModuleType("verl.trainer.ppo")
        ppo_module.__path__ = []
        core_algos_module = ModuleType("verl.trainer.ppo.core_algos")

        def aggregate(
            loss_mat: torch.Tensor,
            loss_mask: torch.Tensor,
            loss_agg_mode: str,
        ) -> torch.Tensor:
            del loss_agg_mode
            return (loss_mat * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

        def policy_loss_fn(
            *,
            old_log_prob: torch.Tensor,
            log_prob: torch.Tensor,
            advantages: torch.Tensor,
            response_mask: torch.Tensor,
            loss_agg_mode: str,
            config: Any,
            rollout_is_weights: torch.Tensor | None,
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            ratio = torch.exp(log_prob - old_log_prob)
            clip_ratio_low = float(config.get("clip_ratio_low", 0.2))
            clip_ratio_high = float(config.get("clip_ratio_high", 0.2))
            unclipped = -advantages * ratio
            clipped = -advantages * torch.clamp(
                ratio,
                1.0 - clip_ratio_low,
                1.0 + clip_ratio_high,
            )
            loss_mat = torch.maximum(unclipped, clipped)
            if rollout_is_weights is not None:
                loss_mat = loss_mat * rollout_is_weights
            return aggregate(loss_mat, response_mask, loss_agg_mode), {}

        core_algos_module.agg_loss = aggregate
        core_algos_module.get_policy_loss_fn = lambda _mode: policy_loss_fn
        core_algos_module.kl_penalty = lambda **_kwargs: None
        isolated_names = ("mopd_verl.full_gradient.actor_loss",)
        saved_modules = {
            name: sys.modules.pop(name)
            for name in isolated_names
            if name in sys.modules
        }
        try:
            with patch.dict(
                sys.modules,
                {
                    "verl": verl_module,
                    "verl.trainer": trainer_module,
                    "verl.trainer.ppo": ppo_module,
                    "verl.trainer.ppo.core_algos": core_algos_module,
                    "verl.utils": utils_module,
                    "verl.utils.device": device_module,
                },
            ):
                yield
        finally:
            for name in isolated_names:
                sys.modules.pop(name, None)
            sys.modules.update(saved_modules)

    def test_eopd_adds_gated_fkl_to_opd_with_full_token_denominator(self) -> None:
        with self._stubbed_verl():
            from mopd_verl.full_gradient.actor_loss import (
                build_actor_micro_batch_loss,
            )

            chosen_log_probs = torch.full(
                (1, 3),
                -1.0,
                requires_grad=True,
            )
            student_topk_log_probs = torch.tensor(
                [[[-0.7, -2.0], [-1.0, -1.8], [-0.4, -2.4]]],
                requires_grad=True,
            )
            teacher_topk_log_probs = torch.tensor(
                [[[-0.2, -1.8], [-0.6, -0.9], [-0.1, -2.2]]]
            )
            teacher_entropy = torch.tensor([[0.8, 0.79, 1.2]])
            teacher_chosen_log_probs = torch.full((1, 3), -0.5)
            teacher_topk_ids = torch.tensor([[[3, 7], [2, 9], [1, 4]]])

            class MicroBatch:
                batch = {
                    "response_mask": torch.ones(1, 3),
                    "old_log_probs": torch.full((1, 3), -2.0),
                    "math_teacher_log_prob": teacher_chosen_log_probs,
                    "math_teacher_topk_ids": teacher_topk_ids,
                    "math_teacher_topk_logprobs": teacher_topk_log_probs,
                    "math_teacher_entropy": teacher_entropy,
                }
                non_tensor_batch: dict[str, object] = {}
                meta_info = {"temperature": 1.0}

                def to(self, _device: object) -> "MicroBatch":
                    return self

            class Actor:
                config = {
                    "entropy_coeff": 0.0,
                    "kl_loss_coef": 0.0,
                    "loss_agg_mode": "token-mean",
                    "use_kl_loss": False,
                    "policy_loss": {
                        "distill_loss_builder": "eopd",
                        "distill_mode": "chosen_token_policy_gradient",
                        "eopd_entropy_threshold": 0.8,
                        "eopd_forward_kl_weight": 1.0,
                        "eopd_topk_k": 2,
                    },
                }
                forward_kwargs: dict[str, object] = {}

                def _forward_micro_batch(
                    self,
                    _model_inputs: dict[str, object],
                    **kwargs: object,
                ) -> tuple[object, ...]:
                    self.forward_kwargs = kwargs
                    return (
                        None,
                        chosen_log_probs,
                        None,
                        None,
                        student_topk_log_probs,
                    )

            actor = Actor()
            result = build_actor_micro_batch_loss(
                actor,
                MicroBatch(),
                loss_scale_factor=1.0,
                on_policy=False,
                include_metrics=True,
                return_teacher_student_cross_entropy=True,
                return_configured_token_loss=True,
            )
            MicroBatch.batch["rollout_is_weights"] = torch.ones(1, 3)
            with self.assertRaisesRegex(ValueError, "rollout IS"):
                build_actor_micro_batch_loss(
                    actor,
                    MicroBatch(),
                    loss_scale_factor=1.0,
                    on_policy=False,
                )
            MicroBatch.batch.pop("rollout_is_weights")

        token_fkl = eopd_forward_kl_matrix(
            student_full_vocab_log_probs=student_topk_log_probs,
            teacher_topk_log_probs=teacher_topk_log_probs,
        )
        gate = torch.tensor([[0.0, 0.0, 1.0]])
        expected_pg = torch.tensor(-1.8)
        expected_fkl = (token_fkl * gate).sum() / 3.0
        expected_ce = eopd_teacher_student_cross_entropy_matrix(
            student_full_vocab_log_probs=student_topk_log_probs.detach(),
            teacher_topk_log_probs=teacher_topk_log_probs,
        )

        torch.testing.assert_close(result.loss, expected_pg + expected_fkl)
        torch.testing.assert_close(
            result.configured_token_loss,
            torch.full((1, 3), -1.5) + token_fkl.detach() * gate,
        )
        torch.testing.assert_close(
            result.teacher_student_cross_entropy,
            expected_ce,
        )
        self.assertAlmostEqual(
            result.metrics["actor/eopd_high_entropy_ratio"],
            1 / 3,
        )
        torch.testing.assert_close(
            actor.forward_kwargs["gather_topk_ids"],
            teacher_topk_ids,
        )
        self.assertTrue(actor.forward_kwargs["normalize_gathered_topk"])
        self.assertEqual(
            actor.forward_kwargs["topk_logprob_mode"],
            TOPK_LOGPROB_MODE_FULL_VOCAB,
        )


if __name__ == "__main__":
    unittest.main()
