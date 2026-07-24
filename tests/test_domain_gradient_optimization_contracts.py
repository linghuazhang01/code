from __future__ import annotations

import sys
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch


class DomainGradientOptimizationContractTests(unittest.TestCase):
    def _torch(self) -> Any:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is unavailable: {exc}")
        return torch

    def test_loss_mass_selection_is_global_and_deterministic(self) -> None:
        from mopd_verl.domain_gradient.token_selection import (
            RankedToken,
            select_tail_loss_mass,
            select_top_k,
            select_top_loss_mass,
        )

        tokens = (
            RankedToken(owner_rank=1, owner_index=0, loss_abs=3.0),
            RankedToken(owner_rank=0, owner_index=0, loss_abs=5.0),
            RankedToken(owner_rank=0, owner_index=1, loss_abs=1.0),
            RankedToken(owner_rank=1, owner_index=1, loss_abs=1.0),
        )

        self.assertEqual(
            select_top_k(tokens, 2),
            (tokens[1], tokens[0]),
        )
        self.assertEqual(
            select_top_loss_mass(tokens, 0.8),
            (tokens[1], tokens[0]),
        )
        self.assertEqual(
            select_tail_loss_mass(tokens, 0.2, minimum_tokens=1),
            (tokens[2], tokens[3]),
        )

    @contextmanager
    def _stubbed_verl(self, torch: Any) -> Iterator[None]:
        """Load the small audit modules without the full Ray/verl runtime."""

        verl_module = ModuleType("verl")
        verl_module.__path__ = []
        verl_module.DataProto = object
        utils_module = ModuleType("verl.utils")
        utils_module.__path__ = []
        device_module = ModuleType("verl.utils.device")
        device_module.get_device_id = lambda: torch.device("cpu")
        device_module.get_torch_device = lambda: torch.cpu
        trainer_module = ModuleType("verl.trainer")
        trainer_module.__path__ = []
        ppo_module = ModuleType("verl.trainer.ppo")
        ppo_module.__path__ = []
        core_algos_module = ModuleType("verl.trainer.ppo.core_algos")
        core_algos_module.agg_loss = (
            lambda loss_mat, loss_mask, loss_agg_mode: (
                loss_mat * loss_mask
            ).sum()
            / loss_mask.sum().clamp(min=1.0)
        )

        def policy_loss_fn(
            *,
            old_log_prob: Any,
            log_prob: Any,
            advantages: Any,
            response_mask: Any,
            loss_agg_mode: str,
            config: Any,
            rollout_is_weights: Any,
        ) -> tuple[Any, dict[str, Any]]:
            del config
            ratio = torch.exp(log_prob - old_log_prob)
            loss_mat = -advantages * ratio
            if rollout_is_weights is not None:
                loss_mat = loss_mat * rollout_is_weights
            return (
                core_algos_module.agg_loss(
                    loss_mat,
                    response_mask,
                    loss_agg_mode,
                ),
                {},
            )

        core_algos_module.get_policy_loss_fn = lambda _mode: policy_loss_fn
        core_algos_module.kl_penalty = lambda **_kwargs: None
        dtensor_module = ModuleType("torch.distributed.tensor")

        class DummyDTensor:
            pass

        dtensor_module.DTensor = DummyDTensor
        isolated_names = (
            "mopd_verl.domain_gradient.audit",
            "mopd_verl.domain_gradient.geometry",
            "mopd_verl.domain_gradient.state",
            "mopd_verl.full_gradient.actor_loss",
        )
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
                    "torch.distributed.tensor": dtensor_module,
                },
            ):
                yield
        finally:
            for name in isolated_names:
                sys.modules.pop(name, None)
            sys.modules.update(saved_modules)

    def test_training_forward_reuses_detached_topk_cross_entropy(self) -> None:
        torch = self._torch()
        with self._stubbed_verl(torch):
            from mopd_verl.full_gradient.actor_loss import (
                build_actor_micro_batch_loss,
            )
            from mopd_verl.topk_distill import (
                topk_distill_loss_matrix,
                topk_teacher_student_cross_entropy_matrix,
            )

            student_logits = torch.tensor(
                [
                    [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]],
                    [[1.5, 0.5, -0.5], [-0.5, 0.5, 1.5]],
                    [[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]],
                ],
                requires_grad=True,
            )
            student_log_probs = torch.log_softmax(student_logits, dim=-1)
            teacher_log_probs = {
                "math": torch.log_softmax(torch.full_like(student_logits, 0.0), dim=-1),
                "code": torch.log_softmax(torch.full_like(student_logits, 1.0), dim=-1),
                "science": torch.log_softmax(torch.full_like(student_logits, 2.0), dim=-1),
            }
            for index, domain in enumerate(("math", "code", "science")):
                teacher_log_probs[domain][index] = torch.log_softmax(
                    torch.tensor(
                        [[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]],
                        dtype=torch.float32,
                    )
                    + index,
                    dim=-1,
                )

            class MicroBatch:
                def __init__(self) -> None:
                    self.batch = {
                        "response_mask": torch.ones(3, 2),
                        "math_teacher_topk_ids": torch.zeros(3, 2, 3, dtype=torch.long),
                        "math_teacher_topk_logprobs": teacher_log_probs["math"],
                        "code_teacher_topk_ids": torch.zeros(3, 2, 3, dtype=torch.long),
                        "code_teacher_topk_logprobs": teacher_log_probs["code"],
                        "science_teacher_topk_ids": torch.zeros(3, 2, 3, dtype=torch.long),
                        "science_teacher_topk_logprobs": teacher_log_probs["science"],
                    }
                    self.non_tensor_batch = {
                        "opd_teacher": ["math", "code", "science"]
                    }
                    self.meta_info = {"temperature": 1.0}

                def to(self, _device: object) -> "MicroBatch":
                    return self

            class Actor:
                config = {
                    "entropy_coeff": 0.0,
                    "kl_loss_coef": 0.0,
                    "loss_agg_mode": "token-mean",
                    "policy_loss": {
                        "distill_loss_builder": "topk_kl",
                        "distill_mode": "topk_renormalized_reverse_kl",
                        "multi_teacher_distill": True,
                        "topk_distill_support_source": "teacher",
                        "topk_distill_temperature": 1.0,
                    },
                    "use_kl_loss": False,
                }

                def _forward_micro_batch(
                    self,
                    _model_inputs: dict[str, object],
                    **_kwargs: object,
                ) -> tuple[object, ...]:
                    return (
                        None,
                        torch.zeros(3, 2),
                        None,
                        None,
                        student_log_probs,
                    )

            result = build_actor_micro_batch_loss(
                Actor(),
                MicroBatch(),
                loss_scale_factor=1.0,
                on_policy=True,
                return_teacher_student_cross_entropy=True,
                return_configured_token_loss=True,
            )
            baseline = build_actor_micro_batch_loss(
                Actor(),
                MicroBatch(),
                loss_scale_factor=1.0,
                on_policy=True,
            )
            prefix_micro_batch = MicroBatch()
            prefix_micro_batch.batch["teacher_prefix_mask"] = torch.tensor(
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
            )

            class PrefixActor(Actor):
                config = {
                    **Actor.config,
                    "policy_loss": {
                        **Actor.config["policy_loss"],
                        "teacher_prefix_enabled": True,
                        "teacher_prefix_loss_region": "suffix_only",
                    },
                }

            prefix_result = build_actor_micro_batch_loss(
                PrefixActor(),
                prefix_micro_batch,
                loss_scale_factor=1.0,
                on_policy=True,
                return_configured_token_loss=True,
            )
            selected_teacher = torch.stack(
                [teacher_log_probs[domain][index] for index, domain in enumerate(("math", "code", "science"))]
            )
            expected = topk_teacher_student_cross_entropy_matrix(
                student_topk_log_probs=student_log_probs.detach(),
                teacher_topk_log_probs=selected_teacher,
                include_tail=False,
                temperature=1.0,
            )
            expected_token_loss = topk_distill_loss_matrix(
                student_topk_log_probs=student_log_probs.detach(),
                teacher_topk_log_probs=selected_teacher,
                mode="topk_renormalized_reverse_kl",
                include_tail=False,
                temperature=1.0,
            )
            weighted_micro_batch = MicroBatch()
            rollout_is_weights = torch.tensor(
                [[2.0, 0.5], [1.5, 0.25], [0.75, 1.25]]
            )
            weighted_micro_batch.batch["rollout_is_weights"] = (
                rollout_is_weights
            )
            weighted_result = build_actor_micro_batch_loss(
                Actor(),
                weighted_micro_batch,
                loss_scale_factor=1.0,
                on_policy=True,
                return_configured_token_loss=True,
            )

            self.assertIsNotNone(result.teacher_student_cross_entropy)
            torch.testing.assert_close(
                result.teacher_student_cross_entropy,
                expected,
            )
            self.assertFalse(result.teacher_student_cross_entropy.requires_grad)
            self.assertIsNone(result.teacher_student_cross_entropy.grad_fn)
            self.assertEqual(result.teacher_student_cross_entropy.shape, (3, 2))
            self.assertIsNotNone(result.configured_token_loss)
            torch.testing.assert_close(
                result.configured_token_loss,
                expected_token_loss,
            )
            torch.testing.assert_close(
                weighted_result.configured_token_loss,
                expected_token_loss * rollout_is_weights,
            )
            torch.testing.assert_close(
                weighted_result.loss,
                (expected_token_loss * rollout_is_weights).mean(),
            )
            self.assertFalse(result.configured_token_loss.requires_grad)
            torch.testing.assert_close(
                result.configured_token_loss_mask,
                torch.ones(3, 2),
            )
            self.assertIsNone(baseline.teacher_student_cross_entropy)
            self.assertIsNone(baseline.configured_token_loss)
            torch.testing.assert_close(
                prefix_result.configured_token_loss_mask,
                torch.tensor(
                    [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
                ),
            )
            torch.testing.assert_close(result.loss, baseline.loss)
            result.loss.backward()
            self.assertIsNotNone(student_logits.grad)

    def test_empty_prefix_respects_configured_prefix_only_region(
        self,
    ) -> None:
        torch = self._torch()
        from mopd_verl.topk_distill import teacher_prefix_masks

        prefix_loss_mask, suffix_loss_mask, active = (
            teacher_prefix_masks(
                {
                    "teacher_prefix_mask": torch.zeros(2, 3),
                    "student_suffix_mask": torch.ones(2, 3),
                },
                torch.ones(2, 3),
                {
                    "teacher_prefix_enabled": True,
                    "teacher_prefix_loss_region": "prefix_only",
                },
            )
        )

        self.assertTrue(active)
        self.assertEqual(prefix_loss_mask.sum().item(), 0.0)
        self.assertEqual(suffix_loss_mask.sum().item(), 0.0)

    def test_micro_batch_contributions_sum_but_observations_remain_rows(self) -> None:
        self._torch()
        from mopd_verl.full_gradient.loss_support import (
            aggregate_actor_micro_batch_metrics,
        )

        contributions, observations = aggregate_actor_micro_batch_metrics(
            (
                {
                    "actor/pg_loss": 0.25,
                    "actor/topk_distill_loss": 0.10,
                    "actor/teacher_prefix_token_count": 2.0,
                    "actor/clipfrac": 0.10,
                    "actor/kl_coef": 0.30,
                },
                {
                    "actor/pg_loss": 0.75,
                    "actor/topk_distill_loss": 0.20,
                    "actor/teacher_prefix_token_count": 3.0,
                    "actor/clipfrac": 0.30,
                    "actor/kl_coef": 0.30,
                },
            )
        )

        self.assertEqual(
            set(contributions),
            {
                "actor/pg_loss",
                "actor/topk_distill_loss",
                "actor/teacher_prefix_token_count",
            },
        )
        self.assertAlmostEqual(contributions["actor/pg_loss"], 1.0)
        self.assertAlmostEqual(contributions["actor/topk_distill_loss"], 0.30)
        self.assertAlmostEqual(
            contributions["actor/teacher_prefix_token_count"],
            5.0,
        )
        self.assertEqual(
            observations,
            (
                {"actor/clipfrac": 0.10, "actor/kl_coef": 0.30},
                {"actor/clipfrac": 0.30, "actor/kl_coef": 0.30},
            ),
        )

    def test_chosen_token_config_reports_reverse_kl_distillation_loss(self) -> None:
        torch = self._torch()
        with self._stubbed_verl(torch):
            from mopd_verl.full_gradient.actor_loss import (
                build_actor_micro_batch_loss,
            )

            student_log_prob = torch.tensor(
                [[-1.0, -2.0]],
                requires_grad=True,
            )

            class MicroBatch:
                batch = {
                    "response_mask": torch.ones(1, 2),
                    "math_teacher_log_prob": torch.tensor([[-1.5, -1.0]]),
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
                    "policy_loss": {
                        "distill_loss_builder": (
                            "chosen_token_reverse_kl"
                        ),
                        "only_reverse_kl_advantages": True,
                        "lambda_vals": 1.0,
                    },
                    "use_kl_loss": False,
                }

                def _forward_micro_batch(
                    self,
                    _model_inputs: dict[str, object],
                    **_kwargs: object,
                ) -> tuple[object, ...]:
                    return None, student_log_prob

            result = build_actor_micro_batch_loss(
                Actor(),
                MicroBatch(),
                loss_scale_factor=1.0,
                on_policy=True,
                return_configured_token_loss=True,
            )

        torch.testing.assert_close(
            result.configured_token_loss,
            torch.tensor([[0.5, -1.0]]),
        )
        torch.testing.assert_close(
            result.configured_token_loss_mask,
            torch.ones(1, 2),
        )

    def test_bf16_cancellation_metrics_expose_unstable_closure(self) -> None:
        torch = self._torch()
        with self._stubbed_verl(torch):
            from mopd_verl.domain_gradient.geometry import (
                domain_metrics_from_gram,
            )

            actor = SimpleNamespace(config={"fsdp_config": {"fsdp_size": -1}})
            metrics = domain_metrics_from_gram(
                actor,
                ("math", "code"),
                total_sq=1.0,
                domain_sq={"math": 1_000_001.0, "code": 1_000_000.25},
                domain_total_dot={"math": 1.0, "code": -0.5},
                pair_dot={("math", "code"): -1_000_000.5},
                closure_threshold=0.02,
                all_vectors_fp32=False,
                storage_dtype="bfloat16",
            )

        prefix = (
            "global/pre_reweight_full_grad_closure/"
            "domain_sum_vs_pre_reweight_audit_total"
        )
        self.assertIn(
            "global/pre_reweight_full_grad/total_grad_norm",
            metrics,
        )
        self.assertIn(
            "math/pre_reweight_full_grad/grad_norm",
            metrics,
        )
        self.assertIn(
            "global/pre_reweight_full_grad_alignment/"
            "math_vs_pre_reweight_total/"
            "pre_reweight_full_grad_cosine_domain_total",
            metrics,
        )
        self.assertIn(
            "global/pre_reweight_full_grad_conflict/"
            "math_vs_code/pre_reweight_full_grad_cosine_train_i_k",
            metrics,
        )
        self.assertFalse(
            any("/full_grad/" in key for key in metrics)
        )
        self.assertAlmostEqual(metrics[f"{prefix}/diff_norm"], 0.5)
        self.assertGreater(metrics[f"{prefix}/domain_vector_norm_sum"], 1_999.0)
        self.assertGreater(metrics[f"{prefix}/domain_norm_sum_over_total_norm"], 1_999.0)
        self.assertLess(metrics[f"{prefix}/diff_norm_over_domain_vector_norm_sum"], 1e-3)
        self.assertGreater(metrics[f"{prefix}/estimated_storage_roundoff_rel_l2"], 0.02)
        self.assertEqual(metrics[f"{prefix}/storage_roundoff_may_exceed_threshold"], 1.0)
        self.assertFalse(
            any("domain_sum_vs_training" in key for key in metrics)
        )

    def test_bf16_roundoff_bound_includes_independent_total_storage(self) -> None:
        torch = self._torch()
        with self._stubbed_verl(torch):
            from mopd_verl.domain_gradient.geometry import (
                domain_metrics_from_gram,
            )

            actor = SimpleNamespace(config={"fsdp_config": {"fsdp_size": -1}})
            metrics = domain_metrics_from_gram(
                actor,
                ("math", "code"),
                total_sq=1.0,
                domain_sq={"math": 4.0, "code": 6.25},
                domain_total_dot={"math": 1.0, "code": 0.0},
                pair_dot={("math", "code"): -4.5},
                closure_threshold=0.02,
                all_vectors_fp32=False,
                storage_dtype="bfloat16",
            )

        prefix = (
            "global/pre_reweight_full_grad_closure/"
            "domain_sum_vs_pre_reweight_audit_total"
        )
        estimate = metrics[f"{prefix}/estimated_storage_roundoff_rel_l2"]
        self.assertAlmostEqual(estimate, 0.00390625 * (4.5 + 1.0))
        self.assertEqual(
            metrics[f"{prefix}/storage_roundoff_may_exceed_threshold"],
            1.0,
        )

    def test_soft_response_mask_domain_gradients_sum_to_total(self) -> None:
        torch = self._torch()
        from mopd_verl.full_gradient.loss_support import gate_tensor_gradient

        with self._stubbed_verl(torch):
            from mopd_verl.domain_gradient.audit import DomainGradientAudit

            response_mask = torch.tensor(
                [[1.0, 0.50, 0.0], [0.25, 0.75, 0.10]],
                dtype=torch.float32,
            )
            micro_batch = SimpleNamespace(
                batch={"response_mask": response_mask},
                non_tensor_batch={"domain": ["math", "code"]},
            )
            math_gate = DomainGradientAudit._domain_gradient_mask(
                micro_batch,
                "math",
            )
            code_gate = DomainGradientAudit._domain_gradient_mask(
                micro_batch,
                "code",
            )

        self.assertTrue(
            torch.equal(
                math_gate,
                torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                code_gate,
                torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            )
        )

        coefficients = torch.tensor(
            [[1.0, -2.0, 3.0], [0.5, 1.5, -1.0]],
            dtype=torch.float32,
        )
        parameter = torch.nn.Parameter(torch.tensor(0.7))

        def gradient(gate: Any | None) -> Any:
            parameter.grad = None
            values = coefficients * parameter
            if gate is not None:
                values = gate_tensor_gradient(values, gate)
            loss = (values.square() * response_mask).sum() / response_mask.sum()
            loss.backward()
            return parameter.grad.detach().clone()

        total = gradient(None)
        math = gradient(math_gate)
        code = gradient(code_gate)
        torch.testing.assert_close(math + code, total, rtol=1e-6, atol=1e-7)

    def test_actor_group_memory_sum_keeps_per_rank_and_global_meanings(self) -> None:
        torch = self._torch()
        with self._stubbed_verl(torch):
            from mopd_verl.domain_gradient import geometry

            actor = SimpleNamespace(config={"fsdp_config": {"fsdp_size": 1}})
            per_rank_bytes = 128.0

            with (
                patch.object(geometry, "get_device_id", return_value=torch.device("cpu")),
                patch.object(geometry.torch.distributed, "is_available", return_value=True),
                patch.object(geometry.torch.distributed, "is_initialized", return_value=True),
                patch.object(
                    geometry.torch.distributed,
                    "all_reduce",
                    side_effect=lambda tensor, op: tensor.mul_(3.0),
                ),
            ):
                actor_group_total = geometry.actor_group_sum(actor, per_rank_bytes)

        self.assertEqual(per_rank_bytes, 128.0)
        self.assertEqual(actor_group_total, 384.0)

    def test_loss_ranked_token_masks_are_selected_globally_per_domain(self) -> None:
        import importlib

        torch = self._torch()
        with self._stubbed_verl(torch):
            audit_module = importlib.import_module(
                "mopd_verl.domain_gradient.audit"
            )
            from mopd_verl.domain_gradient.audit import DomainGradientAudit

            audit = DomainGradientAudit(
                SimpleNamespace(),
                {
                    "enabled": True,
                    "domain_gradient_enabled": True,
                    "domains": ["math", "code"],
                    "token_gradient_enabled": True,
                    "token_gradient_tail_enabled": True,
                    "token_gradient_tail_fraction": 0.2,
                    "token_gradient_tail_min_tokens": 1,
                    "token_gradient_top_p_enabled": True,
                    "token_gradient_top_k": 2,
                    "token_gradient_top_p": 0.8,
                    "token_gradient_loss_abs_selection_enabled": True,
                    "token_gradient_log_tokens_jsonl_enabled": False,
                },
            )
            micro_batch = SimpleNamespace(
                batch={
                    "response_mask": torch.tensor(
                        [
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0],
                        ]
                    )
                },
                non_tensor_batch={"domain": ["math", "code"]},
            )
            result = SimpleNamespace(
                configured_token_loss=torch.tensor(
                    [
                        [8.0, 1.0, 1.0],
                        [4.0, 3.0, 3.0],
                    ]
                ),
                configured_token_loss_mask=torch.ones(2, 3),
            )

            with patch.object(
                audit_module,
                "build_actor_micro_batch_loss",
                return_value=result,
            ):
                selections = audit._loss_ranked_token_selections(
                    [micro_batch],
                    [1.0],
                    on_policy=True,
                    temperature=1.0,
                )
            metrics = audit._token_selection_metrics(selections)

        torch.testing.assert_close(
            selections["math"]["tail"].masks[0],
            torch.tensor(
                [
                    [0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )
        torch.testing.assert_close(
            selections["math"]["top_k"].masks[0],
            torch.tensor(
                [
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )
        torch.testing.assert_close(
            selections["math"]["top_p"].masks[0],
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )
        torch.testing.assert_close(
            selections["code"]["tail"].masks[0],
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        )
        self.assertEqual(metrics["math/token_grad/domain_token_count"], 3.0)
        self.assertEqual(metrics["math/token_grad/tail_loss_abs_mass"], 2.0)
        self.assertEqual(
            metrics["math/token_grad/tail_loss_abs_mass_frac"],
            0.2,
        )
        self.assertEqual(metrics["math/token_grad/top_p_loss_abs_mass"], 8.0)
        self.assertEqual(
            metrics["math/token_grad/top_p_loss_abs_mass_frac"],
            0.8,
        )

    def test_null_top_k_skips_top_k_replay_selection(self) -> None:
        import importlib

        torch = self._torch()
        with self._stubbed_verl(torch):
            audit_module = importlib.import_module(
                "mopd_verl.domain_gradient.audit"
            )
            from mopd_verl.domain_gradient.audit import DomainGradientAudit

            audit = DomainGradientAudit(
                SimpleNamespace(),
                {
                    "enabled": True,
                    "domain_gradient_enabled": True,
                    "domains": ["math"],
                    "token_gradient_enabled": True,
                    "token_gradient_tail_enabled": False,
                    "token_gradient_top_p_enabled": True,
                    "token_gradient_top_k": None,
                    "token_gradient_top_p": 1.0,
                    "token_gradient_loss_abs_selection_enabled": True,
                    "token_gradient_log_tokens_jsonl_enabled": False,
                },
            )
            micro_batch = SimpleNamespace(
                batch={"response_mask": torch.ones(1, 2)},
                non_tensor_batch={"domain": ["math"]},
            )
            result = SimpleNamespace(
                configured_token_loss=torch.tensor([[2.0, 1.0]]),
                configured_token_loss_mask=torch.ones(1, 2),
            )
            with patch.object(
                audit_module,
                "build_actor_micro_batch_loss",
                return_value=result,
            ):
                selections = audit._loss_ranked_token_selections(
                    [micro_batch],
                    [1.0],
                    on_policy=True,
                    temperature=1.0,
                )

        self.assertEqual(set(selections["math"]), {"top_p"})

    def test_tail_gradient_share_uses_projection_on_domain_gradient(self) -> None:
        torch = self._torch()
        with self._stubbed_verl(torch):
            from mopd_verl.domain_gradient.geometry import (
                tail_gradient_metrics_from_gram,
            )

            metrics = tail_gradient_metrics_from_gram(
                domain_sq=25.0,
                tail_sq=9.0,
                tail_domain_dot=12.0,
            )

        self.assertEqual(metrics["tail_grad_norm"], 3.0)
        self.assertEqual(metrics["tail_grad_norm_over_domain_norm"], 0.6)
        self.assertEqual(metrics["tail_grad_cos_to_domain"], 0.8)
        self.assertEqual(
            metrics["tail_grad_signed_projection_share"],
            12.0 / 25.0,
        )

    def test_top_p1_gradient_metrics_are_full_domain_closure(self) -> None:
        torch = self._torch()
        with self._stubbed_verl(torch):
            from mopd_verl.domain_gradient.geometry import (
                top_p1_gradient_metrics_from_domain_sq,
            )

            metrics = top_p1_gradient_metrics_from_domain_sq(25.0)

        self.assertEqual(metrics["top_p1_grad_norm"], 5.0)
        self.assertEqual(metrics["top_p1_grad_norm_over_domain_norm"], 1.0)
        self.assertEqual(metrics["top_p1_grad_cos_to_domain"], 1.0)
        self.assertEqual(
            metrics["top_p1_grad_signed_projection_share"],
            1.0,
        )

    def test_inverse_gradient_norm_controller_uses_bounded_weight_ema(
        self,
    ) -> None:
        from mopd_verl.domain_gradient.weighting import (
            initial_domain_weight_state,
            update_domain_weight_state,
        )

        state = initial_domain_weight_state(("math", "code", "science"))
        updated = update_domain_weight_state(
            state,
            {"math": 1.0, "code": 2.0, "science": 8.0},
            ema_beta=0.9,
            weight_ema_beta=0.9,
            alpha=0.5,
            minimum=1.0 / 3.0,
            maximum=3.0,
        )
        weights = updated.weight_map()
        targets = updated.target_weight_map()

        self.assertGreater(weights["math"], weights["code"])
        self.assertGreater(weights["code"], weights["science"])
        self.assertAlmostEqual(sum(weights.values()) / 3.0, 1.0)
        self.assertTrue(
            all(1.0 / 3.0 <= value <= 3.0 for value in weights.values())
        )
        for domain in state.domains:
            self.assertAlmostEqual(
                weights[domain],
                0.9 + 0.1 * targets[domain],
            )
        self.assertEqual(updated.update_count, 1)

    def test_weight_ema_beta_zero_applies_bounded_target_immediately(
        self,
    ) -> None:
        from mopd_verl.domain_gradient.weighting import (
            initial_domain_weight_state,
            update_domain_weight_state,
        )

        updated = update_domain_weight_state(
            initial_domain_weight_state(("math", "code", "science")),
            {"math": 1.0, "code": 2.0, "science": 8.0},
            ema_beta=0.9,
            weight_ema_beta=0.0,
            alpha=0.5,
            minimum=1.0 / 3.0,
            maximum=3.0,
        )

        self.assertEqual(updated.weights, updated.target_weights)

    def test_legacy_dynamic_weight_state_restores_target_from_weights(
        self,
    ) -> None:
        from mopd_verl.domain_gradient.weighting import DomainWeightState

        restored = DomainWeightState.from_mapping(
            {
                "domains": ("math", "code"),
                "ema_norms": (1.0, 3.0),
                "weights": (1.4, 0.6),
                "update_count": 2,
                "last_updated_step": 12,
            }
        )

        self.assertEqual(restored.target_weights, restored.weights)

    def test_dynamic_domain_weights_gate_each_training_row(self) -> None:
        torch = self._torch()
        with self._stubbed_verl(torch):
            from mopd_verl.domain_gradient.audit import DomainGradientAudit
            from mopd_verl.domain_gradient.weighting import DomainWeightState

            actor = SimpleNamespace(
                _mopd_domain_weight_state=DomainWeightState(
                    domains=("math", "code"),
                    ema_norms=(1.0, 2.0),
                    weights=(1.5, 0.5),
                    update_count=1,
                )
            )
            audit = DomainGradientAudit(
                actor,
                {
                    "enabled": True,
                    "domain_gradient_enabled": False,
                    "domains": ["math", "code"],
                    "dynamic_domain_loss_weighting_enabled": True,
                },
            )
            micro_batch = SimpleNamespace(
                batch={"response_mask": torch.ones(3, 2)},
                non_tensor_batch={"domain": ["math", "code", "math"]},
            )

            mask = audit.training_gradient_mask(micro_batch)
            metrics = audit._dynamic_weight_metrics()

        torch.testing.assert_close(
            mask,
            torch.tensor([[1.5, 1.5], [0.5, 0.5], [1.5, 1.5]]),
        )
        self.assertEqual(
            metrics["math/dynamic_weight/applied_gradient_weight"],
            1.5,
        )
        self.assertEqual(
            metrics["code/dynamic_weight/applied_gradient_weight"],
            0.5,
        )

    def test_dynamic_parity_reference_replays_current_training_masks(self) -> None:
        torch = self._torch()
        with self._stubbed_verl(torch):
            from mopd_verl.domain_gradient.audit import DomainGradientAudit
            from mopd_verl.domain_gradient.weighting import DomainWeightState

            actor = SimpleNamespace(
                _mopd_domain_weight_state=DomainWeightState(
                    domains=("math", "code"),
                    ema_norms=(1.0, 2.0),
                    weights=(1.5, 0.5),
                    update_count=1,
                )
            )
            audit = DomainGradientAudit(
                actor,
                {
                    "enabled": True,
                    "domain_gradient_enabled": True,
                    "domains": ["math", "code"],
                    "dynamic_domain_loss_weighting_enabled": True,
                },
            )
            micro_batch = SimpleNamespace(
                batch={"response_mask": torch.ones(2, 2)},
                non_tensor_batch={"domain": ["math", "code"]},
            )
            replay = MagicMock()
            audit._backward_replay = replay
            expected = (torch.tensor([3.0]),)

            with patch(
                "mopd_verl.domain_gradient.audit.snapshot_gradients",
                return_value=expected,
            ):
                actual = audit._snapshot_training_gradient_reference(
                    SimpleNamespace(),
                    (micro_batch,),
                    (1.0,),
                    on_policy=True,
                    temperature=1.0,
                )

        self.assertIs(actual, expected)
        replay.assert_called_once()
        call = replay.call_args
        self.assertIsNone(call.kwargs["domain"])
        self.assertEqual(len(call.kwargs["gradient_masks"]), 1)
        torch.testing.assert_close(
            call.kwargs["gradient_masks"][0],
            torch.tensor([[1.5, 1.5], [0.5, 0.5]]),
        )

    def test_dynamic_metrics_expose_only_requested_domain_values(self) -> None:
        torch = self._torch()
        with self._stubbed_verl(torch):
            from mopd_verl.domain_gradient.audit import DomainGradientAudit
            from mopd_verl.domain_gradient.weighting import DomainWeightState

            actor = SimpleNamespace(
                _mopd_domain_weight_state=DomainWeightState(
                    domains=("math", "code"),
                    ema_norms=(1.0, 1.0),
                    weights=(1.0, 1.0),
                    update_count=1,
                )
            )
            audit = DomainGradientAudit(
                actor,
                {
                    "domains": ["math", "code"],
                    "dynamic_domain_loss_weighting_enabled": True,
                },
            )

            metrics = audit._dynamic_weight_metrics(
                {"math": 4.0, "code": 9.0}
            )

        self.assertEqual(
            set(metrics),
            {
                "math/dynamic_weight/applied_gradient_weight",
                "math/dynamic_weight/bounded_target_gradient_weight",
                "math/dynamic_weight/ema_grad_norm",
                "math/dynamic_weight/weighted_grad_norm",
                "code/dynamic_weight/applied_gradient_weight",
                "code/dynamic_weight/bounded_target_gradient_weight",
                "code/dynamic_weight/ema_grad_norm",
                "code/dynamic_weight/weighted_grad_norm",
            },
        )
        self.assertEqual(
            metrics["math/dynamic_weight/applied_gradient_weight"],
            1.0,
        )
        self.assertEqual(
            metrics[
                "math/dynamic_weight/bounded_target_gradient_weight"
            ],
            1.0,
        )
        self.assertEqual(
            metrics["math/dynamic_weight/ema_grad_norm"],
            1.0,
        )
        self.assertEqual(
            metrics["math/dynamic_weight/weighted_grad_norm"],
            2.0,
        )

    def test_dynamic_weight_update_runs_at_most_once_per_global_step(self) -> None:
        torch = self._torch()
        with self._stubbed_verl(torch):
            from mopd_verl.domain_gradient.audit import DomainGradientAudit
            from mopd_verl.domain_gradient.weighting import DomainWeightState

            actor = SimpleNamespace(
                _mopd_domain_weight_state=DomainWeightState(
                    domains=("math", "code"),
                    ema_norms=(1.0, 2.0),
                    weights=(1.5, 0.5),
                    update_count=1,
                    last_updated_step=8,
                )
            )
            audit = DomainGradientAudit(
                actor,
                {
                    "enabled": True,
                    "domain_gradient_enabled": True,
                    "domains": ["math", "code"],
                    "step": 8,
                    "dynamic_domain_loss_weighting_enabled": True,
                    "dynamic_domain_loss_weighting_update_enabled": True,
                },
            )

        self.assertFalse(audit._should_update_dynamic_weighting())

    def test_dynamic_weight_state_round_trips_through_optimizer_state(self) -> None:
        torch = self._torch()
        with self._stubbed_verl(torch):
            from mopd_verl.domain_gradient.audit import DomainGradientAudit
            from mopd_verl.domain_gradient.weighting import DomainWeightState

            first_optimizer = SimpleNamespace(param_groups=[{}])
            first_actor = SimpleNamespace(actor_optimizer=first_optimizer)
            first_audit = DomainGradientAudit(
                first_actor,
                {
                    "domains": ["math", "code"],
                    "dynamic_domain_loss_weighting_enabled": True,
                },
            )
            first_audit._weight_state = DomainWeightState(
                domains=("math", "code"),
                ema_norms=(1.0, 3.0),
                weights=(1.4, 0.6),
                target_weights=(1.5, 0.5),
                update_count=2,
                last_updated_step=12,
            )
            first_audit._persist_weight_state()
            serialized_group = dict(first_optimizer.param_groups[0])

            second_optimizer = SimpleNamespace(
                param_groups=[serialized_group]
            )
            second_actor = SimpleNamespace(actor_optimizer=second_optimizer)
            restored = DomainGradientAudit(
                second_actor,
                {
                    "domains": ["math", "code"],
                    "dynamic_domain_loss_weighting_enabled": True,
                },
            )

        self.assertEqual(restored._weight_state, first_audit._weight_state)

    def test_reweighted_total_preserves_unconfigured_domain_residual(self) -> None:
        torch = self._torch()
        with self._stubbed_verl(torch):
            from mopd_verl.domain_gradient.geometry import (
                reweighted_total_vector,
            )

            total = (torch.tensor([3.0, 5.0]),)
            math = (torch.tensor([1.0, 2.0]),)
            reweighted = reweighted_total_vector(
                total,
                {"math": math},
                {"math": 2.0},
            )

        torch.testing.assert_close(
            reweighted[0],
            torch.tensor([4.0, 7.0]),
        )


if __name__ == "__main__":
    unittest.main()
