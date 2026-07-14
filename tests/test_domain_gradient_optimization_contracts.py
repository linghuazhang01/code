from __future__ import annotations

import sys
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


class DomainGradientOptimizationContractTests(unittest.TestCase):
    def _torch(self) -> Any:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is unavailable: {exc}")
        return torch

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

        prefix = "global/full_grad_closure/domain_sum_vs_audit_total"
        self.assertAlmostEqual(metrics[f"{prefix}/diff_norm"], 0.5)
        self.assertGreater(metrics[f"{prefix}/domain_vector_norm_sum"], 1_999.0)
        self.assertGreater(metrics[f"{prefix}/domain_norm_sum_over_total_norm"], 1_999.0)
        self.assertLess(metrics[f"{prefix}/diff_norm_over_domain_vector_norm_sum"], 1e-3)
        self.assertGreater(metrics[f"{prefix}/estimated_storage_roundoff_rel_l2"], 0.02)
        self.assertEqual(metrics[f"{prefix}/storage_roundoff_may_exceed_threshold"], 1.0)
        legacy_prefix = "global/full_grad_closure/domain_sum_vs_training"
        canonical_payload = {
            key.removeprefix(prefix): value
            for key, value in metrics.items()
            if key.startswith(f"{prefix}/")
        }
        legacy_payload = {
            key.removeprefix(legacy_prefix): value
            for key, value in metrics.items()
            if key.startswith(f"{legacy_prefix}/")
        }
        self.assertEqual(canonical_payload, legacy_payload)

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

        prefix = "global/full_grad_closure/domain_sum_vs_audit_total"
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


if __name__ == "__main__":
    unittest.main()
