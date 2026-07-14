from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch

from mopd_verl.domain_gradient.config import DomainGradientConfig


ROOT = Path(__file__).resolve().parents[1]


class DomainGradientConfigTests(unittest.TestCase):
    def test_current_full_gradient_meta_is_supported(self) -> None:
        config = DomainGradientConfig.from_meta(
            {
                "enabled": True,
                "domain_gradient_enabled": True,
                "domains": ["math", "code", "math"],
                "storage_dtype": "bfloat16",
                "step": 4,
                "full_grad_training_parity_freq_steps": 1,
                "full_grad_training_parity_rel_l2_threshold": 1e-5,
                "sequence_masked_target_closure_rel_l2_threshold": 0.02,
            }
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.step, 4)
        self.assertEqual(config.domains, ("math", "code"))
        self.assertTrue(config.parity_enabled)
        self.assertEqual(config.closure_rel_l2_threshold, 0.02)

    def test_training_parity_respects_configured_frequency(self) -> None:
        common = {
            "enabled": True,
            "domain_gradient_enabled": True,
            "domains": ["math", "code"],
            "full_grad_training_parity_freq_steps": 3,
        }

        self.assertFalse(DomainGradientConfig.from_meta({**common, "step": 4}).parity_enabled)
        self.assertTrue(DomainGradientConfig.from_meta({**common, "step": 6}).parity_enabled)

    def test_legacy_nested_gradient_replay_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_gradient_enabled"):
            DomainGradientConfig.from_meta(
                {
                    "enabled": True,
                    "domain_gradient_enabled": True,
                    "domains": ["math"],
                    "sample_gradient_enabled": True,
                }
            )

    def test_audit_frequency_emits_only_current_even_source_steps(self) -> None:
        from mopd_verl.verl_audit import MOPDAuditLogger

        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "full_gradient_enabled": True,
                    "full_gradient_freq_steps": 2,
                    "domains": ["math", "code"],
                }
            }
        )
        configs = [
            DomainGradientConfig.from_meta(
                logger.full_gradient_meta("train", step)[
                    "mopd_full_gradient"
                ]
            )
            for step in range(1, 5)
        ]

        self.assertEqual(
            [config.step for config in configs if config.enabled],
            [2, 4],
        )
        self.assertFalse(configs[0].enabled)
        self.assertFalse(configs[2].enabled)


class DomainGradientSourceTests(unittest.TestCase):
    def test_training_and_audit_share_one_loss_builder(self) -> None:
        actor_source = (
            ROOT / "third_party/verl/verl/workers/actor/dp_actor.py"
        ).read_text(encoding="utf-8")
        audit_source = (
            ROOT / "mopd_verl/domain_gradient/audit.py"
        ).read_text(encoding="utf-8")

        self.assertIn("build_actor_micro_batch_loss(", actor_source)
        self.assertIn("build_actor_micro_batch_loss(", audit_source)
        self.assertNotIn("get_policy_loss_fn", actor_source)
        self.assertNotIn("finalize_fsdp", audit_source)
        self.assertNotIn("no_sync", audit_source)
        self.assertNotIn("total_plus_domain", audit_source)
        self.assertNotIn("floating_response_gradient_mask", audit_source)
        self.assertIn("domain_vectors", audit_source)
        self.assertIn("1 + domain_count", audit_source)

    def test_audit_total_uses_configured_storage_dtype(self) -> None:
        audit_source = (
            ROOT / "mopd_verl/domain_gradient/audit.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn('"float32" if self.config.parity_enabled', audit_source)
        self.assertIn(
            "audit_total = snapshot_gradients(\n"
            "                self.actor,\n"
            "                self.config.storage_dtype,",
            audit_source,
        )

    def test_gradient_override_is_a_pure_domain_selector(self) -> None:
        actor_loss_source = (
            ROOT / "mopd_verl/full_gradient/actor_loss.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("floating_response_gradient_mask(", actor_loss_source)
        self.assertIn("gradient_mask_override.to(", actor_loss_source)

    def test_old_tracker_is_only_a_compatibility_shim(self) -> None:
        source = (ROOT / "mopd_verl/full_gradient/tracker.py").read_text(encoding="utf-8")

        self.assertLess(len(source.splitlines()), 30)
        self.assertIn("DomainGradientAudit", source)

    def test_domain_batch_sampler_takes_precedence_over_plain_sampler(self) -> None:
        source = (
            ROOT / "third_party/verl/verl/trainer/ppo/ray_trainer.py"
        ).read_text(encoding="utf-8")
        create_index = source.index("train_batch_sampler = create_domain_batch_sampler(")
        fallback_index = source.index(
            "if train_batch_sampler is None and train_sampler is None:"
        )

        self.assertLess(create_index, fallback_index)

    def test_split_teacher_config_has_runtime_consumers(self) -> None:
        main_source = (
            ROOT / "third_party/verl/verl/trainer/main_ppo.py"
        ).read_text(encoding="utf-8")
        trainer_source = (
            ROOT / "third_party/verl/verl/trainer/ppo/ray_trainer.py"
        ).read_text(encoding="utf-8")

        self.assertIn("REF_POLICY_POOL_ID", main_source)
        self.assertIn("_configured_teacher_domains(config)", trainer_source)
        self.assertIn("_alias_math_teacher_tensors(batch, self.teacher_domains)", trainer_source)

    def test_patch_script_cannot_inject_retired_tracker_api(self) -> None:
        source = (ROOT / "scripts/apply_gopd_audit_patch.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("prepare_micro_batches", source)
        self.assertNotIn("run_pre_update_audit", source)
        self.assertNotIn("full_grad_training_parity_metrics", source)
        self.assertIn("Automatic patching of a pristine dp_actor.py is retired", source)

    def test_reviewed_patch_entrypoint_is_read_only_and_idempotent(self) -> None:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "apply_gopd_audit_patch.py"),
            str(ROOT / "third_party"),
        ]
        paths = tuple(
            ROOT / "third_party" / relative
            for relative in (
                "verl/verl/trainer/main_ppo.py",
                "verl/verl/trainer/ppo/ray_trainer.py",
                "verl/verl/utils/dataset/rl_dataset.py",
                "verl/verl/workers/actor/dp_actor.py",
                "verl/verl/workers/fsdp_workers.py",
            )
        )
        before = {path: path.read_bytes() for path in paths}

        subprocess.run(command, check=True, capture_output=True, text=True)
        subprocess.run(command, check=True, capture_output=True, text=True)

        after = {path: path.read_bytes() for path in paths}
        self.assertEqual(after, before)

    def test_standalone_ref_worker_uses_ref_fsdp_mesh(self) -> None:
        source = (
            ROOT / "third_party/verl/verl/workers/fsdp_workers.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'self.config.ref.fsdp_config if self.role == "ref"',
            source,
        )
        self.assertIn(
            "create_device_mesh(world_size=world_size, fsdp_size=mesh_fsdp_config.fsdp_size)",
            source,
        )
        self.assertLess(
            source.index("self.role = role"),
            source.index('if self.role == "ref"'),
        )


class GradientGateTorchTests(unittest.TestCase):
    def test_zero_kl_coefficient_disables_kl_compute(self) -> None:
        try:
            from mopd_verl.full_gradient.loss_support import active_kl_loss
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch/verl is unavailable in this environment: {exc}")

        self.assertEqual(
            active_kl_loss({"use_kl_loss": True, "kl_loss_coef": 0.0}),
            (False, 0.0),
        )
        self.assertEqual(
            active_kl_loss({"use_kl_loss": True, "kl_loss_coef": 0.25}),
            (True, 0.25),
        )

    def test_masked_mean_ignores_non_finite_masked_values(self) -> None:
        try:
            import torch
            from mopd_verl.full_gradient.loss_support import masked_mean
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch/verl is unavailable in this environment: {exc}")

        values = torch.tensor([2.0, float("nan"), 4.0])
        mask = torch.tensor([1.0, 0.0, 1.0])

        self.assertEqual(masked_mean(values, mask), 3.0)

    def test_domain_gradient_sum_matches_unmasked_gradient(self) -> None:
        try:
            import torch
            from mopd_verl.full_gradient.loss_support import (
                gate_tensor_gradient,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch/verl is unavailable in this environment: {exc}")

        parameter = torch.nn.Parameter(torch.tensor([0.4, -0.2, 0.7]))
        features = torch.tensor(
            [[1.0, 2.0, -1.0], [0.5, -1.0, 3.0], [-2.0, 0.2, 1.0], [1.5, 0.5, -0.5]]
        )

        def gradient(mask: "torch.Tensor | None") -> "torch.Tensor":
            parameter.grad = None
            values = features @ parameter
            if mask is not None:
                values = gate_tensor_gradient(values, mask)
            loss = values.square().mean()
            loss.backward()
            return parameter.grad.detach().clone()

        total = gradient(None)
        math = gradient(torch.tensor([1.0, 1.0, 0.0, 0.0]))
        code = gradient(torch.tensor([0.0, 0.0, 1.0, 1.0]))

        torch.testing.assert_close(math + code, total, rtol=1e-6, atol=1e-7)

    def test_boolean_response_mask_preserves_domain_gradient_gate(self) -> None:
        try:
            import torch
            from mopd_verl.full_gradient.loss_support import (
                floating_response_gradient_mask,
                gate_tensor_gradient,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch/verl is unavailable in this environment: {exc}")

        parameter = torch.nn.Parameter(torch.tensor([0.3, -0.6]))
        features = torch.tensor([[1.0, 2.0], [-0.5, 1.0], [2.0, -1.0]])
        response_mask = torch.tensor([True, True, True])
        domain_mask = torch.tensor([True, False, True])

        def gradient(mask: "torch.Tensor | None") -> "torch.Tensor":
            parameter.grad = None
            values = features @ parameter
            if mask is not None:
                values = gate_tensor_gradient(values, mask)
            values.square().mean().backward()
            return parameter.grad.detach().clone()

        total = gradient(None)
        domain_weights = floating_response_gradient_mask(domain_mask, response_mask)
        self.assertEqual(domain_weights.tolist(), [1.0, 0.0, 1.0])
        domain = gradient(domain_weights)
        other = gradient(1.0 - domain_weights)

        torch.testing.assert_close(domain + other, total, rtol=1e-6, atol=1e-7)

    def test_audit_state_restores_rng_grad_buffer_and_mode(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is unavailable in this environment: {exc}")

        verl_module = ModuleType("verl")
        verl_module.__path__ = []
        utils_module = ModuleType("verl.utils")
        utils_module.__path__ = []
        device_module = ModuleType("verl.utils.device")
        device_module.get_torch_device = lambda: torch.cpu
        sys.modules.pop("mopd_verl.domain_gradient.state", None)
        with patch.dict(
            sys.modules,
            {
                "verl": verl_module,
                "verl.utils": utils_module,
                "verl.utils.device": device_module,
            },
        ):
            from mopd_verl.domain_gradient.state import AuditState

        module = torch.nn.BatchNorm1d(3)
        module.train()
        parameter = next(module.parameters())
        parameter.grad = torch.ones_like(parameter)
        actor = SimpleNamespace(actor_module=module)
        torch.manual_seed(17)
        state = AuditState.capture(actor)
        expected_random = torch.rand(4)

        module.eval()
        module.running_mean.add_(10.0)
        parameter.grad.zero_()
        state.restore()

        self.assertTrue(module.training)
        torch.testing.assert_close(module.running_mean, torch.zeros_like(module.running_mean))
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
        torch.testing.assert_close(torch.rand(4), expected_random)

    def test_geometry_supports_fp32_and_bf16_parity_snapshots(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is unavailable in this environment: {exc}")

        verl_module = ModuleType("verl")
        verl_module.__path__ = []
        utils_module = ModuleType("verl.utils")
        utils_module.__path__ = []
        device_module = ModuleType("verl.utils.device")
        device_module.get_device_id = lambda: "cpu"
        sys.modules.pop("mopd_verl.domain_gradient.geometry", None)
        with patch.dict(
            sys.modules,
            {
                "verl": verl_module,
                "verl.utils": utils_module,
                "verl.utils.device": device_module,
            },
        ):
            from mopd_verl.domain_gradient import geometry

        module = torch.nn.Linear(3, 1, bias=False)
        parameter = next(module.parameters())
        parameter.grad = torch.tensor([[0.5, -0.25, 1.25]])
        actor = SimpleNamespace(
            actor_module=module,
            config={"fsdp_config": {"fsdp_size": -1}},
            scaler=None,
        )
        reference = geometry.snapshot_gradients(actor)
        compact_reference = geometry.snapshot_gradients(actor, "bfloat16")

        self.assertEqual(reference[0].dtype, torch.float32)
        self.assertEqual(compact_reference[0].dtype, torch.bfloat16)
        self.assertEqual(
            geometry.vector_nbytes(compact_reference) * 2,
            geometry.vector_nbytes(reference),
        )
        self.assertAlmostEqual(
            geometry.vector_dot(actor, reference, reference),
            geometry.vector_squared_norm(actor, reference),
        )
        metrics = geometry.training_parity_metrics(actor, reference, 1e-8)
        compact_metrics = geometry.training_parity_metrics(
            actor,
            compact_reference,
            2e-2,
        )
        prefix = "global/full_grad_training_parity/audit_total_vs_training_total"
        self.assertLessEqual(metrics[f"{prefix}/rel_l2"], 1e-8)
        self.assertEqual(metrics[f"{prefix}/passed"], 1.0)
        self.assertLessEqual(compact_metrics[f"{prefix}/rel_l2"], 2e-2)
        self.assertEqual(compact_metrics[f"{prefix}/passed"], 1.0)


if __name__ == "__main__":
    unittest.main()
