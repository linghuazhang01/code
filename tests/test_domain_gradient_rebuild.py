from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from mopd_verl.domain_gradient.config import DomainGradientConfig

ROOT = Path(__file__).resolve().parents[1]
KNOTS = (
    (-0.0025, 0.0),
    (0.0, 0.2),
    (0.005, 2.0),
    (0.010, 3.0),
    (0.015, 4.0),
)


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

        self.assertFalse(
            DomainGradientConfig.from_meta({**common, "step": 4}).parity_enabled
        )
        self.assertTrue(
            DomainGradientConfig.from_meta({**common, "step": 6}).parity_enabled
        )

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

    def test_tail_token_gradient_replay_is_supported(self) -> None:
        config = DomainGradientConfig.from_meta(
            {
                "enabled": True,
                "domain_gradient_enabled": True,
                "domains": ["math", "code"],
                "token_gradient_enabled": True,
                "token_gradient_tail_fraction": 0.2,
                "token_gradient_tail_min_tokens": 2,
            }
        )

        self.assertTrue(config.enabled)
        self.assertTrue(config.token_gradient_enabled)
        self.assertTrue(config.token_gradient_tail_enabled)
        self.assertEqual(config.token_gradient_tail_fraction, 0.2)
        self.assertEqual(config.token_gradient_tail_min_tokens, 2)
        self.assertFalse(config.token_gradient_top_p_enabled)

    def test_partial_top_p_gradient_statistics_are_supported(self) -> None:
        config = DomainGradientConfig.from_meta(
            {
                "enabled": True,
                "domain_gradient_enabled": True,
                "domains": ["math"],
                "token_gradient_enabled": True,
                "token_gradient_tail_enabled": False,
                "token_gradient_top_p_enabled": True,
                "token_gradient_top_k": None,
                "token_gradient_top_p": 0.5,
                "token_gradient_loss_abs_selection_enabled": True,
                "token_gradient_vocab_size": 128,
            }
        )

        self.assertEqual(config.token_gradient_top_p, 0.5)
        self.assertIsNone(config.token_gradient_top_k)
        self.assertEqual(config.token_gradient_vocab_size, 128)

    def test_token_gradient_master_requires_an_enabled_subset(self) -> None:
        from mopd_verl.verl_audit import MOPDAuditLogger

        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "domains": ["math"],
                    "token_gradient_enabled": True,
                    "token_gradient_tail_enabled": False,
                    "token_gradient_top_p_enabled": False,
                    "token_gradient_freq_steps": 1,
                }
            }
        )

        self.assertFalse(logger.should_compute_token_gradient(1))

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
                logger.full_gradient_meta("train", step)["mopd_full_gradient"]
            )
            for step in range(1, 5)
        ]

        self.assertEqual(
            [config.step for config in configs if config.enabled],
            [2, 4],
        )
        self.assertFalse(configs[0].enabled)
        self.assertFalse(configs[2].enabled)

    def test_dynamic_weighting_keeps_actor_hook_active_between_updates(self) -> None:
        from mopd_verl.verl_audit import MOPDAuditLogger

        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "full_gradient_enabled": False,
                    "domains": ["math", "code"],
                    "dynamic_domain_loss_weighting_enabled": True,
                    "dynamic_domain_loss_weighting_freq_steps": 4,
                }
            }
        )

        step_one_meta = logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
        step_four_meta = logger.full_gradient_meta("train", 4)["mopd_full_gradient"]
        step_one = DomainGradientConfig.from_meta(step_one_meta)
        step_four = DomainGradientConfig.from_meta(step_four_meta)

        self.assertTrue(logger.should_compute_full_gradient(1))
        self.assertFalse(step_one.enabled)
        self.assertTrue(step_one.dynamic_weighting_enabled)
        self.assertFalse(step_one.dynamic_weighting_update_enabled)
        self.assertTrue(step_four.enabled)
        self.assertTrue(step_four.dynamic_weighting_update_enabled)

    def test_dynamic_weighting_accepts_projection_share_source(self) -> None:
        config = DomainGradientConfig.from_meta(
            {
                "domains": ["math", "code", "science"],
                "dynamic_domain_loss_weighting_enabled": True,
                "dynamic_domain_loss_weighting_signal_source": (
                    "domain_gradient_projection_share"
                ),
            }
        )

        self.assertEqual(
            config.dynamic_weighting_signal_source,
            "domain_gradient_projection_share",
        )

    def test_token_weighting_configuration_is_validated(self) -> None:
        config = DomainGradientConfig.from_meta(
            {
                "domains": ["math", "code", "science"],
                "control_token_loss_weighting_enabled": True,
                "control_token_loss_weight": 2.0,
                "control_token_ids": [11, 22, 11],
                "all_domain_shared_token_loss_weighting_enabled": True,
                "all_domain_shared_token_loss_weight": 1.5,
                "all_domain_shared_token_selection_mode": ("cumulative_abs_loss"),
                "all_domain_shared_token_top_k": 200,
            }
        )

        self.assertEqual(config.control_token_ids, (11, 22))
        self.assertEqual(config.control_token_weight, 2.0)
        self.assertEqual(
            config.all_domain_shared_token_selection_mode,
            "cumulative_abs_loss",
        )
        self.assertEqual(config.all_domain_shared_token_top_k, 200)

    def test_online_control_selection_configuration_is_supported(self) -> None:
        config = DomainGradientConfig.from_meta(
            {
                "domains": ["math", "code", "science"],
                "control_token_loss_weighting_enabled": True,
                "control_token_loss_weight": 4.0,
                "control_token_candidate_ids": [30, 10, 30, 20],
                "control_token_normalize_per_domain": True,
                "control_token_online_selection_enabled": True,
                "control_token_online_audit_interval_steps": 3,
                "control_token_online_window_steps": 3,
                "control_token_online_min_mean_occurrences_per_step": 20.0,
                "control_token_online_top_k": 30,
                "control_token_online_selection_mode": "top_speed",
            }
        )

        self.assertTrue(config.control_token_online_selection_enabled)
        self.assertEqual(config.control_token_candidate_ids, (10, 20, 30))
        self.assertEqual(config.control_token_online_audit_interval_steps, 3)
        self.assertEqual(config.control_token_online_window_steps, 3)
        self.assertEqual(
            config.control_token_online_min_mean_occurrences_per_step,
            20.0,
        )
        self.assertEqual(config.control_token_online_top_k, 30)
        self.assertEqual(config.control_token_online_selection_mode, "top_speed")

    def test_online_control_selection_defaults_to_top_loss(self) -> None:
        config = DomainGradientConfig.from_meta(
            {
                "domains": ["math"],
                "control_token_loss_weighting_enabled": True,
                "control_token_candidate_ids": [10, 20],
                "control_token_online_selection_enabled": True,
            }
        )

        self.assertEqual(config.control_token_online_selection_mode, "top_loss")

    def test_online_control_selection_rejects_invalid_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "selection mode"):
            DomainGradientConfig.from_meta(
                {
                    "domains": ["math"],
                    "control_token_loss_weighting_enabled": True,
                    "control_token_candidate_ids": [10, 20],
                    "control_token_online_selection_enabled": True,
                    "control_token_online_selection_mode": "fastest",
                }
            )

    def test_online_top_speed_requires_two_step_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            DomainGradientConfig.from_meta(
                {
                    "domains": ["math"],
                    "control_token_loss_weighting_enabled": True,
                    "control_token_candidate_ids": [10, 20],
                    "control_token_online_selection_enabled": True,
                    "control_token_online_selection_mode": "top_speed",
                    "control_token_online_window_steps": 1,
                }
            )

    def test_domain_online_control_candidates_are_canonicalized(self) -> None:
        config = DomainGradientConfig.from_meta(
            {
                "domains": ["math", "code", "science"],
                "control_token_loss_weighting_enabled": True,
                "domain_control_token_candidate_ids": {
                    "math": [30, 10, 30],
                    "code": [20, 10],
                    "science": [40],
                },
                "control_token_online_selection_enabled": True,
            }
        )

        self.assertEqual(config.control_token_candidate_ids, ())
        self.assertEqual(
            config.effective_domain_candidate_map(),
            {
                "math": (10, 30),
                "code": (10, 20),
                "science": (40,),
            },
        )

    def test_domain_online_control_candidates_require_exact_domains(self) -> None:
        base = {
            "domains": ["math", "code"],
            "control_token_loss_weighting_enabled": True,
            "control_token_online_selection_enabled": True,
        }
        for candidates in (
            {"math": [10]},
            {"math": [10], "code": [20], "science": [30]},
            {"math": [10], "code": []},
        ):
            with self.subTest(candidates=candidates):
                with self.assertRaises((TypeError, ValueError)):
                    DomainGradientConfig.from_meta(
                        {
                            **base,
                            "domain_control_token_candidate_ids": candidates,
                        }
                    )

    def test_online_control_candidates_reject_both_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "either.*candidate"):
            DomainGradientConfig.from_meta(
                {
                    "domains": ["math"],
                    "control_token_loss_weighting_enabled": True,
                    "control_token_candidate_ids": [10],
                    "domain_control_token_candidate_ids": {"math": [10]},
                    "control_token_online_selection_enabled": True,
                }
            )

    def test_online_control_selection_rejects_fixed_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed Control-token IDs"):
            DomainGradientConfig.from_meta(
                {
                    "domains": ["math"],
                    "control_token_loss_weighting_enabled": True,
                    "control_token_ids": [10],
                    "control_token_candidate_ids": [10, 20],
                    "control_token_online_selection_enabled": True,
                }
            )

    def test_domain_phase_control_configuration_is_supported(self) -> None:
        config = DomainGradientConfig.from_meta(
            {
                "domains": ["math", "code"],
                "control_token_loss_weighting_enabled": True,
                "control_token_loss_weight": 2.0,
                "domain_control_token_ids": {
                    "math": [10, 11, 10],
                    "code": [20],
                },
                "control_token_normalize_per_domain": True,
                "control_token_phase_gate_enabled": True,
                "control_token_span_weighting_enabled": True,
                "control_token_phase_gate_window_steps": 5,
                "control_token_phase_gate_ema_beta": 0.9,
                "control_token_phase_gate_temperature": 0.1,
                "control_token_phase_gate_initial": 0.8,
                "control_token_span_length": 16,
                "control_token_span_decay_tau": 8.0,
            }
        )

        self.assertEqual(
            dict(config.domain_control_token_ids),
            {"math": (10, 11), "code": (20,)},
        )
        self.assertTrue(config.control_token_phase_gate_enabled)
        self.assertTrue(config.control_token_span_weighting_enabled)
        self.assertTrue(config.control_token_normalize_per_domain)

    def test_domain_control_speed_configuration_is_supported(self) -> None:
        config = DomainGradientConfig.from_meta(
            {
                "domains": ["math", "code"],
                "control_token_loss_weighting_enabled": True,
                "domain_control_token_ids": {
                    "math": [10, 11],
                    "code": [20],
                },
                "control_token_speed_weighting_enabled": True,
                "control_token_speed_window_steps": 5,
                "control_token_speed_ema_beta": 0.8,
                "control_token_speed_update_interval_steps": 2,
                "control_token_speed_initial_weight": 3.0,
                "control_token_speed_min_occurrences": 128,
                "control_token_speed_weight_knots": KNOTS,
            }
        )

        self.assertTrue(config.control_token_speed_weighting_enabled)
        self.assertEqual(config.control_token_speed_window_steps, 5)
        self.assertEqual(config.control_token_speed_update_interval_steps, 2)
        self.assertEqual(config.control_token_speed_weight_knots, KNOTS)

    def test_speed_weighting_and_phase_gate_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            DomainGradientConfig.from_meta(
                {
                    "domains": ["math"],
                    "control_token_loss_weighting_enabled": True,
                    "domain_control_token_ids": {"math": [10]},
                    "control_token_speed_weighting_enabled": True,
                    "control_token_phase_gate_enabled": True,
                }
            )

    def test_span_weighting_requires_phase_gate(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Successor-span weighting requires",
        ):
            DomainGradientConfig.from_meta(
                {
                    "domains": ["math"],
                    "control_token_loss_weighting_enabled": True,
                    "domain_control_token_ids": {"math": [10]},
                    "control_token_phase_gate_enabled": False,
                    "control_token_span_weighting_enabled": True,
                }
            )

    def test_invalid_shared_token_selection_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "all_domain_shared_token_selection_mode",
        ):
            DomainGradientConfig.from_meta(
                {
                    "domains": ["math", "code"],
                    "all_domain_shared_token_loss_weighting_enabled": True,
                    "all_domain_shared_token_selection_mode": "future_loss",
                }
            )

    def test_token_weighting_keeps_actor_hook_active_without_grad_audit(
        self,
    ) -> None:
        from mopd_verl.verl_audit import MOPDAuditLogger

        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "full_gradient_enabled": False,
                    "domains": ["math", "code", "science"],
                    "control_token_loss_weighting_enabled": True,
                    "control_token_loss_weight": 2.0,
                    "control_token_ids": [10],
                }
            }
        )

        self.assertTrue(logger.should_compute_full_gradient(1))
        meta = logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
        config = DomainGradientConfig.from_meta(meta)
        self.assertFalse(config.enabled)
        self.assertTrue(config.control_token_weighting_enabled)
        self.assertEqual(config.control_token_ids, (10,))


class DomainGradientSourceTests(unittest.TestCase):
    def test_training_and_audit_share_one_loss_builder(self) -> None:
        actor_source = (
            ROOT / "third_party/verl/verl/workers/actor/dp_actor.py"
        ).read_text(encoding="utf-8")
        audit_source = (ROOT / "mopd_verl/domain_gradient/audit.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("build_actor_micro_batch_loss(", actor_source)
        self.assertIn("build_actor_micro_batch_loss(", audit_source)
        self.assertNotIn("get_policy_loss_fn", actor_source)
        self.assertNotIn("finalize_fsdp", audit_source)
        self.assertNotIn("no_sync", audit_source)
        self.assertNotIn("total_plus_domain", audit_source)
        self.assertNotIn("floating_response_gradient_mask", audit_source)
        self.assertIn("domain_vectors", audit_source)
        self.assertIn("+ domain_count", audit_source)

    def test_audit_total_uses_configured_storage_dtype(self) -> None:
        audit_source = (ROOT / "mopd_verl/domain_gradient/audit.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn('"float32" if self.config.parity_enabled', audit_source)
        self.assertIn(
            "audit_total = snapshot_gradients(\n"
            "                self.actor,\n"
            "                self.config.storage_dtype,",
            audit_source,
        )

    def test_gradient_override_is_a_pure_domain_selector(self) -> None:
        actor_loss_source = (ROOT / "mopd_verl/full_gradient/actor_loss.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("floating_response_gradient_mask(", actor_loss_source)
        self.assertIn("gradient_mask_override.to(", actor_loss_source)

    def test_shared_token_weighting_requires_one_full_step_batch(
        self,
    ) -> None:
        actor_source = (
            ROOT / "third_party/verl/verl/workers/actor/dp_actor.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "audit.config.all_domain_shared_token_weighting_enabled",
            actor_source,
        )
        self.assertIn(
            "len(mini_batches) != 1 or self.config.ppo_epochs != 1",
            actor_source,
        )

    def test_old_tracker_is_only_a_compatibility_shim(self) -> None:
        source = (ROOT / "mopd_verl/full_gradient/tracker.py").read_text(
            encoding="utf-8"
        )

        self.assertLess(len(source.splitlines()), 30)
        self.assertIn("DomainGradientAudit", source)

    def test_domain_batch_sampler_takes_precedence_over_plain_sampler(self) -> None:
        source = (ROOT / "third_party/verl/verl/trainer/ppo/ray_trainer.py").read_text(
            encoding="utf-8"
        )
        create_index = source.index(
            "train_batch_sampler = create_domain_batch_sampler("
        )
        fallback_index = source.index(
            "if train_batch_sampler is None and train_sampler is None:"
        )

        self.assertLess(create_index, fallback_index)

    def test_shared_teacher_config_avoids_redundant_tensor_aliases(self) -> None:
        main_source = (ROOT / "third_party/verl/verl/trainer/main_ppo.py").read_text(
            encoding="utf-8"
        )
        trainer_source = (
            ROOT / "third_party/verl/verl/trainer/ppo/ray_trainer.py"
        ).read_text(encoding="utf-8")

        self.assertIn("REF_POLICY_POOL_ID", main_source)
        self.assertIn("_configured_teacher_domains(config)", trainer_source)
        self.assertNotIn("_alias_math_teacher_tensors", trainer_source)

    def test_topk_cross_entropy_reuses_the_production_forward(self) -> None:
        trainer_source = (
            ROOT / "third_party/verl/verl/trainer/ppo/ray_trainer.py"
        ).read_text(encoding="utf-8")
        actor_source = (
            ROOT / "third_party/verl/verl/workers/actor/dp_actor.py"
        ).read_text(encoding="utf-8")
        worker_source = (
            ROOT / "third_party/verl/verl/workers/fsdp_workers.py"
        ).read_text(encoding="utf-8")

        fallback_index = trainer_source.index(
            "if not reuse_training_topk_cross_entropy:"
        )
        standalone_index = trainer_source.index(
            "self.actor_rollout_wg.compute_teacher_student_cross_entropy("
        )
        audit_meta_index = trainer_source.index(
            "self.mopd_audit_logger.full_gradient_meta("
        )
        update_index = trainer_source.index(
            "actor_output = self.actor_rollout_wg.update_actor(batch)"
        )
        reused_ce_index = trainer_source.index(
            'cross_entropy_key = "teacher_student_cross_entropy"'
        )
        training_log_index = trainer_source.index(
            "self.mopd_audit_logger.log_training_step("
        )

        self.assertLess(fallback_index, standalone_index)
        self.assertLess(audit_meta_index, update_index)
        self.assertLess(update_index, reused_ce_index)
        self.assertLess(reused_ce_index, training_log_index)
        self.assertIn(
            'actor_config.strategy in {"fsdp", "fsdp2"}',
            trainer_source,
        )
        self.assertIn(
            "return_teacher_student_cross_entropy=(",
            actor_source,
        )
        self.assertIn(
            "mini_batch_cross_entropy = restore_dynamic_batch(",
            actor_source,
        )
        self.assertIn("return_auxiliary_outputs=True", worker_source)

    def test_configured_token_loss_reuses_the_production_forward(self) -> None:
        trainer_source = (
            ROOT / "third_party/verl/verl/trainer/ppo/ray_trainer.py"
        ).read_text(encoding="utf-8")
        actor_source = (
            ROOT / "third_party/verl/verl/workers/actor/dp_actor.py"
        ).read_text(encoding="utf-8")

        update_index = trainer_source.index(
            "actor_output = self.actor_rollout_wg.update_actor(batch)"
        )
        copy_index = trainer_source.index(
            'configured_token_loss_key = "configured_token_loss"'
        )
        audit_index = trainer_source.index("self.mopd_audit_logger.log_training_step(")

        self.assertLess(update_index, copy_index)
        self.assertLess(copy_index, audit_index)
        self.assertIn("return_configured_token_loss=(", actor_source)
        self.assertIn(
            "mini_batch_configured_token_loss = restore_dynamic_batch(",
            actor_source,
        )
        self.assertIn(
            "configured_token_loss_epoch_batches",
            actor_source,
        )
        self.assertIn(
            'auxiliary_outputs["configured_token_loss_mask"]',
            actor_source,
        )
        self.assertIn(
            ").mean(dim=0)",
            actor_source,
        )

    def test_dynamic_weighting_requires_full_actor_batch_replay(self) -> None:
        actor_source = (
            ROOT / "third_party/verl/verl/workers/actor/dp_actor.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "audit.config.dynamic_weighting_enabled " "and len(mini_batches) != 1",
            actor_source,
        )
        self.assertIn(
            '"entire actor batch."',
            actor_source,
        )

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
        source = (ROOT / "third_party/verl/verl/workers/fsdp_workers.py").read_text(
            encoding="utf-8"
        )

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

    def test_fractional_gradient_gate_is_bitwise_identity_in_bfloat16(
        self,
    ) -> None:
        try:
            import torch

            from mopd_verl.full_gradient.loss_support import (
                gate_tensor_gradient,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch/verl is unavailable in this environment: {exc}")

        values = torch.tensor(
            [[-7.75, -0.125, 0.75], [1.5, 4.25, 15.5]],
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        gate = torch.tensor(
            [1.3425, 0.6575],
            dtype=torch.float32,
        )

        gated_values = gate_tensor_gradient(values, gate)

        self.assertTrue(torch.equal(gated_values, values))
        gated_values.float().sum().backward()
        expected_gradient = (
            gate.to(dtype=torch.bfloat16).unsqueeze(-1).expand_as(values)
        )
        self.assertTrue(torch.equal(values.grad, expected_gradient))

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
        torch.testing.assert_close(
            module.running_mean, torch.zeros_like(module.running_mean)
        )
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
        self.assertAlmostEqual(metrics[f"{prefix}/cosine"], 1.0)
        self.assertAlmostEqual(metrics[f"{prefix}/norm_ratio"], 1.0)
        self.assertAlmostEqual(metrics[f"{prefix}/projection_share"], 1.0)
        self.assertAlmostEqual(metrics[f"{prefix}/diff_norm"], 0.0)
        self.assertLessEqual(metrics[f"{prefix}/rel_l2"], 1e-8)
        self.assertEqual(metrics[f"{prefix}/passed"], 1.0)
        self.assertLessEqual(compact_metrics[f"{prefix}/rel_l2"], 2e-2)
        self.assertEqual(compact_metrics[f"{prefix}/passed"], 1.0)


if __name__ == "__main__":
    unittest.main()
