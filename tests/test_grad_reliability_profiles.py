from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from mopd_verl.config_profiles import list_config_profiles
from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import MOPDConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "test_grad_configs"
MATRIX_PATH = CONFIG_DIR / "mopd_grad_reliability_qwen0p6b_8b_matrix.yaml"
EXPECTED_PROFILE_NAMES = (
    "aw2_fsdp1_audit_on",
    "aw2_fsdp1_audit_off",
    "aw2_fsdp2_audit_on",
    "aw4_fsdp2_audit_on",
    "aw4_fsdp2_audit_off",
)
STUDENT_PATH = "/root/autodl-tmp/models/Qwen3-0.6B"
TEACHER_PATH = "/root/autodl-tmp/models/Qwen3-8B"


class GradientReliabilityProfileTests(unittest.TestCase):
    def test_matrix_contains_exactly_five_profiles(self) -> None:
        self.assertEqual(
            list_config_profiles(MATRIX_PATH),
            EXPECTED_PROFILE_NAMES,
        )
        self.assertEqual(
            tuple(
                CONFIG_DIR.glob(
                    "mopd_grad_reliability_qwen0p6b_0p6b_aw*.yaml"
                )
            ),
            (),
        )

    def _load(
        self,
        fsdp_size: int,
        audit_enabled: bool,
    ) -> tuple[Path, MOPDConfig]:
        audit_mode = "audit_on" if audit_enabled else "audit_off"
        profile = f"aw2_fsdp{fsdp_size}_{audit_mode}"
        return MATRIX_PATH, load_config(f"{MATRIX_PATH}::{profile}")

    def _load_hybrid(self, audit_enabled: bool) -> tuple[Path, MOPDConfig]:
        audit_mode = "audit_on" if audit_enabled else "audit_off"
        profile = f"aw4_fsdp2_{audit_mode}"
        return MATRIX_PATH, load_config(f"{MATRIX_PATH}::{profile}")

    def test_two_gpu_student_one_gpu_teacher_profiles(self) -> None:
        for fsdp_size in (1, 2):
            with self.subTest(fsdp_size=fsdp_size):
                _, config = self._load(fsdp_size, audit_enabled=True)
                rendered = format_command(build_command(config))

                self.assertEqual(config.model.student_path, STUDENT_PATH)
                self.assertEqual(config.model.primary_teacher_path, TEACHER_PATH)
                self.assertEqual(config.model.math_teacher_path, TEACHER_PATH)
                self.assertEqual(config.model.code_teacher_path, TEACHER_PATH)
                self.assertNotEqual(
                    config.model.student_path,
                    config.model.primary_teacher_path,
                )
                self.assertIsNone(config.model.secondary_teacher_path)
                self.assertEqual(config.model.teacher_model_device, "gpu")

                self.assertEqual(config.trainer.n_gpus_per_node, 2)
                self.assertEqual(config.trainer.nnodes, 1)
                self.assertTrue(config.worker_placement.separate_ref_policy)
                self.assertEqual(
                    config.worker_placement.actor_rollout.n_gpus_per_node,
                    2,
                )
                self.assertEqual(config.worker_placement.actor_rollout.nnodes, 1)
                self.assertEqual(
                    config.worker_placement.ref_policy.n_gpus_per_node,
                    1,
                )
                self.assertEqual(config.worker_placement.ref_policy.nnodes, 1)
                self.assertEqual(
                    int(config.worker_placement.actor_rollout.n_gpus_per_node or 0)
                    + int(config.worker_placement.ref_policy.n_gpus_per_node or 0),
                    3,
                )
                self.assertEqual(config.actor.fsdp_size, fsdp_size)
                self.assertFalse(config.rollout.do_sample)
                self.assertEqual(config.rollout.temperature, 1.0)
                self.assertIn("trainer.resume_mode=disable", config.extra_overrides)

                self.assertIn(
                    f"actor_rollout_ref.actor.fsdp_config.fsdp_size={fsdp_size}",
                    rendered,
                )
                self.assertIn(
                    "+actor_rollout_ref.worker_placement.separate_ref_policy=true",
                    rendered,
                )
                self.assertIn(
                    "+actor_rollout_ref.worker_placement.actor_rollout.n_gpus_per_node=2",
                    rendered,
                )
                self.assertIn(
                    "+actor_rollout_ref.worker_placement.ref_policy.n_gpus_per_node=1",
                    rendered,
                )
                self.assertIn(
                    f"+actor_rollout_ref.ref.model.path={TEACHER_PATH}",
                    rendered,
                )
                self.assertIn(
                    "actor_rollout_ref.rollout.do_sample=False",
                    rendered,
                )
                self.assertIn(
                    "actor_rollout_ref.rollout.temperature=1.0",
                    rendered,
                )
                self.assertIn("trainer.resume_mode=disable", rendered)

    def test_audit_profiles_exercise_accumulation_and_direct_vectors(self) -> None:
        for fsdp_size in (1, 2):
            with self.subTest(fsdp_size=fsdp_size):
                _, config = self._load(fsdp_size, audit_enabled=True)

                self.assertTrue(config.audit.enabled)
                self.assertTrue(config.audit.full_gradient_enabled)
                self.assertEqual(config.audit.full_gradient_freq_steps, 2)
                self.assertEqual(
                    config.audit.full_gradient_micro_batch_size_per_gpu,
                    1,
                )
                self.assertEqual(config.audit.full_gradient_storage_dtype, "bfloat16")
                self.assertEqual(
                    config.audit.full_grad_training_parity_rel_l2_threshold,
                    2e-2,
                )
                self.assertEqual(config.audit.domains, ["math", "code"])
                self.assertEqual(config.actor.ppo_micro_batch_size_per_gpu, 1)
                self.assertEqual(config.data.train_batch_size, 16)
                self.assertEqual(config.trainer.total_training_steps, 4)
                self.assertEqual(config.data.train_batch_size % 2, 0)

    def test_size_one_has_a_matching_audit_off_control(self) -> None:
        _, audit_on = self._load(1, audit_enabled=True)
        _, audit_off = self._load(1, audit_enabled=False)

        self.assertTrue(audit_on.audit.enabled)
        self.assertFalse(audit_off.audit.enabled)
        self.assertEqual(audit_on.actor, audit_off.actor)
        self.assertEqual(audit_on.data, audit_off.data)
        self.assertEqual(audit_on.model, audit_off.model)
        self.assertEqual(audit_on.rollout, audit_off.rollout)
        self.assertEqual(
            audit_on.rollout_correction,
            audit_off.rollout_correction,
        )
        self.assertEqual(audit_on.extra_overrides, audit_off.extra_overrides)
        self.assertEqual(
            audit_on.worker_placement,
            audit_off.worker_placement,
        )
        self.assertEqual(
            replace(
                audit_on.trainer,
                experiment_name="ignored",
                default_local_dir="ignored",
            ),
            replace(
                audit_off.trainer,
                experiment_name="ignored",
                default_local_dir="ignored",
            ),
        )

    def test_four_actor_hybrid_shard_profile_contract(self) -> None:
        _, audit_on = self._load_hybrid(audit_enabled=True)
        _, audit_off = self._load_hybrid(audit_enabled=False)
        rendered = format_command(build_command(audit_on))

        self.assertEqual(audit_on.trainer.n_gpus_per_node, 4)
        self.assertEqual(audit_on.trainer.nnodes, 1)
        self.assertEqual(audit_on.actor.fsdp_size, 2)
        self.assertTrue(audit_on.worker_placement.separate_ref_policy)
        self.assertEqual(
            audit_on.worker_placement.actor_rollout.n_gpus_per_node,
            4,
        )
        self.assertEqual(audit_on.worker_placement.actor_rollout.nnodes, 1)
        self.assertEqual(
            audit_on.worker_placement.ref_policy.n_gpus_per_node,
            1,
        )
        self.assertEqual(audit_on.worker_placement.ref_policy.nnodes, 1)
        total_gpus = (
            int(audit_on.worker_placement.actor_rollout.n_gpus_per_node or 0)
            + int(audit_on.worker_placement.ref_policy.n_gpus_per_node or 0)
        )
        self.assertEqual(total_gpus, 5)
        self.assertEqual(4 // audit_on.actor.fsdp_size, 2)
        self.assertTrue(audit_on.audit.enabled)
        self.assertFalse(audit_off.audit.enabled)
        self.assertTrue(audit_on.audit.full_gradient_enabled)
        self.assertEqual(audit_on.audit.full_gradient_freq_steps, 2)
        self.assertEqual(
            audit_on.audit.full_gradient_micro_batch_size_per_gpu,
            1,
        )
        self.assertEqual(audit_on.audit.full_gradient_storage_dtype, "bfloat16")
        self.assertEqual(
            audit_on.audit.full_grad_training_parity_rel_l2_threshold,
            2e-2,
        )
        self.assertEqual(audit_on.audit.domains, ["math", "code"])
        self.assertEqual(audit_on.actor.ppo_micro_batch_size_per_gpu, 1)
        self.assertEqual(audit_on.data.train_batch_size, 16)
        self.assertEqual(audit_on.trainer.total_training_steps, 4)
        self.assertFalse(audit_on.rollout.do_sample)
        self.assertEqual(audit_on.rollout.temperature, 1.0)
        self.assertIn("trainer.resume_mode=disable", audit_on.extra_overrides)
        self.assertIn(
            "actor_rollout_ref.actor.fsdp_config.fsdp_size=2",
            rendered,
        )
        self.assertIn(
            "+actor_rollout_ref.worker_placement.actor_rollout.n_gpus_per_node=4",
            rendered,
        )
        self.assertIn(
            "+actor_rollout_ref.worker_placement.ref_policy.n_gpus_per_node=1",
            rendered,
        )
        self.assertIn("trainer.resume_mode=disable", rendered)

        for field_name in (
            "runtime",
            "data",
            "model",
            "actor",
            "rollout",
            "rollout_correction",
            "worker_placement",
            "paper_eval",
            "ray_kwargs",
            "extra_overrides",
        ):
            with self.subTest(field=field_name):
                self.assertEqual(
                    getattr(audit_on, field_name),
                    getattr(audit_off, field_name),
                )
        self.assertEqual(
            replace(
                audit_on.trainer,
                experiment_name="ignored",
                default_local_dir="ignored",
            ),
            replace(
                audit_off.trainer,
                experiment_name="ignored",
                default_local_dir="ignored",
            ),
        )

    def test_all_reliability_profiles_use_finite_logprob_temperature(self) -> None:
        for profile in EXPECTED_PROFILE_NAMES:
            with self.subTest(profile=profile):
                config = load_config(f"{MATRIX_PATH}::{profile}")
                self.assertGreater(config.rollout.temperature, 0.0)
                if not config.rollout.do_sample:
                    self.assertIn(
                        "actor_rollout_ref.rollout.do_sample=False",
                        format_command(build_command(config)),
                    )


if __name__ == "__main__":
    unittest.main()
