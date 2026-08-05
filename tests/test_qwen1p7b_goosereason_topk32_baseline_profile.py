from __future__ import annotations

from pathlib import Path
import unittest

from mopd_verl.domain_sampling import allocate_domain_batch_counts
from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "mopd_qwen1p7b_nonthinking_goosereason4b_instruct_8gpu_"
    "math_code_science_topk32_reverse_kl_baseline.yaml"
)
STUDENT_PATH = "../models/Qwen3-1.7B"
TEACHER_PATH = "../models/Nemotron-Research-GooseReason-4B-Instruct"


class Qwen1p7BGooseReasonTopK32BaselineProfileTests(unittest.TestCase):
    def test_profile_is_the_non_thinking_sft_to_goosereason_pair(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertEqual(config.model.student_path, STUDENT_PATH)
        self.assertIsNone(config.model.student_base_path)
        self.assertNotIn("-Base", config.model.student_path)
        self.assertFalse(config.data.enable_thinking)
        self.assertEqual(config.model.primary_teacher_path, TEACHER_PATH)
        self.assertIsNone(config.model.secondary_teacher_path)
        self.assertEqual(
            config.model.domain_teacher_paths,
            {
                "math": TEACHER_PATH,
                "code": TEACHER_PATH,
                "science": TEACHER_PATH,
            },
        )

    def test_profile_uses_balanced_three_domain_sampling(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertEqual(config.data.train_batch_size, 504)
        self.assertEqual(config.actor.ppo_mini_batch_size, 504)
        self.assertEqual(
            allocate_domain_batch_counts(
                config.data.train_batch_size,
                config.data.domain_sampling_weights,
            ),
            {"math": 168, "code": 168, "science": 168},
        )
        self.assertTrue(config.rollout.do_sample)
        self.assertEqual(config.rollout.n, 1)
        self.assertEqual(config.rollout.temperature, 1.0)
        self.assertEqual(config.rollout.top_p, 1.0)
        self.assertEqual(config.rollout_correction.rollout_is, "null")
        self.assertIsNone(config.rollout_correction.rollout_rs)
        self.assertFalse(config.rollout_correction.bypass_mode)

    def test_profile_uses_only_topk32_renormalized_reverse_kl(self) -> None:
        config = load_config(CONFIG_PATH)
        actor = config.actor

        self.assertEqual(actor.distill_loss_builder, "topk_kl")
        self.assertEqual(actor.distill_mode, "topk_renormalized_reverse_kl")
        self.assertTrue(actor.topk_distill_enabled)
        self.assertEqual(actor.topk_distill_kl_direction, "reverse")
        self.assertEqual(actor.topk_distill_k, 32)
        self.assertEqual(actor.topk_distill_support_source, "teacher")
        self.assertFalse(actor.topk_distill_tail_bucket)
        self.assertEqual(actor.topk_distill_temperature, 1.0)
        self.assertEqual(actor.topk_distill_loss_weight, 1.0)
        self.assertFalse(actor.teacher_prefix_enabled)
        self.assertFalse(actor.only_reverse_kl_advantages)
        self.assertEqual(actor.kl_loss_coef, 0)
        self.assertEqual(actor.entropy_coeff, 0)

    def test_profile_disables_mopd_tlpd_method_paths(self) -> None:
        config = load_config(CONFIG_PATH)
        audit = config.audit

        self.assertFalse(audit.enabled)
        self.assertFalse(audit.full_gradient_enabled)
        self.assertFalse(audit.sample_gradient_enabled)
        self.assertFalse(audit.token_gradient_enabled)
        self.assertFalse(audit.token_gap_enabled)
        self.assertFalse(audit.entropy_enabled)
        self.assertFalse(audit.logp_vector_enabled)
        self.assertFalse(audit.logp_abs_vector_enabled)
        self.assertFalse(audit.dynamic_domain_loss_weighting_enabled)
        self.assertFalse(audit.control_token_loss_weighting_enabled)
        self.assertFalse(audit.all_domain_shared_token_loss_weighting_enabled)
        self.assertFalse(config.domain_budgeting.enabled)
        self.assertFalse(config.paper_eval.enabled)
        self.assertEqual(
            config.extra_overrides,
            [
                "custom_reward_function.path=mopd_verl/mixed_reward.py",
                "custom_reward_function.name=compute_score",
                "trainer.resume_mode=disable",
            ],
        )
        self.assertIn("baseline", config.trainer.experiment_name)
        self.assertIn("baseline", config.trainer.default_local_dir)
        for forbidden_name in ("reweight", "dynamic", "auditall"):
            self.assertNotIn(forbidden_name, config.trainer.experiment_name)
            self.assertNotIn(forbidden_name, config.trainer.default_local_dir)

    def test_rendered_command_contains_baseline_and_no_method_overrides(self) -> None:
        config = load_config(CONFIG_PATH)
        rendered = format_command(build_command(config))

        self.assertIn(
            "actor_rollout_ref.actor.policy_loss.distill_mode="
            "topk_renormalized_reverse_kl",
            rendered,
        )
        self.assertIn(
            "actor_rollout_ref.actor.policy_loss.topk_distill_k=32",
            rendered,
        )
        self.assertIn(
            "actor_rollout_ref.actor.policy_loss.topk_distill_tail_bucket=false",
            rendered,
        )
        self.assertIn("data.apply_chat_template_kwargs.enable_thinking=False", rendered)
        self.assertIn("algorithm.rollout_correction.rollout_is=null", rendered)
        self.assertNotIn("algorithm.rollout_correction.rollout_is=token", rendered)
        self.assertIn("trainer.resume_mode=disable", rendered)
        self.assertNotIn("+mopd_audit.", rendered)
        self.assertNotIn("+mopd_domain_budgeting.", rendered)


if __name__ == "__main__":
    unittest.main()
