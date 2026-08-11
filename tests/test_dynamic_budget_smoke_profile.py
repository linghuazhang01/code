from __future__ import annotations

import unittest
from pathlib import Path

import torch

from mopd_verl.config_profiles import load_raw_config
from mopd_verl.domain_budgeting_config import parse_domain_budgeting_config
from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config
from mopd_verl.topk_distill import (
    select_teacher_tensor_by_domain,
    topk_distill_loss_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "test_grad_configs"
    / "mopd_dynamic_budget_qwen0p6b_8b_aw2_fsdp2_b16_4step_3gpu_smoke.yaml"
)
EXPECTED_DOMAINS = {"math", "code", "science", "if"}


class DynamicBudgetSmokeProfileTests(unittest.TestCase):
    def test_profile_contract(self) -> None:
        raw = load_raw_config(CONFIG_PATH)
        config = load_config(CONFIG_PATH)
        budget = parse_domain_budgeting_config(raw["domain_budgeting"])
        rendered = format_command(build_command(config))

        self.assertEqual(set(config.data.domain_train_files), EXPECTED_DOMAINS)
        self.assertEqual(set(config.data.domain_sampling_weights), EXPECTED_DOMAINS)
        self.assertEqual(set(config.model.domain_teacher_paths), EXPECTED_DOMAINS)
        self.assertEqual(config.audit.domains, ["math", "code", "science", "if"])
        self.assertEqual(config.data.train_batch_size, 16)
        self.assertEqual(config.data.dataloader_num_workers, 0)
        self.assertTrue(config.data.domain_sampling_replacement)
        self.assertEqual(config.actor.ppo_mini_batch_size, 16)
        self.assertEqual(config.actor.ppo_epochs, 1)
        self.assertEqual(config.actor.fsdp_size, 2)
        self.assertEqual(config.actor.loss_agg_mode, "seq-mean-token-mean")
        self.assertEqual(config.actor.distill_loss_builder, "topk_kl")
        self.assertEqual(config.actor.distill_mode, "topk_renormalized_reverse_kl")
        self.assertEqual(config.actor.topk_distill_kl_direction, "reverse")
        self.assertEqual(config.actor.topk_distill_k, 32)
        self.assertEqual(config.actor.entropy_coeff, 0)
        self.assertEqual(config.actor.kl_loss_coef, 0)
        self.assertFalse(config.audit.enabled)
        self.assertFalse(config.audit.dynamic_domain_loss_weighting_enabled)
        self.assertTrue(config.trainer.val_before_train)
        self.assertEqual(config.trainer.test_freq, 1)
        self.assertEqual(config.trainer.save_freq, 2)
        self.assertEqual(config.trainer.total_training_steps, 4)
        self.assertEqual(config.trainer.n_gpus_per_node, 2)
        self.assertEqual(
            config.worker_placement.actor_rollout.n_gpus_per_node,
            2,
        )
        self.assertEqual(config.worker_placement.ref_policy.n_gpus_per_node, 1)

        self.assertTrue(budget.enabled)
        self.assertEqual(set(budget.domains), EXPECTED_DOMAINS)
        self.assertEqual(set(budget.teacher_scores), EXPECTED_DOMAINS)
        self.assertTrue(budget.teacher_scores_calibrated)
        self.assertEqual(budget.variance_update_freq_steps, 1)
        self.assertEqual(budget.variance_min_samples, 2)
        self.assertEqual(budget.min_samples_per_domain, 2)
        self.assertEqual(budget.gap_normalization_floor, 0.05)

        for override in (
            "+mopd_domain_budgeting.enabled=true",
            "+mopd_domain_budgeting.teacher_scores_calibrated=true",
            "+mopd_domain_budgeting.variance_update_freq_steps=1",
            "+mopd_domain_budgeting.min_samples_per_domain=2",
            "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
            "actor_rollout_ref.actor.policy_loss.distill_mode="
            "topk_renormalized_reverse_kl",
        ):
            self.assertIn(override, rendered)

    def test_forward_kl_renormalization_remains_covered_by_unit_test(self) -> None:
        torch.manual_seed(42)
        teacher_scores = torch.randn(2, 3, 32, dtype=torch.float32)
        student_scores = torch.randn(2, 3, 32, dtype=torch.float32)

        actual = topk_distill_loss_matrix(
            student_topk_log_probs=student_scores,
            teacher_topk_log_probs=teacher_scores,
            mode="topk_renormalized_forward_kl",
            include_tail=False,
            temperature=1.0,
        )
        teacher_log_q = torch.log_softmax(teacher_scores, dim=-1)
        student_log_q = torch.log_softmax(student_scores, dim=-1)
        expected = (teacher_log_q.exp() * (teacher_log_q - student_log_q)).sum(dim=-1)

        torch.testing.assert_close(actual, expected)

    def test_shared_teacher_aliases_return_original_tensor(self) -> None:
        shared = torch.randn(4, 3, 2)
        inputs = {
            "math_teacher_topk_logprobs": shared,
            "code_teacher_topk_logprobs": shared,
            "science_teacher_topk_logprobs": shared,
            "if_teacher_topk_logprobs": shared,
            "opd_teacher": ["math", "code", "science", "if"],
        }

        selected = select_teacher_tensor_by_domain(
            inputs,
            {"multi_teacher_distill": True},
            suffix="topk_logprobs",
        )

        self.assertIs(selected, shared)

    def test_distinct_teacher_tensors_still_select_rows(self) -> None:
        math = torch.zeros(2, 3)
        code = torch.ones(2, 3)
        inputs = {
            "math_teacher_log_prob": math,
            "code_teacher_log_prob": code,
            "opd_teacher": ["math", "code"],
        }

        selected = select_teacher_tensor_by_domain(
            inputs,
            {"multi_teacher_distill": True},
            suffix="log_prob",
        )

        torch.testing.assert_close(selected, torch.stack((math[0], code[1])))


if __name__ == "__main__":
    unittest.main()
