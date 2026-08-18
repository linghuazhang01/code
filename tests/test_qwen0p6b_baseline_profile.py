from __future__ import annotations

import unittest
from pathlib import Path

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / (
        "mopd_qwen0p6b_30b_a3b_instruct_2507_4gpu_math_code_science_"
        "topk32_baseline_b528.yaml"
    )
)


class Qwen0p6bBaselineProfileTests(unittest.TestCase):
    def test_is_one_student_three_teacher_batch_528_profile(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertEqual(config.model.student_path, "../mopd/models/Qwen3-0.6B")
        self.assertEqual(config.data.train_batch_size, 528)
        self.assertEqual(config.actor.ppo_mini_batch_size, 528)
        self.assertEqual(config.actor.ppo_epochs, 1)
        self.assertEqual(config.runtime.slurm_allocation_gpus, 4)
        self.assertEqual(config.worker_placement.actor_rollout.n_gpus_per_node, 1)
        self.assertEqual(config.worker_placement.ref_policy.n_gpus_per_node, 3)
        self.assertEqual(config.trainer.n_gpus_per_node, 1)

    def test_disables_every_reweight_family(self) -> None:
        audit = load_config(CONFIG_PATH).audit

        self.assertFalse(audit.control_token_loss_weighting_enabled)
        self.assertFalse(audit.control_token_online_selection_enabled)
        self.assertEqual(audit.control_token_ids, [])
        self.assertEqual(audit.domain_control_token_ids, {})
        self.assertEqual(audit.control_token_candidate_ids, [])
        self.assertEqual(audit.domain_control_token_candidate_ids, {})
        self.assertFalse(audit.control_token_speed_weighting_enabled)
        self.assertFalse(audit.control_token_phase_gate_enabled)
        self.assertFalse(audit.control_token_span_weighting_enabled)
        self.assertFalse(audit.dynamic_domain_loss_weighting_enabled)
        self.assertFalse(audit.all_domain_shared_token_loss_weighting_enabled)
        config = load_config(CONFIG_PATH)
        self.assertFalse(config.region_dpo.enabled)
        self.assertFalse(config.domain_budgeting.enabled)
        self.assertEqual(
            config.data.domain_sampling_weights,
            {"math": 1.0, "code": 1.0, "science": 1.0},
        )

    def test_preserves_topk32_reverse_kl_and_unique_outputs(self) -> None:
        config = load_config(CONFIG_PATH)
        rendered = format_command(build_command(config))

        self.assertEqual(config.actor.distill_mode, "topk_renormalized_reverse_kl")
        self.assertEqual(config.actor.topk_distill_k, 32)
        self.assertIn("qwen0p6b", config.runtime.wandb_run_id)
        self.assertIn("qwen0p6b", config.audit.output_dir)
        self.assertIn("qwen0p6b", config.trainer.default_local_dir)
        self.assertIn("Qwen3-0.6B", rendered)
        self.assertIn(
            "+mopd_audit.control_token_loss_weighting_enabled=false", rendered
        )


if __name__ == "__main__":
    unittest.main()
