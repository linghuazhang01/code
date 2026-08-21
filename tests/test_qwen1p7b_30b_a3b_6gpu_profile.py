from __future__ import annotations

import unittest
from pathlib import Path

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "mopd_qwen1p7b_30b_a3b_instruct_2507_6gpu_math_code_science_topk32_baseline_525.yaml"
)


class Qwen1p7bFrom30bA3bSixGpuProfileTests(unittest.TestCase):
    def test_uses_the_server_assets_and_fresh_tracking(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertEqual(
            config.runtime.python_bin,
            "/home/shuang_qiu/env/miniconda3/envs/mopd-verl/bin/python",
        )
        self.assertEqual(config.runtime.env_file, "../mopd/code/.env.local")
        self.assertIsNone(config.runtime.wandb_run_id)
        self.assertEqual(config.runtime.wandb_resume, "never")
        self.assertEqual(config.model.student_path, "../mopd/models/Qwen3-1.7B")
        self.assertEqual(
            config.model.primary_teacher_path,
            "../mopd/models/Qwen3-30B-A3B-Instruct-2507",
        )
        self.assertIn("trainer.resume_mode=disable", config.extra_overrides)

    def test_six_gpu_topology_is_internally_consistent(self) -> None:
        config = load_config(CONFIG_PATH)
        actor_gpus = config.worker_placement.actor_rollout.n_gpus_per_node
        teacher_gpus = config.worker_placement.ref_policy.n_gpus_per_node

        self.assertTrue(config.worker_placement.separate_ref_policy)
        self.assertEqual(actor_gpus, 5)
        self.assertEqual(teacher_gpus, 1)
        self.assertEqual(actor_gpus + teacher_gpus, 6)
        self.assertEqual(config.trainer.n_gpus_per_node, actor_gpus)
        self.assertEqual(config.rollout.tensor_model_parallel_size, 1)
        self.assertEqual(
            actor_gpus % config.rollout.tensor_model_parallel_size,
            0,
        )

    def test_balanced_batch_matches_the_actor_world_size(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertEqual(config.data.train_batch_size, 525)
        self.assertEqual(config.actor.ppo_mini_batch_size, 525)
        self.assertEqual(config.data.domain_sampling_weights, {
            "math": 1,
            "code": 1,
            "science": 1,
        })
        self.assertEqual(config.data.train_batch_size // 3, 175)
        self.assertEqual(config.data.train_batch_size % 5, 0)

    def test_keeps_topk32_reverse_kl_without_dynamic_reweighting(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertTrue(config.actor.topk_distill_enabled)
        self.assertEqual(config.actor.topk_distill_k, 32)
        self.assertEqual(config.actor.topk_distill_kl_direction, "reverse")
        self.assertEqual(
            config.actor.distill_mode,
            "topk_renormalized_reverse_kl",
        )
        self.assertFalse(config.actor.topk_distill_tail_bucket)
        self.assertFalse(config.audit.dynamic_domain_loss_weighting_enabled)

    def test_audit_contract_matches_the_declared_profile(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertTrue(config.audit.enabled)
        self.assertTrue(config.audit.full_gradient_enabled)
        self.assertTrue(config.audit.token_gap_enabled)
        self.assertTrue(config.audit.entropy_enabled)
        self.assertFalse(config.audit.sample_gradient_enabled)
        self.assertFalse(config.audit.token_gradient_enabled)

    def test_rendered_command_contains_the_fixed_contract(self) -> None:
        config = load_config(CONFIG_PATH)
        rendered = format_command(build_command(config))

        expected_fragments = (
            "data.train_batch_size=525",
            "actor_rollout_ref.actor.ppo_mini_batch_size=525",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "trainer.n_gpus_per_node=5",
            "+actor_rollout_ref.worker_placement.actor_rollout.n_gpus_per_node=5",
            "+actor_rollout_ref.worker_placement.ref_policy.n_gpus_per_node=1",
            "trainer.resume_mode=disable",
        )
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, rendered)


if __name__ == "__main__":
    unittest.main()
