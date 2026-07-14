from __future__ import annotations

import unittest
from pathlib import Path

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


class FSDPAccumulationParityProfileTests(unittest.TestCase):
    def test_profile_compares_replay_with_normal_backward(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "mopd_fsdp_accumulation_parity_2gpu_b16_2step_smoke.yaml"
        )
        config = load_config(config_path)
        command = format_command(build_command(config))

        self.assertEqual(config.trainer.n_gpus_per_node, 2)
        self.assertEqual(config.actor.fsdp_size, 1)
        self.assertFalse(config.worker_placement.separate_ref_policy)
        self.assertEqual(config.model.student_path, "../models/Qwen3-4B")
        self.assertEqual(config.model.math_teacher_path, "../models/Qwen3-0.6B")
        self.assertEqual(config.model.code_teacher_path, "../models/Qwen3-0.6B")
        self.assertEqual(config.data.train_batch_size, 16)
        self.assertEqual(config.data.max_response_length, 128)
        self.assertEqual(config.actor.ppo_mini_batch_size, 16)
        self.assertEqual(config.actor.ppo_micro_batch_size_per_gpu, 1)
        self.assertEqual(config.rollout.name, "hf")
        self.assertEqual(config.rollout.tensor_model_parallel_size, 1)
        self.assertEqual(config.trainer.total_training_steps, 2)
        self.assertTrue(config.audit.full_gradient_enabled)
        self.assertEqual(config.audit.full_gradient_freq_steps, 1)
        self.assertEqual(config.audit.full_grad_training_parity_freq_steps, 1)
        self.assertTrue(config.audit.sequence_masked_target_enabled)
        self.assertTrue(config.audit.sequence_masked_target_use_as_primary)
        self.assertFalse(hasattr(config.audit, "training_gradient_from_domain_sum_enabled"))
        self.assertFalse(config.audit.sample_gradient_enabled)
        self.assertFalse(config.audit.token_gradient_enabled)

        self.assertIn("trainer.n_gpus_per_node=2", command)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.fsdp_size=1", command)
        self.assertIn("+mopd_audit.full_gradient_freq_steps=1", command)
        self.assertIn("+mopd_audit.full_grad_training_parity_freq_steps=1", command)
        self.assertNotIn("training_gradient_from_domain_sum", command)

    def test_normal_backward_and_fsdp_mesh_contracts_are_explicit(self) -> None:
        root = Path(__file__).resolve().parents[1]
        actor_source = (
            root / "third_party" / "verl" / "verl" / "workers" / "actor" / "dp_actor.py"
        ).read_text(encoding="utf-8")
        worker_source = (
            root / "third_party" / "verl" / "verl" / "workers" / "fsdp_workers.py"
        ).read_text(encoding="utf-8")

        self.assertIn("build_actor_micro_batch_loss(", actor_source)
        self.assertIn("audit.run_before_training(", actor_source)
        self.assertIn("audit.compare_training_gradient()", actor_source)
        self.assertNotIn("finalize_fsdp", actor_source)
        self.assertNotIn("no_sync", actor_source)
        self.assertIn(
            'self.config.ref.fsdp_config if self.role == "ref"',
            worker_source,
        )


if __name__ == "__main__":
    unittest.main()
