from __future__ import annotations

import unittest
from pathlib import Path

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


class MOPDProfileTests(unittest.TestCase):
    def assert_a800_profile(
        self,
        config_name: str,
        batch_size: int,
        gpu_count: int,
        tensor_parallel: int,
        cpu_count: int,
        output_name: str,
    ) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / config_name
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertEqual(config.data.train_batch_size, batch_size)
        self.assertEqual(config.data.max_response_length, 16384)
        self.assertEqual(config.actor.ppo_mini_batch_size, batch_size)
        self.assertEqual(config.actor.ppo_micro_batch_size_per_gpu, 1)
        self.assertTrue(config.actor.gradient_checkpointing)
        self.assertEqual(config.actor.fsdp_size, 1)
        self.assertEqual(config.rollout.tensor_model_parallel_size, tensor_parallel)
        self.assertEqual(config.rollout.gpu_memory_utilization, 0.8)
        self.assertEqual(config.trainer.n_gpus_per_node, gpu_count)
        self.assertEqual(config.trainer.total_training_steps, 10)
        self.assertEqual(config.ray_kwargs.ray_init.num_cpus, cpu_count)
        self.assertTrue(config.audit.full_gradient_enabled)
        self.assertTrue(config.audit.sample_gradient_enabled)
        self.assertTrue(config.audit.sample_gradient_norm_enabled)
        self.assertFalse(config.audit.sample_gradient_cos_enabled)
        self.assertIn(f"data.train_batch_size={batch_size}", rendered)
        self.assertIn(f"actor_rollout_ref.actor.ppo_mini_batch_size={batch_size}", rendered)
        self.assertIn(f"trainer.n_gpus_per_node={gpu_count}", rendered)
        self.assertIn(f"actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_parallel}", rendered)
        self.assertIn(f"+mopd_audit.output_dir=audit/{output_name}", rendered)
        self.assertIn(f"trainer.default_local_dir=checkpoints/{output_name}", rendered)

    def test_dual_a800_profile_matches_current_diagnostic_run(self) -> None:
        self.assert_a800_profile(
            "mopd_formal_dual_a800.yaml",
            batch_size=256,
            gpu_count=2,
            tensor_parallel=2,
            cpu_count=8,
            output_name="formal_dual_a800",
        )

    def test_scaled_a800_profiles_keep_audit_semantics(self) -> None:
        cases = [
            ("mopd_formal_4gpu_a800.yaml", 512, 4, 4, 16, "formal_4gpu_a800"),
            ("mopd_formal_8gpu_a800.yaml", 1024, 8, 4, 32, "formal_8gpu_a800"),
        ]

        for config_name, batch_size, gpu_count, tensor_parallel, cpu_count, output_name in cases:
            with self.subTest(config=config_name):
                self.assert_a800_profile(
                    config_name,
                    batch_size=batch_size,
                    gpu_count=gpu_count,
                    tensor_parallel=tensor_parallel,
                    cpu_count=cpu_count,
                    output_name=output_name,
                )

    def test_explicit_rollout_model_lengths_cover_prompt_and_response(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"

        for config_path in sorted(config_dir.glob("*.yaml")):
            with self.subTest(config=config_path.name):
                config = load_config(config_path)
                max_model_len = config.rollout.max_model_len
                if max_model_len is None:
                    continue

                required_context = config.data.max_prompt_length + config.data.max_response_length
                self.assertGreaterEqual(max_model_len, required_context)
