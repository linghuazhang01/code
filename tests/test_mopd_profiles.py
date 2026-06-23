from __future__ import annotations

import unittest
from pathlib import Path

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


class MOPDProfileTests(unittest.TestCase):
    FORMAL_PROFILE_SPECS = (
        ("all", 2, 256, 2, 8),
        ("all", 4, 512, 4, 16),
        ("all", 8, 1024, 4, 32),
        ("loss_only", 2, 256, 2, 8),
        ("loss_only", 4, 512, 4, 16),
        ("loss_only", 8, 1024, 4, 32),
        ("off", 2, 256, 2, 8),
        ("off", 4, 512, 4, 16),
        ("off", 8, 1024, 4, 32),
    )

    def assert_formal_profile(
        self,
        audit_mode: str,
        gpu_count: int,
        train_batch_size: int,
        tensor_parallel_size: int,
        cpu_count: int,
    ) -> None:
        suffix = f"{gpu_count}gpu"
        config_name = f"mopd_formal_audit_{audit_mode}_{suffix}.yaml"
        output_name = f"formal_audit_{audit_mode}_{suffix}"
        config_path = Path(__file__).resolve().parents[1] / "configs" / config_name
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertEqual(config.data.train_batch_size, train_batch_size)
        self.assertEqual(config.data.max_response_length, 16384)
        self.assertEqual(config.actor.ppo_mini_batch_size, train_batch_size)
        self.assertEqual(config.actor.ppo_micro_batch_size_per_gpu, 1)
        self.assertTrue(config.actor.gradient_checkpointing)
        self.assertEqual(config.actor.fsdp_size, 1)
        self.assertEqual(config.rollout.tensor_model_parallel_size, tensor_parallel_size)
        self.assertEqual(config.rollout.gpu_memory_utilization, 0.6 if audit_mode in {"all", "loss_only"} else 0.8)
        self.assertEqual(config.rollout.max_num_seqs, 8 if gpu_count == 2 else 16)
        self.assertEqual(config.trainer.n_gpus_per_node, gpu_count)
        self.assertEqual(config.trainer.total_training_steps, 10)
        self.assertEqual(config.ray_kwargs.ray_init.num_cpus, cpu_count)
        self.assertTrue(config.actor.topk_distill_enabled)
        self.assertEqual(config.actor.topk_distill_support_source, "teacher")
        self.assertEqual(config.actor.topk_distill_k, 5)
        self.assertFalse(config.actor.topk_distill_tail_bucket)
        self.assertIn(f"data.train_batch_size={train_batch_size}", rendered)
        self.assertIn(f"actor_rollout_ref.actor.ppo_mini_batch_size={train_batch_size}", rendered)
        self.assertIn(f"trainer.n_gpus_per_node={gpu_count}", rendered)
        self.assertIn(f"actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_parallel_size}", rendered)
        if audit_mode in {"all", "loss_only"}:
            self.assertIn(f"+mopd_audit.output_dir=audit/{output_name}", rendered)
        else:
            self.assertNotIn("+mopd_audit.", rendered)
        self.assertIn(f"trainer.default_local_dir=checkpoints/{output_name}", rendered)

    def test_formal_profiles_have_expected_gpu_scaling(self) -> None:
        for spec in self.FORMAL_PROFILE_SPECS:
            with self.subTest(audit_mode=spec[0], gpu_count=spec[1]):
                self.assert_formal_profile(*spec)

    def test_audit_all_profile_enables_all_audit_families(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"

        for gpu_count in (2, 4, 8):
            with self.subTest(gpu_count=gpu_count):
                config = load_config(config_dir / f"mopd_formal_audit_all_{gpu_count}gpu.yaml")
                self.assertTrue(config.audit.enabled)
                self.assertTrue(config.audit.full_gradient_enabled)
                self.assertTrue(config.audit.sample_gradient_enabled)
                self.assertTrue(config.audit.sample_gradient_norm_enabled)
                self.assertTrue(config.audit.sample_gradient_cos_enabled)
                self.assertTrue(config.audit.token_gap_enabled)
                self.assertTrue(config.audit.token_gap_vocab_vector_enabled)
                self.assertTrue(config.audit.entropy_enabled)
                self.assertTrue(config.audit.entropy_vocab_vector_enabled)
                self.assertTrue(config.audit.token_conflict_enabled)
                self.assertTrue(config.audit.token_gradient_enabled)
                self.assertTrue(config.audit.token_gradient_gap_selection_enabled)
                self.assertTrue(config.audit.token_gradient_gap_abs_selection_enabled)
                self.assertTrue(config.audit.token_gradient_loss_abs_selection_enabled)
                self.assertEqual(config.audit.token_gradient_top_k, 100)
                self.assertEqual(config.audit.token_gradient_top_p, 0.10)

    def test_loss_only_profile_uses_only_loss_token_gradient_selection(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"

        for gpu_count in (2, 4, 8):
            with self.subTest(gpu_count=gpu_count):
                config = load_config(config_dir / f"mopd_formal_audit_loss_only_{gpu_count}gpu.yaml")
                rendered = format_command(build_command(config))

                self.assertTrue(config.audit.enabled)
                self.assertTrue(config.audit.full_gradient_enabled)
                self.assertTrue(config.audit.sample_gradient_enabled)
                self.assertTrue(config.audit.sample_gradient_norm_enabled)
                self.assertTrue(config.audit.sample_gradient_cos_enabled)
                self.assertTrue(config.audit.token_gap_enabled)
                self.assertTrue(config.audit.token_gap_vocab_vector_enabled)
                self.assertTrue(config.audit.entropy_enabled)
                self.assertTrue(config.audit.entropy_vocab_vector_enabled)
                self.assertTrue(config.audit.token_conflict_enabled)
                self.assertTrue(config.audit.token_gradient_enabled)
                self.assertFalse(config.audit.token_gradient_gap_selection_enabled)
                self.assertFalse(config.audit.token_gradient_gap_abs_selection_enabled)
                self.assertTrue(config.audit.token_gradient_loss_abs_selection_enabled)
                self.assertEqual(config.audit.token_gradient_top_k, 100)
                self.assertEqual(config.audit.token_gradient_top_p, 0.10)
                self.assertIn("+mopd_audit.token_gradient_gap_selection_enabled=false", rendered)
                self.assertIn("+mopd_audit.token_gradient_gap_abs_selection_enabled=false", rendered)
                self.assertIn("+mopd_audit.token_gradient_loss_abs_selection_enabled=true", rendered)

    def test_metric_smoke_profile_tracks_all_audit_outputs(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_audit_all_smoke.yaml"
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertEqual(config.data.train_batch_size, 32)
        self.assertEqual(config.data.max_response_length, 16384)
        self.assertEqual(config.data.val_batch_size, 32)
        self.assertEqual(config.actor.ppo_mini_batch_size, 32)
        self.assertEqual(config.rollout.tensor_model_parallel_size, 2)
        self.assertEqual(config.trainer.n_gpus_per_node, 2)
        self.assertEqual(config.trainer.total_epochs, 1)
        self.assertEqual(config.trainer.total_training_steps, 1)
        self.assertEqual(config.audit.output_dir, "audit/formal_audit_all_smoke")
        self.assertEqual(config.paper_eval.output_dir, "eval_outputs/paper_suite/formal_audit_all_smoke")
        self.assertEqual(config.trainer.default_local_dir, "checkpoints/formal_audit_all_smoke")
        self.assertTrue(config.audit.enabled)
        self.assertTrue(config.audit.full_gradient_enabled)
        self.assertTrue(config.audit.sample_gradient_enabled)
        self.assertTrue(config.audit.sample_gradient_cos_enabled)
        self.assertTrue(config.audit.token_gap_vocab_vector_enabled)
        self.assertTrue(config.audit.entropy_vocab_vector_enabled)
        self.assertTrue(config.audit.token_gradient_enabled)
        self.assertTrue(config.audit.token_gradient_gap_selection_enabled)
        self.assertTrue(config.audit.token_gradient_gap_abs_selection_enabled)
        self.assertTrue(config.audit.token_gradient_loss_abs_selection_enabled)
        self.assertEqual(config.audit.token_gradient_top_k, 100)
        self.assertEqual(config.audit.token_gradient_top_p, 0.10)
        self.assertIsNone(config.audit.token_gap_vocab_size)
        self.assertIn("data.train_batch_size=32", rendered)
        self.assertIn("data.max_response_length=16384", rendered)
        self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=32", rendered)
        self.assertIn("+mopd_audit.output_dir=audit/formal_audit_all_smoke", rendered)
        self.assertIn("+mopd_audit.token_gradient_gap_selection_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_gap_abs_selection_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_loss_abs_selection_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_top_k=100", rendered)
        self.assertNotIn("token_gradient_top_k_per_sample", rendered)
        self.assertNotIn("token_gradient_max_samples_per_domain", rendered)
        self.assertNotIn("token_gradient_min_teacher_diff", rendered)

    def test_metric_smoke_profile_can_use_loss_only_token_gradient_selection(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_audit_loss_only_smoke.yaml"
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertEqual(config.data.train_batch_size, 32)
        self.assertEqual(config.data.max_response_length, 16384)
        self.assertEqual(config.data.val_batch_size, 32)
        self.assertEqual(config.actor.ppo_mini_batch_size, 32)
        self.assertEqual(config.rollout.tensor_model_parallel_size, 2)
        self.assertEqual(config.trainer.n_gpus_per_node, 2)
        self.assertEqual(config.trainer.total_epochs, 1)
        self.assertEqual(config.trainer.total_training_steps, 1)
        self.assertEqual(config.audit.output_dir, "audit/formal_audit_loss_only_smoke")
        self.assertEqual(config.paper_eval.output_dir, "eval_outputs/paper_suite/formal_audit_loss_only_smoke")
        self.assertEqual(config.trainer.default_local_dir, "checkpoints/formal_audit_loss_only_smoke")
        self.assertTrue(config.audit.enabled)
        self.assertTrue(config.audit.full_gradient_enabled)
        self.assertTrue(config.audit.sample_gradient_enabled)
        self.assertTrue(config.audit.sample_gradient_cos_enabled)
        self.assertTrue(config.audit.token_gap_vocab_vector_enabled)
        self.assertTrue(config.audit.entropy_vocab_vector_enabled)
        self.assertTrue(config.audit.token_gradient_enabled)
        self.assertFalse(config.audit.token_gradient_gap_selection_enabled)
        self.assertFalse(config.audit.token_gradient_gap_abs_selection_enabled)
        self.assertTrue(config.audit.token_gradient_loss_abs_selection_enabled)
        self.assertEqual(config.audit.token_gradient_top_k, 100)
        self.assertEqual(config.audit.token_gradient_top_p, 0.10)
        self.assertIsNone(config.audit.token_gap_vocab_size)
        self.assertIn("data.train_batch_size=32", rendered)
        self.assertIn("data.max_response_length=16384", rendered)
        self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=32", rendered)
        self.assertIn("+mopd_audit.output_dir=audit/formal_audit_loss_only_smoke", rendered)
        self.assertIn("+mopd_audit.token_gradient_gap_selection_enabled=false", rendered)
        self.assertIn("+mopd_audit.token_gradient_gap_abs_selection_enabled=false", rendered)
        self.assertIn("+mopd_audit.token_gradient_loss_abs_selection_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_top_k=100", rendered)
        self.assertNotIn("token_gradient_top_k_per_sample", rendered)
        self.assertNotIn("token_gradient_max_samples_per_domain", rendered)
        self.assertNotIn("token_gradient_min_teacher_diff", rendered)

    def test_audit_off_profile_disables_audit_families(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"

        for gpu_count in (2, 4, 8):
            with self.subTest(gpu_count=gpu_count):
                config = load_config(config_dir / f"mopd_formal_audit_off_{gpu_count}gpu.yaml")
                self.assertFalse(config.audit.enabled)
                self.assertFalse(config.audit.log_sample_level)
                self.assertFalse(config.audit.log_validation_metrics)
                self.assertFalse(config.audit.full_gradient_enabled)
                self.assertFalse(config.audit.sample_gradient_enabled)
                self.assertFalse(config.audit.sample_gradient_norm_enabled)
                self.assertFalse(config.audit.sample_gradient_cos_enabled)
                self.assertFalse(config.audit.sample_gradient_log_sample_level)
                self.assertFalse(config.audit.token_gap_enabled)
                self.assertFalse(config.audit.token_gap_vocab_vector_enabled)
                self.assertFalse(config.audit.entropy_enabled)
                self.assertFalse(config.audit.entropy_vocab_vector_enabled)
                self.assertFalse(config.audit.token_conflict_enabled)
                self.assertFalse(config.audit.token_gradient_enabled)

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
