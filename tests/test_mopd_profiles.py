from __future__ import annotations

import unittest
from pathlib import Path

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


class MOPDProfileTests(unittest.TestCase):
    FORMAL_PROFILE_SPECS = (
        ("all", 2, 256, 2, 0.7, 8, 8),
        ("all", 4, 512, 4, 0.7, 16, 16),
        ("all", 6, 768, 2, 0.6, 24, 24),
        ("loss_only", 2, 256, 2, 0.7, 8, 8),
        ("loss_only", 4, 512, 4, 0.7, 16, 16),
        ("loss_only", 6, 768, 2, 0.6, 24, 24),
        ("loss_only", 8, 1024, 4, 0.7, 16, 32),
        ("off", 2, 256, 2, 0.7, 8, 8),
        ("off", 4, 512, 4, 0.7, 16, 16),
        ("off", 6, 768, 2, 0.6, 24, 24),
        ("off", 8, 1024, 4, 0.7, 16, 32),
    )

    def assert_formal_profile(
        self,
        audit_mode: str,
        gpu_count: int,
        train_batch_size: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        cpu_count: int,
    ) -> None:
        suffix = f"{gpu_count}gpu"
        config_name = f"mopd_formal_audit_{audit_mode}_{suffix}.yaml"
        output_name = f"formal_audit_{audit_mode}_{suffix}"
        config_path = Path(__file__).resolve().parents[1] / "configs" / config_name
        config = load_config(config_path)
        rendered = format_command(build_command(config))
        expected_steps = 200
        expected_save_freq = 5

        self.assertEqual(config.data.train_batch_size, train_batch_size)
        expected_response_length = 10240 if audit_mode == "loss_only" and gpu_count == 6 else 16384
        self.assertEqual(config.data.max_response_length, expected_response_length)
        self.assertEqual(config.actor.ppo_mini_batch_size, train_batch_size)
        self.assertEqual(config.actor.ppo_micro_batch_size_per_gpu, 1)
        self.assertTrue(config.actor.gradient_checkpointing)
        expected_fsdp_size = 2 if audit_mode == "loss_only" and gpu_count == 6 else 1
        self.assertEqual(config.actor.fsdp_size, expected_fsdp_size)
        self.assertEqual(config.rollout.tensor_model_parallel_size, tensor_parallel_size)
        self.assertEqual(config.rollout.gpu_memory_utilization, gpu_memory_utilization)
        self.assertEqual(config.rollout.max_num_seqs, max_num_seqs)
        expected_max_model_len = 12288 if audit_mode == "loss_only" and gpu_count == 6 else None
        self.assertEqual(config.rollout.max_model_len, expected_max_model_len)
        self.assertEqual(config.trainer.n_gpus_per_node, gpu_count)
        self.assertEqual(config.trainer.total_training_steps, expected_steps)
        self.assertEqual(config.trainer.save_freq, expected_save_freq)
        self.assertEqual(config.ray_kwargs.ray_init.num_cpus, cpu_count)
        self.assertIsNone(config.audit.max_samples_per_domain)
        self.assertTrue(config.actor.topk_distill_enabled)
        self.assertEqual(config.actor.topk_distill_support_source, "teacher")
        self.assertEqual(config.actor.topk_distill_k, 32)
        self.assertFalse(config.actor.topk_distill_tail_bucket)
        self.assertIn(f"data.train_batch_size={train_batch_size}", rendered)
        self.assertIn(f"data.max_response_length={expected_response_length}", rendered)
        self.assertIn(f"actor_rollout_ref.actor.ppo_mini_batch_size={train_batch_size}", rendered)
        self.assertIn(f"trainer.n_gpus_per_node={gpu_count}", rendered)
        self.assertIn(f"trainer.save_freq={expected_save_freq}", rendered)
        self.assertIn(f"trainer.total_training_steps={expected_steps}", rendered)
        self.assertIn(f"actor_rollout_ref.actor.fsdp_config.fsdp_size={expected_fsdp_size}", rendered)
        self.assertIn(f"actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_parallel_size}", rendered)
        if expected_max_model_len is not None:
            self.assertIn(f"actor_rollout_ref.rollout.max_model_len={expected_max_model_len}", rendered)
        if audit_mode in {"all", "loss_only"}:
            self.assertIn(f"+mopd_audit.output_dir=audit/{output_name}", rendered)
            self.assertIn("+mopd_audit.max_samples_per_domain=null", rendered)
            self.assertNotIn("token_conflict", rendered)
        else:
            self.assertNotIn("+mopd_audit.", rendered)
        self.assertIn(f"trainer.default_local_dir=checkpoints/{output_name}", rendered)

    def test_formal_profiles_have_expected_gpu_scaling(self) -> None:
        for spec in self.FORMAL_PROFILE_SPECS:
            with self.subTest(audit_mode=spec[0], gpu_count=spec[1]):
                self.assert_formal_profile(*spec)

    def test_audit_all_profile_keeps_observation_metrics_and_domain_gradient(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"

        for gpu_count in (2, 4, 6, 8):
            with self.subTest(gpu_count=gpu_count):
                config = load_config(config_dir / f"mopd_formal_audit_all_{gpu_count}gpu.yaml")
                self.assertTrue(config.audit.enabled)
                self.assertTrue(config.audit.full_gradient_enabled)
                self.assertEqual(config.actor.fsdp_size, 1)
                self.assertFalse(config.audit.sample_gradient_enabled)
                self.assertTrue(config.audit.sample_gradient_norm_enabled)
                self.assertTrue(config.audit.sample_gradient_cos_enabled)
                self.assertTrue(config.audit.token_gap_enabled)
                self.assertTrue(config.audit.token_gap_vocab_vector_enabled)
                self.assertTrue(config.audit.entropy_enabled)
                self.assertTrue(config.audit.entropy_vocab_vector_enabled)
                self.assertFalse(config.audit.token_gradient_enabled)
                self.assertTrue(config.audit.token_gradient_gap_selection_enabled)
                self.assertTrue(config.audit.token_gradient_gap_abs_selection_enabled)
                self.assertTrue(config.audit.token_gradient_loss_abs_selection_enabled)
                self.assertIsNone(config.audit.max_samples_per_domain)
                self.assertEqual(config.audit.token_gradient_top_k, 100)
                self.assertEqual(config.audit.token_gradient_top_p, 0.10)

    def test_eight_gpu_opd_profile_uses_split_placement_and_audit_only_ce(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "mopd_formal_audit_all_8gpu.yaml"
        )
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertEqual(config.actor.distill_loss_builder, "policy_gradient")
        self.assertEqual(config.actor.distill_mode, "chosen_token_policy_gradient")
        self.assertFalse(config.actor.topk_distill_enabled)
        self.assertEqual(config.actor.topk_distill_loss_weight, 0.0)
        self.assertEqual(config.trainer.n_gpus_per_node, 6)
        self.assertEqual(config.worker_placement.actor_rollout.n_gpus_per_node, 6)
        self.assertEqual(config.worker_placement.ref_policy.n_gpus_per_node, 2)
        self.assertTrue(config.audit.topk_teacher_student_cross_entropy_vocab_enabled)
        self.assertEqual(config.audit.topk_teacher_student_cross_entropy_vocab_freq_steps, 1)
        self.assertEqual(config.audit.topk_teacher_student_cross_entropy_k, 32)
        self.assertFalse(config.audit.topk_teacher_student_cross_entropy_include_tail)
        self.assertTrue(config.audit.logp_abs_vector_enabled)
        self.assertEqual(config.audit.logp_abs_vector_freq_steps, 1)
        self.assertIn(
            "+mopd_audit.topk_teacher_student_cross_entropy_vocab_enabled=true",
            rendered,
        )
        self.assertIn("+mopd_audit.logp_abs_vector_enabled=true", rendered)

    def test_eight_gpu_post_training_teacher_profile_contract(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "mopd_qwen4b_rl_teacher_8gpu_math_code.yaml"
        )
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertEqual(
            config.model.math_teacher_path,
            "../models/Qwen3-4B-Non-Thinking-RL-Math-Step500",
        )
        self.assertEqual(
            config.model.code_teacher_path,
            "../models/Qwen3-4B-Non-Thinking-RL-Code-Step300",
        )
        self.assertEqual(
            config.model.domain_teacher_paths,
            {
                "math": "../models/Qwen3-4B-Non-Thinking-RL-Math-Step500",
                "code": "../models/Qwen3-4B-Non-Thinking-RL-Code-Step300",
            },
        )
        self.assertEqual(config.data.train_batch_size, 768)
        self.assertEqual(config.actor.ppo_mini_batch_size, 768)
        self.assertEqual(config.trainer.n_gpus_per_node, 6)
        self.assertEqual(config.worker_placement.actor_rollout.n_gpus_per_node, 6)
        self.assertEqual(config.worker_placement.ref_policy.n_gpus_per_node, 2)
        self.assertEqual(config.audit.full_gradient_freq_steps, 2)
        self.assertTrue(config.audit.entropy_vocab_vector_enabled)
        self.assertTrue(config.audit.logp_vector_enabled)
        self.assertTrue(config.audit.logp_abs_vector_enabled)
        self.assertTrue(config.audit.vocab_per_occurrence_mean_vector_enabled)
        self.assertTrue(config.audit.logp_vocab_per_occurrence_mean_vector_enabled)
        self.assertTrue(config.audit.logp_abs_vocab_per_occurrence_mean_vector_enabled)
        self.assertTrue(config.audit.entropy_vocab_per_occurrence_mean_vector_enabled)
        self.assertIn("+mopd_audit.logp_vector_enabled=true", rendered)

    def test_loss_only_profile_keeps_loss_selection_without_nested_gradient_replay(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"

        for gpu_count in (2, 4, 6, 8):
            with self.subTest(gpu_count=gpu_count):
                config = load_config(config_dir / f"mopd_formal_audit_loss_only_{gpu_count}gpu.yaml")
                rendered = format_command(build_command(config))
                fsdp2_sequence_replay_profile = gpu_count == 6
                expected_token_gradient_top_p = 0.15 if fsdp2_sequence_replay_profile else 0.10

                self.assertTrue(config.audit.enabled)
                self.assertTrue(config.audit.full_gradient_enabled)
                self.assertEqual(config.actor.fsdp_size, 2 if fsdp2_sequence_replay_profile else 1)
                self.assertFalse(config.audit.sample_gradient_enabled)
                self.assertEqual(config.audit.sample_gradient_norm_enabled, not fsdp2_sequence_replay_profile)
                self.assertEqual(config.audit.sample_gradient_cos_enabled, not fsdp2_sequence_replay_profile)
                self.assertEqual(
                    config.audit.sample_gradient_log_sample_level,
                    not fsdp2_sequence_replay_profile,
                )
                self.assertTrue(config.audit.token_gap_enabled)
                self.assertTrue(config.audit.token_gap_vocab_vector_enabled)
                self.assertTrue(config.audit.entropy_enabled)
                self.assertTrue(config.audit.entropy_vocab_vector_enabled)
                self.assertFalse(config.audit.token_gradient_enabled)
                self.assertFalse(config.audit.token_gradient_gap_selection_enabled)
                self.assertFalse(config.audit.token_gradient_gap_abs_selection_enabled)
                self.assertTrue(config.audit.token_gradient_loss_abs_selection_enabled)
                self.assertIsNone(config.audit.max_samples_per_domain)
                self.assertEqual(config.audit.token_gradient_top_k, 100)
                self.assertEqual(config.audit.token_gradient_top_p, expected_token_gradient_top_p)
                self.assertIn("+mopd_audit.token_gradient_gap_selection_enabled=false", rendered)
                self.assertIn("+mopd_audit.token_gradient_gap_abs_selection_enabled=false", rendered)
                self.assertIn("+mopd_audit.token_gradient_loss_abs_selection_enabled=true", rendered)
                self.assertIn(
                    f"+mopd_audit.token_gradient_top_p={expected_token_gradient_top_p}",
                    rendered,
                )
                if fsdp2_sequence_replay_profile:
                    self.assertIn("actor_rollout_ref.actor.fsdp_config.fsdp_size=2", rendered)
                    self.assertIn("+mopd_audit.sequence_masked_target_use_as_primary=true", rendered)
                    self.assertIn("+mopd_audit.sample_gradient_enabled=false", rendered)
                    self.assertIn("+mopd_audit.full_gradient_direct_recompute_enabled=false", rendered)

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
        self.assertFalse(config.audit.sample_gradient_enabled)
        self.assertTrue(config.audit.sample_gradient_cos_enabled)
        self.assertTrue(config.audit.token_gap_vocab_vector_enabled)
        self.assertTrue(config.audit.entropy_vocab_vector_enabled)
        self.assertFalse(config.audit.token_gradient_enabled)
        self.assertTrue(config.audit.token_gradient_gap_selection_enabled)
        self.assertTrue(config.audit.token_gradient_gap_abs_selection_enabled)
        self.assertTrue(config.audit.token_gradient_loss_abs_selection_enabled)
        self.assertIsNone(config.audit.max_samples_per_domain)
        self.assertEqual(config.audit.token_gradient_top_k, 100)
        self.assertEqual(config.audit.token_gradient_top_p, 0.10)
        self.assertIsNone(config.audit.token_gap_vocab_size)
        self.assertIn("data.train_batch_size=32", rendered)
        self.assertIn("data.max_response_length=16384", rendered)
        self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=32", rendered)
        self.assertIn("+mopd_audit.output_dir=audit/formal_audit_all_smoke", rendered)
        self.assertIn("+mopd_audit.max_samples_per_domain=null", rendered)
        self.assertIn("+mopd_audit.token_gradient_gap_selection_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_gap_abs_selection_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_loss_abs_selection_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_top_k=100", rendered)
        self.assertNotIn("token_conflict", rendered)
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
        self.assertFalse(config.audit.sample_gradient_enabled)
        self.assertTrue(config.audit.sample_gradient_cos_enabled)
        self.assertTrue(config.audit.token_gap_vocab_vector_enabled)
        self.assertTrue(config.audit.entropy_vocab_vector_enabled)
        self.assertFalse(config.audit.token_gradient_enabled)
        self.assertFalse(config.audit.token_gradient_gap_selection_enabled)
        self.assertFalse(config.audit.token_gradient_gap_abs_selection_enabled)
        self.assertTrue(config.audit.token_gradient_loss_abs_selection_enabled)
        self.assertIsNone(config.audit.max_samples_per_domain)
        self.assertEqual(config.audit.token_gradient_top_k, 100)
        self.assertEqual(config.audit.token_gradient_top_p, 0.10)
        self.assertIsNone(config.audit.token_gap_vocab_size)
        self.assertIn("data.train_batch_size=32", rendered)
        self.assertIn("data.max_response_length=16384", rendered)
        self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=32", rendered)
        self.assertIn("+mopd_audit.output_dir=audit/formal_audit_loss_only_smoke", rendered)
        self.assertIn("+mopd_audit.max_samples_per_domain=null", rendered)
        self.assertIn("+mopd_audit.token_gradient_gap_selection_enabled=false", rendered)
        self.assertIn("+mopd_audit.token_gradient_gap_abs_selection_enabled=false", rendered)
        self.assertIn("+mopd_audit.token_gradient_loss_abs_selection_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_top_k=100", rendered)
        self.assertNotIn("token_conflict", rendered)
        self.assertNotIn("token_gradient_top_k_per_sample", rendered)
        self.assertNotIn("token_gradient_max_samples_per_domain", rendered)
        self.assertNotIn("token_gradient_min_teacher_diff", rendered)

    def test_preferred_gradient_smoke_enables_dynamic_tail_and_top_p1(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "test_grad_configs"
            / (
                "mopd_dynamic_weight_qwen0p6b_0p6b_aw2_fsdpsize2_"
                "tail_topp1_b16_4step_smoke.yaml"
            )
        )
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertEqual(config.data.train_batch_size, 16)
        self.assertEqual(config.data.max_response_length, 512)
        self.assertTrue(config.data.load_parquet_direct)
        self.assertFalse(config.data.shuffle)
        self.assertEqual(
            config.model.student_path,
            "/root/autodl-tmp/models/Qwen3-0.6B",
        )
        self.assertEqual(
            config.model.primary_teacher_path,
            "/root/autodl-tmp/models/Qwen3-0.6B",
        )
        expected_domains = {"math", "code", "science"}
        self.assertEqual(set(config.data.domain_train_files), expected_domains)
        self.assertEqual(
            set(config.data.domain_sampling_weights),
            expected_domains,
        )
        self.assertEqual(
            set(config.model.domain_teacher_paths),
            expected_domains,
        )
        self.assertEqual(config.audit.domains, ["math", "code", "science"])
        self.assertIn(
            "data/eval_data/science/GPQA/test.parquet",
            config.data.val_files,
        )
        self.assertEqual(config.actor.ppo_mini_batch_size, 16)
        self.assertEqual(config.actor.fsdp_size, 2)
        self.assertEqual(config.actor.distill_loss_builder, "topk_kl")
        self.assertFalse(config.rollout.do_sample)
        self.assertEqual(config.rollout.tensor_model_parallel_size, 1)
        self.assertTrue(config.worker_placement.separate_ref_policy)
        self.assertEqual(
            config.worker_placement.actor_rollout.n_gpus_per_node,
            2,
        )
        self.assertEqual(
            config.worker_placement.ref_policy.n_gpus_per_node,
            1,
        )
        self.assertTrue(config.audit.token_gradient_enabled)
        self.assertEqual(config.audit.token_gradient_freq_steps, 2)
        self.assertTrue(config.audit.token_gradient_tail_enabled)
        self.assertEqual(config.audit.token_gradient_tail_fraction, 0.15)
        self.assertEqual(config.audit.token_gradient_tail_min_tokens, 1)
        self.assertTrue(config.audit.token_gradient_top_p_enabled)
        self.assertIsNone(config.audit.token_gradient_top_k)
        self.assertEqual(config.audit.token_gradient_top_p, 1.0)
        self.assertTrue(
            config.audit.token_gradient_log_tokens_jsonl_enabled
        )
        self.assertTrue(config.audit.token_gradient_loss_abs_selection_enabled)
        self.assertEqual(
            config.audit.full_grad_training_parity_rel_l2_threshold,
            2.0e-2,
        )
        self.assertTrue(config.audit.dynamic_domain_loss_weighting_enabled)
        self.assertEqual(
            config.audit.dynamic_domain_loss_weighting_freq_steps,
            1,
        )
        self.assertEqual(config.runtime.wandb_mode, "disabled")
        self.assertEqual(config.trainer.logger, '["console","tensorboard"]')
        self.assertEqual(config.trainer.total_training_steps, 4)
        self.assertIn("trainer.resume_mode=disable", config.extra_overrides)
        self.assertIn("+mopd_audit.token_gradient_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_tail_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_top_p_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_top_k=null", rendered)
        self.assertIn("+mopd_audit.token_gradient_top_p=1.0", rendered)
        self.assertIn(
            "+mopd_audit.token_gradient_log_tokens_jsonl_enabled=true",
            rendered,
        )
        self.assertIn(
            "+mopd_audit.token_gradient_loss_abs_selection_enabled=true",
            rendered,
        )
        self.assertIn(
            "+mopd_audit.dynamic_domain_loss_weighting_enabled=true",
            rendered,
        )
        self.assertIn(
            "+actor_rollout_ref.worker_placement.separate_ref_policy=true",
            rendered,
        )

    def test_feature_coverage_smoke_exercises_partial_top_p_prefix_and_ppo2(
        self,
    ) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "test_grad_configs"
            / (
                "mopd_feature_coverage_qwen0p6b_0p6b_aw2_fsdpsize2_"
                "top_partial_prefix_ppo2_b8_2step_smoke.yaml"
            )
        )
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertEqual(config.data.train_batch_size, 8)
        self.assertEqual(config.actor.ppo_mini_batch_size, 8)
        self.assertTrue(config.actor.teacher_prefix_enabled)
        self.assertEqual(
            config.actor.teacher_prefix_loss_region,
            "prefix_and_suffix",
        )
        self.assertTrue(config.rollout.teacher_prefix_sampling_enabled)
        self.assertEqual(config.rollout.teacher_prefix_length, 8)
        self.assertEqual(
            config.rollout.teacher_prefix_dataset_key,
            "data_source",
        )
        self.assertTrue(config.audit.token_gradient_enabled)
        self.assertFalse(config.audit.token_gradient_tail_enabled)
        self.assertTrue(config.audit.token_gradient_top_p_enabled)
        self.assertEqual(config.audit.token_gradient_top_k, 10)
        self.assertEqual(config.audit.token_gradient_top_p, 0.5)
        self.assertFalse(config.audit.dynamic_domain_loss_weighting_enabled)
        self.assertEqual(config.trainer.total_training_steps, 2)
        self.assertIn(
            "actor_rollout_ref.actor.ppo_epochs=2",
            config.extra_overrides,
        )
        self.assertIn(
            "actor_rollout_ref.actor.policy_loss."
            "teacher_prefix_loss_region=prefix_and_suffix",
            rendered,
        )
        self.assertIn(
            "actor_rollout_ref.rollout."
            "teacher_prefix_sampling_enabled=True",
            rendered,
        )
        self.assertIn(
            "+mopd_audit.token_gradient_tail_enabled=false",
            rendered,
        )
        self.assertIn(
            "+mopd_audit.token_gradient_top_p_enabled=true",
            rendered,
        )
        self.assertIn("+mopd_audit.token_gradient_top_p=0.5", rendered)

    def test_audit_off_profile_disables_audit_families(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"

        for gpu_count in (2, 4, 6, 8):
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
