from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / (
        "mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_"
        "topk32_control_speed_pwl_u2.yaml"
    )
)
SMOKE_CONFIG_PATH = (
    ROOT
    / "configs"
    / (
        "mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_"
        "topk32_control_speed_pwl_u2_smoke_b24.yaml"
    )
)
REFERENCE_PATH = (
    ROOT
    / "configs"
    / (
        "mopd_qwen1p7b_30b_a3b_instruct_2507_6gpu_math_code_science_"
        "topk32_control_speed_pwl_u2.yaml"
    )
)


class Qwen4bControlSpeedProfileTests(unittest.TestCase):
    def test_uses_expected_models_batch_and_gpu_placement(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertEqual(config.model.student_path, "../mopd/models/Qwen3-4B")
        self.assertEqual(
            set(config.model.domain_teacher_paths.values()),
            {"../mopd/models/Qwen3-30B-A3B-Instruct-2507"},
        )
        self.assertEqual(config.data.train_batch_size, 504)
        self.assertEqual(config.actor.ppo_mini_batch_size, 504)
        self.assertEqual(config.worker_placement.actor_rollout.n_gpus_per_node, 6)
        self.assertEqual(config.worker_placement.ref_policy.n_gpus_per_node, 2)
        self.assertEqual(config.trainer.n_gpus_per_node, 6)
        self.assertEqual(config.rollout.tensor_model_parallel_size, 2)
        self.assertEqual(config.trainer.seed, 42)
        self.assertEqual(config.data.seed, 42)
        self.assertEqual(config.rollout.seed, 42)

    def test_smoke_only_changes_batch_size_and_runtime_namespace(self) -> None:
        full = asdict(load_config(CONFIG_PATH))
        smoke = asdict(load_config(SMOKE_CONFIG_PATH))

        self.assertEqual(smoke["data"]["train_batch_size"], 24)
        self.assertEqual(smoke["actor"]["ppo_mini_batch_size"], 24)
        full["data"]["train_batch_size"] = 24
        full["actor"]["ppo_mini_batch_size"] = 24
        for section, field in (
            ("runtime", "wandb_run_id"),
            ("audit", "output_dir"),
            ("paper_eval", "output_dir"),
            ("trainer", "experiment_name"),
            ("trainer", "default_local_dir"),
        ):
            full[section][field] = smoke[section][field]
        self.assertEqual(smoke, full)

    def test_freezes_the_cross_model_common_support_universe(self) -> None:
        config = load_config(CONFIG_PATH)
        reference = load_config(REFERENCE_PATH)

        self.assertEqual(
            config.audit.domain_control_token_ids,
            reference.audit.domain_control_token_ids,
        )
        self.assertEqual(
            {
                domain: len(token_ids)
                for domain, token_ids in config.audit.domain_control_token_ids.items()
            },
            {"math": 44, "code": 30, "science": 27},
        )
        for token_ids in config.audit.domain_control_token_ids.values():
            self.assertEqual(len(token_ids), len(set(token_ids)))

    def test_enables_only_the_control_speed_reweight_family(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertEqual(
            config.actor.distill_mode,
            "topk_renormalized_reverse_kl",
        )
        self.assertEqual(
            config.audit.loss_variance_signal,
            "topk_renormalized_reverse_kl",
        )
        self.assertTrue(config.audit.control_token_loss_weighting_enabled)
        self.assertTrue(config.audit.control_token_speed_weighting_enabled)
        self.assertTrue(config.audit.control_token_normalize_per_domain)
        self.assertFalse(config.audit.control_token_phase_gate_enabled)
        self.assertFalse(config.audit.control_token_span_weighting_enabled)
        self.assertFalse(config.audit.dynamic_domain_loss_weighting_enabled)
        self.assertFalse(config.audit.all_domain_shared_token_loss_weighting_enabled)
        self.assertFalse(config.audit.full_gradient_enabled)
        self.assertFalse(config.audit.sample_gradient_enabled)
        self.assertFalse(config.audit.token_gradient_enabled)

    def test_matches_the_reference_controller_and_renders_it(self) -> None:
        config = load_config(CONFIG_PATH)
        reference = load_config(REFERENCE_PATH)

        fields = (
            "control_token_speed_window_steps",
            "control_token_speed_ema_beta",
            "control_token_speed_update_interval_steps",
            "control_token_speed_initial_weight",
            "control_token_speed_min_occurrences",
            "control_token_speed_weight_knots",
        )
        for field in fields:
            with self.subTest(field=field):
                self.assertEqual(
                    getattr(config.audit, field),
                    getattr(reference.audit, field),
                )

        rendered = format_command(build_command(config))
        for fragment in (
            "+mopd_audit.control_token_speed_weighting_enabled=true",
            "+mopd_audit.control_token_speed_window_steps=5",
            "+mopd_audit.control_token_speed_ema_beta=0.8",
            "+mopd_audit.control_token_speed_update_interval_steps=2",
            "+mopd_audit.control_token_speed_initial_weight=3.0",
            "+mopd_audit.control_token_speed_min_occurrences=128",
            "+trainer.seed=42",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, rendered)

    def test_resumes_wandb_and_latest_checkpoint(self) -> None:
        config = load_config(CONFIG_PATH)
        smoke = load_config(SMOKE_CONFIG_PATH)

        self.assertEqual(
            config.runtime.wandb_run_id,
            "qwen4b-30b-a3b-8gpu-mcs-control-speed-pwl-u2",
        )
        self.assertEqual(config.runtime.wandb_resume, "allow")
        self.assertIn("trainer.resume_mode=auto", config.extra_overrides)
        self.assertEqual(config.trainer.save_freq, 5)
        self.assertEqual(smoke.runtime.wandb_resume, "allow")
        self.assertIn("trainer.resume_mode=auto", smoke.extra_overrides)
        self.assertNotEqual(
            smoke.runtime.wandb_run_id,
            config.runtime.wandb_run_id,
        )
        self.assertNotEqual(
            smoke.trainer.default_local_dir,
            config.trainer.default_local_dir,
        )

    def test_keeps_complete_response_records(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertTrue(config.audit.response_level_enabled)
        self.assertEqual(config.audit.response_level_freq_steps, 1)
        self.assertEqual(config.audit.response_level_compression, "gzip")
        self.assertFalse(config.trainer.val_before_train)
        self.assertEqual(config.trainer.test_freq, -1)


if __name__ == "__main__":
    unittest.main()
