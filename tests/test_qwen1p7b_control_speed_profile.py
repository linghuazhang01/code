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
        "mopd_qwen1p7b_30b_a3b_instruct_2507_6gpu_math_code_science_"
        "topk32_control_speed_pwl_u2.yaml"
    )
)
EOPD_CONFIG_PATH = (
    ROOT
    / "configs"
    / (
        "mopd_qwen1p7b_30b_a3b_instruct_2507_6gpu_math_code_science_"
        "topk32_eopd_speed.yaml"
    )
)
EXPECTED_KNOTS = [
    [-0.0025, 0.0],
    [0.0, 0.2],
    [0.005, 2.0],
    [0.010, 3.0],
    [0.015, 4.0],
]


class Qwen1p7bControlSpeedProfileTests(unittest.TestCase):
    def test_enables_only_the_simple_speed_controller(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertTrue(config.audit.control_token_loss_weighting_enabled)
        self.assertTrue(config.audit.control_token_speed_weighting_enabled)
        self.assertFalse(config.audit.control_token_phase_gate_enabled)
        self.assertFalse(config.audit.control_token_span_weighting_enabled)
        self.assertFalse(config.audit.dynamic_domain_loss_weighting_enabled)
        self.assertTrue(config.audit.control_token_normalize_per_domain)

    def test_controller_parameters_and_domain_universes_are_frozen(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertEqual(config.audit.control_token_speed_window_steps, 5)
        self.assertEqual(config.audit.control_token_speed_ema_beta, 0.8)
        self.assertEqual(
            config.audit.control_token_speed_update_interval_steps,
            2,
        )
        self.assertEqual(config.audit.control_token_speed_initial_weight, 3.0)
        self.assertEqual(config.audit.control_token_speed_min_occurrences, 128)
        self.assertEqual(
            config.audit.control_token_speed_weight_knots,
            EXPECTED_KNOTS,
        )
        self.assertEqual(
            {
                domain: len(token_ids)
                for domain, token_ids in config.audit.domain_control_token_ids.items()
            },
            {"math": 44, "code": 30, "science": 27},
        )

    def test_profile_does_not_run_validation(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertFalse(config.audit.log_validation_metrics)
        self.assertFalse(config.trainer.val_before_train)
        self.assertEqual(config.trainer.test_freq, -1)

    def test_profile_records_complete_response_token_statistics(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertTrue(config.audit.response_level_enabled)
        self.assertEqual(config.audit.response_level_freq_steps, 1)
        self.assertEqual(config.audit.response_level_compression, "gzip")

    def test_rendered_command_contains_controller_contract(self) -> None:
        config = load_config(CONFIG_PATH)
        rendered = format_command(build_command(config))

        expected_fragments = (
            "+mopd_audit.control_token_speed_weighting_enabled=true",
            "+mopd_audit.control_token_speed_window_steps=5",
            "+mopd_audit.control_token_speed_ema_beta=0.8",
            "+mopd_audit.control_token_speed_update_interval_steps=2",
            "+mopd_audit.control_token_speed_initial_weight=3.0",
            "+mopd_audit.control_token_speed_min_occurrences=128",
            "+mopd_audit.response_level_enabled=true",
            "+mopd_audit.response_level_freq_steps=1",
            "+mopd_audit.response_level_compression=gzip",
            (
                "+mopd_audit.control_token_speed_weight_knots="
                "[[-0.0025, 0], [0, 0.2], [0.005, 2], "
                "[0.01, 3], [0.015, 4]]"
            ),
        )
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, rendered)

    def test_eopd_profile_uses_stable_tracking_and_safe_retention_overrides(
        self,
    ) -> None:
        config = load_config(EOPD_CONFIG_PATH)
        command = build_command(config)

        self.assertEqual(
            config.runtime.wandb_run_id,
            "eopd-control-speed-pwl-u2-20260810",
        )
        self.assertEqual(config.runtime.wandb_resume, "allow")
        self.assertIn("trainer.resume_mode=auto", config.extra_overrides)
        self.assertIn("trainer.max_actor_ckpt_to_keep=5", command)
        self.assertIn("trainer.max_critic_ckpt_to_keep=1", command)
        self.assertNotIn("+trainer.max_actor_ckpt_to_keep=5", command)
        self.assertNotIn("+trainer.max_critic_ckpt_to_keep=1", command)


if __name__ == "__main__":
    unittest.main()
