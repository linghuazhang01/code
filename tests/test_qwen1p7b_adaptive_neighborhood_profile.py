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
        "topk32_structural_codelex_rising_top200_control_adaptive_pl_"
        "w4_b525.yaml"
    )
)


class Qwen1p7bAdaptiveNeighborhoodProfileTests(unittest.TestCase):
    def test_enables_only_fixed_structural_adaptive_control(self) -> None:
        config = load_config(CONFIG_PATH)
        audit = config.audit

        self.assertTrue(audit.control_token_loss_weighting_enabled)
        self.assertTrue(audit.control_token_adaptive_neighborhood_enabled)
        self.assertFalse(audit.control_token_online_selection_enabled)
        self.assertFalse(audit.control_token_speed_weighting_enabled)
        self.assertFalse(audit.control_token_phase_gate_enabled)
        self.assertFalse(audit.control_token_span_weighting_enabled)
        self.assertEqual(
            {
                domain: len(ids)
                for domain, ids in audit.domain_control_token_ids.items()
            },
            {"math": 55, "code": 56, "science": 39},
        )

    def test_controller_parameters_match_the_loss_relative_design(self) -> None:
        audit = load_config(CONFIG_PATH).audit

        self.assertEqual(audit.control_token_adaptive_neighborhood_max_distance, 8)
        self.assertEqual(
            audit.control_token_adaptive_neighborhood_relative_loss_clip_max,
            1.5,
        )
        self.assertEqual(
            audit.control_token_adaptive_neighborhood_relative_loss_threshold,
            0.3,
        )
        self.assertEqual(audit.control_token_adaptive_neighborhood_min_far_tokens, 1)

    def test_rendered_command_contains_the_full_controller_contract(self) -> None:
        rendered = format_command(build_command(load_config(CONFIG_PATH)))
        fragments = (
            "+mopd_audit.control_token_adaptive_neighborhood_enabled=true",
            "+mopd_audit.control_token_adaptive_neighborhood_max_distance=8",
            (
                "+mopd_audit.control_token_adaptive_neighborhood_"
                "relative_loss_clip_max=1.5"
            ),
            (
                "+mopd_audit.control_token_adaptive_neighborhood_"
                "relative_loss_threshold=0.3"
            ),
            "+mopd_audit.control_token_adaptive_neighborhood_min_far_tokens=1",
        )
        for fragment in fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, rendered)


if __name__ == "__main__":
    unittest.main()
