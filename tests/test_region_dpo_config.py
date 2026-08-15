from __future__ import annotations

import unittest

from mopd_verl.region_dpo_config import (
    RegionDPOConfig,
    parse_region_dpo_config,
    validate_region_dpo_config,
    with_control_token_fallback,
)


class RegionDPOConfigTests(unittest.TestCase):
    def test_parse_exposes_budget_controls(self) -> None:
        config = parse_region_dpo_config(
            {
                "enabled": True,
                "points_per_rollout": 3,
                "branches_per_point": 6,
                "max_new_tokens": 128,
                "beta": 0.2,
                "loss_weight": 0.4,
                "min_reward_margin": 0.1,
                "selection_strategy": "uniform",
                "seed": 7,
                "domain_control_token_ids": {
                    "math": [10, 20],
                    "code": [30],
                },
            }
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.points_per_rollout, 3)
        self.assertEqual(config.branches_per_point, 6)
        self.assertEqual(config.max_new_tokens, 128)
        self.assertEqual(config.beta, 0.2)
        self.assertEqual(config.loss_weight, 0.4)
        self.assertEqual(config.min_reward_margin, 0.1)
        self.assertEqual(config.selection_strategy, "uniform")
        self.assertEqual(config.seed, 7)
        self.assertEqual(
            config.domain_control_token_ids,
            {"math": [10, 20], "code": [30]},
        )
        validate_region_dpo_config(config, max_response_length=256)

    def test_frozen_audit_taxonomy_is_default_anchor_source(self) -> None:
        config = with_control_token_fallback(
            RegionDPOConfig(enabled=True),
            control_token_ids=[],
            domain_control_token_ids={"math": [100, 200]},
        )

        self.assertEqual(
            config.domain_control_token_ids,
            {"math": [100, 200]},
        )
        validate_region_dpo_config(config, max_response_length=512)

    def test_explicit_region_taxonomy_is_not_overwritten(self) -> None:
        config = with_control_token_fallback(
            RegionDPOConfig(enabled=True, control_token_ids=[9]),
            control_token_ids=[1, 2],
            domain_control_token_ids={"math": [3]},
        )

        self.assertEqual(config.control_token_ids, [9])
        self.assertEqual(config.domain_control_token_ids, {})

    def test_invalid_enabled_configs_fail_fast(self) -> None:
        cases = (
            (
                RegionDPOConfig(enabled=True),
                "control-token IDs",
            ),
            (
                RegionDPOConfig(
                    enabled=True,
                    control_token_ids=[1],
                    branches_per_point=1,
                ),
                "at least 2",
            ),
            (
                RegionDPOConfig(
                    enabled=True,
                    control_token_ids=[1],
                    max_new_tokens=513,
                ),
                "max_response_length",
            ),
            (
                RegionDPOConfig(
                    enabled=True,
                    control_token_ids=[1],
                    selection_strategy="outcome_ranked",
                ),
                "must be one of",
            ),
        )
        for config, message in cases:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, message):
                    validate_region_dpo_config(
                        config,
                        max_response_length=512,
                    )

    def test_duplicate_token_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_region_dpo_config({"control_token_ids": [1, 1]})


if __name__ == "__main__":
    unittest.main()
