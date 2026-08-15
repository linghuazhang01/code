from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from mopd_verl.config_profiles import load_raw_config
from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "mopd_formal_audit_all_smoke.yaml"


class RegionDPOProfileTests(unittest.TestCase):
    def test_yaml_budget_controls_reach_trainer_and_actor(self) -> None:
        raw = load_raw_config(BASE_CONFIG)
        raw["region_dpo"] = {
            "enabled": True,
            "points_per_rollout": 2,
            "branches_per_point": 5,
            "max_new_tokens": 64,
            "beta": 0.2,
            "loss_weight": 0.3,
            "min_reward_margin": 0.05,
            "selection_strategy": "uniform",
            "seed": 17,
            "domain_control_token_ids": {
                "math": [100, 200],
                "code": [300],
            },
        }
        raw["rollout"]["name"] = "vllm"
        raw["rollout"]["mode"] = "sync"
        raw["rollout"]["calculate_log_probs"] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "region_dpo.yaml"
            path.write_text(
                yaml.safe_dump(raw, sort_keys=False),
                encoding="utf-8",
            )
            config = load_config(path)
            rendered = format_command(build_command(config))

        self.assertTrue(config.region_dpo.enabled)
        self.assertEqual(config.region_dpo.points_per_rollout, 2)
        self.assertEqual(config.region_dpo.branches_per_point, 5)
        for override in (
            "+mopd_region_dpo.enabled=true",
            "+mopd_region_dpo.points_per_rollout=2",
            "+mopd_region_dpo.branches_per_point=5",
            "+mopd_region_dpo.max_new_tokens=64",
            "actor_rollout_ref.actor.policy_loss.region_dpo_enabled=true",
            "actor_rollout_ref.actor.policy_loss.region_dpo_beta=0.2",
            "actor_rollout_ref.actor.policy_loss.region_dpo_loss_weight=0.3",
        ):
            self.assertIn(override, rendered)


if __name__ == "__main__":
    unittest.main()
