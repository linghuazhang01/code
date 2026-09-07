"""Check inherited and matrix training profiles use the measured performance settings."""

from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path

from mopd_verl.config_profiles import list_config_profiles, load_raw_config
from mopd_verl.launch import build_overrides
from mopd_verl.settings import load_config


class PerformanceConfigCoverageTests(unittest.TestCase):
    def test_all_training_profiles_resolve_performance_settings(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        abstract_profiles = {
            "configs/token_selection/math/_common.yaml",
            "configs/token_selection/math/_full_taxonomy.yaml",
        }
        legacy_invalid_profiles = {
            f"configs/baselines/{student}_30b_eopd_native_{placement}.yaml"
            for student in ("qwen1p7b", "qwen4b")
            for placement in ("4gpu_b525", "8gpu_b528")
        }
        expected_teacher = {
            "enabled": True,
            "topk_logprob_chunk_size": 1024,
            "max_micro_batch_size": 32,
            "max_tokens": 57344,
            "moe_dispatch": "stable_sort",
            "memory_margin_gib": 12.0,
        }
        for directory in ("configs", "test_grad_configs"):
            paths = sorted((repo / directory).rglob("*.yaml"))
            for path in paths:
                profiles = list_config_profiles(path)
                references = [f"{path}::{name}" for name in profiles] or [str(path)]
                for reference in references:
                    with self.subTest(profile=reference):
                        raw = load_raw_config(reference)
                        self.assertFalse(raw["rollout"].get("enforce_eager", False))
                        self.assertEqual(raw["rollout"]["max_num_seqs"], 64)
                        self.assertEqual(raw["teacher_performance"], expected_teacher)
                        if str(path.relative_to(repo)) in abstract_profiles:
                            continue
                        if str(path.relative_to(repo)) in legacy_invalid_profiles:
                            with self.assertRaisesRegex(
                                ValueError,
                                r"Paper-native EOPD requires rollout_correction.rollout_is=null",
                            ):
                                load_config(reference)
                            continue
                        config = load_config(reference)
                        self.assertFalse(config.rollout.enforce_eager)
                        self.assertEqual(config.rollout.max_num_seqs, 64)
                        self.assertEqual(asdict(config.teacher_performance), expected_teacher)
                        overrides = build_overrides(config)
                        self.assertIn("actor_rollout_ref.rollout.enforce_eager=False", overrides)
                        self.assertIn("actor_rollout_ref.rollout.max_num_seqs=64", overrides)
                        for key, value in expected_teacher.items():
                            rendered = str(value).lower() if isinstance(value, bool) else str(value)
                            self.assertIn(
                                f"+actor_rollout_ref.ref.teacher_performance.{key}={rendered}",
                                overrides,
                            )


if __name__ == "__main__":
    unittest.main()
