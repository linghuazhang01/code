from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / (
        "mopd_qwen1p7b_30b_a3b_instruct_2507_8gpu_math_code_science_"
        "topk32_control_online_toploss_i3_w3_f20_k30_adaptive_pl_t1p0_"
        "w4_b528.yaml"
    )
)
ONLINE_CONFIG_PATHS = (
    ROOT
    / "configs"
    / (
        "mopd_qwen1p7b_30b_a3b_instruct_2507_6gpu_math_code_science_"
        "topk32_control_online_toploss_i3_w3_f20_k30_adaptive_pl_t1p0_"
        "w4_b525.yaml"
    ),
    ROOT
    / "configs"
    / (
        "mopd_qwen1p7b_30b_a3b_instruct_2507_7gpu_math_code_science_"
        "topk32_control_online_toploss_i3_w3_f20_k30_adaptive_pl_t1p0_"
        "w4_b528.yaml"
    ),
    CONFIG_PATH,
)


class Qwen1p7bOnlineAdaptiveNeighborhoodProfileTests(unittest.TestCase):
    def test_all_online_topologies_cover_one_complete_optimizer_step(self) -> None:
        expected = ((525, 5, 1), (528, 6, 1), (528, 6, 2))
        for path, (batch_size, actor_gpus, ref_gpus) in zip(
            ONLINE_CONFIG_PATHS,
            expected,
            strict=True,
        ):
            with self.subTest(path=path.name):
                config = load_config(path)
                self.assertEqual(config.data.train_batch_size, batch_size)
                self.assertEqual(config.rollout.n, 1)
                self.assertEqual(config.actor.ppo_mini_batch_size, batch_size)
                self.assertEqual(config.actor.ppo_epochs, 1)
                self.assertTrue(
                    config.audit.control_token_online_selection_enabled
                )
                self.assertTrue(
                    config.audit.control_token_adaptive_neighborhood_enabled
                )
                threshold = (
                    config.audit.
                    control_token_adaptive_neighborhood_relative_loss_threshold
                )
                self.assertEqual(threshold, 1.0)
                self.assertEqual(
                    config.worker_placement.actor_rollout.n_gpus_per_node,
                    actor_gpus,
                )
                self.assertEqual(
                    config.worker_placement.ref_policy.n_gpus_per_node,
                    ref_gpus,
                )
                rendered = format_command(build_command(config))
                self.assertIn(
                    "+mopd_audit.control_token_online_top_k=30",
                    rendered,
                )
                self.assertIn(
                    "+mopd_audit.control_token_adaptive_neighborhood_"
                    "relative_loss_threshold=1.0",
                    rendered,
                )

    def test_adaptive_profile_rejects_inexact_step_aggregation_contracts(
        self,
    ) -> None:
        cases = (
            ({"actor": {"ppo_mini_batch_size": 264}}, "one optimizer mini-batch"),
            ({"audit": {"token_gradient_enabled": True}}, "token-gradient"),
        )
        for overlay, message in cases:
            with self.subTest(overlay=overlay), TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "invalid.yaml"
                payload = {"extends": str(CONFIG_PATH.resolve()), **overlay}
                path.write_text(yaml.safe_dump(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_config(path)

    def test_preserves_online_selector_and_enables_adaptive_neighborhood(self) -> None:
        config = load_config(CONFIG_PATH)
        audit = config.audit

        self.assertEqual(config.data.train_batch_size, 528)
        self.assertEqual(config.worker_placement.actor_rollout.n_gpus_per_node, 6)
        self.assertEqual(config.worker_placement.ref_policy.n_gpus_per_node, 2)
        self.assertTrue(audit.control_token_online_selection_enabled)
        self.assertEqual(audit.control_token_online_audit_interval_steps, 3)
        self.assertEqual(audit.control_token_online_window_steps, 3)
        self.assertEqual(audit.control_token_online_top_k, 30)
        self.assertTrue(audit.control_token_adaptive_neighborhood_enabled)
        self.assertEqual(audit.domain_control_token_ids, {})

    def test_adaptive_threshold_is_one(self) -> None:
        audit = load_config(CONFIG_PATH).audit

        self.assertEqual(audit.control_token_adaptive_neighborhood_max_distance, 8)
        self.assertEqual(
            audit.control_token_adaptive_neighborhood_relative_loss_clip_max,
            1.5,
        )
        self.assertEqual(
            audit.control_token_adaptive_neighborhood_relative_loss_threshold,
            1.0,
        )

    def test_rendered_command_contains_online_and_adaptive_contracts(self) -> None:
        rendered = format_command(build_command(load_config(CONFIG_PATH)))
        fragments = (
            "+mopd_audit.control_token_online_selection_enabled=true",
            "+mopd_audit.control_token_online_audit_interval_steps=3",
            "+mopd_audit.control_token_online_window_steps=3",
            "+mopd_audit.control_token_online_top_k=30",
            "+mopd_audit.control_token_adaptive_neighborhood_enabled=true",
            (
                "+mopd_audit.control_token_adaptive_neighborhood_"
                "relative_loss_threshold=1.0"
            ),
        )
        for fragment in fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, rendered)


if __name__ == "__main__":
    unittest.main()
