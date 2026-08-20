from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / (
        "mopd_qwen4b_30b_a3b_instruct_2507_4gpu_math_code_science_"
        "topk32_control_online_toploss_i3_w3_f10_k30_w4_b525.yaml"
    )
)
TOP_SPEED_CONFIG_PATH = (
    ROOT
    / "configs"
    / (
        "mopd_qwen4b_30b_a3b_instruct_2507_4gpu_math_code_science_"
        "topk32_control_online_topspeed_i3_w3_f10_k30_w4_b525.yaml"
    )
)
EXPECTED_DOMAIN_CANDIDATE_SHA256 = {
    "math": "4d923c247102cb4e70674f01fa45c7dffb6ae11d4780ed295c984b0d6bbd6464",
    "code": "836cc9a5dd938a72276ef5457d0cf8c2757b320cab9a58d199926013e69de5c6",
    "science": "4c0ec7a4fae28eca4cfad46ce74c1abe7982eaff6cc1c605aa62658c8c568e05",
}


class Qwen4bOnlineControlProfileTests(unittest.TestCase):
    def test_freezes_four_b_domain_candidates(self) -> None:
        config = load_config(CONFIG_PATH)
        candidates = config.audit.domain_control_token_candidate_ids

        self.assertEqual(config.audit.control_token_candidate_ids, [])
        self.assertEqual(
            {domain: len(token_ids) for domain, token_ids in candidates.items()},
            {"math": 185, "code": 206, "science": 188},
        )
        for domain, token_ids in candidates.items():
            digest = hashlib.sha256(
                ",".join(str(token_id) for token_id in token_ids).encode()
            ).hexdigest()
            self.assertEqual(token_ids, sorted(set(token_ids)))
            self.assertEqual(digest, EXPECTED_DOMAIN_CANDIDATE_SHA256[domain])
        candidate_sets = [set(token_ids) for token_ids in candidates.values()]
        self.assertEqual(len(set().union(*candidate_sets)), 209)
        self.assertEqual(len(set.intersection(*candidate_sets)), 180)
        self.assertEqual(config.audit.control_token_ids, [])
        self.assertEqual(config.audit.domain_control_token_ids, {})

    def test_enables_only_online_control_reweighting(self) -> None:
        config = load_config(CONFIG_PATH)
        audit = config.audit

        self.assertTrue(audit.control_token_loss_weighting_enabled)
        self.assertTrue(audit.control_token_online_selection_enabled)
        self.assertTrue(audit.control_token_normalize_per_domain)
        self.assertEqual(audit.control_token_loss_weight, 4.0)
        self.assertEqual(audit.control_token_online_audit_interval_steps, 3)
        self.assertEqual(audit.control_token_online_window_steps, 3)
        self.assertEqual(
            audit.control_token_online_min_mean_occurrences_per_step,
            10.0,
        )
        self.assertEqual(audit.control_token_online_top_k, 30)
        self.assertEqual(audit.control_token_online_selection_mode, "top_loss")
        self.assertFalse(audit.control_token_speed_weighting_enabled)
        self.assertFalse(audit.control_token_phase_gate_enabled)
        self.assertFalse(audit.control_token_span_weighting_enabled)
        self.assertFalse(audit.dynamic_domain_loss_weighting_enabled)
        self.assertFalse(audit.all_domain_shared_token_loss_weighting_enabled)
        self.assertFalse(config.region_dpo.enabled)
        self.assertFalse(config.domain_budgeting.enabled)

    def test_uses_three_student_and_one_teacher_gpu(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertEqual(config.runtime.slurm_allocation_gpus, 4)
        self.assertEqual(config.model.student_path, "../mopd/models/Qwen3-4B")
        self.assertEqual(config.worker_placement.actor_rollout.n_gpus_per_node, 3)
        self.assertEqual(config.worker_placement.ref_policy.n_gpus_per_node, 1)
        self.assertEqual(config.trainer.n_gpus_per_node, 3)
        self.assertEqual(config.rollout.tensor_model_parallel_size, 1)
        self.assertEqual(config.data.train_batch_size, 525)
        self.assertEqual(config.actor.ppo_mini_batch_size, 525)
        self.assertEqual(config.actor.ppo_epochs, 1)
        self.assertEqual(config.data.train_batch_size % 3, 0)

    def test_rendered_command_contains_selector_and_placement_contract(self) -> None:
        rendered = format_command(build_command(load_config(CONFIG_PATH)))

        for fragment in (
            "+mopd_audit.control_token_online_selection_enabled=true",
            "+mopd_audit.control_token_online_audit_interval_steps=3",
            "+mopd_audit.control_token_online_window_steps=3",
            (
                "+mopd_audit."
                "control_token_online_min_mean_occurrences_per_step=10.0"
            ),
            "+mopd_audit.control_token_online_top_k=30",
            "+mopd_audit.control_token_online_selection_mode=top_loss",
            "+mopd_audit.domain_control_token_candidate_ids=",
            "trainer.n_gpus_per_node=3",
            (
                "+actor_rollout_ref.worker_placement."
                "ref_policy.n_gpus_per_node=1"
            ),
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, rendered)

    def test_uses_unique_run_and_output_paths(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertIn("qwen4b", config.runtime.wandb_run_id)
        self.assertIn("4g", config.runtime.wandb_run_id)
        self.assertIn("f10", config.runtime.wandb_run_id)
        self.assertIn("4gpu", config.audit.output_dir)
        self.assertIn("4gpu", config.trainer.experiment_name)
        self.assertIn("f10", config.trainer.default_local_dir)


class Qwen4bOnlineControlTopSpeedProfileTests(unittest.TestCase):
    def test_reuses_candidate_pool_and_changes_only_ranking_mode(self) -> None:
        top_loss = load_config(CONFIG_PATH)
        top_speed = load_config(TOP_SPEED_CONFIG_PATH)

        self.assertEqual(
            top_speed.audit.domain_control_token_candidate_ids,
            top_loss.audit.domain_control_token_candidate_ids,
        )
        self.assertEqual(
            top_speed.audit.control_token_online_selection_mode,
            "top_speed",
        )
        self.assertEqual(top_speed.audit.control_token_online_top_k, 30)
        self.assertEqual(top_speed.audit.control_token_online_window_steps, 3)
        self.assertEqual(top_speed.audit.control_token_online_audit_interval_steps, 3)
        self.assertEqual(
            top_speed.audit.control_token_online_min_mean_occurrences_per_step,
            10.0,
        )
        self.assertFalse(top_speed.audit.control_token_speed_weighting_enabled)

    def test_renders_mode_and_uses_isolated_outputs(self) -> None:
        config = load_config(TOP_SPEED_CONFIG_PATH)
        rendered = format_command(build_command(config))

        self.assertIn(
            "+mopd_audit.control_token_online_selection_mode=top_speed",
            rendered,
        )
        for value in (
            config.runtime.wandb_run_id,
            config.audit.output_dir,
            config.paper_eval.output_dir,
            config.trainer.experiment_name,
            config.trainer.default_local_dir,
        ):
            self.assertIn("topspeed", value)


if __name__ == "__main__":
    unittest.main()
