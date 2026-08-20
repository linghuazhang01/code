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
        "mopd_qwen1p7b_30b_a3b_instruct_2507_6gpu_math_code_science_"
        "topk32_control_online_toploss_i3_w3_f20_k30_w4_b525.yaml"
    )
)
CONFIG_8GPU_PATH = (
    ROOT
    / "configs"
    / (
        "mopd_qwen1p7b_30b_a3b_instruct_2507_8gpu_math_code_science_"
        "topk32_control_online_toploss_i3_w3_f20_k30_w4_b528.yaml"
    )
)
ALL_CANDIDATES_CONFIG_8GPU_PATH = (
    ROOT
    / "configs"
    / (
        "mopd_qwen1p7b_30b_a3b_instruct_2507_8gpu_math_code_science_"
        "topk32_control_allcand_domaincand_v2_w4_b528.yaml"
    )
)
TOP_SPEED_CONFIG_8GPU_PATH = (
    ROOT
    / "configs"
    / (
        "mopd_qwen1p7b_30b_a3b_instruct_2507_8gpu_math_code_science_"
        "topk32_control_online_topspeed_i3_w3_f20_k30_w4_b528.yaml"
    )
)
EXPECTED_DOMAIN_CANDIDATE_SHA256 = {
    "math": "cdb9c4baa8770aeceda0c533a5889df8385ea1fd42739d78f29532447d040ddf",
    "code": "12f7e1a16efdd20129c11bee086fc825740c010fdda12c6a6c55b7a5cc5d69f2",
    "science": "73f68a93038df626d26a5cc856f19d4188c1016806129066ea3ffe7df45ea948",
}


class Qwen1p7bOnlineControlProfileTests(unittest.TestCase):
    def test_freezes_domain_specific_control_candidates(self) -> None:
        config = load_config(CONFIG_PATH)
        candidates = config.audit.domain_control_token_candidate_ids

        self.assertEqual(config.audit.control_token_candidate_ids, [])
        self.assertEqual(
            {domain: len(token_ids) for domain, token_ids in candidates.items()},
            {"math": 188, "code": 207, "science": 188},
        )
        for domain, token_ids in candidates.items():
            digest = hashlib.sha256(
                ",".join(str(token_id) for token_id in token_ids).encode()
            ).hexdigest()
            self.assertEqual(token_ids, sorted(set(token_ids)))
            self.assertEqual(digest, EXPECTED_DOMAIN_CANDIDATE_SHA256[domain])
        self.assertEqual(len(set().union(*map(set, candidates.values()))), 210)
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
            20.0,
        )
        self.assertEqual(audit.control_token_online_top_k, 30)
        self.assertFalse(audit.control_token_speed_weighting_enabled)
        self.assertFalse(audit.control_token_phase_gate_enabled)
        self.assertFalse(audit.control_token_span_weighting_enabled)
        self.assertFalse(audit.dynamic_domain_loss_weighting_enabled)
        self.assertFalse(audit.all_domain_shared_token_loss_weighting_enabled)

    def test_preserves_qwen1p7b_batch_and_worker_placement(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertEqual(config.model.student_path, "../mopd/models/Qwen3-1.7B")
        self.assertEqual(config.data.train_batch_size, 525)
        self.assertEqual(config.actor.ppo_mini_batch_size, 525)
        self.assertEqual(config.worker_placement.actor_rollout.n_gpus_per_node, 5)
        self.assertEqual(config.worker_placement.ref_policy.n_gpus_per_node, 1)
        self.assertEqual(config.actor.ppo_epochs, 1)

    def test_rendered_command_contains_selector_contract(self) -> None:
        rendered = format_command(build_command(load_config(CONFIG_PATH)))

        for fragment in (
            "+mopd_audit.control_token_online_selection_enabled=true",
            "+mopd_audit.control_token_online_audit_interval_steps=3",
            "+mopd_audit.control_token_online_window_steps=3",
            ("+mopd_audit." "control_token_online_min_mean_occurrences_per_step=20.0"),
            "+mopd_audit.control_token_online_top_k=30",
            "+mopd_audit.domain_control_token_candidate_ids=",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, rendered)


class Qwen1p7bEightGpuOnlineControlProfileTests(unittest.TestCase):
    def test_uses_six_student_and_two_teacher_gpus(self) -> None:
        config = load_config(CONFIG_8GPU_PATH)

        self.assertEqual(config.runtime.slurm_allocation_gpus, 8)
        self.assertEqual(config.worker_placement.actor_rollout.n_gpus_per_node, 6)
        self.assertEqual(config.worker_placement.ref_policy.n_gpus_per_node, 2)
        self.assertEqual(config.trainer.n_gpus_per_node, 6)
        self.assertEqual(config.data.train_batch_size, 528)
        self.assertEqual(config.actor.ppo_mini_batch_size, 528)
        self.assertEqual(config.actor.ppo_epochs, 1)
        self.assertEqual(config.data.train_batch_size % 6, 0)

    def test_preserves_domain_candidates_and_selector_contract(self) -> None:
        reference = load_config(CONFIG_PATH)
        config = load_config(CONFIG_8GPU_PATH)

        self.assertEqual(
            config.audit.domain_control_token_candidate_ids,
            reference.audit.domain_control_token_candidate_ids,
        )
        self.assertTrue(config.audit.control_token_online_selection_enabled)
        self.assertEqual(config.audit.control_token_online_audit_interval_steps, 3)
        self.assertEqual(config.audit.control_token_online_window_steps, 3)
        self.assertEqual(
            config.audit.control_token_online_min_mean_occurrences_per_step,
            20.0,
        )
        self.assertEqual(config.audit.control_token_online_top_k, 30)
        self.assertEqual(config.audit.control_token_online_selection_mode, "top_loss")
        self.assertEqual(config.audit.control_token_loss_weight, 4.0)

    def test_uses_unique_eight_gpu_run_and_output_paths(self) -> None:
        config = load_config(CONFIG_8GPU_PATH)
        rendered = format_command(build_command(config))

        self.assertIn("8g", config.runtime.wandb_run_id)
        self.assertIn("8gpu", config.audit.output_dir)
        self.assertIn("8gpu", config.trainer.experiment_name)
        self.assertIn("8gpu", config.trainer.default_local_dir)
        self.assertIn("trainer.n_gpus_per_node=6", rendered)
        self.assertIn(
            "+actor_rollout_ref.worker_placement.ref_policy.n_gpus_per_node=2",
            rendered,
        )


class Qwen1p7bEightGpuControlBaselineProfileTests(unittest.TestCase):
    def test_all_candidates_profile_weights_the_complete_domain_pools(self) -> None:
        reference = load_config(CONFIG_8GPU_PATH)
        config = load_config(ALL_CANDIDATES_CONFIG_8GPU_PATH)
        audit = config.audit

        self.assertEqual(
            audit.domain_control_token_ids,
            reference.audit.domain_control_token_candidate_ids,
        )
        self.assertEqual(audit.domain_control_token_candidate_ids, {})
        self.assertEqual(audit.control_token_ids, [])
        self.assertEqual(audit.control_token_candidate_ids, [])
        self.assertTrue(audit.control_token_loss_weighting_enabled)
        self.assertFalse(audit.control_token_online_selection_enabled)
        self.assertTrue(audit.control_token_normalize_per_domain)
        self.assertEqual(audit.control_token_loss_weight, 4.0)
        self.assertFalse(audit.control_token_speed_weighting_enabled)

    def test_top_speed_profile_changes_only_the_online_ranking_signal(self) -> None:
        reference = load_config(CONFIG_8GPU_PATH)
        config = load_config(TOP_SPEED_CONFIG_8GPU_PATH)
        audit = config.audit

        self.assertEqual(
            audit.domain_control_token_candidate_ids,
            reference.audit.domain_control_token_candidate_ids,
        )
        self.assertEqual(audit.domain_control_token_ids, {})
        self.assertTrue(audit.control_token_online_selection_enabled)
        self.assertEqual(audit.control_token_online_selection_mode, "top_speed")
        self.assertEqual(audit.control_token_online_audit_interval_steps, 3)
        self.assertEqual(audit.control_token_online_window_steps, 3)
        self.assertEqual(
            audit.control_token_online_min_mean_occurrences_per_step,
            20.0,
        )
        self.assertEqual(audit.control_token_online_top_k, 30)
        self.assertEqual(audit.control_token_loss_weight, 4.0)
        self.assertFalse(audit.control_token_speed_weighting_enabled)

    def test_baselines_preserve_topology_and_use_isolated_outputs(self) -> None:
        reference = load_config(CONFIG_8GPU_PATH)
        profiles = (
            (load_config(ALL_CANDIDATES_CONFIG_8GPU_PATH), "allcand"),
            (load_config(TOP_SPEED_CONFIG_8GPU_PATH), "topspeed"),
        )

        for config, marker in profiles:
            with self.subTest(marker=marker):
                self.assertEqual(config.data.train_batch_size, 528)
                self.assertEqual(config.actor.ppo_mini_batch_size, 528)
                self.assertEqual(
                    config.worker_placement,
                    reference.worker_placement,
                )
                self.assertEqual(config.model, reference.model)
                for value in (
                    config.runtime.wandb_run_id,
                    config.audit.output_dir,
                    config.paper_eval.output_dir,
                    config.trainer.experiment_name,
                    config.trainer.default_local_dir,
                ):
                    self.assertIn(marker, value)

        top_speed_rendered = format_command(build_command(profiles[1][0]))
        self.assertIn(
            "+mopd_audit.control_token_online_selection_mode=top_speed",
            top_speed_rendered,
        )


if __name__ == "__main__":
    unittest.main()
