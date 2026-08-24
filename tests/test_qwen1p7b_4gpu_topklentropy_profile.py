from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / (
        "mopd_qwen1p7b_30b_a3b_instruct_2507_4gpu_math_code_science_"
        "topk32_control_online_topklentropy_i3_w3_f20_k30_w4_b528.yaml"
    )
)
REFERENCE_CONFIG_PATH = (
    ROOT
    / "configs"
    / (
        "mopd_qwen1p7b_30b_a3b_instruct_2507_7gpu_math_code_science_"
        "topk32_control_online_topklentropy_i3_w3_f20_k30_w4_b528.yaml"
    )
)
STABLE_ENTROPY_OVERRIDE = (
    "actor_rollout_ref.actor.entropy_from_logits_with_chunking=true"
)


def _normalized_config(path: Path) -> dict[str, Any]:
    config = asdict(load_config(path))
    config["runtime"].pop("wandb_run_id")
    config["runtime"].pop("slurm_allocation_gpus")
    config.pop("worker_placement")
    config["audit"].pop("output_dir")
    config["paper_eval"].pop("output_dir")
    config["trainer"].pop("experiment_name")
    config["trainer"].pop("n_gpus_per_node")
    config["trainer"].pop("default_local_dir")
    return config


class Qwen1p7bFourGpuKlEntropyControlProfileTests(unittest.TestCase):
    def test_uses_three_student_and_one_teacher_gpu(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertEqual(config.runtime.slurm_allocation_gpus, 4)
        self.assertEqual(config.worker_placement.actor_rollout.n_gpus_per_node, 3)
        self.assertEqual(config.worker_placement.ref_policy.n_gpus_per_node, 1)
        self.assertEqual(config.trainer.n_gpus_per_node, 3)
        self.assertEqual(config.data.train_batch_size, 528)
        self.assertEqual(config.actor.ppo_mini_batch_size, 528)
        self.assertEqual(config.data.train_batch_size % 3, 0)
        self.assertEqual(
            config.extra_overrides.count(STABLE_ENTROPY_OVERRIDE),
            1,
        )

    def test_preserves_the_kl_entropy_selector_contract(self) -> None:
        reference = load_config(REFERENCE_CONFIG_PATH)
        config = load_config(CONFIG_PATH)
        audit = config.audit

        self.assertEqual(
            _normalized_config(CONFIG_PATH),
            _normalized_config(REFERENCE_CONFIG_PATH),
        )
        self.assertEqual(
            audit.domain_control_token_candidate_ids,
            reference.audit.domain_control_token_candidate_ids,
        )
        self.assertTrue(audit.control_token_online_selection_enabled)
        self.assertEqual(
            audit.control_token_online_selection_mode,
            "top_kl_student_entropy",
        )
        self.assertEqual(audit.control_token_online_weight_mode, "fixed")
        self.assertEqual(audit.control_token_loss_weight, 4.0)
        self.assertEqual(audit.control_token_online_audit_interval_steps, 3)
        self.assertEqual(audit.control_token_online_window_steps, 3)
        self.assertEqual(
            audit.control_token_online_min_mean_occurrences_per_step,
            20.0,
        )
        self.assertEqual(audit.control_token_online_top_k, 30)

    def test_renders_placement_and_uses_isolated_outputs(self) -> None:
        config = load_config(CONFIG_PATH)
        rendered = format_command(build_command(config))

        self.assertIn("trainer.n_gpus_per_node=3", rendered)
        self.assertIn("actor_rollout_ref.model.use_remove_padding=True", rendered)
        self.assertIn(STABLE_ENTROPY_OVERRIDE, rendered)
        self.assertIn(
            "+actor_rollout_ref.worker_placement."
            "actor_rollout.n_gpus_per_node=3",
            rendered,
        )
        self.assertIn(
            "+actor_rollout_ref.worker_placement.ref_policy.n_gpus_per_node=1",
            rendered,
        )
        for value in (
            config.runtime.wandb_run_id,
            config.audit.output_dir,
            config.paper_eval.output_dir,
            config.trainer.experiment_name,
            config.trainer.default_local_dir,
        ):
            self.assertIn("topklentropy", value)
            self.assertTrue("4g" in value or "4gpu" in value)


if __name__ == "__main__":
    unittest.main()
