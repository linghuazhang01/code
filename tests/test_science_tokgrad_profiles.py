from __future__ import annotations

import unittest
from pathlib import Path

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
PROFILE_EXPECTATIONS = {
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_science_"
    "topk32_tokgrad_topp10_b168.yaml": (4, 2, 42),
    "mopd_qwen4b_30b_a3b_instruct_2507_8gpu_science_"
    "topk32_tokgrad_topp10_b168.yaml": (6, 1, 28),
}


class ScienceTokenGradientProfileTests(unittest.TestCase):
    def test_profile_contracts(self) -> None:
        experiment_names: set[str] = set()
        audit_output_dirs: set[str] = set()
        evaluation_output_dirs: set[str] = set()
        checkpoint_dirs: set[str] = set()

        for filename, expected in PROFILE_EXPECTATIONS.items():
            with self.subTest(config=filename):
                actor_gpus, fsdp_size, per_rank_batch = expected
                config = load_config(CONFIG_DIR / filename)
                rendered = format_command(build_command(config))

                self.assertEqual(
                    config.data.domain_train_files,
                    {
                        "science": [
                            "data/G-OPD-Training-Data/Science/train.parquet"
                        ]
                    },
                )
                self.assertEqual(
                    config.data.domain_sampling_weights,
                    {"science": 1.0},
                )
                self.assertEqual(config.audit.domains, ["science"])
                self.assertEqual(config.data.train_batch_size, 168)
                self.assertEqual(config.actor.ppo_mini_batch_size, 168)
                self.assertEqual(168 % actor_gpus, 0)
                self.assertEqual(168 // actor_gpus, per_rank_batch)
                self.assertEqual(
                    config.worker_placement.actor_rollout.n_gpus_per_node,
                    actor_gpus,
                )
                self.assertEqual(
                    config.worker_placement.ref_policy.n_gpus_per_node,
                    2,
                )
                self.assertEqual(config.trainer.n_gpus_per_node, actor_gpus)
                self.assertEqual(config.actor.fsdp_size, fsdp_size)

                self.assertEqual(config.actor.distill_loss_builder, "topk_kl")
                self.assertEqual(
                    config.actor.distill_mode,
                    "topk_renormalized_reverse_kl",
                )
                self.assertTrue(config.actor.topk_distill_enabled)
                self.assertEqual(config.actor.topk_distill_k, 32)
                self.assertEqual(
                    config.actor.topk_distill_kl_direction,
                    "reverse",
                )
                self.assertEqual(
                    config.actor.topk_distill_support_source,
                    "teacher",
                )
                self.assertFalse(config.actor.topk_distill_tail_bucket)
                self.assertEqual(config.actor.topk_distill_temperature, 1.0)

                self.assertTrue(config.audit.token_gradient_enabled)
                self.assertEqual(config.audit.token_gradient_freq_steps, 1)
                self.assertFalse(config.audit.token_gradient_tail_enabled)
                self.assertFalse(
                    config.audit.token_gradient_gap_selection_enabled
                )
                self.assertFalse(
                    config.audit.token_gradient_gap_abs_selection_enabled
                )
                self.assertTrue(config.audit.token_gradient_top_p_enabled)
                self.assertIsNone(config.audit.token_gradient_top_k)
                self.assertEqual(config.audit.token_gradient_top_p, 0.10)
                self.assertTrue(
                    config.audit.token_gradient_loss_abs_selection_enabled
                )
                self.assertTrue(
                    config.audit.token_gradient_log_tokens_jsonl_enabled
                )
                self.assertFalse(
                    config.audit.dynamic_domain_loss_weighting_enabled
                )

                experiment_names.add(config.trainer.experiment_name)
                audit_output_dirs.add(config.audit.output_dir)
                evaluation_output_dirs.add(config.paper_eval.output_dir)
                checkpoint_dirs.add(config.trainer.default_local_dir)

                for override in (
                    "+mopd_audit.token_gradient_enabled=true",
                    "+mopd_audit.token_gradient_freq_steps=1",
                    "+mopd_audit.token_gradient_tail_enabled=false",
                    "+mopd_audit.token_gradient_top_p_enabled=true",
                    "+mopd_audit.token_gradient_top_k=null",
                    "+mopd_audit.token_gradient_top_p=0.1",
                    "+mopd_audit.token_gradient_log_tokens_jsonl_enabled=true",
                ):
                    self.assertIn(override, rendered)

        expected_profile_count = len(PROFILE_EXPECTATIONS)
        self.assertEqual(len(experiment_names), expected_profile_count)
        self.assertEqual(len(audit_output_dirs), expected_profile_count)
        self.assertEqual(len(evaluation_output_dirs), expected_profile_count)
        self.assertEqual(len(checkpoint_dirs), expected_profile_count)


if __name__ == "__main__":
    unittest.main()
