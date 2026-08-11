from __future__ import annotations

from pathlib import Path
import unittest

from mopd_verl.domain_sampling import allocate_domain_batch_counts
from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "mopd_qwen1p7b_nonthinking_goosereason4b_instruct_6gpu_h200_"
    "math_code_science_topk32_reverse_kl_baseline.yaml"
)
EIGHT_GPU_CONFIG_PATH = (
    ROOT
    / "configs"
    / "mopd_qwen1p7b_nonthinking_goosereason4b_instruct_8gpu_"
    "math_code_science_topk32_reverse_kl_baseline.yaml"
)
PYTHON_PATH = "/home/shuang_qiu/env/miniconda3/envs/mopd-verl/bin/python"
DATA_PREFIX = "../mopd/code/data/"
MODEL_PREFIX = "../mopd/models/"


class Qwen1p7BGooseReason6GpuServerBaselineTests(unittest.TestCase):
    def test_server_assets_and_runtime_are_explicit(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertEqual(config.runtime.python_bin, PYTHON_PATH)
        self.assertEqual(config.runtime.env_file, "../mopd/code/.env.local")
        self.assertTrue(config.model.student_path.startswith(MODEL_PREFIX))
        self.assertNotIn("-Base", config.model.student_path)
        self.assertTrue(
            config.model.primary_teacher_path.startswith(MODEL_PREFIX)
        )
        self.assertTrue(
            all(path.startswith(DATA_PREFIX) for path in config.data.train_files)
        )
        self.assertTrue(
            all(path.startswith(DATA_PREFIX) for path in config.data.val_files)
        )
        self.assertFalse(config.data.enable_thinking)
        self.assertTrue(config.data.return_raw_chat)

    def test_six_gpu_topology_preserves_balanced_batch(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertEqual(config.trainer.n_gpus_per_node, 6)
        self.assertEqual(config.worker_placement.actor_rollout.n_gpus_per_node, 6)
        self.assertEqual(config.actor.fsdp_size, 1)
        self.assertEqual(config.rollout.tensor_model_parallel_size, 2)
        self.assertEqual(config.data.train_batch_size, 504)
        self.assertEqual(config.actor.ppo_mini_batch_size, 504)
        self.assertEqual(
            allocate_domain_batch_counts(
                config.data.train_batch_size,
                config.data.domain_sampling_weights,
            ),
            {"math": 168, "code": 168, "science": 168},
        )

    def test_objective_is_unweighted_topk32_reverse_kl(self) -> None:
        config = load_config(CONFIG_PATH)
        actor = config.actor

        self.assertEqual(actor.distill_loss_builder, "topk_kl")
        self.assertEqual(actor.distill_mode, "topk_renormalized_reverse_kl")
        self.assertEqual(actor.topk_distill_kl_direction, "reverse")
        self.assertEqual(actor.topk_distill_k, 32)
        self.assertEqual(actor.topk_distill_support_source, "teacher")
        self.assertFalse(actor.topk_distill_tail_bucket)
        self.assertFalse(actor.teacher_prefix_enabled)
        self.assertEqual(actor.kl_loss_coef, 0)
        self.assertEqual(actor.entropy_coeff, 0)
        self.assertEqual(config.rollout_correction.rollout_is, "null")

    def test_lightweight_domain_audit_and_fresh_start_are_enabled(self) -> None:
        config = load_config(CONFIG_PATH)
        audit = config.audit

        self.assertTrue(audit.enabled)
        self.assertEqual(audit.domains, ["math", "code", "science"])
        self.assertIn("domain_metrics", audit.output_dir)
        self.assertFalse(audit.full_gradient_enabled)
        self.assertFalse(audit.sample_gradient_enabled)
        self.assertFalse(audit.token_gradient_enabled)
        self.assertFalse(audit.entropy_enabled)
        self.assertFalse(audit.token_gap_enabled)
        self.assertFalse(audit.dynamic_domain_loss_weighting_enabled)
        self.assertFalse(audit.control_token_loss_weighting_enabled)
        self.assertFalse(audit.all_domain_shared_token_loss_weighting_enabled)
        domain_budgeting = getattr(config, "domain_budgeting", None)
        if domain_budgeting is not None:
            self.assertFalse(domain_budgeting.enabled)
        self.assertFalse(config.paper_eval.enabled)
        self.assertIn("trainer.resume_mode=disable", config.extra_overrides)
        self.assertNotIn("trainer.resume_mode=auto", config.extra_overrides)

    def test_server_profile_preserves_eight_gpu_baseline_semantics(self) -> None:
        server = load_config(CONFIG_PATH)
        baseline = load_config(EIGHT_GPU_CONFIG_PATH)

        self.assertEqual(server.actor, baseline.actor)
        self.assertEqual(server.rollout, baseline.rollout)
        self.assertEqual(server.rollout_correction, baseline.rollout_correction)
        self.assertTrue(server.audit.enabled)
        self.assertFalse(baseline.audit.enabled)
        self.assertFalse(server.audit.dynamic_domain_loss_weighting_enabled)
        self.assertFalse(server.audit.control_token_loss_weighting_enabled)
        self.assertFalse(server.audit.all_domain_shared_token_loss_weighting_enabled)
        self.assertEqual(
            getattr(server, "domain_budgeting", None),
            getattr(baseline, "domain_budgeting", None),
        )
        self.assertEqual(server.extra_overrides[:2], baseline.extra_overrides[:2])
        self.assertEqual(
            server.data.domain_sampling_weights,
            baseline.data.domain_sampling_weights,
        )
        self.assertEqual(server.data.train_batch_size, baseline.data.train_batch_size)
        self.assertEqual(server.data.max_prompt_length, baseline.data.max_prompt_length)
        self.assertEqual(
            server.data.max_response_length,
            baseline.data.max_response_length,
        )

    def test_rendered_command_matches_server_contract(self) -> None:
        config = load_config(CONFIG_PATH)
        rendered = format_command(build_command(config))

        self.assertTrue(rendered.startswith(PYTHON_PATH))
        self.assertIn("algorithm.rollout_correction.rollout_is=null", rendered)
        self.assertIn(
            "actor_rollout_ref.actor.policy_loss.distill_mode="
            "topk_renormalized_reverse_kl",
            rendered,
        )
        self.assertIn(
            "+actor_rollout_ref.worker_placement.actor_rollout."
            "n_gpus_per_node=6",
            rendered,
        )
        self.assertIn("+mopd_audit.enabled=true", rendered)
        self.assertIn("+mopd_audit.domains=", rendered)
        self.assertIn("+mopd_audit.full_gradient_enabled=false", rendered)
        self.assertIn("+mopd_audit.token_gradient_enabled=false", rendered)
        self.assertIn("trainer.resume_mode=disable", rendered)
        self.assertNotIn("+mopd_domain_budgeting.", rendered)


if __name__ == "__main__":
    unittest.main()
