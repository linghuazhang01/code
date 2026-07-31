from __future__ import annotations

from pathlib import Path
import unittest

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
PROFILE_NAME = (
    "mopd_qwen4b_30b_a3b_instruct_2507_5gpu_"
    "math_code_science_topk32_reweight_projection_share.yaml"
)
SIX_GPU_PROFILE_NAME = PROFILE_NAME.replace("_5gpu_", "_6gpu_")


class Qwen30BInstruct2507FiveGpuProfileTests(unittest.TestCase):
    def test_five_gpu_profile_contract(self) -> None:
        config = load_config(CONFIG_DIR / PROFILE_NAME)
        command = build_command(config)
        rendered = format_command(command)

        self.assertEqual(command[0], "python3")
        self.assertEqual(config.trainer.n_gpus_per_node, 4)
        self.assertTrue(config.worker_placement.separate_ref_policy)
        self.assertEqual(
            config.worker_placement.actor_rollout.n_gpus_per_node,
            4,
        )
        self.assertEqual(
            config.worker_placement.ref_policy.n_gpus_per_node,
            1,
        )
        self.assertEqual(config.actor.fsdp_size, 1)
        self.assertEqual(config.rollout.tensor_model_parallel_size, 2)
        self.assertEqual(config.data.train_batch_size, 504)
        self.assertEqual(config.actor.ppo_mini_batch_size, 504)
        self.assertEqual(
            config.data.train_batch_size
            % (
                config.trainer.n_gpus_per_node
                * len(config.data.domain_sampling_weights)
            ),
            0,
        )
        self.assertEqual(
            config.rollout.max_model_len,
            config.data.max_prompt_length + config.data.max_response_length,
        )
        self.assertTrue(config.audit.dynamic_domain_loss_weighting_enabled)
        self.assertEqual(
            config.audit.dynamic_domain_loss_weighting_signal_source,
            "domain_gradient_projection_share",
        )
        self.assertEqual(config.audit.full_gradient_freq_steps, 4)
        self.assertEqual(
            config.audit.dynamic_domain_loss_weighting_freq_steps,
            4,
        )
        self.assertEqual(config.audit.token_gradient_freq_steps, 4)
        self.assertIn("trainer.resume_mode=disable", config.extra_overrides)
        self.assertIn(
            "+actor_rollout_ref.worker_placement.ref_policy."
            "n_gpus_per_node=1",
            rendered,
        )

    def test_five_gpu_outputs_do_not_collide_with_six_gpu_profile(self) -> None:
        five_gpu = load_config(CONFIG_DIR / PROFILE_NAME)
        six_gpu = load_config(CONFIG_DIR / SIX_GPU_PROFILE_NAME)

        output_pairs = (
            (five_gpu.audit.output_dir, six_gpu.audit.output_dir),
            (five_gpu.paper_eval.output_dir, six_gpu.paper_eval.output_dir),
            (
                five_gpu.trainer.experiment_name,
                six_gpu.trainer.experiment_name,
            ),
            (
                five_gpu.trainer.default_local_dir,
                six_gpu.trainer.default_local_dir,
            ),
        )
        for five_gpu_value, six_gpu_value in output_pairs:
            with self.subTest(value=five_gpu_value):
                self.assertNotEqual(five_gpu_value, six_gpu_value)
                self.assertIn("5gpu", five_gpu_value)
                self.assertNotIn("6gpu", five_gpu_value)


if __name__ == "__main__":
    unittest.main()
