from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

from mopd_verl.launch import build_command, format_command, main, run_command
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
        self.assertEqual(
            config.audit.dynamic_domain_loss_weighting_min,
            1.0 / 3.0,
        )
        self.assertEqual(
            config.audit.dynamic_domain_loss_weighting_max,
            3.0,
        )
        resume_overrides = [
            override
            for override in config.extra_overrides
            if override.startswith("trainer.resume_mode=")
        ]
        self.assertEqual(resume_overrides, ["trainer.resume_mode=auto"])
        self.assertEqual(
            config.trainer.default_local_dir,
            "checkpoints/MOPD/qwen4b-from-30b-a3b-instruct-2507-"
            "math-code-science-topk32-reweight-projection-share-5gpu",
        )
        self.assertEqual(
            config.trainer.experiment_name,
            "qwen4b-from-30b-a3b-instruct-2507-math-code-science-topk32-"
            "reweight-projection-share-5gpu_20260731_131456",
        )
        self.assertEqual(config.runtime.wandb_entity, "lz101-rice-university")
        self.assertEqual(config.runtime.wandb_run_id, "qhtj51n5")
        self.assertEqual(config.runtime.wandb_resume, "must")
        self.assertIn(
            "+actor_rollout_ref.worker_placement.ref_policy."
            "n_gpus_per_node=1",
            rendered,
        )

    def test_five_gpu_profile_resumes_exact_wandb_run(self) -> None:
        config = load_config(CONFIG_DIR / PROFILE_NAME)
        with (
            patch.dict("mopd_verl.launch.os.environ", {}, clear=True),
            patch("mopd_verl.launch._read_env_file", return_value={}),
            patch(
                "mopd_verl.launch.subprocess.call",
                return_value=0,
            ) as subprocess_call,
        ):
            return_code = run_command(["python3", "-c", "pass"], config)

        self.assertEqual(return_code, 0)
        environment = subprocess_call.call_args.kwargs["env"]
        self.assertEqual(environment["WANDB_ENTITY"], "lz101-rice-university")
        self.assertEqual(environment["WANDB_PROJECT"], "MOPD")
        self.assertEqual(environment["WANDB_RUN_ID"], "qhtj51n5")
        self.assertEqual(environment["WANDB_RESUME"], "must")
        self.assertEqual(environment["WANDB_MODE"], "online")

        output = StringIO()
        with redirect_stdout(output):
            dry_run_code = main(
                [
                    "--config",
                    str(CONFIG_DIR / PROFILE_NAME),
                    "--dry-run",
                ]
            )
        rendered = output.getvalue()
        expected_name = f"trainer.experiment_name={config.trainer.experiment_name}"
        self.assertEqual(dry_run_code, 0)
        self.assertIn(expected_name, rendered)
        self.assertNotIn(f"{expected_name}_", rendered)

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
