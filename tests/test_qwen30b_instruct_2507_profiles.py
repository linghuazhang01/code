from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mopd_verl.domain_sampling import allocate_domain_batch_counts
from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
STUDENT_PATH = "../models/Qwen3-4B"
TEACHER_PATH = "../models/Qwen3-30B-A3B-Instruct-2507"
PROFILE_TRAIN_FILES = {
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math.yaml": {
        "math": ["data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"]
    },
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_code.yaml": {
        "code": ["data/G-OPD-Training-Data/Eurus/code_train.parquet"]
    },
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_if.yaml": {
        "if": ["data/G-OPD-Training-Data/IF/train.parquet"]
    },
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_science.yaml": {
        "science": ["data/G-OPD-Training-Data/Science/train.parquet"]
    },
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code.yaml": {
        "math": ["data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"],
        "code": ["data/G-OPD-Training-Data/Eurus/code_train.parquet"],
    },
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code_science.yaml": {
        "math": ["data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"],
        "code": ["data/G-OPD-Training-Data/Eurus/code_train.parquet"],
        "science": ["data/G-OPD-Training-Data/Science/train.parquet"],
    },
}
PROFILE_VAL_FILES = {
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math.yaml": [
        "data/eval_data/math/AIME24/test.parquet",
        "data/eval_data/math/AIME25/test.parquet",
        "data/eval_data/math/HMMT25Feb/test.parquet",
        "data/eval_data/math/HMMT25Nov/test.parquet",
    ],
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_code.yaml": [
        "data/eval_data/code/HumanEvalPlus/test.parquet",
        "data/eval_data/code/MBPPPlus/test.parquet",
    ],
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_if.yaml": [
        "data/eval_data/ifbench/IFBench_test.parquet"
    ],
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_science.yaml": [
        "data/eval_data/science/gpqa.parquet"
    ],
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code.yaml": [
        "data/eval_data/math/AIME24/test.parquet",
        "data/eval_data/math/AIME25/test.parquet",
        "data/eval_data/math/HMMT25Feb/test.parquet",
        "data/eval_data/math/HMMT25Nov/test.parquet",
        "data/eval_data/code/HumanEvalPlus/test.parquet",
        "data/eval_data/code/MBPPPlus/test.parquet",
    ],
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code_science.yaml": [
        "data/eval_data/math/AIME24/test.parquet",
        "data/eval_data/math/AIME25/test.parquet",
        "data/eval_data/math/HMMT25Feb/test.parquet",
        "data/eval_data/math/HMMT25Nov/test.parquet",
        "data/eval_data/code/HumanEvalPlus/test.parquet",
        "data/eval_data/code/MBPPPlus/test.parquet",
        "data/eval_data/science/gpqa.parquet",
    ],
}


class Qwen30BInstruct2507ProfileTests(unittest.TestCase):
    def test_profile_contracts(self) -> None:
        experiment_names: set[str] = set()
        output_dirs: set[str] = set()
        paper_eval_output_dirs: set[str] = set()
        checkpoint_dirs: set[str] = set()

        for filename, expected_train_files in PROFILE_TRAIN_FILES.items():
            with self.subTest(config=filename):
                config = load_config(CONFIG_DIR / filename)
                rendered = format_command(build_command(config))
                expected_domains = set(expected_train_files)
                expected_batch_size = 504 if len(expected_domains) == 3 else 512

                self.assertEqual(config.data.domain_train_files, expected_train_files)
                self.assertEqual(config.data.val_files, PROFILE_VAL_FILES[filename])
                self.assertEqual(set(config.data.domain_sampling_weights), expected_domains)
                expected_domain_count = expected_batch_size // len(expected_domains)
                self.assertEqual(
                    allocate_domain_batch_counts(
                        expected_batch_size,
                        config.data.domain_sampling_weights,
                    ),
                    {domain: expected_domain_count for domain in config.data.domain_sampling_weights},
                )
                self.assertEqual(set(config.audit.domains), expected_domains)
                self.assertEqual(config.model.student_path, STUDENT_PATH)
                self.assertEqual(config.model.primary_teacher_path, TEACHER_PATH)
                self.assertEqual(config.model.math_teacher_path, TEACHER_PATH)
                self.assertEqual(config.model.code_teacher_path, TEACHER_PATH)
                self.assertIsNone(config.model.secondary_teacher_path)
                self.assertEqual(config.model.teacher_model_device, "gpu")
                self.assertEqual(config.model.attn_implementation, "flash_attention_2")
                for domain in expected_domains:
                    self.assertEqual(config.model.domain_teacher_paths[domain], TEACHER_PATH)

                self.assertEqual(config.data.train_batch_size, expected_batch_size)
                self.assertEqual(config.data.max_response_length, 16384)
                self.assertEqual(config.actor.ppo_mini_batch_size, expected_batch_size)
                self.assertEqual(config.actor.ppo_micro_batch_size_per_gpu, 1)
                self.assertFalse(config.data.enable_thinking)
                self.assertEqual(config.actor.fsdp_size, 2)
                self.assertTrue(config.actor.param_offload)
                self.assertTrue(config.actor.optimizer_offload)
                self.assertEqual(config.rollout.tensor_model_parallel_size, 2)
                self.assertEqual(config.rollout.max_model_len, 18432)
                self.assertGreaterEqual(config.rollout.max_num_batched_tokens, 18432)
                self.assertGreaterEqual(config.actor.ppo_max_token_len_per_gpu, 18432)
                self.assertTrue(config.rollout.do_sample)
                self.assertGreater(config.rollout.temperature, 0.0)
                self.assertGreaterEqual(
                    config.rollout.max_model_len or 0,
                    config.data.max_prompt_length + config.data.max_response_length,
                )

                actor_gpus = config.worker_placement.actor_rollout.n_gpus_per_node
                teacher_gpus = config.worker_placement.ref_policy.n_gpus_per_node
                self.assertTrue(config.worker_placement.separate_ref_policy)
                self.assertEqual(actor_gpus, 4)
                self.assertEqual(teacher_gpus, 2)
                self.assertEqual(config.trainer.n_gpus_per_node, actor_gpus)
                self.assertEqual((actor_gpus or 0) + (teacher_gpus or 0), 6)
                self.assertEqual((actor_gpus or 0) // (config.actor.fsdp_size or 1), 2)
                self.assertEqual(
                    config.data.train_batch_size % (actor_gpus * len(expected_domains)),
                    0,
                )

                self.assertTrue(config.audit.enabled)
                self.assertTrue(config.audit.full_gradient_enabled)
                self.assertEqual(config.audit.full_gradient_freq_steps, 2)
                self.assertEqual(config.audit.full_grad_training_parity_freq_steps, 1)
                self.assertEqual(config.audit.full_grad_training_parity_rel_l2_threshold, 2e-2)
                self.assertEqual(config.audit.full_gradient_storage_dtype, "bfloat16")
                self.assertTrue(config.audit.token_gap_enabled)
                self.assertTrue(config.audit.token_gap_vocab_vector_enabled)
                self.assertTrue(config.audit.vocab_per_occurrence_mean_vector_enabled)
                self.assertTrue(config.audit.logp_vocab_per_occurrence_mean_vector_enabled)
                self.assertTrue(config.audit.logp_abs_vocab_per_occurrence_mean_vector_enabled)
                self.assertTrue(config.audit.entropy_vocab_per_occurrence_mean_vector_enabled)
                self.assertFalse(config.audit.entropy_enabled)
                self.assertTrue(config.audit.entropy_vocab_vector_enabled)
                self.assertFalse(config.audit.topk_teacher_student_cross_entropy_vocab_enabled)
                self.assertTrue(config.audit.logp_vector_enabled)
                self.assertEqual(config.audit.logp_vector_freq_steps, 1)
                self.assertTrue(config.audit.logp_abs_vector_enabled)
                self.assertFalse(config.audit.sample_gradient_enabled)
                self.assertFalse(config.audit.token_gradient_enabled)
                self.assertFalse(config.audit.token_conflict_enabled)

                self.assertIn(TEACHER_PATH, rendered)
                self.assertIn("data.max_response_length=16384", rendered)
                self.assertIn("actor_rollout_ref.rollout.max_model_len=18432", rendered)
                self.assertIn(
                    "+actor_rollout_ref.model.override_config."
                    "attn_implementation=flash_attention_2",
                    rendered,
                )
                self.assertIn("actor_rollout_ref.rollout.enforce_eager=True", rendered)
                self.assertNotIn(
                    "+actor_rollout_ref.model.override_config.attn_implementation=eager",
                    rendered,
                )
                self.assertIn(
                    "+mopd_audit.vocab_per_occurrence_mean_vector_enabled=true",
                    rendered,
                )
                self.assertIn(
                    "+mopd_audit.logp_vocab_per_occurrence_mean_vector_enabled=true",
                    rendered,
                )
                self.assertIn(
                    "+mopd_audit.logp_abs_vocab_per_occurrence_mean_vector_enabled=true",
                    rendered,
                )
                self.assertIn(
                    "+mopd_audit.entropy_vocab_per_occurrence_mean_vector_enabled=true",
                    rendered,
                )
                self.assertIn("+mopd_audit.logp_vector_enabled=true", rendered)
                if expected_domains - {"math", "code"}:
                    self.assertIn("actor_rollout_ref.ref.model.teacher_paths", rendered)
                self.assertIn("actor_rollout_ref.actor.fsdp_config.fsdp_size=2", rendered)
                self.assertIn(
                    "+actor_rollout_ref.worker_placement.actor_rollout.n_gpus_per_node=4",
                    rendered,
                )
                self.assertIn(
                    "+actor_rollout_ref.worker_placement.ref_policy.n_gpus_per_node=2",
                    rendered,
                )

                experiment_names.add(config.trainer.experiment_name)
                output_dirs.add(config.audit.output_dir)
                paper_eval_output_dirs.add(config.paper_eval.output_dir)
                checkpoint_dirs.add(config.trainer.default_local_dir)

        self.assertEqual(len(experiment_names), len(PROFILE_TRAIN_FILES))
        self.assertEqual(len(output_dirs), len(PROFILE_TRAIN_FILES))
        self.assertEqual(len(paper_eval_output_dirs), len(PROFILE_TRAIN_FILES))
        self.assertEqual(len(checkpoint_dirs), len(PROFILE_TRAIN_FILES))

    def test_attention_implementation_can_be_overridden(self) -> None:
        source_path = (
            CONFIG_DIR / "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code_science.yaml"
        )
        source_text = source_path.read_text(encoding="utf-8")
        default_setting = "  attn_implementation: flash_attention_2\n"
        self.assertEqual(source_text.count(default_setting), 1)
        with TemporaryDirectory() as temp_dir:
            override_path = Path(temp_dir) / source_path.name
            override_path.write_text(
                source_text.replace(default_setting, "  attn_implementation: sdpa\n"),
                encoding="utf-8",
            )
            config = load_config(override_path)

        rendered = format_command(build_command(config))

        self.assertEqual(config.model.attn_implementation, "sdpa")
        self.assertIn(
            "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
