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
MATH_CODE_SCIENCE_TOPK32_PROFILE = (
    "mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_topk32.yaml"
)
PROFILE_PREFIX = "mopd_qwen4b_30b_a3b_instruct_2507"
BASE_PROFILE_TRAIN_FILES = {
    "math": {
        "math": ["data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"]
    },
    "code": {
        "code": ["data/G-OPD-Training-Data/Eurus/code_train.parquet"]
    },
    "if": {
        "if": ["data/G-OPD-Training-Data/IF/train.parquet"]
    },
    "science": {
        "science": ["data/G-OPD-Training-Data/Science/train.parquet"]
    },
    "math_code": {
        "math": ["data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"],
        "code": ["data/G-OPD-Training-Data/Eurus/code_train.parquet"],
    },
    "math_code_science": {
        "math": ["data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"],
        "code": ["data/G-OPD-Training-Data/Eurus/code_train.parquet"],
        "science": ["data/G-OPD-Training-Data/Science/train.parquet"],
    },
}
BASE_PROFILE_VAL_FILES = {
    "math": [
        "data/eval_data/math/AIME24/test.parquet",
        "data/eval_data/math/AIME25/test.parquet",
        "data/eval_data/math/HMMT25Feb/test.parquet",
        "data/eval_data/math/HMMT25Nov/test.parquet",
    ],
    "code": [
        "data/eval_data/code/HumanEvalPlus/test.parquet",
        "data/eval_data/code/MBPPPlus/test.parquet",
    ],
    "if": ["data/eval_data/if/IFBench/test.parquet"],
    "science": ["data/eval_data/science/GPQA/test.parquet"],
    "math_code": [
        "data/eval_data/math/AIME24/test.parquet",
        "data/eval_data/math/AIME25/test.parquet",
        "data/eval_data/math/HMMT25Feb/test.parquet",
        "data/eval_data/math/HMMT25Nov/test.parquet",
        "data/eval_data/code/HumanEvalPlus/test.parquet",
        "data/eval_data/code/MBPPPlus/test.parquet",
    ],
    "math_code_science": [
        "data/eval_data/math/AIME24/test.parquet",
        "data/eval_data/math/AIME25/test.parquet",
        "data/eval_data/math/HMMT25Feb/test.parquet",
        "data/eval_data/math/HMMT25Nov/test.parquet",
        "data/eval_data/code/HumanEvalPlus/test.parquet",
        "data/eval_data/code/MBPPPlus/test.parquet",
        "data/eval_data/science/GPQA/test.parquet",
    ],
}
SIX_GPU_BASE_PROFILES = {
    f"{PROFILE_PREFIX}_6gpu_{profile_name}.yaml"
    for profile_name in BASE_PROFILE_TRAIN_FILES
}
EIGHT_GPU_BASE_PROFILES = {
    f"{PROFILE_PREFIX}_8gpu_{profile_name}.yaml"
    for profile_name in BASE_PROFILE_TRAIN_FILES
}
PROFILE_TRAIN_FILES = {
    **{
        f"{PROFILE_PREFIX}_{gpu_count}gpu_{profile_name}.yaml": train_files
        for gpu_count in (6, 8)
        for profile_name, train_files in BASE_PROFILE_TRAIN_FILES.items()
    },
    MATH_CODE_SCIENCE_TOPK32_PROFILE: BASE_PROFILE_TRAIN_FILES[
        "math_code_science"
    ],
}
PROFILE_VAL_FILES = {
    **{
        f"{PROFILE_PREFIX}_{gpu_count}gpu_{profile_name}.yaml": val_files
        for gpu_count in (6, 8)
        for profile_name, val_files in BASE_PROFILE_VAL_FILES.items()
    },
    MATH_CODE_SCIENCE_TOPK32_PROFILE: BASE_PROFILE_VAL_FILES[
        "math_code_science"
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
                is_eight_gpu_base = filename in EIGHT_GPU_BASE_PROFILES
                expected_batch_size = (
                    504
                    if is_eight_gpu_base or len(expected_domains) == 3
                    else 512
                )
                is_eight_gpu = "_8gpu_" in filename
                expected_actor_gpus = 6 if is_eight_gpu else 4
                expected_fsdp_size = 1 if is_eight_gpu else 2
                expected_full_gradient_freq = (
                    4 if filename == MATH_CODE_SCIENCE_TOPK32_PROFILE else 2
                )

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
                self.assertEqual(config.actor.fsdp_size, expected_fsdp_size)
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
                self.assertEqual(actor_gpus, expected_actor_gpus)
                self.assertEqual(teacher_gpus, 2)
                self.assertEqual(config.trainer.n_gpus_per_node, actor_gpus)
                self.assertEqual(
                    (actor_gpus or 0) + (teacher_gpus or 0),
                    expected_actor_gpus + 2,
                )
                self.assertEqual(
                    (actor_gpus or 0) // (config.actor.fsdp_size or 1),
                    expected_actor_gpus // expected_fsdp_size,
                )
                self.assertEqual(
                    config.data.train_batch_size % (actor_gpus * len(expected_domains)),
                    0,
                )

                if filename == MATH_CODE_SCIENCE_TOPK32_PROFILE:
                    self.assertEqual(config.actor.distill_loss_builder, "topk_kl")
                    self.assertEqual(
                        config.actor.distill_mode,
                        "topk_renormalized_reverse_kl",
                    )
                    self.assertTrue(config.actor.topk_distill_enabled)
                    self.assertEqual(config.actor.topk_distill_k, 32)
                    self.assertEqual(
                        config.actor.topk_distill_support_source,
                        "teacher",
                    )
                    self.assertEqual(config.actor.topk_distill_kl_direction, "reverse")
                    self.assertFalse(config.actor.topk_distill_tail_bucket)
                    self.assertEqual(config.actor.topk_distill_temperature, 1.0)
                    self.assertEqual(config.actor.topk_distill_loss_weight, 1.0)
                    self.assertEqual(config.actor.topk_distill_logprob_chunk_size, 16)
                    self.assertEqual(config.actor.topk_distill_logprob_mode, "sparse")

                self.assertTrue(config.audit.enabled)
                self.assertTrue(config.audit.full_gradient_enabled)
                self.assertEqual(
                    config.audit.full_gradient_freq_steps,
                    expected_full_gradient_freq,
                )
                self.assertEqual(config.audit.full_grad_training_parity_freq_steps, 1)
                self.assertEqual(config.audit.full_grad_training_parity_rel_l2_threshold, 2e-2)
                self.assertEqual(config.audit.full_gradient_storage_dtype, "bfloat16")
                self.assertTrue(config.audit.token_gap_enabled)
                self.assertTrue(config.audit.token_gap_vocab_vector_enabled)
                self.assertTrue(config.audit.vocab_per_occurrence_mean_vector_enabled)
                self.assertTrue(config.audit.logp_vocab_per_occurrence_mean_vector_enabled)
                self.assertTrue(config.audit.logp_abs_vocab_per_occurrence_mean_vector_enabled)
                self.assertTrue(config.audit.entropy_vocab_per_occurrence_mean_vector_enabled)
                self.assertEqual(
                    config.audit.entropy_enabled,
                    filename == MATH_CODE_SCIENCE_TOPK32_PROFILE,
                )
                self.assertTrue(config.audit.entropy_vocab_vector_enabled)
                self.assertEqual(
                    config.audit.topk_teacher_student_cross_entropy_vocab_enabled,
                    filename == MATH_CODE_SCIENCE_TOPK32_PROFILE,
                )
                if filename == MATH_CODE_SCIENCE_TOPK32_PROFILE:
                    self.assertEqual(
                        config.audit.topk_teacher_student_cross_entropy_vocab_freq_steps,
                        1,
                    )
                    self.assertEqual(
                        config.audit.topk_teacher_student_cross_entropy_k,
                        32,
                    )
                    self.assertFalse(
                        config.audit.topk_teacher_student_cross_entropy_include_tail,
                    )
                    self.assertEqual(
                        config.audit.topk_teacher_student_cross_entropy_temperature,
                        1.0,
                    )
                self.assertTrue(config.audit.logp_vector_enabled)
                self.assertEqual(config.audit.logp_vector_freq_steps, 1)
                self.assertTrue(config.audit.logp_abs_vector_enabled)
                self.assertFalse(config.audit.sample_gradient_enabled)
                self.assertEqual(
                    config.audit.token_gradient_enabled,
                    filename == MATH_CODE_SCIENCE_TOPK32_PROFILE,
                )
                if filename == MATH_CODE_SCIENCE_TOPK32_PROFILE:
                    self.assertEqual(config.audit.token_gradient_freq_steps, 4)
                    self.assertTrue(config.audit.token_gradient_tail_enabled)
                    self.assertEqual(
                        config.audit.token_gradient_tail_fraction,
                        0.10,
                    )
                    self.assertEqual(
                        config.audit.token_gradient_tail_min_tokens,
                        1,
                    )
                    self.assertTrue(config.audit.token_gradient_top_p_enabled)
                    self.assertIsNone(config.audit.token_gradient_top_k)
                    self.assertTrue(
                        config.audit.token_gradient_log_tokens_jsonl_enabled
                    )
                self.assertFalse(
                    config.audit.dynamic_domain_loss_weighting_enabled
                )

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
                self.assertIn(
                    "+mopd_audit.token_gradient_tail_fraction=0.1",
                    rendered,
                )
                if filename == MATH_CODE_SCIENCE_TOPK32_PROFILE:
                    self.assertIn(
                        "+mopd_audit.token_gradient_tail_enabled=true",
                        rendered,
                    )
                    self.assertIn(
                        "+mopd_audit.token_gradient_top_p_enabled=true",
                        rendered,
                    )
                    self.assertIn(
                        "+mopd_audit.token_gradient_top_k=null",
                        rendered,
                    )
                    self.assertIn(
                        "+mopd_audit."
                        "token_gradient_log_tokens_jsonl_enabled=true",
                        rendered,
                    )
                self.assertIn(
                    "+mopd_audit.dynamic_domain_loss_weighting_enabled=false",
                    rendered,
                )
                if expected_domains - {"math", "code"}:
                    self.assertIn("actor_rollout_ref.ref.model.teacher_paths", rendered)
                self.assertIn(
                    "actor_rollout_ref.actor.fsdp_config.fsdp_size="
                    f"{expected_fsdp_size}",
                    rendered,
                )
                self.assertIn(
                    "+actor_rollout_ref.worker_placement.actor_rollout.n_gpus_per_node="
                    f"{expected_actor_gpus}",
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
