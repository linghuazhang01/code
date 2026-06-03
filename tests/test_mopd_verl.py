from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mopd_verl.audit_proxy import extract_teacher_domains, extract_validation_datasets
from mopd_verl.launch import build_command, format_command
from mopd_verl.prepare_data import (
    evalplus_jsonl_to_verl_parquet,
    lcb_jsonl_to_verl_parquet,
    math_eval_jsonl_to_verl_parquet,
    merge_teacher_data,
    prepare_paper_eval_data,
    teacher_counts,
    validate_sample_ids,
    validate_teacher_labels,
)
from mopd_verl.settings import load_config
from mopd_verl.smoke_data import write_smoke_data
from mopd_verl.verl_audit import MOPDAuditLogger


class MOPDVerlTests(unittest.TestCase):
    def test_validation_dataset_and_teacher_domain_are_separate(self) -> None:
        non_tensor = {
            "data_source": ["AIME2024", "codeforces"],
            "ability": ["math", "code"],
        }

        self.assertEqual(extract_teacher_domains(non_tensor, 2), ["math", "code"])
        self.assertEqual(extract_validation_datasets(non_tensor, 2), ["AIME2024", "codeforces"])

    def test_default_command_contains_multi_teacher_setting(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_math_code.yaml"
        config = load_config(config_path)
        command = build_command(config)
        rendered = format_command(command)

        self.assertIn("actor_rollout_ref.model.path=Qwen/Qwen3-4B", rendered)
        self.assertIn("+actor_rollout_ref.ref.model.path=Qwen3-4B-Non-Thinking-RL-Math", rendered)
        self.assertIn("+actor_rollout_ref.ref.model.base_model_path=Qwen3-4B-Non-Thinking-RL-Code", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.multi_teacher_distill=true", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.lambda_vals=1.25", rendered)
        self.assertIn("../G-OPD-Training-Data/PaperEval/HMMT25Feb/test.parquet", rendered)
        self.assertIn("../G-OPD-Training-Data/PaperEval/HMMT25Nov/test.parquet", rendered)
        self.assertIn("../G-OPD-Training-Data/PaperEval/HumanEvalPlus/test.parquet", rendered)
        self.assertIn("../G-OPD-Training-Data/PaperEval/MBPPPlus/test.parquet", rendered)
        self.assertIn("../G-OPD-Training-Data/PaperEval/LiveCodeBench/test.parquet", rendered)

    def test_audit_smoke_command_contains_tensorboard_and_audit_settings(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_audit_smoke.yaml"
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertIn('trainer.logger=["console","tensorboard"]', rendered)
        self.assertIn("data.validation_shuffle=False", rendered)
        self.assertNotIn("actor_rollout_ref.rollout.seed", rendered)
        self.assertIn("actor_rollout_ref.rollout.enable_chunked_prefill=False", rendered)
        self.assertIn("actor_rollout_ref.rollout.val_kwargs.do_sample=False", rendered)
        self.assertIn("+mopd_audit.enabled=true", rendered)
        self.assertIn("+mopd_audit.output_dir=/root/autodl-tmp/opd_mopd/audit/smoke", rendered)
        self.assertIn("+mopd_audit.tensorboard_layout=domain_category", rendered)
        self.assertIn("+mopd_audit.tensorboard_prune_mode=core", rendered)
        self.assertIn("+mopd_audit.max_samples_per_domain=8", rendered)
        self.assertIn("+mopd_audit.full_gradient_enabled=false", rendered)
        self.assertIn("+mopd_audit.full_gradient_train_max_samples_per_domain=null", rendered)

    def test_formal_command_enables_full_parameter_gradient_audit(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_single_a800.yaml"
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertIn("+mopd_audit.full_gradient_enabled=true", rendered)
        self.assertIn("+mopd_audit.full_gradient_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.full_gradient_train_max_samples_per_domain=null", rendered)
        self.assertIn("+mopd_audit.full_gradient_validation_max_samples_per_domain=null", rendered)
        self.assertIn("+mopd_audit.full_gradient_validation_files=", rendered)
        self.assertIn("PaperEval/AIME25/test.parquet", rendered)
        self.assertIn("PaperEval/HumanEvalPlus/test.parquet", rendered)
        self.assertIn("+mopd_audit.full_gradient_validation_batch_size=16", rendered)
        self.assertIn("+mopd_audit.full_gradient_micro_batch_size_per_gpu=1", rendered)
        self.assertIn("+mopd_audit.tensorboard_prune_mode=core", rendered)
        self.assertIn("PaperEval/AIME24/test.parquet", rendered)
        self.assertIn("PaperEval/AIME25/test.parquet", rendered)
        self.assertIn("PaperEval/HMMT25Feb/test.parquet", rendered)
        self.assertIn("PaperEval/HMMT25Nov/test.parquet", rendered)
        self.assertIn("PaperEval/MBPPPlus/test.parquet", rendered)
        self.assertIn("PaperEval/LiveCodeBench/test.parquet", rendered)
        self.assertIn("+paper_eval.enabled=true", rendered)
        self.assertIn("run_paper_eval_suite.sh", rendered)
        self.assertIn("+paper_eval.datasets=", rendered)
        self.assertIn("humaneval_plus", rendered)
        self.assertIn("mbpp_plus", rendered)
        self.assertIn("lcb", rendered)

    def test_tensorboard_core_filter_keeps_high_signal_metrics(self) -> None:
        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "domains": ["math", "code"],
                    "tensorboard_prune_mode": "core",
                }
            }
        )
        metrics = {
            "math/loss/opd_loss_mean": 0.1,
            "math/loss/opd_loss_p95": 0.9,
            "math/loss/kl_spike_rate": 0.2,
            "math/full_grad_anchor/code/full_grad_cosine_i_j": 0.2,
            "math/full_grad_anchor/code/full_grad_dot_i_j": 10.0,
            "math/full_grad_anchor/code/predicted_val_opd_loss_delta_i_j": -0.001,
            "math/full_grad_anchor/code/alignment_sign_i_j": 1.0,
            "code/full_grad_anchor/validation_anchor_token_count": 64.0,
            "global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k": -0.3,
            "global/full_grad_conflict/math_vs_code/full_grad_dot_train_i_k": -10.0,
            "global/audit/full_gradient_anchor_token_count": 128.0,
            "math/grad_conflict/code/grad_cosine_train_i_k": -0.4,
            "global/grad_conflict/math_vs_code/grad_cosine_train_i_k": -0.4,
            "math/grad/grad_norm": 1.0,
            "math/teacher/teacher_logprob_mean": -0.5,
            "math/teacher/teacher_student_gap_mean": 0.1,
            "math/coverage/duplicate_rate": 0.0,
            "math/coverage/new_sample_rate": 1.0,
            "global/cost/gpu_count": 1.0,
            "global/cost/step_seconds": 2.0,
            "global/validation/val-core_AIME2024_reward_mean_1": 0.3,
            "global/validation_gain/val-core_AIME2024_reward_mean_1": 0.05,
            "global/validation_gain_stats/val-core_AIME2024_reward_mean_1/ci_high": 0.1,
            "rollout_corr/kl": 0.02,
            "rollout_corr/rollout_is_mean": 1.0,
            "rollout_corr/training_ppl": 3.0,
            "actor/lr": 1e-5,
            "actor/kl_loss": 0.01,
            "training/rollout_log_probs_diff": 0.01,
        }

        filtered = logger.filter_tensorboard_metrics(metrics)

        self.assertIn("math/loss/opd_loss_mean", filtered)
        self.assertIn("math/full_grad_anchor/code/full_grad_cosine_i_j", filtered)
        self.assertIn("math/full_grad_anchor/code/predicted_val_opd_loss_delta_i_j", filtered)
        self.assertIn("code/full_grad_anchor/validation_anchor_token_count", filtered)
        self.assertIn("global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k", filtered)
        self.assertIn("global/audit/full_gradient_anchor_token_count", filtered)
        self.assertIn("math/teacher/teacher_student_gap_mean", filtered)
        self.assertIn("math/coverage/duplicate_rate", filtered)
        self.assertIn("global/cost/step_seconds", filtered)
        self.assertIn("global/validation_gain/val-core_AIME2024_reward_mean_1", filtered)
        self.assertIn("rollout_corr/kl", filtered)
        self.assertIn("actor/lr", filtered)
        self.assertNotIn("math/loss/opd_loss_p95", filtered)
        self.assertNotIn("math/loss/kl_spike_rate", filtered)
        self.assertNotIn("math/full_grad_anchor/code/full_grad_dot_i_j", filtered)
        self.assertNotIn("math/full_grad_anchor/code/alignment_sign_i_j", filtered)
        self.assertNotIn("global/full_grad_conflict/math_vs_code/full_grad_dot_train_i_k", filtered)
        self.assertNotIn("math/grad_conflict/code/grad_cosine_train_i_k", filtered)
        self.assertNotIn("global/grad_conflict/math_vs_code/grad_cosine_train_i_k", filtered)
        self.assertNotIn("math/grad/grad_norm", filtered)
        self.assertNotIn("math/teacher/teacher_logprob_mean", filtered)
        self.assertNotIn("math/coverage/new_sample_rate", filtered)
        self.assertNotIn("global/cost/gpu_count", filtered)
        self.assertNotIn("global/validation/val-core_AIME2024_reward_mean_1", filtered)
        self.assertNotIn("global/validation_gain_stats/val-core_AIME2024_reward_mean_1/ci_high", filtered)
        self.assertNotIn("rollout_corr/rollout_is_mean", filtered)
        self.assertNotIn("rollout_corr/training_ppl", filtered)
        self.assertNotIn("actor/kl_loss", filtered)
        self.assertNotIn("training/rollout_log_probs_diff", filtered)

    def test_audit_logger_uses_domain_category_tags_for_validation_gain_and_cost(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": str(Path(temp_dir) / "audit"),
                        "domains": ["math", "code"],
                    }
                }
            )
            metrics = logger.log_validation_metrics({"val/math/score": 0.2, "val/code/pass@k": 0.1}, step=0)
            metrics.update(logger.log_validation_metrics({"val/math/score": 0.25}, step=1))
            metrics.update(
                logger.log_training_cost(
                    {"timing_s/step": 2.0, "perf/total_num_tokens": 7, "perf/max_memory_allocated_gb": 1.5},
                    step=0,
                    n_gpus=1,
                )
            )

            expected_metric_keys = [
                "math/validation_gain/score",
                "math/validation_gain_stats/score/variance",
                "global/cost/gpu_seconds_step",
            ]
            for key in expected_metric_keys:
                self.assertIn(key, metrics)
            self.assertNotIn("math/validation/score", metrics)
            self.assertNotIn("code/validation/pass_k", metrics)
            self.assertNotIn("mopd/validation/val_math_score", metrics)

    def test_audit_logger_emits_domain_category_metrics_on_synthetic_batch(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        class SyntheticBatch:
            def __init__(self, output_dir: Path) -> None:
                self.batch = {
                    "response_mask": torch.tensor(
                        [[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.float32
                    ),
                    "old_log_probs": torch.tensor(
                        [[-0.2, -0.4, -0.3], [-0.5, -0.3, 0.0], [-0.6, -0.8, -0.4], [-0.7, 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    "ref_log_prob": torch.tensor(
                        [[-0.3, -0.5, -0.35], [-0.45, -0.4, 0.0], [-0.9, -0.7, -0.5], [-0.8, 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    "base_ref_log_prob": torch.tensor(
                        [[-0.1, -0.2, -0.3], [-0.4, -0.5, 0.0], [-0.5, -0.6, -0.3], [-0.6, 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    "base_log_prob": torch.tensor(
                        [[-0.25, -0.45, -0.4], [-0.55, -0.35, 0.0], [-0.65, -0.85, -0.45], [-0.75, 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    "advantages": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.2, 0.1, 0.0], [0.4, 0.2, 0.1], [0.5, 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    "token_level_scores": torch.tensor(
                        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                }
                self.non_tensor_batch = {
                    "domain": ["math", "math", "code", "code"],
                    "opd_teacher": ["math", "math", "code", "code"],
                    "ability": ["math", "math", "code", "code"],
                    "data_source": ["AIME2024", "AIME2024", "code", "code"],
                    "sample_id": [f"sample-{idx}" for idx in range(4)],
                    "output_dir": str(output_dir),
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "audit"
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": str(output_dir),
                        "domains": ["math", "code"],
                        "tier2_window_size": 3,
                        "calibration_bins": 3,
                    },
                    "actor_rollout_ref": {"actor": {"policy_loss": {"lambda_vals": 1.0}}},
                }
            )
            batch = SyntheticBatch(output_dir)
            self.assertEqual(logger.log_validation_anchor_batch(batch=batch, step=0), {})
            metrics = logger.log_training_step(batch=batch, step=0, lr="1e-5")
            metrics.update(logger.log_validation_metrics({"val/math/score": 0.2, "val/code/score": 0.1}, step=0))
            metrics.update(logger.log_validation_metrics({"val/math/score": 0.25, "val/code/score": 0.05}, step=1))
            metrics.update(
                logger.log_training_cost(
                    {"timing_s/step": 2.0, "perf/total_num_tokens": 7, "perf/max_memory_allocated_gb": 1.5},
                    step=0,
                    n_gpus=1,
                )
            )

            expected_metric_keys = [
                "math/loss/sample_loss_variance_mean",
                "math/calibration/calibration_error",
                "global/cost/gpu_seconds_step",
                "math/validation_gain_stats/score/variance",
            ]
            for key in expected_metric_keys:
                self.assertIn(key, metrics)
            removed_metric_keys = [
                "math/grad_conflict/code/grad_cosine_train_i_k",
                "math/grad/gradient_signal",
                "math/grad/gradient_noise",
                "math/grad/grad_norm",
                "math/grad_anchor/code/anchor_grad_cosine_i_j",
                "global/grad_conflict/math_vs_code/grad_cosine_train_i_k",
                "math/loss/kl_spike_rate",
                "math/coverage/new_sample_rate",
                "global/reliability/rank_stability_across_windows",
            ]
            for key in removed_metric_keys:
                self.assertNotIn(key, metrics)

            expected_files = [
                "domain_step_metrics.jsonl",
                "loss_variance_domain_step.jsonl",
                "loss_variance_sample.jsonl",
                "validation_probe.jsonl",
                "validation_gain_variance.jsonl",
                "training_cost.jsonl",
            ]
            for filename in expected_files:
                self.assertTrue((output_dir / filename).exists(), filename)
            removed_files = [
                "trend_stability.jsonl",
                "gradient_noise.jsonl",
                "rank_stability.jsonl",
                "teacher_logits_reliability.jsonl",
                "calibration.jsonl",
                "sample_influence.jsonl",
                "coverage_diversity.jsonl",
                "shadow_probe.jsonl",
                "validation_anchor.jsonl",
                "gradient_anchor_alignment.jsonl",
                "domain_conflict.jsonl",
            ]
            for filename in removed_files:
                self.assertFalse((output_dir / filename).exists(), filename)

    def test_validation_anchor_scheduler_marks_full_gradient_anchor_step(self) -> None:
        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "domains": ["math", "code"],
                    "full_gradient_enabled": True,
                    "validation_anchor_enabled": True,
                    "validation_anchor_refresh_steps": 0,
                }
            }
        )

        self.assertTrue(logger.should_update_validation_anchor(0))
        self.assertEqual(logger.log_validation_anchor_batch(batch=object(), step=0), {})
        self.assertTrue(logger.should_update_validation_anchor(0))
        self.assertFalse(logger.should_update_validation_anchor(1))

    def test_merge_teacher_data_adds_extra_info_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            math_path = temp_path / "math.parquet"
            code_path = temp_path / "code.parquet"
            output_path = temp_path / "merged.parquet"

            pd.DataFrame(
                [
                    {"data_source": "math", "prompt": "1+1", "extra_info": {"index": 0}},
                    {"data_source": "math", "prompt": "2+2", "extra_info": None},
                ]
            ).to_parquet(math_path, index=False)
            pd.DataFrame(
                [
                    {"data_source": "code", "prompt": "write add", "extra_info": {"index": 2}},
                ]
            ).to_parquet(code_path, index=False)

            merge_teacher_data(math_path, code_path, output_path)
            self.assertEqual(teacher_counts(output_path), {"code": 1, "math": 2})
            sample_validation = validate_sample_ids(output_path)
            self.assertTrue(sample_validation.is_valid)

    def test_math_eval_jsonl_to_verl_parquet_adds_validation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            jsonl_path = temp_path / "test.jsonl"
            parquet_path = temp_path / "test.parquet"
            jsonl_path.write_text('{"id":"p0","problem":"Find 1+1.","answer":"2"}\n', encoding="utf-8")

            row_count = math_eval_jsonl_to_verl_parquet(jsonl_path, parquet_path, "HMMT25Feb")
            frame = pd.read_parquet(parquet_path)

            self.assertEqual(row_count, 1)
            self.assertEqual(frame.iloc[0]["data_source"], "HMMT25Feb")
            self.assertEqual(frame.iloc[0]["ability"], "math")
            self.assertEqual(frame.iloc[0]["reward_model"]["ground_truth"], "2")
            self.assertIn("\\boxed{}", frame.iloc[0]["prompt"][0]["content"])
            self.assertEqual(frame.iloc[0]["extra_info"]["validation_dataset"], "HMMT25Feb")
            self.assertEqual(frame.iloc[0]["extra_info"]["sample_id"], "validation:HMMT25Feb:p0")

    def test_evalplus_jsonl_to_verl_parquet_adds_code_validation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            jsonl_path = temp_path / "HumanEvalPlus.jsonl"
            parquet_path = temp_path / "test.parquet"
            evalplus_record = {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n",
                "entry_point": "add",
                "canonical_solution": "    return a + b\n",
                "base_input": [[1, 2]],
                "plus_input": [[3, 4]],
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            }
            jsonl_path.write_text(json.dumps(evalplus_record) + "\n", encoding="utf-8")

            row_count = evalplus_jsonl_to_verl_parquet(jsonl_path, parquet_path, "HumanEvalPlus")
            frame = pd.read_parquet(parquet_path)
            ground_truth = frame.iloc[0]["reward_model"]["ground_truth"]

            self.assertEqual(row_count, 1)
            self.assertEqual(frame.iloc[0]["data_source"], "HumanEvalPlus")
            self.assertEqual(frame.iloc[0]["ability"], "code")
            self.assertEqual(frame.iloc[0]["extra_info"]["opd_teacher"], "code")
            self.assertEqual(frame.iloc[0]["extra_info"]["validation_dataset"], "HumanEvalPlus")
            self.assertIn('"entry_point": "add"', ground_truth)
            self.assertIn('"assert_case"', ground_truth)
            self.assertIn("check(add)", ground_truth)

    def test_lcb_jsonl_to_verl_parquet_adds_code_validation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            jsonl_path = temp_path / "test.jsonl"
            parquet_path = temp_path / "test.parquet"
            record = {
                "question_title": "Add numbers",
                "question_content": "Implement add.",
                "platform": "leetcode",
                "question_id": "q0",
                "contest_id": "c0",
                "contest_date": "2025-01-01T00:00:00",
                "starter_code": "def add(a, b):",
                "difficulty": "easy",
                "public_test_cases": '[{"input":"1\\n2","output":"3","testtype":"stdin"}]',
                "private_test_cases": '[{"input":"3\\n4","output":"7","testtype":"stdin"}]',
                "metadata": "{}",
            }
            jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            row_count = lcb_jsonl_to_verl_parquet([jsonl_path], parquet_path)
            frame = pd.read_parquet(parquet_path)

            self.assertEqual(row_count, 1)
            self.assertEqual(frame.iloc[0]["data_source"], "LiveCodeBench")
            self.assertEqual(frame.iloc[0]["ability"], "code")
            self.assertEqual(frame.iloc[0]["extra_info"]["validation_dataset"], "LiveCodeBench")
            self.assertIn('"outputs": ["3"]', frame.iloc[0]["reward_model"]["ground_truth"])

    def test_prepare_paper_eval_data_writes_all_validation_parquets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "G-OPD"
            for relative in ["aime24", "aime25", "hmmt25_feb", "hmmt25_nov"]:
                data_dir = root / "data" / relative
                data_dir.mkdir(parents=True, exist_ok=True)
                (data_dir / "test.jsonl").write_text(
                    '{"id":"0","problem":"Compute 1+1.","answer":"2"}\n',
                    encoding="utf-8",
                )
            code_data_dir = root / "code_eval" / "data"
            code_data_dir.mkdir(parents=True, exist_ok=True)
            evalplus_record = {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n",
                "entry_point": "add",
                "canonical_solution": "    return a + b\n",
                "base_input": [[1, 2]],
                "plus_input": [[3, 4]],
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            }
            evalplus_row = json.dumps(evalplus_record) + "\n"
            (code_data_dir / "HumanEvalPlus.jsonl").write_text(evalplus_row, encoding="utf-8")
            mbpp_record = {
                **evalplus_record,
                "task_id": "Mbpp/0",
                "assertion": "assert add(1, 2) == 3",
            }
            (code_data_dir / "MbppPlus.jsonl").write_text(json.dumps(mbpp_record) + "\n", encoding="utf-8")
            lcb_dir = root / "code_eval" / "coding" / "LiveCodeBench" / "code_generation_lite"
            lcb_dir.mkdir(parents=True, exist_ok=True)
            lcb_record = {
                "question_title": "Add numbers",
                "question_content": "Implement add.",
                "platform": "leetcode",
                "question_id": "q0",
                "contest_id": "c0",
                "contest_date": "2025-01-01T00:00:00",
                "starter_code": "def add(a, b):",
                "difficulty": "easy",
                "public_test_cases": '[{"input":"1\\n2","output":"3","testtype":"stdin"}]',
                "private_test_cases": "[]",
                "metadata": "{}",
            }
            (lcb_dir / "test.jsonl").write_text(json.dumps(lcb_record) + "\n", encoding="utf-8")

            counts = prepare_paper_eval_data(root)
            output_root = root / "G-OPD-Training-Data" / "PaperEval"

            self.assertEqual(
                counts,
                {
                    "aime24": 1,
                    "aime25": 1,
                    "hmmt25_feb": 1,
                    "hmmt25_nov": 1,
                    "humaneval_plus": 1,
                    "mbpp_plus": 1,
                    "lcb": 1,
                },
            )
            self.assertTrue((output_root / "AIME24" / "test.parquet").exists())
            self.assertTrue((output_root / "AIME25" / "test.parquet").exists())
            self.assertTrue((output_root / "HMMT25Feb" / "test.parquet").exists())
            self.assertTrue((output_root / "HMMT25Nov" / "test.parquet").exists())
            self.assertTrue((output_root / "HumanEvalPlus" / "test.parquet").exists())
            self.assertTrue((output_root / "MBPPPlus" / "test.parquet").exists())
            self.assertTrue((output_root / "LiveCodeBench" / "test.parquet").exists())

    def test_validate_teacher_labels_rejects_missing_or_invalid_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.parquet"
            pd.DataFrame(
                [
                    {"prompt": "valid", "extra_info": {"opd_teacher": "math"}},
                    {"prompt": "missing", "extra_info": {"index": 1}},
                    {"prompt": "bad", "extra_info": {"opd_teacher": "science"}},
                ]
            ).to_parquet(path, index=False)

            validation = validate_teacher_labels(path)
            self.assertFalse(validation.is_valid)
            self.assertEqual(validation.counts, {"code": 0, "math": 1})
            self.assertEqual(len(validation.invalid_rows), 2)

    def test_validate_sample_ids_rejects_missing_duplicate_or_mismatched_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad_sample_ids.parquet"
            pd.DataFrame(
                [
                    {"prompt": "valid", "extra_info": {"opd_teacher": "math", "domain": "math", "sample_id": "m0"}},
                    {"prompt": "missing", "extra_info": {"opd_teacher": "math", "domain": "math"}},
                    {"prompt": "duplicate", "extra_info": {"opd_teacher": "math", "domain": "math", "sample_id": "m0"}},
                    {"prompt": "mismatch", "extra_info": {"opd_teacher": "code", "domain": "math", "sample_id": "c0"}},
                ]
            ).to_parquet(path, index=False)

            validation = validate_sample_ids(path)
            self.assertFalse(validation.is_valid)
            self.assertEqual(validation.duplicate_count, 1)
            self.assertEqual(len(validation.invalid_rows), 3)

    def test_smoke_data_contains_both_teacher_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_smoke_data(temp_dir)

            self.assertEqual(teacher_counts(paths["train"]), {"code": 1, "math": 1})
            self.assertEqual(teacher_counts(paths["val"]), {"code": 1, "math": 1})
            self.assertTrue(validate_sample_ids(paths["train"]).is_valid)
            self.assertTrue(validate_sample_ids(paths["val"]).is_valid)


if __name__ == "__main__":
    unittest.main()
