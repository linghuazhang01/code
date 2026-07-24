from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


class AuditVocabCosineTests(unittest.TestCase):
    def test_logp_alias_and_occurrence_normalized_vectors(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")
        from mopd_verl.verl_audit import MOPDAuditLogger

        class SyntheticBatch:
            def __init__(self) -> None:
                self.batch = {
                    "old_log_probs": torch.full((2, 2), -3.0, dtype=torch.float32),
                    "math_teacher_log_prob": torch.tensor(
                        [[-2.0, -1.0], [-3.0, -3.0]],
                        dtype=torch.float32,
                    ),
                    "code_teacher_log_prob": torch.tensor(
                        [[-3.0, -3.0], [-4.0, -1.0]],
                        dtype=torch.float32,
                    ),
                    "response_mask": torch.ones((2, 2), dtype=torch.float32),
                    "responses": torch.tensor([[1, 1], [1, 2]], dtype=torch.long),
                    "student_entropy": torch.tensor(
                        [[0.5, 1.5], [2.0, 4.0]],
                        dtype=torch.float32,
                    ),
                }
                self.non_tensor_batch = {
                    "opd_teacher": ["math", "code"],
                    "sample_id": ["m0", "c0"],
                }

        with tempfile.TemporaryDirectory() as output_dir:
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": output_dir,
                        "domains": ["math", "code"],
                        "log_sample_level": False,
                        "token_gap_enabled": False,
                        "logp_vector_enabled": True,
                        "logp_vector_freq_steps": 2,
                        "logp_abs_vector_enabled": True,
                        "logp_abs_vector_freq_steps": 2,
                        "entropy_enabled": False,
                        "entropy_vocab_vector_enabled": True,
                        "entropy_vocab_vector_freq_steps": 2,
                        "vocab_per_occurrence_mean_vector_enabled": True,
                        "token_gap_vocab_size": 4,
                    },
                    "actor_rollout_ref": {
                        "actor": {"policy_loss": {"lambda_vals": 1.0}},
                    },
                }
            )
            inactive_metrics = logger.log_training_step(SyntheticBatch(), step=1, lr=0.01)
            metrics = logger.log_training_step(SyntheticBatch(), step=2, lr=0.01)
            output_path = Path(output_dir)
            logp_rows = [
                json.loads(line)
                for line in (output_path / "logp_vocab_vectors.jsonl").read_text().splitlines()
            ]
            logp_occurrence_rows = [
                json.loads(line)
                for line in (output_path / "logp_vectors.jsonl").read_text().splitlines()
            ]
            logp_abs_rows = [
                json.loads(line)
                for line in (output_path / "logp_abs_vocab_vectors.jsonl").read_text().splitlines()
            ]
            logp_abs_occurrence_rows = [
                json.loads(line)
                for line in (output_path / "logp_abs_vectors.jsonl").read_text().splitlines()
            ]
            entropy_rows = [
                json.loads(line)
                for line in (output_path / "entropy_vocab_vectors.jsonl").read_text().splitlines()
            ]
            entropy_distribution_exists = (
                output_path / "entropy_distribution_vectors.jsonl"
            ).exists()
            token_gap_occurrence_exists = (output_path / "token_gap_vectors.jsonl").exists()
            token_gap_vocab_exists = (output_path / "token_gap_vocab_vectors.jsonl").exists()

        self.assertFalse(any(key.startswith("global/logp_vocab_cosine/") for key in inactive_metrics))
        self.assertIn("global/logp_vocab_cosine/math_vs_code/logp_sum_cosine", metrics)
        self.assertIn("global/logp_vocab_cosine/math_vs_code/logp_mean_cosine", metrics)
        self.assertFalse(entropy_distribution_exists)
        self.assertFalse(token_gap_occurrence_exists)
        self.assertFalse(token_gap_vocab_exists)
        self.assertFalse(any("/entropy/" in key for key in metrics))

        math_logp = next(row for row in logp_rows if row["domain"] == "math")
        self.assertEqual(math_logp["token_count_vector_vocab"], [0, 2, 0, 0])
        self.assertEqual(math_logp["logp_sum_vector_vocab"], [0.0, 3.0, 0.0, 0.0])
        self.assertEqual(math_logp["logp_mean_vector_vocab"], [0.0, 1.5, 0.0, 0.0])
        self.assertNotIn("logp_abs_sum_vector_vocab", math_logp)
        self.assertNotIn("gap_signed_sum_vector_vocab", math_logp)
        math_occurrence = next(row for row in logp_occurrence_rows if row["domain"] == "math")
        self.assertEqual(math_occurrence["logp_vector_domain"], [1.0, 2.0])
        self.assertNotIn("logp_abs_vector_domain", math_occurrence)
        self.assertNotIn("gap_signed_vector_domain", math_occurrence)

        math_logp_abs = next(row for row in logp_abs_rows if row["domain"] == "math")
        self.assertEqual(math_logp_abs["logp_abs_sum_vector_vocab"], [0.0, 3.0, 0.0, 0.0])
        self.assertNotIn("logp_sum_vector_vocab", math_logp_abs)
        self.assertNotIn("gap_signed_sum_vector_vocab", math_logp_abs)
        math_abs_occurrence = next(
            row for row in logp_abs_occurrence_rows if row["domain"] == "math"
        )
        self.assertEqual(math_abs_occurrence["logp_abs_vector_domain"], [1.0, 2.0])
        self.assertNotIn("logp_vector_domain", math_abs_occurrence)

        math_entropy = next(row for row in entropy_rows if row["domain"] == "math")
        self.assertEqual(
            math_entropy["student_entropy_sum_vector_vocab"],
            [0.0, 2.0, 0.0, 0.0],
        )
        self.assertEqual(
            math_entropy["student_entropy_mean_vector_vocab"],
            [0.0, 1.0, 0.0, 0.0],
        )

    def test_occurrence_normalized_vectors_can_be_disabled(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")
        from mopd_verl.verl_audit import MOPDAuditLogger

        class SyntheticBatch:
            def __init__(self) -> None:
                self.batch = {
                    "old_log_probs": torch.full((2, 2), -3.0, dtype=torch.float32),
                    "math_teacher_log_prob": torch.tensor(
                        [[-2.0, -1.0], [-3.0, -3.0]],
                        dtype=torch.float32,
                    ),
                    "code_teacher_log_prob": torch.tensor(
                        [[-3.0, -3.0], [-4.0, -1.0]],
                        dtype=torch.float32,
                    ),
                    "response_mask": torch.ones((2, 2), dtype=torch.float32),
                    "responses": torch.tensor([[1, 1], [1, 2]], dtype=torch.long),
                    "student_entropy": torch.tensor(
                        [[0.5, 1.5], [2.0, 4.0]],
                        dtype=torch.float32,
                    ),
                }
                self.non_tensor_batch = {
                    "opd_teacher": ["math", "code"],
                    "sample_id": ["m0", "c0"],
                }

        with tempfile.TemporaryDirectory() as output_dir:
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": output_dir,
                        "domains": ["math", "code"],
                        "log_sample_level": False,
                        "token_gap_enabled": True,
                        "token_gap_vocab_vector_enabled": True,
                        "logp_vector_enabled": True,
                        "entropy_enabled": False,
                        "entropy_vocab_vector_enabled": True,
                        "vocab_per_occurrence_mean_vector_enabled": False,
                        "token_gap_vocab_size": 4,
                    },
                    "actor_rollout_ref": {
                        "actor": {"policy_loss": {"lambda_vals": 1.0}},
                    },
                }
            )
            metrics = logger.log_training_step(SyntheticBatch(), step=1, lr=0.01)
            row = json.loads(
                (Path(output_dir) / "logp_vocab_vectors.jsonl").read_text().splitlines()[0]
            )
            token_gap_row = json.loads(
                (Path(output_dir) / "token_gap_vocab_vectors.jsonl").read_text().splitlines()[0]
            )
            entropy_row = json.loads(
                (Path(output_dir) / "entropy_vocab_vectors.jsonl").read_text().splitlines()[0]
            )

        self.assertIn("logp_sum_vector_vocab", row)
        self.assertNotIn("logp_mean_vector_vocab", row)
        self.assertIn("gap_signed_sum_vector_vocab", token_gap_row)
        self.assertNotIn("gap_signed_mean_vector_vocab", token_gap_row)
        self.assertIn("student_entropy_sum_vector_vocab", entropy_row)
        self.assertNotIn("student_entropy_mean_vector_vocab", entropy_row)
        self.assertIn(
            "global/logp_vocab_cosine/math_vs_code/logp_sum_cosine",
            metrics,
        )
        self.assertFalse(any(key.endswith("/logp_mean_cosine") for key in metrics))
        self.assertFalse(any(key.endswith("/gap_signed_mean_cosine") for key in metrics))
        self.assertFalse(any(key.endswith("/student_entropy_mean_cosine") for key in metrics))

    def test_topk_only_respects_entropy_occurrence_override(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")
        from mopd_verl.verl_audit import MOPDAuditLogger

        class SyntheticBatch:
            def __init__(self) -> None:
                self.batch = {
                    "old_log_probs": torch.full((2, 2), -3.0, dtype=torch.float32),
                    "math_teacher_log_prob": torch.full((2, 2), -2.0, dtype=torch.float32),
                    "code_teacher_log_prob": torch.full((2, 2), -1.0, dtype=torch.float32),
                    "response_mask": torch.ones((2, 2), dtype=torch.float32),
                    "responses": torch.tensor([[1, 2], [1, 3]], dtype=torch.long),
                    "student_entropy": torch.tensor(
                        [[0.5, 0.6], [0.7, 0.8]],
                        dtype=torch.float32,
                    ),
                    "teacher_student_cross_entropy": torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0]],
                        dtype=torch.float32,
                    ),
                }
                self.non_tensor_batch = {
                    "opd_teacher": ["math", "code"],
                    "sample_id": ["m0", "c0"],
                }

        with tempfile.TemporaryDirectory() as output_dir:
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": output_dir,
                        "domains": ["math", "code"],
                        "log_sample_level": False,
                        "token_gap_enabled": False,
                        "entropy_enabled": False,
                        "entropy_vocab_vector_enabled": False,
                        "entropy_vocab_per_occurrence_mean_vector_enabled": False,
                        "topk_teacher_student_cross_entropy_vocab_enabled": True,
                        "topk_teacher_student_cross_entropy_vocab_freq_steps": 2,
                        "token_gap_vocab_size": 4,
                    },
                    "actor_rollout_ref": {
                        "actor": {"policy_loss": {"lambda_vals": 1.0}},
                    },
                }
            )
            inactive_metrics = logger.log_training_step(SyntheticBatch(), step=1, lr=0.01)
            metrics = logger.log_training_step(SyntheticBatch(), step=2, lr=0.01)

        self.assertFalse(
            any(key.startswith("global/entropy_vocab_cosine/") for key in inactive_metrics)
        )

        self.assertIn(
            "global/entropy_vocab_cosine/math_vs_code/"
            "teacher_student_cross_entropy_sum_cosine",
            metrics,
        )
        self.assertNotIn(
            "global/entropy_vocab_cosine/math_vs_code/"
            "teacher_student_cross_entropy_mean_cosine",
            metrics,
        )
        self.assertNotIn(
            "global/entropy_vocab_cosine/math_vs_code/student_entropy_sum_cosine",
            metrics,
        )
        self.assertNotIn(
            "global/entropy_vocab_cosine/math_vs_code/student_entropy_mean_cosine",
            metrics,
        )

    def test_disabled_token_gap_vocab_vector_emits_no_domain_cosine(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")
        from mopd_verl.verl_audit import MOPDAuditLogger

        class SyntheticBatch:
            def __init__(self) -> None:
                self.batch = {
                    "old_log_probs": torch.full((2, 2), -3.0, dtype=torch.float32),
                    "math_teacher_log_prob": torch.tensor(
                        [[-2.0, -2.0], [-3.0, -3.0]],
                        dtype=torch.float32,
                    ),
                    "code_teacher_log_prob": torch.tensor(
                        [[-3.0, -3.0], [-1.0, -1.0]],
                        dtype=torch.float32,
                    ),
                    "response_mask": torch.ones((2, 2), dtype=torch.float32),
                    "responses": torch.tensor([[1, 2], [1, 3]], dtype=torch.long),
                }
                self.non_tensor_batch = {
                    "opd_teacher": ["math", "code"],
                    "sample_id": ["m0", "c0"],
                }

        with tempfile.TemporaryDirectory() as output_dir:
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": output_dir,
                        "domains": ["math", "code"],
                        "log_sample_level": False,
                        "token_gap_enabled": True,
                        "token_gap_vocab_vector_enabled": False,
                    },
                    "actor_rollout_ref": {
                        "actor": {"policy_loss": {"lambda_vals": 1.0}},
                    },
                }
            )
            metrics = logger.log_training_step(SyntheticBatch(), step=1, lr=0.01)

        self.assertFalse(
            any(key.startswith("global/token_gap_vocab_cosine/") for key in metrics)
        )


if __name__ == "__main__":
    unittest.main()
