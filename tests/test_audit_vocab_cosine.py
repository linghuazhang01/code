from __future__ import annotations

import tempfile
import unittest


class AuditVocabCosineTests(unittest.TestCase):
    def test_topk_only_emits_cross_entropy_domain_cosine(self) -> None:
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
                        "token_conflict_enabled": False,
                        "entropy_enabled": False,
                        "entropy_vocab_vector_enabled": False,
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
        self.assertIn(
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
                        "token_conflict_enabled": False,
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
