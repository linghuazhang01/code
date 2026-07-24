from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PROFILE_PATH = (
    ROOT
    / "test_grad_configs"
    / "mopd_grad_reliability_qwen0p6b_0p6b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml"
)


def _synthetic_batch(torch_module: Any) -> Any:
    class SyntheticBatch:
        def __init__(self) -> None:
            self.batch = {
                "old_log_probs": torch_module.full((2, 2), -3.0, dtype=torch_module.float32),
                "math_teacher_log_prob": torch_module.tensor(
                    [[-2.0, -1.0], [-3.0, -3.0]],
                    dtype=torch_module.float32,
                ),
                "code_teacher_log_prob": torch_module.tensor(
                    [[-3.0, -3.0], [-4.0, -1.0]],
                    dtype=torch_module.float32,
                ),
                "response_mask": torch_module.ones((2, 2), dtype=torch_module.float32),
                "responses": torch_module.tensor([[1, 1], [1, 2]], dtype=torch_module.long),
                "student_entropy": torch_module.tensor(
                    [[0.5, 1.5], [2.0, 4.0]],
                    dtype=torch_module.float32,
                ),
            }
            self.non_tensor_batch = {
                "opd_teacher": ["math", "code"],
                "sample_id": ["m0", "c0"],
            }

    return SyntheticBatch()


def _read_domain_row(output_dir: Path, filename: str, domain: str = "math") -> dict[str, Any]:
    rows = [json.loads(line) for line in (output_dir / filename).read_text().splitlines()]
    return next(row for row in rows if row["domain"] == domain)


class AuditOccurrenceOverrideTests(unittest.TestCase):
    def test_family_overrides_control_json_and_cosine_independently(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")
        from mopd_verl.verl_audit import MOPDAuditLogger

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
                        "logp_abs_vector_enabled": True,
                        "entropy_enabled": False,
                        "entropy_vocab_vector_enabled": True,
                        "vocab_per_occurrence_mean_vector_enabled": False,
                        "logp_vocab_per_occurrence_mean_vector_enabled": True,
                        "entropy_vocab_per_occurrence_mean_vector_enabled": True,
                        "token_gap_vocab_size": 4,
                    },
                    "actor_rollout_ref": {
                        "actor": {"policy_loss": {"lambda_vals": 1.0}},
                    },
                }
            )
            metrics = logger.log_training_step(_synthetic_batch(torch), step=1, lr=0.01)
            output_path = Path(output_dir)
            token_gap_row = _read_domain_row(output_path, "token_gap_vocab_vectors.jsonl")
            logp_row = _read_domain_row(output_path, "logp_vocab_vectors.jsonl")
            logp_abs_row = _read_domain_row(output_path, "logp_abs_vocab_vectors.jsonl")
            entropy_row = _read_domain_row(output_path, "entropy_vocab_vectors.jsonl")

        self.assertFalse(logger.vocab_per_occurrence_mean_vector_enabled)
        self.assertTrue(logger.logp_vocab_per_occurrence_mean_vector_enabled)
        self.assertFalse(logger.logp_abs_vocab_per_occurrence_mean_vector_enabled)
        self.assertTrue(logger.entropy_vocab_per_occurrence_mean_vector_enabled)

        self.assertIn("gap_signed_sum_vector_vocab", token_gap_row)
        self.assertNotIn("gap_signed_mean_vector_vocab", token_gap_row)
        self.assertIn("logp_mean_vector_vocab", logp_row)
        self.assertNotIn("logp_abs_mean_vector_vocab", logp_abs_row)
        self.assertIn("student_entropy_mean_vector_vocab", entropy_row)

        self.assertFalse(any(key.endswith("/gap_signed_mean_cosine") for key in metrics))
        self.assertIn("global/logp_vocab_cosine/math_vs_code/logp_mean_cosine", metrics)
        self.assertFalse(any(key.endswith("/logp_abs_mean_cosine") for key in metrics))
        self.assertIn(
            "global/entropy_vocab_cosine/math_vs_code/student_entropy_mean_cosine",
            metrics,
        )

    def test_explicit_false_overrides_enabled_global(self) -> None:
        from mopd_verl.verl_audit import MOPDAuditLogger

        with tempfile.TemporaryDirectory() as output_dir:
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": output_dir,
                        "vocab_per_occurrence_mean_vector_enabled": True,
                        "logp_abs_vocab_per_occurrence_mean_vector_enabled": False,
                    }
                }
            )

        self.assertTrue(logger.logp_vocab_per_occurrence_mean_vector_enabled)
        self.assertFalse(logger.logp_abs_vocab_per_occurrence_mean_vector_enabled)
        self.assertTrue(logger.entropy_vocab_per_occurrence_mean_vector_enabled)

    def test_invalid_occurrence_override_is_rejected(self) -> None:
        from mopd_verl.verl_audit import MOPDAuditLogger

        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(ValueError, "optional boolean"):
                MOPDAuditLogger(
                    {
                        "mopd_audit": {
                            "enabled": True,
                            "output_dir": output_dir,
                            "logp_vocab_per_occurrence_mean_vector_enabled": 2,
                        }
                    }
                )

    def test_legacy_profile_renders_nullable_overrides(self) -> None:
        config = load_config(LEGACY_PROFILE_PATH)
        rendered = format_command(build_command(config))

        self.assertIsNone(config.audit.logp_vocab_per_occurrence_mean_vector_enabled)
        self.assertIsNone(config.audit.logp_abs_vocab_per_occurrence_mean_vector_enabled)
        self.assertIsNone(config.audit.entropy_vocab_per_occurrence_mean_vector_enabled)
        self.assertIn(
            "+mopd_audit.logp_vocab_per_occurrence_mean_vector_enabled=null",
            rendered,
        )
        self.assertIn(
            "+mopd_audit.logp_abs_vocab_per_occurrence_mean_vector_enabled=null",
            rendered,
        )
        self.assertIn(
            "+mopd_audit.entropy_vocab_per_occurrence_mean_vector_enabled=null",
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
