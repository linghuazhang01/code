from __future__ import annotations

import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mopd_verl.audit_io import step_jsonl_dir
from mopd_verl.verl_audit import MOPDAuditLogger


def _response_batch() -> SimpleNamespace:
    return SimpleNamespace(
        batch={
            "old_log_probs": torch.tensor(
                [[-1.0, -2.0, -9.0], [-3.0, -4.0, -5.0]]
            ),
            "response_mask": torch.tensor(
                [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
            ),
            "responses": torch.tensor([[11, 12, 0], [21, 22, 23]]),
            "configured_token_loss": torch.tensor(
                [[0.1, 0.2, 99.0], [0.3, 0.4, 0.5]]
            ),
            "configured_token_loss_mask": torch.tensor(
                [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
            ),
            "student_entropy": torch.tensor(
                [[1.1, 1.2, 9.9], [2.1, 2.2, 2.3]]
            ),
            "math_teacher_log_prob": torch.tensor(
                [[-0.5, -1.5, -8.0], [0.0, 0.0, 0.0]]
            ),
            "code_teacher_log_prob": torch.tensor(
                [[0.0, 0.0, 0.0], [-2.5, -3.5, -4.5]]
            ),
            "math_teacher_entropy": torch.tensor(
                [[0.6, 0.7, 8.8], [0.0, 0.0, 0.0]]
            ),
            "code_teacher_entropy": torch.tensor(
                [[0.0, 0.0, 0.0], [1.6, 1.7, 1.8]]
            ),
            "teacher_student_cross_entropy": torch.tensor(
                [[1.4, 1.5, 7.7], [2.4, 2.5, 2.6]]
            ),
        },
        non_tensor_batch={
            "opd_teacher": ["math", "code"],
            "sample_id": ["math-sample", "code-sample"],
        },
        meta_info={
            "mopd_configured_token_loss_name": (
                "topk_renormalized_reverse_kl"
            ),
            "mopd_configured_token_loss_epoch_reduction": "mean",
            "mopd_configured_token_loss_epoch_count": 1,
            "topk_distill_support_source": "teacher",
            "teacher_topk_k": 32,
            "topk_distill_include_tail": False,
            "topk_distill_temperature": 1.0,
        },
    )


def _logger(output_dir: str, compression: str) -> MOPDAuditLogger:
    return MOPDAuditLogger(
        {
            "mopd_audit": {
                "enabled": True,
                "output_dir": output_dir,
                "domains": ["math", "code"],
                "log_sample_level": False,
                "token_gap_enabled": False,
                "entropy_enabled": False,
                "entropy_vocab_vector_enabled": False,
                "response_level_enabled": True,
                "response_level_freq_steps": 1,
                "response_level_compression": compression,
                "domain_control_token_ids": {
                    "math": [12],
                    "code": [21, 23],
                },
            },
            "actor_rollout_ref": {
                "actor": {"policy_loss": {"lambda_vals": 1.0}}
            },
        }
    )


def test_complete_response_records_are_step_scoped_and_aligned() -> None:
    with TemporaryDirectory() as output_dir:
        logger = _logger(output_dir, compression="none")
        logger.log_training_step(_response_batch(), step=7, lr=1e-5)
        logger.log_training_step(_response_batch(), step=7, lr=1e-5)
        directory = step_jsonl_dir(output_dir, 7)
        math_lines = (directory / "response_records_math.jsonl").read_text().splitlines()
        code_lines = (directory / "response_records_code.jsonl").read_text().splitlines()
        domain_lines = (directory / "domain_step_metrics.jsonl").read_text().splitlines()
        math_row = json.loads(math_lines[0])
        code_row = json.loads(code_lines[0])
        manifest = json.loads(
            (directory / "response_manifest.json").read_text()
        )

        assert not (Path(output_dir) / "domain_step_metrics.jsonl").exists()
        assert (directory / "domain_step_metrics.jsonl").exists()

    assert len(math_lines) == 1
    assert len(code_lines) == 1
    assert len(domain_lines) == 2
    assert math_row["response_token_ids"] == [11, 12]
    assert math_row["response_positions"] == [0, 1]
    assert math_row["control_token_mask"] == [False, True]
    assert math_row["control_token_positions"] == [1]
    assert math_row["student_entropy"] == pytest.approx([1.1, 1.2])
    assert math_row["teacher_entropy"] == pytest.approx([0.6, 0.7])
    assert math_row["teacher_student_cross_entropy"] == pytest.approx(
        [1.4, 1.5]
    )
    assert math_row["student_token_log_prob"] == pytest.approx([-1.0, -2.0])
    assert math_row["teacher_token_log_prob"] == pytest.approx([-0.5, -1.5])
    assert math_row["configured_token_loss"] == pytest.approx([0.1, 0.2])

    assert code_row["response_token_ids"] == [21, 22, 23]
    assert code_row["control_token_mask"] == [True, False, True]
    assert code_row["configured_token_loss_mask"] == [1.0, 0.0, 1.0]
    assert manifest["response_count"] == 2
    assert manifest["teacher_student_cross_entropy"] == {
        "include_tail": False,
        "scope": "topk",
        "support_source": "teacher",
        "temperature": 1.0,
        "topk": 32,
    }
    assert manifest["configured_token_loss"]["name"] == (
        "topk_renormalized_reverse_kl"
    )


def test_response_records_support_gzip_without_losing_json_schema() -> None:
    with TemporaryDirectory() as output_dir:
        logger = _logger(output_dir, compression="gzip")
        logger.log_training_step(_response_batch(), step=3, lr=1e-5)
        path = (
            step_jsonl_dir(output_dir, 3)
            / "response_records_math.jsonl.gz"
        )
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            row = json.loads(handle.readline())

    assert row["schema_version"] == 1
    assert row["step"] == 3
    assert row["sample_id"] == "math-sample"


def test_missing_required_response_metric_is_reported_without_partial_file() -> None:
    with TemporaryDirectory() as output_dir:
        logger = _logger(output_dir, compression="none")
        batch = _response_batch()
        del batch.batch["math_teacher_entropy"]
        del batch.batch["code_teacher_entropy"]

        metrics = logger.log_training_step(batch, step=9, lr=1e-5)
        directory = step_jsonl_dir(output_dir, 9)
        error = json.loads(
            (directory / "audit_errors.jsonl").read_text().strip()
        )

        assert not (directory / "response_records_math.jsonl").exists()

        logger.log_training_step(_response_batch(), step=9, lr=1e-5)
        assert not (directory / "audit_errors.jsonl").exists()

    assert metrics["global/audit/error"] == 1.0
    assert "teacher_entropy" in error["error"]
