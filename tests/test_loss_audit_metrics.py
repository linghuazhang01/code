from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mopd_verl.verl_audit import MOPDAuditLogger


def test_configured_loss_gap_and_entropy_are_aggregated_by_domain() -> None:
    with TemporaryDirectory() as output_dir:
        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "output_dir": str(Path(output_dir) / "audit"),
                    "domains": ["math", "code"],
                    "entropy_enabled": True,
                    "entropy_freq_steps": 1,
                    "token_gap_enabled": True,
                    "token_gap_freq_steps": 1,
                    "log_sample_level": False,
                },
                "actor_rollout_ref": {
                    "actor": {
                        "policy_loss": {
                            "lambda_vals": 1.0,
                        }
                    }
                },
            }
        )
        batch = SimpleNamespace(
            batch={
                "old_log_probs": torch.zeros(2, 3),
                "response_mask": torch.tensor(
                    [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
                ),
                "configured_token_loss": torch.tensor(
                    [[1.0, 2.0, 999.0], [10.0, 20.0, 30.0]]
                ),
                "math_teacher_log_prob": torch.tensor(
                    [[0.5, -0.5, 7.0], [0.0, 0.0, 0.0]]
                ),
                "code_teacher_log_prob": torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
                ),
                "student_entropy": torch.tensor(
                    [[0.3, 0.5, 8.0], [0.7, 0.9, 1.1]]
                ),
                "math_teacher_entropy": torch.tensor(
                    [[0.1, 0.2, 9.0], [0.0, 0.0, 0.0]]
                ),
                "code_teacher_entropy": torch.tensor(
                    [[0.0, 0.0, 0.0], [0.4, 0.5, 0.6]]
                ),
            },
            non_tensor_batch={
                "opd_teacher": ["math", "code"],
                "sample_id": ["math-1", "code-1"],
            },
            meta_info={
                "mopd_configured_token_loss_name": (
                    "topk_renormalized_reverse_kl"
                )
            },
        )

        result = logger._compute_training_rows(batch, step=1, lr=1e-5)

    metrics, domain_rows, variance_rows = result[:3]
    rows = {row["domain"]: row for row in domain_rows}

    assert rows["math"]["token_opd_loss_mean"] == pytest.approx(1.5)
    assert rows["math"]["token_opd_loss_sum"] == pytest.approx(3.0)
    assert rows["math"]["token_opd_loss_p05"] == pytest.approx(1.05)
    assert rows["math"]["token_opd_loss_p50"] == pytest.approx(1.5)
    assert rows["math"]["token_opd_loss_p95"] == pytest.approx(1.95)
    assert rows["code"]["token_opd_loss_mean"] == pytest.approx(20.0)
    assert rows["code"]["token_opd_loss_sum"] == pytest.approx(60.0)

    assert rows["math"]["teacher_student_gap_mean"] == pytest.approx(0.0)
    assert rows["math"]["teacher_student_gap_sum"] == pytest.approx(0.0)
    assert rows["math"]["teacher_student_gap_p05"] == pytest.approx(-0.45)
    assert rows["math"]["teacher_student_gap_p50"] == pytest.approx(0.0)
    assert rows["math"]["teacher_student_gap_p95"] == pytest.approx(0.45)
    assert rows["code"]["teacher_student_gap_mean"] == pytest.approx(2.0)
    assert rows["code"]["teacher_student_gap_sum"] == pytest.approx(6.0)

    assert metrics["math/entropy/teacher_entropy_mean"] == pytest.approx(0.15)
    assert metrics["code/entropy/teacher_entropy_mean"] == pytest.approx(0.5)
    assert metrics["math/entropy/student_entropy_mean"] == pytest.approx(0.4)
    assert metrics["code/entropy/student_entropy_mean"] == pytest.approx(0.9)
    assert metrics["global/loss/token_opd_loss_sum"] == pytest.approx(63.0)
    assert metrics["global/teacher/teacher_student_gap_sum"] == pytest.approx(
        6.0
    )
    assert all(
        row["loss_name"] == "topk_renormalized_reverse_kl"
        for row in variance_rows
    )


def test_missing_teacher_log_probs_do_not_create_synthetic_gap() -> None:
    with TemporaryDirectory() as output_dir:
        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "output_dir": str(Path(output_dir) / "audit"),
                    "domains": ["math"],
                }
            }
        )
        batch = SimpleNamespace(
            batch={
                "old_log_probs": torch.tensor([[-1.0, -2.0]]),
                "response_mask": torch.ones(1, 2),
                "configured_token_loss": torch.ones(1, 2),
            },
            non_tensor_batch={"opd_teacher": ["math"]},
            meta_info={},
        )

        with pytest.raises(
            ValueError,
            match="refusing to substitute synthetic zeros",
        ):
            logger._compute_training_rows(batch, step=1, lr=1e-5)


def test_missing_configured_loss_cannot_inherit_configured_loss_metadata() -> None:
    with TemporaryDirectory() as output_dir:
        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "output_dir": str(Path(output_dir) / "audit"),
                    "domains": ["math"],
                }
            }
        )
        batch = SimpleNamespace(
            batch={
                "old_log_probs": torch.tensor([[0.0, 0.0]]),
                "response_mask": torch.ones(1, 2),
                "math_teacher_log_prob": torch.tensor([[-1.0, -2.0]]),
            },
            non_tensor_batch={"opd_teacher": ["math"]},
            meta_info={
                "mopd_configured_token_loss_name": (
                    "topk_renormalized_reverse_kl"
                ),
                "mopd_configured_token_loss_epoch_reduction": "mean",
                "mopd_configured_token_loss_epoch_count": 4,
            },
        )

        variance_rows = logger._compute_training_rows(
            batch,
            step=1,
            lr=1e-5,
        )[2]

    assert len(variance_rows) == 1
    assert variance_rows[0]["loss_name"] == (
        "chosen_token_reverse_kl_fallback"
    )
    assert variance_rows[0]["loss_epoch_reduction"] == "single_forward"
    assert variance_rows[0]["loss_epoch_count"] == 1



def test_configured_loss_mask_excludes_inactive_teacher_prefix_tokens() -> None:
    with TemporaryDirectory() as output_dir:
        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "output_dir": str(Path(output_dir) / "audit"),
                    "domains": ["math"],
                }
            }
        )
        batch = SimpleNamespace(
            batch={
                "old_log_probs": torch.zeros(1, 3),
                "response_mask": torch.ones(1, 3),
                "configured_token_loss": torch.tensor([[0.0, 2.0, 4.0]]),
                "configured_token_loss_mask": torch.tensor(
                    [[0.0, 1.0, 1.0]]
                ),
                "math_teacher_log_prob": torch.zeros(1, 3),
            },
            non_tensor_batch={"opd_teacher": ["math"]},
            meta_info={},
        )

        metrics = logger._compute_training_rows(
            batch,
            step=1,
            lr=1e-5,
        )[0]

    assert metrics["math/loss/token_opd_loss_mean"] == pytest.approx(3.0)
    assert metrics["math/loss/token_opd_loss_sum"] == pytest.approx(6.0)
    assert metrics["math/loss/token_opd_loss_p50"] == pytest.approx(3.0)
