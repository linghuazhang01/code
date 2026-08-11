from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mopd_verl.audit_io import step_jsonl_dir
from mopd_verl.verl_audit import MOPDAuditLogger


def _domain_row(output_dir: Path, domain: str, step: int = 3) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in (step_jsonl_dir(output_dir, step) / "entropy_vocab_vectors.jsonl")
        .read_text()
        .splitlines()
    ]
    return next(row for row in rows if row["domain"] == domain)


def test_teacher_entropy_is_grouped_by_token_id_with_eopd_gate_counts() -> None:
    with TemporaryDirectory() as output_dir:
        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "output_dir": output_dir,
                    "domains": ["math", "code"],
                    "tensorboard_prune_mode": "core",
                    "log_sample_level": False,
                    "token_gap_enabled": False,
                    "entropy_enabled": False,
                    "entropy_vocab_vector_enabled": True,
                    "entropy_vocab_vector_freq_steps": 1,
                    "entropy_vocab_per_occurrence_mean_vector_enabled": True,
                    "token_gap_vocab_size": 4,
                },
                "actor_rollout_ref": {
                    "actor": {
                        "policy_loss": {
                            "eopd_entropy_threshold": 0.8,
                        }
                    }
                },
            }
        )
        batch = SimpleNamespace(
            batch={
                "old_log_probs": torch.zeros(2, 4),
                "response_mask": torch.tensor(
                    [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]]
                ),
                "responses": torch.tensor([[1, 1, 2, 0], [1, 3, 3, 0]]),
                "student_entropy": torch.ones(2, 4),
                "math_teacher_log_prob": torch.zeros(2, 4),
                "code_teacher_log_prob": torch.zeros(2, 4),
                "math_teacher_entropy": torch.tensor(
                    [[0.7, 0.8, 1.0, 9.0], [0.0, 0.0, 0.0, 0.0]]
                ),
                "code_teacher_entropy": torch.tensor(
                    [[0.0, 0.0, 0.0, 0.0], [0.81, 0.2, 1.2, 9.0]]
                ),
            },
            non_tensor_batch={
                "opd_teacher": ["math", "code"],
                "sample_id": ["math-1", "code-1"],
            },
            meta_info={},
        )

        metrics = logger.log_training_step(batch, step=3, lr=1e-5)
        path = Path(output_dir)
        math_row = _domain_row(path, "math")
        code_row = _domain_row(path, "code")

    assert math_row["teacher_entropy_sum_vector_vocab"] == pytest.approx(
        {"1": 1.5, "2": 1.0}
    )
    assert math_row["teacher_entropy_mean_vector_vocab"] == pytest.approx(
        {"1": 0.75, "2": 1.0}
    )
    assert math_row["vector_storage"] == "sparse_token_id_dict"
    assert math_row["eopd_high_entropy_count_vector_vocab"] == {"1": 1, "2": 1}
    assert math_row["eopd_entropy_threshold"] == pytest.approx(0.8)
    assert math_row["eopd_high_entropy_occurrence_count"] == 2
    assert math_row["eopd_high_entropy_occurrence_ratio"] == pytest.approx(2 / 3)
    assert math_row["eopd_high_entropy_distinct_token_id_count"] == 2

    assert code_row["eopd_high_entropy_count_vector_vocab"] == {"1": 1, "3": 1}
    assert code_row["eopd_high_entropy_occurrence_count"] == 2
    assert code_row["eopd_high_entropy_distinct_token_id_count"] == 2

    assert metrics[
        "math/entropy/eopd_high_entropy_occurrence_count"
    ] == pytest.approx(2.0)
    assert metrics[
        "math/entropy/eopd_high_entropy_occurrence_ratio"
    ] == pytest.approx(2 / 3)
    assert metrics[
        "math/entropy/eopd_high_entropy_distinct_token_id_count"
    ] == pytest.approx(2.0)
