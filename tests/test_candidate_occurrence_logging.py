from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mopd_verl.audit_candidate_occurrence_logging import (
    CANDIDATE_OCCURRENCE_MANIFEST,
    POSITION_ARRAY_FIELDS,
    CandidateOccurrenceBatch,
    iter_candidate_positions,
    reference_sampling_seed,
    write_candidate_occurrence_log,
)
from mopd_verl.audit_io import step_jsonl_dir
from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import AuditConfig, load_config
from mopd_verl.verl_audit import MOPDAuditLogger


# Two math responses, one code response; padded length 6.
MASK = torch.tensor(
    [
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0],
    ],
    dtype=torch.float32,
)
IDS = torch.tensor(
    [
        [5, 7, 9, 5, 5, 0],
        [7, 1, 2, 3, 4, 9],
        [5, 8, 8, 0, 0, 0],
    ]
)
CANDIDATES = {"math": (5, 7), "code": (8,)}


def _matrix(offset: float) -> torch.Tensor:
    return torch.arange(18, dtype=torch.float32).reshape(3, 6) / 10 + offset


def _weights() -> torch.Tensor:
    # Row 0: base 0.5, ID-5 positions up-weighted to 2.0, ID-7 left at base.
    # Row 1: base 0.5, ID-7 up-weighted. Row 2: unweighted (1.0).
    return torch.tensor(
        [
            [2.0, 0.5, 0.5, 2.0, 0.0, 0.0],
            [2.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    ) * MASK


def _batch(step: int = 4, **overrides) -> CandidateOccurrenceBatch:
    values = dict(
        step=step,
        domains=("math", "math", "code"),
        sample_ids=("m0", "m1", "c0"),
        response_token_ids=IDS,
        response_mask=MASK,
        student_log_prob=-_matrix(1.0),
        teacher_log_prob=-_matrix(0.5) * 2,
        configured_token_loss=_matrix(0.0),
        student_entropy=_matrix(2.0),
        teacher_entropy=_matrix(3.0),
        teacher_student_cross_entropy=_matrix(4.0),
        applied_token_weight=_weights(),
        applied_weight_path="current_step",
        signals={"signal_timing": "pre_update"},
    )
    values.update(overrides)
    return CandidateOccurrenceBatch(**values)


def _rows(directory: Path, domain: str) -> list[dict]:
    with gzip.open(
        directory / f"candidate_occurrences_{domain}.jsonl.gz",
        mode="rt",
        encoding="utf-8",
    ) as handle:
        return [json.loads(line) for line in handle]


def test_writer_aligns_fields_and_filters_candidates_per_domain(tmp_path) -> None:
    totals = write_candidate_occurrence_log(
        output_dir=tmp_path,
        batch=_batch(),
        domain_candidate_token_ids=CANDIDATES,
        candidate_source="domain_control_token_candidate_ids",
        reference_fraction=0.0,
        compression="gzip",
    )
    directory = step_jsonl_dir(tmp_path, 4)
    math_rows = _rows(directory, "math")
    code_rows = _rows(directory, "code")

    assert [row["response_uid"] for row in math_rows] == [
        "step=4:batch_index=0",
        "step=4:batch_index=1",
    ]
    assert code_rows[0]["response_uid"] == "step=4:batch_index=2"
    for row in math_rows + code_rows:
        lengths = {len(row[name]) for name in POSITION_ARRAY_FIELDS if row[name] is not None}
        assert len(lengths) == 1
        assert row["position_eligible"] is None

    first = math_rows[0]
    # Padded ID-5 at column 4 is invalid and must not appear; 9 is not a math candidate.
    assert first["response_position"] == [0, 1, 3]
    assert first["token_id"] == [5, 7, 5]
    assert first["is_candidate"] == [True, True, True]
    assert first["valid_token_count"] == 4
    assert first["relative_position"] == pytest.approx([0.0, 0.25, 0.75])
    assert first["configured_token_loss"] == pytest.approx([0.0, 0.1, 0.3])
    assert first["abs_logp_gap"] == pytest.approx([0.0, 0.1, 0.3])
    assert first["student_entropy"] == pytest.approx([2.0, 2.1, 2.3])
    assert first["teacher_entropy"] == pytest.approx([3.0, 3.1, 3.3])
    assert first["teacher_student_cross_entropy"] == pytest.approx([4.0, 4.1, 4.3])
    assert first["applied_weight"] == pytest.approx([2.0, 0.5, 2.0])
    assert first["row_base_weight"] == pytest.approx(0.5)
    assert first["selected"] == [True, False, True]

    second = math_rows[1]
    assert second["response_position"] == [0]
    assert second["selected"] == [True]

    code = code_rows[0]
    assert code["response_position"] == [1, 2]
    assert code["token_id"] == [8, 8]
    assert code["selected"] == [False, False]

    loss = _matrix(0.0).double()
    weights = _weights().double()
    math_valid = MASK[:2].double()
    assert totals["math"]["valid_token_count"] == 10
    assert totals["math"]["candidate_position_count"] == 4
    assert totals["math"]["noncandidate_valid_position_count"] == 6
    assert totals["math"]["reference_position_count"] == 0
    assert totals["math"]["selected_candidate_position_count"] == 3
    assert totals["math"]["selected_noncandidate_valid_position_count"] == 0
    assert totals["math"]["sum_configured_token_loss"] == pytest.approx(
        float((loss[:2] * math_valid).sum())
    )
    assert totals["math"]["sum_weighted_configured_token_loss"] == pytest.approx(
        float((loss[:2] * weights[:2] * math_valid).sum())
    )
    assert totals["code"]["valid_token_count"] == 3
    assert totals["code"]["candidate_position_count"] == 2
    assert totals["code"]["selected_candidate_position_count"] == 0

    manifest = json.loads((directory / CANDIDATE_OCCURRENCE_MANIFEST).read_text())
    assert manifest["totals_by_domain"] == json.loads(json.dumps(totals))
    assert manifest["candidate_token_ids"]["by_domain"] == {
        "math": [5, 7],
        "code": [8],
    }
    assert manifest["applied_weight"]["available"] is True
    assert manifest["applied_weight"]["path"] == "current_step"
    assert set(manifest["per_position_fields"]) == set(POSITION_ARRAY_FIELDS)
    assert "candidate position" in manifest["definition"]

    flat = list(
        iter_candidate_positions(directory / "candidate_occurrences_math.jsonl.gz")
    )
    assert len(flat) == 4
    assert flat[2]["response_uid"] == "step=4:batch_index=0"
    assert flat[2]["response_position"] == 3
    assert flat[3]["applied_weight"] == pytest.approx(2.0)


def _big_batch(step: int) -> CandidateOccurrenceBatch:
    generator = torch.Generator().manual_seed(0)
    rows, width = 40, 250
    ids = torch.randint(0, 50, (rows, width), generator=generator)
    mask = torch.ones(rows, width)
    mask[:, 200:] = 0
    values = torch.rand(rows, width, generator=generator)
    return CandidateOccurrenceBatch(
        step=step,
        domains=tuple("math" if i % 2 == 0 else "code" for i in range(rows)),
        sample_ids=tuple(f"s{i}" for i in range(rows)),
        response_token_ids=ids,
        response_mask=mask,
        student_log_prob=-values,
        teacher_log_prob=-values,
        configured_token_loss=values,
    )


def _reference_keys(output_dir: Path, step: int) -> dict[str, set]:
    directory = step_jsonl_dir(output_dir, step)
    keys = {}
    for domain in ("math", "code"):
        keys[domain] = {
            (row["response_uid"], row["response_position"])
            for row in iter_candidate_positions(
                directory / f"candidate_occurrences_{domain}.jsonl.gz"
            )
            if not row["is_candidate"]
        }
    return keys


def test_reference_sample_is_deterministic_by_step_and_sized(tmp_path) -> None:
    kwargs = dict(
        domain_candidate_token_ids={"math": (1, 2, 3), "code": (4,)},
        candidate_source="test",
        reference_fraction=0.05,
        compression="gzip",
    )
    first = write_candidate_occurrence_log(
        output_dir=tmp_path / "a", batch=_big_batch(9), **kwargs
    )
    write_candidate_occurrence_log(
        output_dir=tmp_path / "b", batch=_big_batch(9), **kwargs
    )
    write_candidate_occurrence_log(
        output_dir=tmp_path / "c", batch=_big_batch(10), **kwargs
    )
    a = _reference_keys(tmp_path / "a", 9)
    b = _reference_keys(tmp_path / "b", 9)
    c = _reference_keys(tmp_path / "c", 10)

    assert a == b
    strip = lambda keys: {(uid.split(":")[1], pos) for uid, pos in keys}  # noqa: E731
    assert strip(a["math"]) != strip(c["math"])
    for domain in ("math", "code"):
        total = first[domain]
        assert total["reference_position_count"] == round(
            0.05 * total["noncandidate_valid_position_count"]
        )
        assert len(a[domain]) == total["reference_position_count"]
        assert total["reference_seed"] == reference_sampling_seed(9, domain)
        assert total["valid_token_count"] == 20 * 200
        # Without returned weights the writer must not fabricate them.
        assert total["sum_weighted_configured_token_loss"] is None
        assert total["selected_candidate_position_count"] is None
    flat = list(
        iter_candidate_positions(
            step_jsonl_dir(tmp_path / "a", 9) / "candidate_occurrences_math.jsonl.gz"
        )
    )
    assert all(row["applied_weight"] is None for row in flat)
    assert all(row["selected"] is None for row in flat)
    assert all(row["response_position"] < 200 for row in flat)
    candidate_ids = {1, 2, 3}
    assert all((row["token_id"] in candidate_ids) == row["is_candidate"] for row in flat)


def _audit_batch(with_weights: bool) -> SimpleNamespace:
    tensors = {
        "old_log_probs": -_matrix(1.0),
        "response_mask": MASK,
        "responses": IDS,
        "configured_token_loss": _matrix(0.0),
        "configured_token_loss_mask": MASK,
        "student_entropy": _matrix(2.0),
        "math_teacher_log_prob": -_matrix(0.5),
        "code_teacher_log_prob": -_matrix(0.5),
        "math_teacher_entropy": _matrix(3.0),
        "code_teacher_entropy": _matrix(3.0),
        "teacher_student_cross_entropy": _matrix(4.0),
    }
    if with_weights:
        tensors["mopd_applied_token_weight"] = _weights()
    return SimpleNamespace(
        batch=tensors,
        non_tensor_batch={
            "opd_teacher": ["math", "math", "code"],
            "sample_id": ["m0", "m1", "c0"],
        },
        meta_info={
            "mopd_configured_token_loss_name": "topk_renormalized_reverse_kl",
            "topk_distill_support_source": "teacher",
            "teacher_topk_k": 32,
        },
    )


def _audit_logger(output_dir: Path, **extra) -> MOPDAuditLogger:
    audit = {
        "enabled": True,
        "output_dir": str(output_dir),
        "domains": ["math", "code"],
        "log_sample_level": False,
        "token_gap_enabled": False,
        "entropy_enabled": False,
        "candidate_occurrence_log_enabled": True,
        "domain_control_token_candidate_ids": {"math": [5, 7], "code": [8]},
        "control_token_online_selection_timing": "current_step",
    }
    audit.update(extra)
    return MOPDAuditLogger(
        {
            "mopd_audit": audit,
            "actor_rollout_ref": {"actor": {"policy_loss": {"lambda_vals": 1.0}}},
        }
    )


def test_audit_logger_writes_returned_weights_and_nulls_missing_ones(tmp_path) -> None:
    logger = _audit_logger(tmp_path)
    metrics = logger.log_training_step(_audit_batch(True), step=2, lr=1e-5)
    assert "mopd/global/audit/error" not in metrics
    directory = step_jsonl_dir(tmp_path, 2)
    manifest = json.loads((directory / CANDIDATE_OCCURRENCE_MANIFEST).read_text())
    rows = _rows(directory, "math")
    assert rows[0]["applied_weight"] == pytest.approx([2.0, 0.5, 2.0])
    assert rows[0]["selected"] == [True, False, True]
    assert manifest["applied_weight"]["path"] == "current_step"
    assert manifest["candidate_token_ids"]["source"] == (
        "domain_control_token_candidate_ids"
    )
    assert not (directory / "response_records_math.jsonl.gz").exists()

    logger.log_training_step(_audit_batch(False), step=3, lr=1e-5)
    directory = step_jsonl_dir(tmp_path, 3)
    manifest = json.loads((directory / CANDIDATE_OCCURRENCE_MANIFEST).read_text())
    assert manifest["applied_weight"]["available"] is False
    assert "did not return" in manifest["applied_weight"]["unavailable_reason"]
    assert _rows(directory, "math")[0]["applied_weight"] is None


def test_audit_logger_respects_frequency(tmp_path) -> None:
    logger = _audit_logger(tmp_path, candidate_occurrence_log_freq_steps=5)
    assert not logger.should_log_candidate_occurrence(3)
    assert logger.should_log_candidate_occurrence(5)
    logger.log_training_step(_audit_batch(True), step=3, lr=1e-5)
    assert not (step_jsonl_dir(tmp_path, 3) / CANDIDATE_OCCURRENCE_MANIFEST).exists()


def test_candidate_occurrence_switch_defaults_off() -> None:
    audit = AuditConfig()
    assert audit.candidate_occurrence_log_enabled is False
    assert audit.candidate_occurrence_log_freq_steps == 1
    assert audit.candidate_occurrence_reference_fraction == 0.01
    assert audit.candidate_occurrence_log_compression == "gzip"


def test_candidate_occurrence_switch_defaults_off_in_logger(tmp_path) -> None:
    logger = MOPDAuditLogger(
        {"mopd_audit": {"enabled": True, "output_dir": str(tmp_path)}}
    )
    assert logger.candidate_occurrence_log_enabled is False
    assert not logger.should_log_candidate_occurrence(1)


def test_existing_profile_keeps_switch_off_and_launch_renders_it() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        str(
            root
            / "configs/token_selection/math/taxonomy/"
            "top32kl_next_step_full_taxonomy_split_topp0p05_i1_w1_5gpu_b256.yaml"
        )
    )
    assert config.audit.candidate_occurrence_log_enabled is False
    rendered = format_command(build_command(config))
    if config.audit.enabled:
        assert "+mopd_audit.candidate_occurrence_log_enabled=false" in rendered
        assert "+mopd_audit.candidate_occurrence_reference_fraction=0.01" in rendered


def test_full_vocabulary_candidate_scope_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="configured candidate set"):
        _audit_logger(
            tmp_path,
            control_token_online_candidate_scope="full_vocabulary",
            control_token_online_candidate_vocab_size=32,
        )
