"""Real-batch audit regression for the student-entropy selection baselines.

The runs q1p7b-{math,code}-opdnative-ent{top20,tail80}-toklog-8g8c-b256-s42 (2026-10-21/22) trained with every audit
step failing: the configs inherited audit.loss_variance_signal=policy_gradient_distillation_signal, while the trainer
names the entropy-weighted loss policy_gradient_distillation_signal+entropy_token_weighting. verl_audit refused each
batch before it recorded training_accuracy or wrote any per-token file; training went on, so only audit_errors.jsonl
and training_cost.jsonl were left. Checking YAML switches or the selection algorithm did not catch this, so these tests
push a trainer-shaped batch, built from the real configs, through MOPDAuditLogger.log_training_step and read the files
that land on disk.
"""

from __future__ import annotations

import ast
import gzip
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from mopd_verl.audit_io import step_jsonl_dir
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from mopd_verl.token_baselines import (
    build_token_baseline_weights,
    configured_token_loss_name,
    validate_loss_variance_signal,
)
from mopd_verl.verl_audit import MOPDAuditLogger

ROOT = Path(__file__).resolve().parents[1]
ENTROPY_CONFIGS = sorted((ROOT / "configs").rglob("*_ent_*_toklog_8gpu_8c_b256.yaml"))
ENTROPY_SIGNAL = "policy_gradient_distillation_signal+entropy_token_weighting"
OLD_SIGNAL = "policy_gradient_distillation_signal"
POLICY_LOSS_PREFIX = "actor_rollout_ref.actor.policy_loss."
RESPONSE_LENGTHS = (12, 9, 5)  # per domain; the batch is padded to the longest
CORRECT = (1.0, 0.0, 1.0)  # binary score at the last valid token -> training_accuracy 2/3
PER_TOKEN_FILES = (
    "domain_step_metrics.jsonl",
    "loss_variance_domain_step.jsonl",
    "token_gap_vectors.jsonl",
    "token_gap_vocab_vectors.jsonl",
    "logp_vectors.jsonl",
    "logp_abs_vectors.jsonl",
    "logp_vocab_vectors.jsonl",
    "logp_abs_vocab_vectors.jsonl",
    "entropy_distribution_vectors.jsonl",
    "entropy_vocab_vectors.jsonl",
    "topk_teacher_student_cross_entropy_vocab_vectors.jsonl",
    "response_manifest.json",
)


def _config_id(path: Path) -> str:
    return str(path.relative_to(ROOT / "configs"))


def _policy_loss(config) -> dict:
    """actor_rollout_ref.actor.policy_loss as the launched trainer receives it (launch.py overrides)."""
    policy_loss = {}
    for argument in build_command(config):
        key, _, value = argument.lstrip("+").partition("=")
        if key.startswith(POLICY_LOSS_PREFIX):
            policy_loss[key[len(POLICY_LOSS_PREFIX):]] = yaml.safe_load(value)
    return policy_loss


def _logger(config, output_dir: Path, policy_loss: dict, **audit_overrides) -> MOPDAuditLogger:
    audit = {**asdict(config.audit), "output_dir": str(output_dir), **audit_overrides}
    return MOPDAuditLogger({"mopd_audit": audit, "actor_rollout_ref": {"actor": {"policy_loss": policy_loss}}})


def _teacher_prefixes(domains: list[str], policy_loss: dict) -> list[str]:
    """Key prefix of each domain's teacher tensors in the trainer batch.

    The first reference worker always writes math_teacher_{log_prob,entropy} (fsdp_workers.py) and the second
    code_teacher_*, so a single-teacher run (Math-only or Code-only, multi_teacher_distill false) carries its only
    teacher under the math_ prefix whatever the domain.
    """
    return list(domains) if policy_loss["multi_teacher_distill"] else ["math"] * len(domains)


def _trainer_batch(domains: list[str], policy_loss: dict) -> SimpleNamespace:
    """The tensors the trainer has attached by the time it calls the audit (toklog on)."""
    generator = torch.Generator().manual_seed(0)
    lengths = [length for _ in domains for length in RESPONSE_LENGTHS]
    batch_size, width = len(lengths), max(RESPONSE_LENGTHS)
    mask = torch.zeros(batch_size, width)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1.0
    old_log_probs = -3.0 * torch.rand(batch_size, width, generator=generator) * mask
    student_entropy = 2.0 * torch.rand(batch_size, width, generator=generator) * mask
    tensors = {
        "responses": torch.randint(10, 200, (batch_size, width), generator=generator) * mask.long(),
        "response_mask": mask,
        "old_log_probs": old_log_probs,
        "student_entropy": student_entropy,
        "teacher_student_cross_entropy": (1.0 + torch.rand(batch_size, width, generator=generator)) * mask,
    }
    teacher_log_probs = torch.zeros_like(old_log_probs)
    for index, prefix in enumerate(_teacher_prefixes(domains, policy_loss)):
        domain_teacher = (old_log_probs + 0.5 * torch.randn(batch_size, width, generator=generator)) * mask
        tensors[f"{prefix}_teacher_log_prob"] = domain_teacher
        tensors[f"{prefix}_teacher_entropy"] = torch.rand(batch_size, width, generator=generator) * mask
        rows = slice(index * len(RESPONSE_LENGTHS), (index + 1) * len(RESPONSE_LENGTHS))
        teacher_log_probs[rows] = domain_teacher[rows]
    # Native OPD signal log pS - log pT, reweighted by the configured entropy selection (0/1, unnormalised).
    weights = build_token_baseline_weights(
        student_entropy=student_entropy, response_mask=mask, policy_loss_config=policy_loss
    ).weights
    tensors["configured_token_loss"] = (old_log_probs - teacher_log_probs) * weights
    tensors["configured_token_loss_mask"] = mask.clone()
    tensors["advantages"] = -tensors["configured_token_loss"]
    scores = torch.zeros(batch_size, width)
    for row, length in enumerate(lengths):
        scores[row, length - 1] = CORRECT[row % len(CORRECT)]
    tensors["token_level_scores"] = scores
    tensors["token_level_rewards"] = scores.clone()
    labels = [domain for domain in domains for _ in RESPONSE_LENGTHS]
    return SimpleNamespace(
        batch=tensors,
        non_tensor_batch={
            "opd_teacher": np.array(labels, dtype=object),
            "sample_id": np.array([f"{label}-{row}" for row, label in enumerate(labels)], dtype=object),
            "uid": np.array([f"uid-{row}" for row in range(batch_size)], dtype=object),
        },
        meta_info={
            "mopd_configured_token_loss_name": configured_token_loss_name(policy_loss),
            "mopd_configured_token_loss_epoch_reduction": "mean",
            "mopd_configured_token_loss_epoch_count": 1,
            "topk_distill_support_source": "teacher",
            "teacher_topk_k": 32,
            "topk_distill_include_tail": False,
            "topk_distill_temperature": 1.0,
        },
    )


def _jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.mark.parametrize("config_path", ENTROPY_CONFIGS, ids=_config_id)
def test_entropy_run_batch_writes_accuracy_and_per_token_audit(config_path: Path, tmp_path: Path) -> None:
    config = load_config(config_path)
    domains = list(config.audit.domains)
    policy_loss = _policy_loss(config)
    assert configured_token_loss_name(policy_loss) == config.audit.loss_variance_signal == ENTROPY_SIGNAL
    assert f"+mopd_audit.loss_variance_signal={ENTROPY_SIGNAL}" in build_command(config)

    logger = _logger(config, tmp_path, policy_loss)
    batch = _trainer_batch(domains, policy_loss)
    metrics = logger.log_training_step(batch, step=1, lr=1e-6)
    directory = step_jsonl_dir(tmp_path, 1)

    assert "global/audit/error" not in metrics
    assert not (directory / "audit_errors.jsonl").exists()

    # training_accuracy: metric (also after the W&B core filter), its own record, and the domain row.
    logged = logger.filter_tensorboard_metrics(metrics)
    for key in ["global/reward/training_accuracy"] + [f"{domain}/reward/training_accuracy" for domain in domains]:
        assert metrics[key] == pytest.approx(2 / 3), key
        assert logged[key] == pytest.approx(2 / 3), key
    accuracy_rows = {row["domain"]: row for row in _jsonl(directory / "training_accuracy.jsonl")}
    assert accuracy_rows["all"]["response_count"] == len(domains) * len(RESPONSE_LENGTHS)
    for domain in domains:
        assert (accuracy_rows[domain]["response_count"], accuracy_rows[domain]["correct_count"]) == (3, 2)
        assert accuracy_rows[domain]["source"] == "token_level_scores"

    for name in PER_TOKEN_FILES:
        assert (directory / name).stat().st_size > 0, name
    domain_rows = {row["domain"]: row for row in _jsonl(directory / "domain_step_metrics.jsonl")}
    tokens = sum(RESPONSE_LENGTHS)
    for domain in domains:
        row = domain_rows[domain]
        assert row["training_accuracy"] == pytest.approx(2 / 3)
        assert row["domain_token_count"] == tokens
        for key in ("gap_signed_mean", "gap_abs_mean", "teacher_student_gap_mean", "student_entropy_mean", "teacher_entropy_mean"):
            assert math.isfinite(row[key]), (domain, key)
    for name, field in (
        ("token_gap_vectors.jsonl", "gap_signed_vector_domain"),
        ("token_gap_vectors.jsonl", "gap_abs_vector_domain"),
        ("logp_vectors.jsonl", "logp_vector_domain"),
        ("logp_abs_vectors.jsonl", "logp_abs_vector_domain"),
    ):
        vectors = {row["domain"]: row for row in _jsonl(directory / name)}
        assert set(vectors) == set(domains), name
        assert all(len(vectors[domain][field]) == tokens for domain in domains), (name, field)
    assert {row["domain"] for row in _jsonl(directory / "entropy_distribution_vectors.jsonl")} == set(domains)

    # Per-response records: one row per response, every per-token array aligned with the response length.
    manifest = json.loads((directory / "response_manifest.json").read_text())
    assert manifest["configured_token_loss"]["name"] == ENTROPY_SIGNAL
    assert manifest["response_count"] == len(domains) * len(RESPONSE_LENGTHS)
    weights = (batch.batch["configured_token_loss"] != 0).float()
    prefixes = _teacher_prefixes(domains, policy_loss)
    for index, domain in enumerate(domains):
        records = _jsonl(directory / f"response_records_{domain}.jsonl.gz")
        assert [len(record["response_token_ids"]) for record in records] == list(RESPONSE_LENGTHS)
        for offset, record in enumerate(records):
            row = index * len(RESPONSE_LENGTHS) + offset
            length = RESPONSE_LENGTHS[offset]
            for field in ("student_entropy", "teacher_entropy", "student_token_log_prob", "teacher_token_log_prob", "configured_token_loss"):
                assert len(record[field]) == length, field
            assert record["student_token_log_prob"] == pytest.approx(batch.batch["old_log_probs"][row, :length].tolist())
            assert record["teacher_token_log_prob"] == pytest.approx(
                batch.batch[f"{prefixes[index]}_teacher_log_prob"][row, :length].tolist()
            )
            assert record["configured_token_loss"] == pytest.approx(batch.batch["configured_token_loss"][row, :length].tolist())
        # The entropy selection zeroes the loss on unselected positions; the records keep that.
        selected = weights[index * 3 : index * 3 + 3]
        assert 0 < float(selected.sum()) < float(batch.batch["response_mask"][index * 3 : index * 3 + 3].sum())


@pytest.mark.parametrize("config_path", ENTROPY_CONFIGS, ids=_config_id)
def test_old_entropy_signal_is_refused_at_config_load(config_path: Path, tmp_path: Path) -> None:
    stale = tmp_path / "stale_signal.yaml"
    stale.write_text(f"extends: {config_path}\naudit:\n  loss_variance_signal: {OLD_SIGNAL}\n")
    with pytest.raises(ValueError, match="does not match the production token loss"):
        load_config(stale)


def test_signal_check_is_kept_and_audit_failure_still_records_training_accuracy(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen1p7b_suite/math_code/opd_native_ent_top20_toklog_8gpu_8c_b256.yaml")
    policy_loss = _policy_loss(config)
    # The 2026-10-21 state: old signal in the audit, entropy-weighted name from the trainer.
    logger = _logger(config, tmp_path, policy_loss, loss_variance_signal=OLD_SIGNAL)
    metrics = logger.log_training_step(_trainer_batch(["math", "code"], policy_loss), step=3, lr=1e-6)
    directory = step_jsonl_dir(tmp_path, 3)

    assert metrics["global/audit/error"] == 1.0
    for key in ("global/reward/training_accuracy", "math/reward/training_accuracy", "code/reward/training_accuracy"):
        assert metrics[key] == pytest.approx(2 / 3), key
    assert {row["domain"] for row in _jsonl(directory / "training_accuracy.jsonl")} == {"all", "math", "code"}
    (error,) = _jsonl(directory / "audit_errors.jsonl")
    assert error["stage"] == "training"
    assert OLD_SIGNAL in error["error"] and ENTROPY_SIGNAL in error["error"]
    assert not (directory / "token_gap_vectors.jsonl").exists()
    assert not (directory / "response_manifest.json").exists()

    with pytest.raises(ValueError, match="does not match the production token loss"):
        validate_loss_variance_signal(OLD_SIGNAL, configured_token_loss_name(policy_loss))
    validate_loss_variance_signal("opd_loss_token", configured_token_loss_name(policy_loss))


def test_trainer_checks_the_signal_at_startup_with_the_shared_loss_name() -> None:
    source = (ROOT / "third_party/verl/verl/trainer/ppo/ray_trainer.py").read_text()
    trainer = next(
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer"
    )
    methods = {node.name: node for node in trainer.body if isinstance(node, ast.FunctionDef)}

    def calls(function: ast.FunctionDef) -> dict[str, int]:
        """First line on which each function name is called."""
        first: dict[str, int] = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                first[name] = min(node.lineno, first.get(name, node.lineno))
        return first

    init_calls = calls(methods["__init__"])
    assert "validate_loss_variance_signal" in init_calls and "configured_token_loss_name" in init_calls
    assert init_calls["validate_loss_variance_signal"] < init_calls["_create_dataloader"]
    # The per-step batch name comes from the same helper, so the two cannot drift apart.
    assert "configured_token_loss_name" in calls(methods["_fit_impl"])
    assert "configured_distill_loss_name" not in source
