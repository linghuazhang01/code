"""Contracts for the Math+Code equal-weight Top-32 OPD toklog configs at global batch 528.

Qwen3-1.7B: configs/qwen1p7b_suite/math_code/opd_toklog_*; Qwen3-4B: configs/qwen4b_suite/math_code/opd_toklog_*.
Every file logs per-token data (token gap / entropy / logp vectors, response records) on top of the plain OPD run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mopd_verl.config_profiles import load_raw_config
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SUITE_1P7B = CONFIGS / "qwen1p7b_suite/math_code"
SUITE_4B = CONFIGS / "qwen4b_suite/math_code"
# (1.7B file, 4B toklog file, 4B plain file, GPUs, actor GPUs)
LAYOUTS = [
    ("opd_toklog_3gpu_3c.yaml", "opd_toklog_3gpu_3c.yaml", "opd_3gpu_3c.yaml", 3, 3),
    ("opd_toklog_4gpu_3s1t.yaml", "opd_toklog_4gpu_3s1t.yaml", "opd_4gpu_3s1t.yaml", 4, 3),
    ("opd_toklog_8gpu_8c.yaml", "opd_toklog_8gpu_6s2t.yaml", "opd_8gpu_6s2t.yaml", 8, 8),
]
IDENTITY_KEYS = (
    "audit.output_dir",
    "paper_eval.output_dir",
    "huggingface_checkpoint.path_prefix",
    "runtime.wandb_run_id",
    "trainer.experiment_name",
    "trainer.default_local_dir",
)
PER_TOKEN_AUDIT = (
    "token_gap_enabled",
    "token_gap_vocab_vector_enabled",
    "entropy_enabled",
    "entropy_vocab_vector_enabled",
    "entropy_vocab_per_occurrence_mean_vector_enabled",
    "logp_vector_enabled",
    "logp_abs_vector_enabled",
    "logp_vocab_per_occurrence_mean_vector_enabled",
    "logp_abs_vocab_per_occurrence_mean_vector_enabled",
    "topk_teacher_student_cross_entropy_vocab_enabled",
    "response_level_enabled",
    "log_sample_level",
)


def _flat(tree: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flat(value, path))
        else:
            out[path] = value
    return out


def _changed(left: Path, right: Path) -> set[str]:
    a, b = _flat(load_raw_config(left)), _flat(load_raw_config(right))
    return {key for key in set(a) | set(b) if a.get(key) != b.get(key)}


def _assert_per_token_audit(cfg) -> None:
    for key in PER_TOKEN_AUDIT:
        assert getattr(cfg.audit, key) is True, key
    assert cfg.audit.response_level_freq_steps == 1
    assert cfg.audit.token_gap_freq_steps == 1
    # logging only: the objective stays the equal-weight Top-32 RKL
    assert cfg.audit.control_token_loss_weighting_enabled is False
    assert cfg.audit.control_token_online_selection_enabled is False


@pytest.mark.parametrize("name,_4b,_plain,gpus,actors", LAYOUTS)
def test_1p7b_b528_toklog_contract(name: str, _4b: str, _plain: str, gpus: int, actors: int) -> None:
    cfg = load_config(SUITE_1P7B / name)
    assert cfg.model.student_path.endswith("Qwen3-1.7B")
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == 528
    assert 528 % actors == 0
    assert cfg.worker_placement.actor_rollout.n_gpus_per_node == actors
    assert cfg.runtime.slurm_allocation_gpus == gpus
    assert cfg.actor.distill_mode == "topk_renormalized_reverse_kl"
    assert cfg.actor.topk_distill_enabled and cfg.actor.topk_distill_k == 32
    assert cfg.actor.token_baseline_method == "none"
    assert cfg.actor.loss_agg_mode == "token-mean"
    assert cfg.actor.multi_teacher_distill is True
    _assert_per_token_audit(cfg)
    assert cfg.trainer.total_training_steps == 60
    assert cfg.huggingface_checkpoint.steps == (60,)
    run = cfg.trainer.experiment_name
    assert run.startswith("q1p7b-mc-opd-toklog-") and run.endswith("-b528-s42")
    assert cfg.runtime.wandb_run_id == run and len(run) <= 64
    assert cfg.huggingface_checkpoint.path_prefix == f"checkpoints/baselines/math_code/{run}"
    assert cfg.trainer.default_local_dir == f"checkpoints/MOPD/{run}"
    assert cfg.audit.output_dir == f"audit/{run}"
    command = build_command(cfg)
    assert "data.train_batch_size=528" in command
    assert "trainer.total_training_steps=60" in command
    assert "+mopd_audit.token_gap_vocab_vector_enabled=true" in command
    assert "+mopd_audit.log_sample_level=true" in command


@pytest.mark.parametrize("name,name_4b,_plain,gpus,actors", LAYOUTS)
def test_1p7b_b528_differs_from_4b_toklog_only_in_student_identities_and_memory(
    name: str, name_4b: str, _plain: str, gpus: int, actors: int
) -> None:
    changed = _changed(SUITE_1P7B / name, SUITE_4B / name_4b)
    allowed = set(IDENTITY_KEYS) | {"model.student_path", "rollout.gpu_memory_utilization"}
    assert changed <= allowed, sorted(changed - allowed)
    assert "model.student_path" in changed


@pytest.mark.parametrize("_1p7b,name_4b,plain,gpus,actors", LAYOUTS)
def test_4b_math_code_toklog_only_adds_logging_to_the_plain_528_run(
    _1p7b: str, name_4b: str, plain: str, gpus: int, actors: int
) -> None:
    cfg = load_config(SUITE_4B / name_4b)
    assert cfg.model.student_path.endswith("Qwen3-4B")
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == 528
    assert cfg.trainer.total_training_steps == 60 and cfg.huggingface_checkpoint.steps == (60,)
    _assert_per_token_audit(cfg)
    run = cfg.trainer.experiment_name
    assert run.startswith("q4b-mc-opd-toklog-") and run.endswith("-b528-s42")
    assert cfg.runtime.wandb_run_id == run and len(run) <= 64
    changed = _changed(SUITE_4B / name_4b, SUITE_4B / plain)
    audit_keys = {key for key in changed if key.startswith("audit.")}
    assert changed - audit_keys <= set(IDENTITY_KEYS), sorted(changed - audit_keys - set(IDENTITY_KEYS))
    # the loss-defining audit field is untouched
    assert "audit.loss_variance_signal" not in changed


def test_b528_run_names_are_unique() -> None:
    runs = []
    for name, name_4b, plain, _gpus, _actors in LAYOUTS:
        runs.append(load_config(SUITE_1P7B / name).trainer.experiment_name)
        runs.append(load_config(SUITE_4B / name_4b).trainer.experiment_name)
        runs.append(load_config(SUITE_4B / plain).trainer.experiment_name)
    assert len(set(runs)) == len(runs)
