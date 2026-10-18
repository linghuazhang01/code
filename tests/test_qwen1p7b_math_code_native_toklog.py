"""Contracts for the Qwen3-1.7B Math+Code native OPD ("OPD naive") configs with per-token audit.

configs/qwen1p7b_suite/math_code/opd_native_toklog_{3gpu_3c_b255,4gpu_4c_b256,8gpu_8c_b256}.yaml
Chosen-token policy-gradient OPD (no Top-32, no weighting, no selection), colocated teacher, 60 steps.
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
# (file, GPUs, batch, 4B file with the same layout or None)
LAYOUTS = [
    ("opd_native_toklog_3gpu_3c_b255.yaml", 3, 255, "opd_native_3gpu_3c.yaml"),
    ("opd_native_toklog_4gpu_4c_b256.yaml", 4, 256, None),
    ("opd_native_toklog_8gpu_8c_b256.yaml", 8, 256, "opd_native_8gpu_8c.yaml"),
]
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
IDENTITY_KEYS = (
    "audit.output_dir",
    "paper_eval.output_dir",
    "huggingface_checkpoint.path_prefix",
    "runtime.wandb_run_id",
    "trainer.experiment_name",
    "trainer.default_local_dir",
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


@pytest.mark.parametrize("name,gpus,batch,_4b", LAYOUTS)
def test_native_toklog_contract(name: str, gpus: int, batch: int, _4b: str | None) -> None:
    cfg = load_config(SUITE_1P7B / name)
    assert cfg.model.student_path.endswith("Qwen3-1.7B")
    # native chosen-token policy gradient, not the Top-32 RKL
    assert cfg.actor.distill_loss_builder == "policy_gradient"
    assert cfg.actor.distill_mode == "chosen_token_policy_gradient"
    assert cfg.actor.topk_distill_enabled is False
    assert cfg.actor.token_baseline_method == "none"
    assert cfg.actor.loss_agg_mode == "token-mean"
    assert cfg.audit.loss_variance_signal == "policy_gradient_distillation_signal"
    # per-token audit on, logging only
    for key in PER_TOKEN_AUDIT:
        assert getattr(cfg.audit, key) is True, key
    assert cfg.audit.response_level_freq_steps == 1 and cfg.audit.token_gap_freq_steps == 1
    assert cfg.audit.control_token_loss_weighting_enabled is False
    assert cfg.audit.control_token_online_selection_enabled is False
    # colocated layout, batch rule (3 GPUs 255; 4 and 8 GPUs 256), vLLM 0.6
    assert cfg.worker_placement.separate_ref_policy is False
    assert cfg.worker_placement.actor_rollout.n_gpus_per_node == gpus
    assert cfg.runtime.slurm_allocation_gpus == cfg.trainer.n_gpus_per_node == gpus
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == batch
    assert f"actor_rollout_ref.ref.fsdp_config.fsdp_size={gpus}" in cfg.extra_overrides
    assert cfg.rollout.gpu_memory_utilization == 0.6
    assert cfg.trainer.total_training_steps == 60 and cfg.huggingface_checkpoint.steps == (60,)
    # names state what the file runs
    assert name == f"opd_native_toklog_{gpus}gpu_{gpus}c_b{batch}.yaml"
    run = cfg.trainer.experiment_name
    assert run == f"q1p7b-mc-opdnative-toklog-{gpus}g{gpus}c-b{batch}-s42"
    assert cfg.runtime.wandb_run_id == run and len(run) <= 64
    assert cfg.huggingface_checkpoint.path_prefix == f"checkpoints/baselines/math_code/{run}"
    assert cfg.trainer.default_local_dir == f"checkpoints/MOPD/{run}"
    assert cfg.audit.output_dir == f"audit/{run}"
    command = build_command(cfg)
    assert f"data.train_batch_size={batch}" in command
    assert "trainer.total_training_steps=60" in command
    assert "+mopd_audit.token_gap_vocab_vector_enabled=true" in command
    assert "+mopd_audit.log_sample_level=true" in command


@pytest.mark.parametrize("name,_gpus,_batch,name_4b", [row for row in LAYOUTS if row[3]])
def test_native_toklog_differs_from_4b_native_only_in_student_batch_and_names(
    name: str, _gpus: int, _batch: int, name_4b: str
) -> None:
    changed = _changed(SUITE_1P7B / name, SUITE_4B / name_4b)
    allowed = set(IDENTITY_KEYS) | {
        "model.student_path",
        "data.train_batch_size",
        "actor.ppo_mini_batch_size",
    }
    assert changed <= allowed, sorted(changed - allowed)
    assert "model.student_path" in changed


def test_native_toklog_run_names_are_unique() -> None:
    runs = [load_config(SUITE_1P7B / name).trainer.experiment_name for name, *_ in LAYOUTS]
    assert len(set(runs)) == len(runs)


def test_4b_math_code_native_has_per_token_audit() -> None:
    for name in ("opd_native_3gpu_3c.yaml", "opd_native_8gpu_8c.yaml"):
        cfg = load_config(SUITE_4B / name)
        for key in PER_TOKEN_AUDIT:
            assert getattr(cfg.audit, key) is True, (name, key)
        assert cfg.actor.distill_mode == "chosen_token_policy_gradient"
        assert cfg.audit.control_token_loss_weighting_enabled is False
