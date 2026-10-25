"""Student-entropy token-selection baselines (top 20% / tail 80%) for Qwen3-4B native OPD, Math-only and Code-only.

configs/qwen4b_suite/{math,code}/opd_native_ent_{top20,tail80}_toklog_{4gpu_4c,8gpu_8c}_b256.yaml
Qwen3-4B twins of configs/token_selection/{math/taxonomy,code}/native_opd_ent_{top20,tail80}_toklog_*_b256.yaml; the
equal-weight reference is the 3-GPU naive run q4b-{math,code}-opdnative-toklog-3g3c-b255-s42.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mopd_verl.launch import build_command
from mopd_verl.settings import load_config

from test_entropy_selection_baselines import SELECTION, TOPOLOGY
from test_qwen1p7b_math_code_native_toklog import IDENTITY_KEYS, PER_TOKEN_AUDIT, _changed

SUITE = Path(__file__).resolve().parents[1] / "configs/qwen4b_suite"
BASE = "opd_native_toklog_3gpu_3c.yaml"
DOMAINS = ("math", "code")
ARMS = {"top20": ("enttop20", 0.2, "topk"), "tail80": ("enttail80", 0.8, "bottomk")}
# layout: (run-name tag, GPUs); every layout is colocated, batch 256.
LAYOUTS = {"8gpu_8c": ("8g8c", 8), "4gpu_4c": ("4g4c", 4)}
CASES = [(d, a, l) for d in DOMAINS for a in ARMS for l in LAYOUTS]


def _path(domain: str, arm: str, layout: str) -> Path:
    return SUITE / domain / f"opd_native_ent_{arm}_toklog_{layout}_b256.yaml"


@pytest.mark.parametrize("domain,arm,layout", CASES)
def test_q4b_entropy_baseline_contract(domain: str, arm: str, layout: str) -> None:
    tag, ratio, mode = ARMS[arm]; layout_tag, gpus = LAYOUTS[layout]
    cfg = load_config(_path(domain, arm, layout))
    assert cfg.model.student_path.endswith("Qwen3-4B")
    assert cfg.actor.distill_loss_builder == "policy_gradient"
    assert cfg.actor.distill_mode == "chosen_token_policy_gradient"
    assert cfg.actor.topk_distill_enabled is False
    assert cfg.actor.loss_agg_mode == "token-mean"
    assert (cfg.actor.token_baseline_method, cfg.actor.token_baseline_retention_ratio, cfg.actor.token_baseline_selection_mode) == ("entropy", ratio, mode)
    assert cfg.actor.token_baseline_normalize_weights is False  # author 2026-10-21: no per-response mean-one rescale
    assert cfg.audit.domains == [domain]
    # The trainer names the weighted loss <base>+entropy_token_weighting; any other signal fails every audit step.
    assert cfg.audit.loss_variance_signal == "policy_gradient_distillation_signal+entropy_token_weighting"
    assert cfg.audit.control_token_loss_weighting_enabled is False and cfg.audit.control_token_online_selection_enabled is False
    for key in PER_TOKEN_AUDIT:
        assert getattr(cfg.audit, key) is True, key
    assert cfg.worker_placement.separate_ref_policy is False
    assert cfg.worker_placement.actor_rollout.n_gpus_per_node == cfg.trainer.n_gpus_per_node == cfg.runtime.slurm_allocation_gpus == gpus
    assert f"actor_rollout_ref.ref.fsdp_config.fsdp_size={gpus}" in cfg.extra_overrides
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == 256
    assert cfg.rollout.gpu_memory_utilization == 0.6
    assert cfg.trainer.save_freq == 1
    assert cfg.trainer.total_training_steps == 60 and cfg.huggingface_checkpoint.steps == (60,)
    run = cfg.trainer.experiment_name
    assert run == f"q4b-{domain}-opdnative-{tag}-toklog-{layout_tag}-b256-s42"
    assert cfg.runtime.wandb_run_id == run and len(run) <= 64
    assert cfg.huggingface_checkpoint.path_prefix == f"checkpoints/qwen4b/{domain}/{run}"
    assert cfg.trainer.default_local_dir == f"checkpoints/MOPD/{run}" and cfg.audit.output_dir == f"audit/{run}"
    assert cfg.paper_eval.output_dir == f"eval_outputs/paper_suite/{run}"
    command = build_command(cfg)
    assert "actor_rollout_ref.actor.policy_loss.token_baseline_method=entropy" in command
    assert f"actor_rollout_ref.actor.policy_loss.token_baseline_retention_ratio={ratio}" in command
    assert f"actor_rollout_ref.actor.policy_loss.token_baseline_selection_mode={mode}" in command
    assert "actor_rollout_ref.actor.policy_loss.token_baseline_normalize_weights=False" in command
    assert "data.train_batch_size=256" in command


@pytest.mark.parametrize("domain,arm,layout", CASES)
def test_q4b_entropy_baseline_differs_from_equal_weight_only_in_topology_selection_and_names(domain: str, arm: str, layout: str) -> None:
    changed = _changed(_path(domain, arm, layout), SUITE / domain / BASE)
    allowed = set(IDENTITY_KEYS) | SELECTION | TOPOLOGY
    assert changed <= allowed, sorted(changed - allowed)


def test_q4b_entropy_baseline_run_names_are_unique() -> None:
    runs = [load_config(_path(*case)).trainer.experiment_name for case in CASES]
    assert len(set(runs)) == len(runs) == 8
