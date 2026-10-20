"""Contracts for the Qwen3-1.7B Math+Code native OPD ("OPD naive") v3-pool ignore / only configs with per-token audit.

configs/qwen1p7b_suite/math_code/opd_native_v3pool_{ignore,only}_toklog_8gpu_8c_b256.yaml
Same as opd_native_toklog_8gpu_8c_b256.yaml except the fixed v3-pool position weights (pool 0 / others 1, or pool 1 / others 0)
and the run names; pools identical to the Top-32 RKL v3 ignore/only runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mopd_verl.config_profiles import load_raw_config
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config

from test_qwen1p7b_math_code_native_toklog import IDENTITY_KEYS, PER_TOKEN_AUDIT, _changed

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SUITE = CONFIGS / "qwen1p7b_suite/math_code"
EQUAL = SUITE / "opd_native_toklog_8gpu_8c_b256.yaml"
TOP32_V3 = CONFIGS / "token_selection/math_code/taxonomy/_mopd_math_code_v3pool_fixed_8gpu_8c.yaml"
ARMS = [("ignore", "v3ign", 0.0, 1.0), ("only", "v3only", 1.0, 0.0)]
WEIGHTING_KEYS = {
    "audit.control_token_loss_weighting_enabled",
    "audit.control_token_normalize_per_domain",
    "audit.control_token_phase_gate_enabled",
    "audit.control_token_span_weighting_enabled",
    "audit.control_token_speed_weighting_enabled",
    "audit.control_token_loss_weight",
    "audit.control_token_other_weight",
    "audit.domain_control_token_ids.math",
    "audit.domain_control_token_ids.code",
}


@pytest.mark.parametrize("arm,tag,pool_w,other_w", ARMS)
def test_native_v3pool_contract(arm: str, tag: str, pool_w: float, other_w: float) -> None:
    name = f"opd_native_v3pool_{arm}_toklog_8gpu_8c_b256.yaml"
    cfg = load_config(SUITE / name)
    assert cfg.model.student_path.endswith("Qwen3-1.7B")
    assert cfg.actor.distill_loss_builder == "policy_gradient"
    assert cfg.actor.distill_mode == "chosen_token_policy_gradient"
    assert cfg.actor.topk_distill_enabled is False
    assert cfg.actor.loss_agg_mode == "token-mean"
    audit = cfg.audit
    assert audit.control_token_loss_weighting_enabled is True
    assert audit.control_token_online_selection_enabled is False
    assert audit.control_token_normalize_per_domain is False
    assert audit.control_token_ids == []
    assert (audit.control_token_loss_weight, audit.control_token_other_weight) == (pool_w, other_w)
    top32 = load_raw_config(TOP32_V3)["audit"]["domain_control_token_ids"]
    for domain, size in (("math", 98), ("code", 167)):
        assert audit.domain_control_token_ids[domain] == top32[domain]
        assert len(set(audit.domain_control_token_ids[domain])) == size
    for key in PER_TOKEN_AUDIT:
        assert getattr(audit, key) is True, key
    assert cfg.worker_placement.separate_ref_policy is False
    assert cfg.worker_placement.actor_rollout.n_gpus_per_node == cfg.trainer.n_gpus_per_node == 8
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == 256
    assert cfg.rollout.gpu_memory_utilization == 0.6
    assert cfg.trainer.total_training_steps == 60 and cfg.huggingface_checkpoint.steps == (60,)
    run = cfg.trainer.experiment_name
    assert run == f"q1p7b-mc-opdnative-{tag}-toklog-8g8c-b256-s42"
    assert cfg.runtime.wandb_run_id == run and len(run) <= 64
    assert cfg.huggingface_checkpoint.path_prefix == f"checkpoints/math_code/{run}"
    assert cfg.trainer.default_local_dir == f"checkpoints/MOPD/{run}"
    assert cfg.audit.output_dir == f"audit/{run}"
    command = build_command(cfg)
    assert "data.train_batch_size=256" in command
    assert "+mopd_audit.control_token_loss_weighting_enabled=true" in command
    assert f"+mopd_audit.control_token_loss_weight={pool_w}" in command
    assert f"+mopd_audit.control_token_other_weight={other_w}" in command


@pytest.mark.parametrize("arm", [arm for arm, *_ in ARMS])
def test_native_v3pool_differs_from_equal_weight_only_in_weighting_and_names(arm: str) -> None:
    changed = _changed(SUITE / f"opd_native_v3pool_{arm}_toklog_8gpu_8c_b256.yaml", EQUAL)
    assert changed - set(IDENTITY_KEYS) <= WEIGHTING_KEYS, sorted(changed - set(IDENTITY_KEYS) - WEIGHTING_KEYS)
    assert {"audit.control_token_loss_weight", "audit.control_token_other_weight", "audit.control_token_loss_weighting_enabled"} <= changed


def test_native_v3pool_run_names_are_unique() -> None:
    runs = {load_config(SUITE / f"opd_native_v3pool_{arm}_toklog_8gpu_8c_b256.yaml").trainer.experiment_name for arm, *_ in ARMS}
    runs.add(load_config(EQUAL).trainer.experiment_name)
    assert len(runs) == 3
