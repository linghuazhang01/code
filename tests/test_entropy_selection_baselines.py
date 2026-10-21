"""Student-entropy token-selection baselines (top 20% / tail 80%) for Qwen3-1.7B native OPD, and the bottomk selection mode.

configs/qwen1p7b_suite/math_code/opd_native_ent_{top20,tail80}_toklog_8gpu_8c_b256.yaml
configs/token_selection/{math/taxonomy,code}/native_opd_ent_{top20,tail80}_toklog_8gpu_8c_b256.yaml
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from mopd_verl.token_baseline_math import select_positions
from mopd_verl.token_baselines import build_token_baseline_weights, token_baseline_selection_mode

from test_qwen1p7b_math_code_native_toklog import IDENTITY_KEYS, PER_TOKEN_AUDIT, _changed

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    # setting: (directory, file prefix, equal-weight base, domains, run prefix, HF folder, topology changed vs base)
    "mc": (ROOT / "configs/qwen1p7b_suite/math_code", "opd_native_ent", "opd_native_toklog_8gpu_8c_b256.yaml", ["math", "code"], "mc", "math_code", False),
    "math": (ROOT / "configs/token_selection/math/taxonomy", "native_opd_ent", "native_opd_toklog_3gpu_3c_b255.yaml", ["math"], "math", "math", True),
    "code": (ROOT / "configs/token_selection/code", "native_opd_ent", "native_opd_toklog_3gpu_3c_b255.yaml", ["code"], "code", "code", True),
}
ARMS = {"top20": ("enttop20", 0.2, "topk"), "tail80": ("enttail80", 0.8, "bottomk")}
CASES = [(s, a) for s in SETTINGS for a in ARMS]
TOPOLOGY = {"runtime.slurm_allocation_gpus", "data.train_batch_size", "actor.ppo_mini_batch_size", "trainer.n_gpus_per_node",
            "worker_placement.actor_rollout.n_gpus_per_node", "extra_overrides", "ray_kwargs.ray_init.num_cpus"}
SELECTION = {"actor.token_baseline_method", "actor.token_baseline_retention_ratio", "actor.token_baseline_selection_mode"}


def _path(setting: str, arm: str) -> Path:
    directory, prefix, *_ = SETTINGS[setting]
    return directory / f"{prefix}_{arm}_toklog_8gpu_8c_b256.yaml"


@pytest.mark.parametrize("n", list(range(1, 61)) + [127, 1000, 16384])
def test_bottomk_is_the_exact_complement_of_topk(n: int) -> None:
    torch.manual_seed(n)
    score = torch.rand(3, n); score[1, : n // 2] = 0.5  # ties are broken the same way in both modes
    mask = torch.ones(3, n); mask[2, n // 3:] = 0.0
    top = select_positions(score=score, mask=mask, retention_ratio=0.2, selection_mode="topk")
    tail = select_positions(score=score, mask=mask, retention_ratio=0.8, selection_mode="bottomk")
    assert torch.equal(top + tail, mask)
    assert float((top * tail).sum()) == 0.0


def test_bottomk_keeps_the_lowest_scores_and_full_retention_keeps_all() -> None:
    score = torch.tensor([[0.1, 0.9, 0.7, 0.2, 0.5, 0.3, 0.8, 0.4, 0.6, 0.0]])
    mask = torch.ones_like(score)
    tail = select_positions(score=score, mask=mask, retention_ratio=0.8, selection_mode="bottomk")
    assert tail[0].tolist() == [1, 0, 1, 1, 1, 1, 0, 1, 1, 1]  # drops the two highest (0.9 at 1, 0.8 at 6)
    assert torch.equal(select_positions(score=score, mask=mask, retention_ratio=1.0, selection_mode="bottomk"), mask)


def test_entropy_tail_weights_are_mean_one_per_response() -> None:
    entropy = torch.tensor([[0.1, 0.9, 0.7, 0.2, 0.5, 0.3, 0.8, 0.4, 0.6, 0.0]])
    result = build_token_baseline_weights(student_entropy=entropy, response_mask=torch.ones_like(entropy), policy_loss_config={
        "token_baseline_method": "entropy", "token_baseline_retention_ratio": 0.8, "token_baseline_selection_mode": "bottomk"})
    torch.testing.assert_close(result.weights.sum(), torch.tensor(10.0))
    assert result.weights[0, 1] == 0 and result.weights[0, 6] == 0
    torch.testing.assert_close(result.weights[0, 0], torch.tensor(1.25))


def test_selection_mode_validation() -> None:
    assert token_baseline_selection_mode({"token_baseline_selection_mode": "bottomk"}) == "bottomk"
    with pytest.raises(ValueError, match="token_baseline_selection_mode"):
        token_baseline_selection_mode({"token_baseline_selection_mode": "tail"})


@pytest.mark.parametrize("setting,arm", CASES)
def test_entropy_baseline_contract(setting: str, arm: str) -> None:
    _, _, _, domains, prefix, hf, _ = SETTINGS[setting]; tag, ratio, mode = ARMS[arm]
    cfg = load_config(_path(setting, arm))
    assert cfg.model.student_path.endswith("Qwen3-1.7B")
    assert cfg.actor.distill_loss_builder == "policy_gradient"
    assert cfg.actor.distill_mode == "chosen_token_policy_gradient"
    assert cfg.actor.topk_distill_enabled is False
    assert (cfg.actor.token_baseline_method, cfg.actor.token_baseline_retention_ratio, cfg.actor.token_baseline_selection_mode) == ("entropy", ratio, mode)
    assert cfg.audit.domains == domains
    assert cfg.audit.control_token_loss_weighting_enabled is False and cfg.audit.control_token_online_selection_enabled is False
    for key in PER_TOKEN_AUDIT:
        assert getattr(cfg.audit, key) is True, key
    assert cfg.worker_placement.separate_ref_policy is False
    assert cfg.worker_placement.actor_rollout.n_gpus_per_node == cfg.trainer.n_gpus_per_node == cfg.runtime.slurm_allocation_gpus == 8
    assert "actor_rollout_ref.ref.fsdp_config.fsdp_size=8" in cfg.extra_overrides
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == 256
    assert cfg.rollout.gpu_memory_utilization == 0.6
    assert cfg.trainer.total_training_steps == 60 and cfg.huggingface_checkpoint.steps == (60,)
    run = cfg.trainer.experiment_name
    assert run == f"q1p7b-{prefix}-opdnative-{tag}-toklog-8g8c-b256-s42"
    assert cfg.runtime.wandb_run_id == run and len(run) <= 64
    assert cfg.huggingface_checkpoint.path_prefix == f"checkpoints/{hf}/{run}"
    assert cfg.trainer.default_local_dir == f"checkpoints/MOPD/{run}" and cfg.audit.output_dir == f"audit/{run}"
    command = build_command(cfg)
    assert "actor_rollout_ref.actor.policy_loss.token_baseline_method=entropy" in command
    assert f"actor_rollout_ref.actor.policy_loss.token_baseline_retention_ratio={ratio}" in command
    assert f"actor_rollout_ref.actor.policy_loss.token_baseline_selection_mode={mode}" in command
    assert "data.train_batch_size=256" in command


@pytest.mark.parametrize("setting,arm", CASES)
def test_entropy_baseline_differs_from_equal_weight_only_in_topology_selection_and_names(setting: str, arm: str) -> None:
    directory, _, base, *_, topology_changed = SETTINGS[setting]
    allowed = set(IDENTITY_KEYS) | SELECTION | (TOPOLOGY if topology_changed else set())
    changed = _changed(_path(setting, arm), directory / base)
    assert changed <= allowed, sorted(changed - allowed)


def test_entropy_baseline_run_names_are_unique() -> None:
    runs = [load_config(_path(s, a)).trainer.experiment_name for s, a in CASES]
    assert len(set(runs)) == len(runs) == 6
