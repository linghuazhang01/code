"""R1: Top-32 RKL OPD with full 结果结构 ∪ 推理阻力 up-weighted ×4, mean one (Math-only and Code-only, 1.7B),
on 3gpu_3c b255, 4gpu_4c b256 and 8gpu_8c b256, plus the 4/8-GPU equal-weight Top-32 references."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from mopd_verl.domain_gradient.phase_control import phase_token_weights
from mopd_verl.settings import load_config

ROOT = Path(__file__).resolve().parents[1]
RECORD = json.loads((ROOT / "configs/token_selection/list12_classes.json").read_text())
FOLDERS = {
    "math": ROOT / "configs/token_selection/math/taxonomy",
    "code": ROOT / "configs/token_selection/code",
}
TOPOLOGIES = {"3gpu_3c": (3, 255, "3g3c"), "4gpu_4c": (4, 256, "4g4c"), "8gpu_8c": (8, 256, "8g8c")}
CASES = [(d, t) for d in sorted(FOLDERS) for t in TOPOLOGIES]
BASE = "uniform_top32kl_toklog_3gpu_3c_b255.yaml"
NAME_KEYS = {"wandb_run_id", "wandb_resume", "experiment_name", "default_local_dir", "output_dir", "path_prefix"}
TOPOLOGY_KEYS = {
    "actor.ppo_mini_batch_size", "data.train_batch_size", "extra_overrides", "ray_kwargs.ray_init.num_cpus",
    "runtime.slurm_allocation_gpus", "trainer.n_gpus_per_node", "worker_placement.actor_rollout.n_gpus_per_node",
}
WEIGHT_KEYS = {"audit.control_token_loss_weighting_enabled", "audit.domain_control_token_ids",
               "region_dpo.domain_control_token_ids"}  # region_dpo is off and only mirrors the control IDs


def _r1(domain: str, topo: str) -> Path:
    return FOLDERS[domain] / f"top32kl_result_brake_fixed4_toklog_{topo}_b{TOPOLOGIES[topo][1]}.yaml"


def _reference(domain: str, topo: str) -> Path:
    if topo == "3gpu_3c":
        return FOLDERS[domain] / BASE
    return FOLDERS[domain] / f"uniform_top32kl_toklog_{topo}_b{TOPOLOGIES[topo][1]}.yaml"


def _flat(data: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in data.items():
        if isinstance(value, dict) and key != "domain_control_token_ids":
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def _diff(a: Path, b: Path) -> set[str]:
    left, right = _flat(asdict(load_config(a))), _flat(asdict(load_config(b)))
    changed = {k for k in left.keys() | right.keys() if left.get(k) != right.get(k)}
    return {k for k in changed if k.rsplit(".", 1)[-1] not in NAME_KEYS}


def _pool(domain: str) -> set[int]:
    full = RECORD["domains"][domain]["full_classes"]
    return set(full["result"]["ids"]) | set(full["brake"]["ids"])


@pytest.mark.parametrize("domain,topo", CASES)
def test_pool_and_weights(domain: str, topo: str) -> None:
    audit = load_config(_r1(domain, topo)).audit
    assert audit.control_token_loss_weighting_enabled
    assert not audit.control_token_online_selection_enabled
    assert audit.control_token_normalize_per_domain
    assert (audit.control_token_loss_weight, audit.control_token_other_weight) == (4.0, 1.0)
    assert audit.control_token_ids == []
    assert list(audit.domain_control_token_ids) == [domain]
    ids = audit.domain_control_token_ids[domain]
    assert len(ids) == len(set(ids))
    assert set(ids) == _pool(domain)
    assert 151645 in ids  # <|im_end|> is in 结果结构
    assert 25521 in ids  # first bytes of " ✅"; author kept it in 结果结构 (2026-11-17)
    assert len(ids) == {"math": 76, "code": 78}[domain]


@pytest.mark.parametrize("domain,topo", CASES)
def test_r1_differs_from_its_reference_only_in_weighting(domain: str, topo: str) -> None:
    assert _diff(_r1(domain, topo), _reference(domain, topo)) <= WEIGHT_KEYS
    assert not load_config(_r1(domain, topo)).region_dpo.enabled


@pytest.mark.parametrize("domain,topo", [c for c in CASES if c[1] != "3gpu_3c"])
def test_reference_differs_from_3gpu_base_only_in_topology(domain: str, topo: str) -> None:
    assert _diff(_reference(domain, topo), FOLDERS[domain] / BASE) <= TOPOLOGY_KEYS


@pytest.mark.parametrize("domain,topo", CASES)
def test_topology_names_and_schedule(domain: str, topo: str) -> None:
    n, batch, tag = TOPOLOGIES[topo]
    cfg = load_config(_r1(domain, topo))
    suffix = "-s42"
    run = f"q1p7b-{domain}-top32kl-resbrake-fixed4-toklog-{tag}-b{batch}{suffix}"
    assert cfg.runtime.wandb_run_id == cfg.trainer.experiment_name == run
    assert cfg.huggingface_checkpoint.path_prefix == f"checkpoints/{domain}/{run}"
    assert cfg.audit.output_dir == f"audit/{run}"
    assert list(cfg.huggingface_checkpoint.steps) == [60]
    assert cfg.trainer.total_training_steps == 60
    assert cfg.trainer.save_freq == 1
    assert cfg.trainer.seed == 42
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == batch
    assert cfg.trainer.n_gpus_per_node == cfg.worker_placement.actor_rollout.n_gpus_per_node == n
    assert cfg.runtime.slurm_allocation_gpus == n
    assert not cfg.worker_placement.separate_ref_policy
    assert cfg.rollout.gpu_memory_utilization == 0.6
    assert f"actor_rollout_ref.ref.fsdp_config.fsdp_size={n}" in cfg.extra_overrides
    ref = load_config(_reference(domain, topo))
    assert ref.runtime.wandb_run_id != run


def test_fixed_weights_are_mean_one() -> None:
    token_ids = torch.tensor([[10, 11, 12, 13], [14, 10, 15, 16]])
    weights = phase_token_weights(
        token_ids, torch.ones_like(token_ids), ["math", "math"],
        domain_token_ids={"math": [10]}, control_weight=4.0, phase_enabled=False,
        span_enabled=False, phase_gates={}, span_length=0, span_decay_tau=1.0,
        normalize_per_domain=True, other_weight=1.0,
    )
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    torch.testing.assert_close(weights[0, 0] / weights[0, 1], torch.tensor(4.0))
