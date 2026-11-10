"""Contracts for the Qwen3-4B native OPD class ignore configs (author 2026-11-10).

configs/qwen4b_suite/{math,code}/list12_class/opd_native_{full_push,full_brake,full_result,full_solve,l12_func,l12_layout}_ignore_toklog_{2gpu_2c,4gpu_4c}_b256.yaml
4B twins of the six 1.7B class ignore runs (内容 left out), built by scripts/build_q4b_list12_class_configs.py from
configs/token_selection/list12_classes.json. Equal-weight reference: the 3-GPU naive run q4b-{math,code}-opdnative-toklog-3g3c-b255-s42.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mopd_verl.domain_gradient.phase_control import phase_token_weights
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config

from test_qwen1p7b_math_code_native_toklog import IDENTITY_KEYS, PER_TOKEN_AUDIT, _changed
from test_qwen1p7b_native_list12_pool import TOPOLOGY, WEIGHTING

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs/qwen4b_suite"
RECORD = json.loads((ROOT / "configs/token_selection/list12_classes.json").read_text())["domains"]
LEAVES = [("full", "push"), ("full", "brake"), ("full", "result"), ("full", "solve"), ("l12", "func"), ("l12", "layout")]
POOLS = {"l12": "classes", "full": "full_classes"}
TOPOLOGIES = [("2gpu_2c", 2, "2g2c"), ("4gpu_4c", 4, "4g4c")]
CASES = [(dom, pool, cls, *topo) for dom in ("math", "code") for pool, cls in LEAVES for topo in TOPOLOGIES]
PAIRS = [(dom, pool, cls) for dom in ("math", "code") for pool, cls in LEAVES]


def _path(dom: str, pool: str, cls: str, topo: str) -> Path:
    return SUITE / dom / "list12_class" / f"opd_native_{pool}_{cls}_ignore_toklog_{topo}_b256.yaml"


@pytest.mark.parametrize("dom,pool,cls,topo,gpus,short", CASES)
def test_contract(dom: str, pool: str, cls: str, topo: str, gpus: int, short: str) -> None:
    cfg = load_config(_path(dom, pool, cls, topo))
    assert cfg.model.student_path.endswith("Qwen3-4B")
    assert cfg.actor.distill_loss_builder == "policy_gradient" and cfg.actor.distill_mode == "chosen_token_policy_gradient"
    assert cfg.actor.topk_distill_enabled is False and cfg.actor.token_baseline_method == "none"
    assert cfg.worker_placement.separate_ref_policy is False
    assert cfg.worker_placement.actor_rollout.n_gpus_per_node == cfg.trainer.n_gpus_per_node == cfg.runtime.slurm_allocation_gpus == gpus
    assert f"actor_rollout_ref.ref.fsdp_config.fsdp_size={gpus}" in cfg.extra_overrides
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == 256
    assert cfg.rollout.gpu_memory_utilization == 0.6
    assert cfg.trainer.total_training_steps == 60 and cfg.trainer.save_freq == 1 and cfg.huggingface_checkpoint.steps == (60,)
    assert cfg.data.seed == cfg.rollout.seed == cfg.trainer.seed == 42
    for key in PER_TOKEN_AUDIT:
        assert getattr(cfg.audit, key) is True, key
    run = cfg.trainer.experiment_name
    assert run == f"q4b-{dom}-opdnative-{pool}-{cls}-ign-toklog-{short}-b256-s42"
    assert cfg.runtime.wandb_run_id == run and len(run) <= 64
    assert cfg.huggingface_checkpoint.path_prefix == f"checkpoints/qwen4b/{dom}/{run}"
    assert cfg.trainer.default_local_dir == f"checkpoints/MOPD/{run}" and cfg.audit.output_dir == f"audit/{run}"
    audit = cfg.audit
    assert audit.domains == [dom] and audit.control_token_ids == []
    assert audit.control_token_loss_weighting_enabled is True and audit.control_token_online_selection_enabled is False
    assert audit.control_token_normalize_per_domain is False
    assert not (audit.control_token_phase_gate_enabled or audit.control_token_span_weighting_enabled or audit.control_token_speed_weighting_enabled)
    assert (audit.control_token_loss_weight, audit.control_token_other_weight) == (0.0, 1.0)
    assert audit.domain_control_token_ids == {dom: RECORD[dom][POOLS[pool]][cls]["ids"]}
    command = build_command(cfg)
    assert "+mopd_audit.control_token_loss_weight=0.0" in command and "+mopd_audit.control_token_other_weight=1.0" in command


@pytest.mark.parametrize("dom,pool,cls", PAIRS)
def test_differs_from_4b_equal_weight_reference_only_in_weighting_topology_and_names(dom: str, pool: str, cls: str) -> None:
    allowed = set(IDENTITY_KEYS) | TOPOLOGY | WEIGHTING | {"audit.control_token_online_selection_enabled"}
    for topo, *_ in TOPOLOGIES:
        changed = _changed(_path(dom, pool, cls, topo), SUITE / dom / "opd_native_toklog_3gpu_3c.yaml")
        assert changed <= allowed, sorted(changed - allowed)


@pytest.mark.parametrize("dom,pool,cls", PAIRS)
def test_2gpu_differs_from_4gpu_only_in_topology_and_names(dom: str, pool: str, cls: str) -> None:
    changed = _changed(_path(dom, pool, cls, "2gpu_2c"), _path(dom, pool, cls, "4gpu_4c"))
    assert changed <= set(IDENTITY_KEYS) | TOPOLOGY, sorted(changed - set(IDENTITY_KEYS) - TOPOLOGY)


@pytest.mark.parametrize("dom,pool,cls", PAIRS)
def test_same_pool_and_weights_as_the_1p7b_run(dom: str, pool: str, cls: str) -> None:
    twin = load_config(ROOT / "configs/token_selection" / dom / "list12_class" / f"native_opd_{pool}_{cls}_ignore_toklog_4gpu_4c_b256.yaml").audit
    audit = load_config(_path(dom, pool, cls, "4gpu_4c")).audit
    assert audit.domain_control_token_ids == twin.domain_control_token_ids
    assert (audit.control_token_loss_weight, audit.control_token_other_weight) == (twin.control_token_loss_weight, twin.control_token_other_weight)


@pytest.mark.parametrize("dom,pool,cls,topo,gpus,short", CASES)
def test_position_weights(dom: str, pool: str, cls: str, topo: str, gpus: int, short: str) -> None:
    audit = load_config(_path(dom, pool, cls, topo)).audit
    ids = audit.domain_control_token_ids[dom]
    other = 99999
    assert other not in ids
    tokens = torch.tensor([[ids[0], other, ids[-1], 0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    weights = phase_token_weights(tokens, mask, [dom], domain_token_ids=audit.domain_control_token_ids, control_weight=audit.control_token_loss_weight,
                                  phase_enabled=False, span_enabled=False, phase_gates={}, span_length=0, span_decay_tau=1.0,
                                  normalize_per_domain=audit.control_token_normalize_per_domain, other_weight=audit.control_token_other_weight)
    torch.testing.assert_close((weights * mask)[0], torch.tensor([0.0, 1.0, 0.0, 0.0]))


def test_no_content_class_and_unique_run_names() -> None:
    assert not list(SUITE.glob("*/list12_class/*content*"))
    runs = [load_config(_path(*case[:4])).trainer.experiment_name for case in CASES]
    assert len(set(runs)) == len(runs) == 24
