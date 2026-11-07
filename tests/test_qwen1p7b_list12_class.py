"""Contracts for the Qwen3-1.7B native OPD ignore / only configs on the seven list-12 classes and their full sets (author 2026-11-04).

configs/token_selection/{math,code}/list12_class/native_opd_{l12,full}_<class>_{ignore,only}_toklog_{4gpu_4c,8gpu_8c}_b256.yaml
Classes: 推理推力 / 推理阻力 / 结果结构 / 解题结构 / 虚词 / 内容 / 版面 (configs/token_selection/list12_classes.json, built by
scripts/build_list12_class_configs.py). l12 = the list-12 IDs of a class; full = every ID of the class passing the list-12 occurrence gate.
Same weighting as the list-12 ignore / only runs; only the pool and, at 4 GPUs, the topology differ.
Equal-weight references at the same topology: {math/taxonomy,code}/native_opd_toklog_{4gpu_4c,8gpu_8c}_b256.yaml.
Extra leaves (author 2026-11-06): code/list12_class/native_opd_full_{push,brake}_ignore_toklog_3gpu_3c_b255.yaml, 3 colocated GPUs, batch 255,
vLLM memory 0.8, W&B offline; reference code/native_opd_toklog_3gpu_3c_b255.yaml.
Their 2-GPU reruns from step 0 (author 2026-11-06): native_opd_full_{push,brake}_ignore_toklog_2gpu_2c_b256.yaml, 2 colocated GPUs,
batch 256, vLLM memory 0.6, W&B offline, parent ../_native_opd_toklog_2gpu_2c_b256.yaml (no 2-GPU equal-weight run).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from mopd_verl.domain_gradient.phase_control import phase_token_weights
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config

from test_qwen1p7b_math_code_native_toklog import IDENTITY_KEYS, PER_TOKEN_AUDIT, _changed
from test_qwen1p7b_native_list12_pool import SHA, SIZE, TOPOLOGY, WEIGHTING

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "configs/token_selection"
RECORD = json.loads((SELECTION / "list12_classes.json").read_text())["domains"]
BASE = {"math": SELECTION / "math/taxonomy", "code": SELECTION / "code"}
EOS = 151645
CLASSES = ("push", "brake", "result", "solve", "func", "content", "layout")
TOPOLOGIES = [("4gpu_4c", 4, "4g4c"), ("8gpu_8c", 8, "8g8c")]
ARMS = [("ignore", "ign", 0.0, 1.0), ("only", "only", 1.0, 0.0)]
POOLS = {"l12": "classes", "full": "full_classes"}
FULL_SIZE = {"math": 1257, "code": 1387}
CASES = [(dom, pool, cls, *arm, *topo) for dom in ("math", "code") for pool in POOLS for cls in CLASSES for arm in ARMS for topo in TOPOLOGIES]
TRIPLES = [(dom, pool, cls, arm) for dom in ("math", "code") for pool in POOLS for cls in CLASSES for arm, *_ in ARMS]


def _path(dom: str, pool: str, cls: str, arm: str, topo: str) -> Path:
    return SELECTION / dom / "list12_class" / f"native_opd_{pool}_{cls}_{arm}_toklog_{topo}_b256.yaml"


def _layout(cfg, gpus: int) -> None:
    assert cfg.model.student_path.endswith("Qwen3-1.7B")
    assert cfg.actor.distill_mode == "chosen_token_policy_gradient"
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


@pytest.mark.parametrize("dom,pool,cls,arm,tag,pool_w,other_w,topo,gpus,short", CASES)
def test_contract(dom: str, pool: str, cls: str, arm: str, tag: str, pool_w: float, other_w: float, topo: str, gpus: int, short: str) -> None:
    cfg = load_config(_path(dom, pool, cls, arm, topo))
    _layout(cfg, gpus)
    run = cfg.trainer.experiment_name
    assert run == f"q1p7b-{dom}-opdnative-{pool}-{cls}-{tag}-toklog-{short}-b256-s42"
    assert cfg.runtime.wandb_run_id == run and len(run) <= 64
    assert cfg.huggingface_checkpoint.path_prefix == f"checkpoints/{dom}/{run}"
    assert cfg.trainer.default_local_dir == f"checkpoints/MOPD/{run}" and cfg.audit.output_dir == f"audit/{run}"
    audit = cfg.audit
    assert audit.domains == [dom] and audit.control_token_ids == []
    assert audit.control_token_loss_weighting_enabled is True and audit.control_token_online_selection_enabled is False
    assert audit.control_token_normalize_per_domain is False
    assert (audit.control_token_loss_weight, audit.control_token_other_weight) == (pool_w, other_w)
    assert audit.domain_control_token_ids == {dom: RECORD[dom][POOLS[pool]][cls]["ids"]}
    command = build_command(cfg)
    assert f"+mopd_audit.control_token_loss_weight={pool_w}" in command
    assert f"+mopd_audit.control_token_other_weight={other_w}" in command


@pytest.mark.parametrize("dom,pool,cls,arm", TRIPLES)
def test_8gpu_differs_from_list12_only_in_pool_and_names(dom: str, pool: str, cls: str, arm: str) -> None:
    allowed = set(IDENTITY_KEYS) | {f"audit.domain_control_token_ids.{dom}"}
    changed = _changed(_path(dom, pool, cls, arm, "8gpu_8c"), BASE[dom] / f"native_opd_list12_{arm}_toklog_8gpu_8c_b256.yaml")
    assert changed <= allowed, sorted(changed - allowed)


@pytest.mark.parametrize("dom,pool,cls,arm", TRIPLES)
def test_4gpu_differs_from_8gpu_only_in_topology_and_names(dom: str, pool: str, cls: str, arm: str) -> None:
    allowed = set(IDENTITY_KEYS) | TOPOLOGY
    changed = _changed(_path(dom, pool, cls, arm, "4gpu_4c"), _path(dom, pool, cls, arm, "8gpu_8c"))
    assert changed <= allowed, sorted(changed - allowed)


@pytest.mark.parametrize("dom,pool,cls,arm,tag,pool_w,other_w,topo,gpus,short", CASES)
def test_position_weights(dom: str, pool: str, cls: str, arm: str, tag: str, pool_w: float, other_w: float, topo: str, gpus: int, short: str) -> None:
    audit = load_config(_path(dom, pool, cls, arm, topo)).audit
    ids = audit.domain_control_token_ids[dom]
    other = 99999
    assert other not in ids
    tokens = torch.tensor([[ids[0], other, ids[-1], 0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    weights = phase_token_weights(tokens, mask, [dom], domain_token_ids=audit.domain_control_token_ids, control_weight=audit.control_token_loss_weight,
                                  phase_enabled=False, span_enabled=False, phase_gates={}, span_length=0, span_decay_tau=1.0,
                                  normalize_per_domain=audit.control_token_normalize_per_domain, other_weight=audit.control_token_other_weight)
    torch.testing.assert_close((weights * mask)[0], torch.tensor([pool_w, other_w, pool_w, 0.0]))


@pytest.mark.parametrize("dom", ["math", "code"])
def test_seven_classes_partition_list12(dom: str) -> None:
    classes = RECORD[dom]["classes"]
    assert tuple(classes) == CLASSES
    ids = [i for c in classes.values() for i in c["ids"]]
    assert len(ids) == len(set(ids)) == SIZE[dom] == RECORD[dom]["list12_n_ids"]
    assert hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest() == SHA[dom] == RECORD[dom]["list12_sha256"]
    for c in classes.values():
        assert c["ids"] == sorted(set(c["ids"])) and len(c["ids"]) == c["n_ids"]
        assert hashlib.sha256(",".join(map(str, c["ids"])).encode()).hexdigest() == c["sha256"]
    assert [k for k, c in classes.items() if EOS in c["ids"]] == ["result"]


@pytest.mark.parametrize("dom", ["math", "code"])
def test_full_sets_contain_list12_classes(dom: str) -> None:
    full, l12 = RECORD[dom]["full_classes"], RECORD[dom]["classes"]
    assert tuple(full) == CLASSES
    ids = [i for c in full.values() for i in c["ids"]]
    assert len(ids) == len(set(ids)) == FULL_SIZE[dom]
    for k, c in full.items():
        assert c["ids"] == sorted(set(c["ids"])) and len(c["ids"]) == c["n_ids"]
        assert hashlib.sha256(",".join(map(str, c["ids"])).encode()).hexdigest() == c["sha256"]
        assert set(l12[k]["ids"]) <= set(c["ids"])
    assert [k for k, c in full.items() if EOS in c["ids"]] == ["result"]


@pytest.mark.parametrize("dom", ["math", "code"])
@pytest.mark.parametrize("topo,gpus,short", TOPOLOGIES)
def test_equal_weight_reference(dom: str, topo: str, gpus: int, short: str) -> None:
    path = BASE[dom] / f"native_opd_toklog_{topo}_b256.yaml"
    cfg = load_config(path)
    _layout(cfg, gpus)
    assert cfg.audit.control_token_loss_weighting_enabled is False
    assert cfg.trainer.experiment_name == cfg.runtime.wandb_run_id == f"q1p7b-{dom}-opdnative-toklog-{short}-b256-s42"
    assert cfg.huggingface_checkpoint.path_prefix == f"checkpoints/{dom}/{cfg.trainer.experiment_name}"
    changed = _changed(path, BASE[dom] / "native_opd_toklog_3gpu_3c_b255.yaml")
    allowed = set(IDENTITY_KEYS) | TOPOLOGY | {"runtime.wandb_resume"}
    assert changed <= allowed, sorted(changed - allowed)


def test_run_names_are_unique() -> None:
    runs = [load_config(_path(dom, pool, cls, arm, topo)).trainer.experiment_name for dom, pool, cls, arm, _, _, _, topo, _, _ in CASES]
    assert len(set(runs)) == len(runs) == 112


@pytest.mark.parametrize("cls", ["push", "brake"])
def test_code_3gpu_full_ignore(cls: str) -> None:
    path = SELECTION / "code/list12_class" / f"native_opd_full_{cls}_ignore_toklog_3gpu_3c_b255.yaml"
    cfg = load_config(path)
    run = f"q1p7b-code-opdnative-full-{cls}-ign-toklog-3g3c-b255-s42"
    assert cfg.trainer.experiment_name == cfg.runtime.wandb_run_id == run and len(run) <= 64
    assert cfg.huggingface_checkpoint.path_prefix == f"checkpoints/code/{run}" and cfg.audit.output_dir == f"audit/{run}"
    assert cfg.worker_placement.actor_rollout.n_gpus_per_node == cfg.trainer.n_gpus_per_node == cfg.runtime.slurm_allocation_gpus == 3
    assert "actor_rollout_ref.ref.fsdp_config.fsdp_size=3" in cfg.extra_overrides and cfg.worker_placement.separate_ref_policy is False
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == 255
    assert cfg.rollout.gpu_memory_utilization == 0.8 and cfg.runtime.wandb_mode == "offline"
    assert cfg.trainer.total_training_steps == 60 and cfg.trainer.save_freq == 1 and cfg.huggingface_checkpoint.steps == (60,)
    assert (cfg.audit.control_token_loss_weight, cfg.audit.control_token_other_weight) == (0.0, 1.0)
    assert cfg.audit.domain_control_token_ids == {"code": RECORD["code"]["full_classes"][cls]["ids"]}
    allowed = set(IDENTITY_KEYS) | WEIGHTING | {"audit.control_token_online_selection_enabled", "rollout.gpu_memory_utilization", "runtime.wandb_mode"}
    changed = _changed(path, BASE["code"] / "native_opd_toklog_3gpu_3c_b255.yaml")
    assert changed <= allowed, sorted(changed - allowed)
    changed = _changed(path, _path("code", "full", cls, "ignore", "4gpu_4c"))
    assert changed <= set(IDENTITY_KEYS) | TOPOLOGY | {"rollout.gpu_memory_utilization", "runtime.wandb_mode"}, sorted(changed - set(IDENTITY_KEYS) - TOPOLOGY)


@pytest.mark.parametrize("cls", ["push", "brake"])
def test_code_2gpu_full_ignore(cls: str) -> None:
    path = SELECTION / "code/list12_class" / f"native_opd_full_{cls}_ignore_toklog_2gpu_2c_b256.yaml"
    cfg = load_config(path)
    run = f"q1p7b-code-opdnative-full-{cls}-ign-toklog-2g2c-b256-s42"
    assert cfg.trainer.experiment_name == cfg.runtime.wandb_run_id == run and len(run) <= 64
    assert cfg.huggingface_checkpoint.path_prefix == f"checkpoints/code/{run}" and cfg.audit.output_dir == f"audit/{run}"
    assert cfg.worker_placement.actor_rollout.n_gpus_per_node == cfg.trainer.n_gpus_per_node == cfg.runtime.slurm_allocation_gpus == 2
    assert "actor_rollout_ref.ref.fsdp_config.fsdp_size=2" in cfg.extra_overrides and cfg.worker_placement.separate_ref_policy is False
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == 256
    assert cfg.rollout.gpu_memory_utilization == 0.6 and cfg.runtime.wandb_mode == "offline"
    assert cfg.trainer.total_training_steps == 60 and cfg.trainer.save_freq == 1 and cfg.huggingface_checkpoint.steps == (60,)
    assert (cfg.audit.control_token_loss_weight, cfg.audit.control_token_other_weight) == (0.0, 1.0)
    assert cfg.audit.domain_control_token_ids == {"code": RECORD["code"]["full_classes"][cls]["ids"]}
    changed = _changed(path, _path("code", "full", cls, "ignore", "4gpu_4c"))
    assert changed <= set(IDENTITY_KEYS) | TOPOLOGY | {"runtime.wandb_mode"}, sorted(changed - set(IDENTITY_KEYS) - TOPOLOGY)


def test_code_2gpu_push_resume() -> None:
    path = SELECTION / "code/list12_class/native_opd_full_push_ignore_toklog_2gpu_2c_b256_resume.yaml"
    base = SELECTION / "code/list12_class/native_opd_full_push_ignore_toklog_2gpu_2c_b256.yaml"
    cfg = load_config(path)
    assert "trainer.resume_mode=auto" in cfg.extra_overrides and "trainer.resume_mode=disable" not in cfg.extra_overrides
    assert [o for o in cfg.extra_overrides if "resume_mode" not in o] == [o for o in load_config(base).extra_overrides if "resume_mode" not in o]
    assert _changed(path, base) <= {"extra_overrides", "runtime.wandb_resume"}
