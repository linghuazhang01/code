"""Contracts for the Qwen3-1.7B Math-only / Code-only native OPD ("OPD naive") v3-pool ignore / only configs with per-token audit.

configs/token_selection/math/taxonomy/native_opd_v3pool_{ignore,only}_toklog_8gpu_8c_b256.yaml
configs/token_selection/code/native_opd_v3pool_{ignore,only}_toklog_8gpu_8c_b256.yaml
Each extends the domain's equal-weight naive file native_opd_toklog_3gpu_3c_b255.yaml, moves it to 8 colocated GPUs at batch 256
and adds fixed v3-pool position weights (pool 0 / others 1, or pool 1 / others 0); pools identical to the Top-32 v3 runs.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mopd_verl.config_profiles import load_raw_config
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from mopd_verl.verl_audit import MOPDAuditLogger

from test_qwen1p7b_math_code_native_toklog import IDENTITY_KEYS, PER_TOKEN_AUDIT, _changed

ROOT = Path(__file__).resolve().parents[1]
DIRS = {"math": ROOT / "configs/token_selection/math/taxonomy", "code": ROOT / "configs/token_selection/code"}
SIZE = {"math": 98, "code": 167}
ARMS = [("ignore", "v3ign", 0.0, 1.0), ("only", "v3only", 1.0, 0.0)]
CASES = [(dom, *arm) for dom in DIRS for arm in ARMS]
ALLOWED = set(IDENTITY_KEYS) | {
    # topology: 3 -> 8 colocated GPUs, batch 255 -> 256
    "runtime.slurm_allocation_gpus", "data.train_batch_size", "actor.ppo_mini_batch_size", "trainer.n_gpus_per_node",
    "worker_placement.actor_rollout.n_gpus_per_node", "extra_overrides", "ray_kwargs.ray_init.num_cpus",
    # weighting
    "audit.control_token_loss_weighting_enabled", "audit.control_token_normalize_per_domain", "audit.control_token_loss_weight",
    "audit.control_token_other_weight", "audit.domain_control_token_ids.math", "audit.domain_control_token_ids.code",
}


def _path(dom: str, arm: str) -> Path:
    return DIRS[dom] / f"native_opd_v3pool_{arm}_toklog_8gpu_8c_b256.yaml"


@pytest.mark.parametrize("dom,arm,tag,pool_w,other_w", CASES)
def test_single_domain_native_v3pool_contract(dom: str, arm: str, tag: str, pool_w: float, other_w: float) -> None:
    cfg = load_config(_path(dom, arm))
    assert cfg.model.student_path.endswith("Qwen3-1.7B")
    assert cfg.actor.distill_loss_builder == "policy_gradient"
    assert cfg.actor.distill_mode == "chosen_token_policy_gradient"
    assert cfg.actor.topk_distill_enabled is False
    assert cfg.actor.loss_agg_mode == "token-mean"
    audit = cfg.audit
    assert audit.domains == [dom] and set(cfg.data.domain_train_files) == {dom}
    assert audit.control_token_loss_weighting_enabled is True
    assert audit.control_token_online_selection_enabled is False
    assert audit.control_token_normalize_per_domain is False
    assert audit.control_token_ids == []
    assert (audit.control_token_loss_weight, audit.control_token_other_weight) == (pool_w, other_w)
    top32 = load_raw_config(DIRS[dom] / "_v3pool_fixed_top32kl_3gpu_3c_b255.yaml")["audit"]["domain_control_token_ids"]
    assert audit.domain_control_token_ids == top32 and len(set(top32[dom])) == SIZE[dom]
    for key in PER_TOKEN_AUDIT:
        assert getattr(audit, key) is True, key
    assert cfg.worker_placement.separate_ref_policy is False
    assert cfg.worker_placement.actor_rollout.n_gpus_per_node == cfg.trainer.n_gpus_per_node == cfg.runtime.slurm_allocation_gpus == 8
    assert "actor_rollout_ref.ref.fsdp_config.fsdp_size=8" in cfg.extra_overrides
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == 256
    assert cfg.rollout.gpu_memory_utilization == 0.6
    assert cfg.trainer.total_training_steps == 60 and cfg.huggingface_checkpoint.steps == (60,)
    run = cfg.trainer.experiment_name
    assert run == f"q1p7b-{dom}-opdnative-{tag}-toklog-8g8c-b256-s42"
    assert cfg.runtime.wandb_run_id == run and len(run) <= 64
    assert cfg.huggingface_checkpoint.path_prefix == f"checkpoints/{dom}/{run}"
    assert cfg.trainer.default_local_dir == f"checkpoints/MOPD/{run}"
    assert audit.output_dir == f"audit/{run}"
    command = build_command(cfg)
    assert "data.train_batch_size=256" in command
    assert f"+mopd_audit.control_token_loss_weight={pool_w}" in command
    assert f"+mopd_audit.control_token_other_weight={other_w}" in command


@pytest.mark.parametrize("dom,arm", [(dom, arm) for dom in DIRS for arm, *_ in ARMS])
def test_single_domain_native_v3pool_differs_from_equal_weight_only_in_topology_weighting_and_names(dom: str, arm: str) -> None:
    changed = _changed(_path(dom, arm), DIRS[dom] / "native_opd_toklog_3gpu_3c_b255.yaml")
    assert changed <= ALLOWED, sorted(changed - ALLOWED)


def test_single_domain_native_v3pool_run_names_are_unique() -> None:
    runs = [load_config(_path(dom, arm)).trainer.experiment_name for dom in DIRS for arm, *_ in ARMS]
    assert len(set(runs)) == len(runs) == 4


@pytest.mark.parametrize("dom,arm", [(dom, arm) for dom in DIRS for arm, *_ in ARMS])
def test_single_domain_gradient_mask(tmp_path: Path, dom: str, arm: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import ModuleType
    try:
        import verl.utils.device  # noqa: F401
    except ModuleNotFoundError:  # same stub as tests/test_v3pool_mask_configs.py
        verl_module, utils_module, device_module = ModuleType("verl"), ModuleType("verl.utils"), ModuleType("verl.utils.device")
        verl_module.__path__, utils_module.__path__ = [], []
        device_module.get_device_id = lambda: torch.device("cpu")
        device_module.get_torch_device = lambda: torch.cpu
        verl_module.DataProto = object
        for name, mod in (("verl", verl_module), ("verl.utils", utils_module), ("verl.utils.device", device_module)):
            monkeypatch.setitem(sys.modules, name, mod)
        for name in [n for n in sys.modules if n.startswith("mopd_verl.domain_gradient.")]:
            monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        from mopd_verl.domain_gradient.audit import DomainGradientAudit
    except ModuleNotFoundError as exc:
        pytest.skip(f"training stack unavailable: {exc}")
    config = load_config(_path(dom, arm))
    logger = MOPDAuditLogger({"mopd_audit": {**asdict(config.audit), "output_dir": str(tmp_path)}})
    meta = logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
    actor = SimpleNamespace(actor_module=torch.nn.Linear(1, 1), actor_optimizer=SimpleNamespace(param_groups=[{}]),
                            config={"policy_loss": asdict(config.actor)})
    audit = DomainGradientAudit(actor, meta)
    pool = config.audit.domain_control_token_ids[dom]
    other = 99999
    assert other not in pool
    ids = torch.tensor([[pool[0], other, pool[-1], 0], [other, pool[1], other, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
    batch = SimpleNamespace(batch={"responses": ids, "response_mask": mask}, non_tensor_batch={"domain": np.array([dom, dom], dtype=object)})
    w = audit.training_gradient_mask(batch)
    assert w is not None
    w = (w.float() * mask.float()).cpu()
    p, o = (1.0, 0.0) if arm == "only" else (0.0, 1.0)
    torch.testing.assert_close(w, torch.tensor([[p, o, p, 0.0], [o, p, o, 0.0]]))
