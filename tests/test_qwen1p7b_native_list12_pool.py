"""Contracts for the Qwen3-1.7B native OPD ("OPD naive") ignore / only configs on the 12-run token list, with per-token audit.

configs/qwen1p7b_suite/math_code/opd_native_list12_{ignore,only}_toklog_8gpu_8c_b256.yaml
configs/token_selection/math/taxonomy/native_opd_list12_{ignore,only}_toklog_8gpu_8c_b256.yaml
configs/token_selection/code/native_opd_list12_{ignore,only}_toklog_8gpu_8c_b256.yaml
Same layout as the v3-pool files (tests/test_qwen1p7b_{math_code,single_domain}_native_v3pool.py); only the pool differs:
Math 78 / Code 119 IDs from analysis-output/token-set-toklog-20261019/tables/d1_token_list_ids.json ("shared").
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from mopd_verl.verl_audit import MOPDAuditLogger

from test_qwen1p7b_math_code_native_toklog import IDENTITY_KEYS, PER_TOKEN_AUDIT, _changed

ROOT = Path(__file__).resolve().parents[1]
SHA = {"math": "e7488da3d3babda2a5f6b3c464bf936e70c857ea2a521f8ed6108e88ea31808d",
       "code": "263bc87e2f03850b6bb0c9b187a4b1bc7c2be97253281bf3efc167adbf6a106f"}
SIZE = {"math": 78, "code": 119}
SETTINGS = {
    # setting: (directory, leaf pattern, equal-weight base, domains, run prefix, HF folder, topology changed vs base)
    "mc": (ROOT / "configs/qwen1p7b_suite/math_code", "opd_native_list12_{arm}_toklog_8gpu_8c_b256.yaml",
           "opd_native_toklog_8gpu_8c_b256.yaml", ["math", "code"], "mc", "math_code", False),
    "math": (ROOT / "configs/token_selection/math/taxonomy", "native_opd_list12_{arm}_toklog_8gpu_8c_b256.yaml",
             "native_opd_toklog_3gpu_3c_b255.yaml", ["math"], "math", "math", True),
    "code": (ROOT / "configs/token_selection/code", "native_opd_list12_{arm}_toklog_8gpu_8c_b256.yaml",
             "native_opd_toklog_3gpu_3c_b255.yaml", ["code"], "code", "code", True),
}
ARMS = [("ignore", "l12ign", 0.0, 1.0), ("only", "l12only", 1.0, 0.0)]
CASES = [(setting, *arm) for setting in SETTINGS for arm in ARMS]
TOPOLOGY = {"runtime.slurm_allocation_gpus", "data.train_batch_size", "actor.ppo_mini_batch_size", "trainer.n_gpus_per_node",
            "worker_placement.actor_rollout.n_gpus_per_node", "extra_overrides", "ray_kwargs.ray_init.num_cpus"}
WEIGHTING = {"audit.control_token_loss_weighting_enabled", "audit.control_token_normalize_per_domain", "audit.control_token_loss_weight",
             "audit.control_token_other_weight", "audit.control_token_phase_gate_enabled", "audit.control_token_span_weighting_enabled",
             "audit.control_token_speed_weighting_enabled", "audit.control_token_ids", "audit.domain_control_token_ids.math",
             "audit.domain_control_token_ids.code"}


def _path(setting: str, arm: str) -> Path:
    directory, leaf, *_ = SETTINGS[setting]
    return directory / leaf.format(arm=arm)


@pytest.mark.parametrize("setting,arm,tag,pool_w,other_w", CASES)
def test_list12_contract(setting: str, arm: str, tag: str, pool_w: float, other_w: float) -> None:
    _, _, _, domains, prefix, hf, _ = SETTINGS[setting]
    cfg = load_config(_path(setting, arm))
    assert cfg.model.student_path.endswith("Qwen3-1.7B")
    assert cfg.actor.distill_loss_builder == "policy_gradient"
    assert cfg.actor.distill_mode == "chosen_token_policy_gradient"
    assert cfg.actor.topk_distill_enabled is False
    audit = cfg.audit
    assert audit.domains == domains
    assert audit.control_token_loss_weighting_enabled is True
    assert audit.control_token_online_selection_enabled is False
    assert audit.control_token_normalize_per_domain is False
    assert audit.control_token_ids == []
    assert (audit.control_token_loss_weight, audit.control_token_other_weight) == (pool_w, other_w)
    assert set(audit.domain_control_token_ids) == set(domains)
    for domain in domains:
        ids = audit.domain_control_token_ids[domain]
        assert ids == sorted(set(ids)) and len(ids) == SIZE[domain]
        assert hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest() == SHA[domain]
    for key in PER_TOKEN_AUDIT:
        assert getattr(audit, key) is True, key
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
    assert cfg.trainer.default_local_dir == f"checkpoints/MOPD/{run}"
    assert audit.output_dir == f"audit/{run}"
    command = build_command(cfg)
    assert "data.train_batch_size=256" in command
    assert f"+mopd_audit.control_token_loss_weight={pool_w}" in command
    assert f"+mopd_audit.control_token_other_weight={other_w}" in command


@pytest.mark.parametrize("setting,arm", [(setting, arm) for setting in SETTINGS for arm, *_ in ARMS])
def test_list12_differs_from_equal_weight_only_in_topology_weighting_and_names(setting: str, arm: str) -> None:
    directory, _, base, *_, topology_changed = SETTINGS[setting]
    allowed = set(IDENTITY_KEYS) | WEIGHTING | (TOPOLOGY if topology_changed else set())
    changed = _changed(_path(setting, arm), directory / base)
    assert changed <= allowed, sorted(changed - allowed)


def test_list12_run_names_are_unique_and_differ_from_v3() -> None:
    runs = [load_config(_path(setting, arm)).trainer.experiment_name for setting in SETTINGS for arm, *_ in ARMS]
    assert len(set(runs)) == len(runs) == 6
    assert not any("v3" in run for run in runs)


@pytest.mark.parametrize("setting,arm", [(setting, arm) for setting in SETTINGS for arm, *_ in ARMS])
def test_list12_gradient_mask(tmp_path: Path, setting: str, arm: str, monkeypatch: pytest.MonkeyPatch) -> None:
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
    config = load_config(_path(setting, arm))
    logger = MOPDAuditLogger({"mopd_audit": {**asdict(config.audit), "output_dir": str(tmp_path)}})
    meta = logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
    actor = SimpleNamespace(actor_module=torch.nn.Linear(1, 1), actor_optimizer=SimpleNamespace(param_groups=[{}]),
                            config={"policy_loss": asdict(config.actor)})
    audit = DomainGradientAudit(actor, meta)
    p, o = (1.0, 0.0) if arm == "only" else (0.0, 1.0)
    other = 99999
    for domain in config.audit.domains:
        pool = config.audit.domain_control_token_ids[domain]
        assert other not in pool
        ids = torch.tensor([[pool[0], other, pool[-1], 0]])
        mask = torch.tensor([[1, 1, 1, 0]])
        batch = SimpleNamespace(batch={"responses": ids, "response_mask": mask}, non_tensor_batch={"domain": np.array([domain], dtype=object)})
        w = audit.training_gradient_mask(batch)
        assert w is not None
        torch.testing.assert_close((w.float() * mask.float()).cpu(), torch.tensor([[p, o, p, 0.0]]))
