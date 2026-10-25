"""Add-only terminate / look-back supplements to the Math and Code list-12 native OPD Only configs."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from mopd_verl.verl_audit import MOPDAuditLogger

from test_qwen1p7b_math_code_native_toklog import IDENTITY_KEYS, PER_TOKEN_AUDIT, _changed

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {
    "math": ROOT / "configs/token_selection/math/taxonomy",
    "code": ROOT / "configs/token_selection/code",
}
MANIFEST = json.loads((ROOT / "configs/token_selection/list12_role_supplements.json").read_text())
ARMS = {"terminate": "term", "lookback": "look", "terminate_lookback": "termlook"}
CASES = [(domain, arm) for domain in DIRECTORIES for arm in ARMS]
BASE_SHA = {
    "math": "e7488da3d3babda2a5f6b3c464bf936e70c857ea2a521f8ed6108e88ea31808d",
    "code": "263bc87e2f03850b6bb0c9b187a4b1bc7c2be97253281bf3efc167adbf6a106f",
}
POOL_SHA = {
    "math": {
        "terminate": "3925da76ecd41a2e6f7b7418edbf219bf0d9e21c712177ca97be109857529324",
        "lookback": "0e19a2c89986967d74d7c98ae21d6943ddc367acbc52345ca9f260fe431cf4c0",
        "terminate_lookback": "9dd7bc9a2160b775dcef8c56a2b43cc2ba358bf5c43301f7d1589c795714b869",
    },
    "code": {
        "terminate": "44611b30675fa1b98e871041b0b8fea6aeafcb438342ce6e4cff3bb92a89bc82",
        "lookback": "74e49b0ce9022fbd5a2e603521ef50a291fec451ff14382152af7069f245b0a4",
        "terminate_lookback": "4df4f50995437de4000e515a787a06193bd535d25ebe531e1421da9e0e2096cb",
    },
}


def _path(domain: str, arm: str | None = None) -> Path:
    suffix = f"_plus_{arm}" if arm else ""
    return DIRECTORIES[domain] / f"native_opd_list12_only{suffix}_toklog_8gpu_8c_b256.yaml"


def _sha(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, sorted(set(ids)))).encode()).hexdigest()


@pytest.mark.parametrize("domain", DIRECTORIES)
def test_frozen_candidates_preserve_original_pool_and_eos(domain: str) -> None:
    info = MANIFEST["domains"][domain]
    original = load_config(_path(domain)).audit.domain_control_token_ids[domain]
    assert _sha(original) == info["baseline_sha256"] == BASE_SHA[domain]
    assert len(original) == info["baseline_count"] == {"math": 78, "code": 119}[domain]
    assert len(info["terminate"]) == 12
    assert len(info["lookback"]) == {"math": 38, "code": 37}[domain]
    for group in ("terminate", "lookback"):
        assert info[group] == sorted(set(info[group]))
    assert set(info["terminate"]).isdisjoint(info["lookback"])
    assert 151645 in original and 151645 in info["terminate"]
    assert MANIFEST["matching"] == "fixed_token_id_all_valid_occurrences"


@pytest.mark.parametrize("domain,arm", CASES)
def test_supplement_config_contract(domain: str, arm: str) -> None:
    cfg = load_config(_path(domain, arm))
    info = MANIFEST["domains"][domain]
    original = set(load_config(_path(domain)).audit.domain_control_token_ids[domain])
    supplement = set(info["terminate"]) | set(info["lookback"]) if arm == "terminate_lookback" else set(info[arm])
    ids = cfg.audit.domain_control_token_ids[domain]
    assert ids == sorted(original | supplement)
    assert len(ids) == info["pools"][arm]["count"]
    assert len(set(ids) - original) == info["pools"][arm]["added_count"]
    assert _sha(ids) == info["pools"][arm]["sha256"] == POOL_SHA[domain][arm]
    assert cfg.model.student_path.endswith("Qwen3-1.7B")
    assert cfg.actor.distill_loss_builder == "policy_gradient"
    assert cfg.actor.distill_mode == "chosen_token_policy_gradient"
    assert cfg.actor.topk_distill_enabled is False
    assert cfg.actor.token_baseline_method == "none" and cfg.actor.loss_agg_mode == "token-mean"
    audit = cfg.audit
    assert audit.domains == [domain] and set(audit.domain_control_token_ids) == {domain}
    assert set(cfg.data.domain_train_files) == {domain}
    assert audit.control_token_loss_weighting_enabled is True
    assert audit.control_token_online_selection_enabled is False
    assert audit.control_token_normalize_per_domain is False
    assert audit.control_token_phase_gate_enabled is False
    assert audit.control_token_span_weighting_enabled is False
    assert audit.control_token_speed_weighting_enabled is False
    assert audit.control_token_ids == []
    assert (audit.control_token_loss_weight, audit.control_token_other_weight) == (1.0, 0.0)
    for key in PER_TOKEN_AUDIT:
        assert getattr(audit, key) is True, key
    assert cfg.worker_placement.separate_ref_policy is False
    assert cfg.worker_placement.actor_rollout.n_gpus_per_node == cfg.trainer.n_gpus_per_node == cfg.runtime.slurm_allocation_gpus == 8
    assert "actor_rollout_ref.ref.fsdp_config.fsdp_size=8" in cfg.extra_overrides
    assert "trainer.resume_mode=disable" in cfg.extra_overrides
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == 256
    assert cfg.rollout.gpu_memory_utilization == 0.6
    assert cfg.teacher_performance.enabled is True and cfg.teacher_performance.topk_logprob_chunk_size == 1024
    assert cfg.teacher_performance.fused_statistics is True and cfg.teacher_performance.compact_topk_ids is True
    assert cfg.trainer.total_training_steps == 60 and cfg.trainer.save_freq == 1
    assert cfg.huggingface_checkpoint.enabled is True and cfg.huggingface_checkpoint.steps == (60,)
    assert cfg.huggingface_checkpoint.private is False
    assert cfg.data.seed == cfg.trainer.seed == cfg.rollout.seed == 42
    run = f"q1p7b-{domain}-opdnative-l12only-{ARMS[arm]}-toklog-8g8c-b256-s42"
    assert cfg.trainer.experiment_name == cfg.runtime.wandb_run_id == run and len(run) <= 64
    assert cfg.huggingface_checkpoint.path_prefix == f"checkpoints/{domain}/{run}"
    assert cfg.trainer.default_local_dir == f"checkpoints/MOPD/{run}"
    assert audit.output_dir == f"audit/{run}"
    assert cfg.paper_eval.output_dir == f"eval_outputs/paper_suite/{run}"
    command = build_command(cfg)
    assert "data.train_batch_size=256" in command
    assert "+mopd_audit.control_token_loss_weight=1.0" in command
    assert "+mopd_audit.control_token_other_weight=0.0" in command
    assert "+mopd_audit.control_token_normalize_per_domain=false" in command
    pool_override = next(arg for arg in command if arg.startswith("+mopd_audit.domain_control_token_ids="))
    assert yaml.safe_load(pool_override.split("=", 1)[1]) == {domain: ids}


@pytest.mark.parametrize("domain,arm", CASES)
def test_supplements_change_only_the_pool_and_run_identity(domain: str, arm: str) -> None:
    changed = _changed(_path(domain, arm), _path(domain))
    assert changed == set(IDENTITY_KEYS) | {f"audit.domain_control_token_ids.{domain}"}


def test_supplement_run_names_and_outputs_are_unique() -> None:
    configs = [load_config(_path(domain, arm)) for domain, arm in CASES]
    assert len({cfg.trainer.experiment_name for cfg in configs}) == 6
    assert len({cfg.audit.output_dir for cfg in configs}) == 6
    assert len({cfg.paper_eval.output_dir for cfg in configs}) == 6
    assert len({cfg.trainer.default_local_dir for cfg in configs}) == 6
    assert len({cfg.huggingface_checkpoint.path_prefix for cfg in configs}) == 6
    originals = {load_config(_path(domain)).trainer.experiment_name for domain in DIRECTORIES}
    assert originals.isdisjoint(cfg.trainer.experiment_name for cfg in configs)


@pytest.mark.parametrize("domain,arm", CASES)
def test_supplement_gradient_mask(tmp_path: Path, domain: str, arm: str, monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        import verl.utils.device  # noqa: F401
    except ModuleNotFoundError:
        verl_module, utils_module, device_module = ModuleType("verl"), ModuleType("verl.utils"), ModuleType("verl.utils.device")
        verl_module.__path__, utils_module.__path__ = [], []
        device_module.get_device_id = lambda: torch.device("cpu")
        device_module.get_torch_device = lambda: torch.cpu
        verl_module.DataProto = object
        for name, mod in (("verl", verl_module), ("verl.utils", utils_module), ("verl.utils.device", device_module)):
            monkeypatch.setitem(sys.modules, name, mod)
        for name in [name for name in sys.modules if name.startswith("mopd_verl.domain_gradient.")]:
            monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        from mopd_verl.domain_gradient.audit import DomainGradientAudit
    except ModuleNotFoundError as exc:
        pytest.skip(f"training stack unavailable: {exc}")
    cfg = load_config(_path(domain, arm))
    logger = MOPDAuditLogger({"mopd_audit": {**asdict(cfg.audit), "output_dir": str(tmp_path)}})
    meta = logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
    actor = SimpleNamespace(actor_module=torch.nn.Linear(1, 1), actor_optimizer=SimpleNamespace(param_groups=[{}]),
                            config={"policy_loss": asdict(cfg.actor)})
    audit = DomainGradientAudit(actor, meta)
    original = set(load_config(_path(domain)).audit.domain_control_token_ids[domain])
    info = MANIFEST["domains"][domain]
    term_id = min(set(info["terminate"]) - original)
    look_id = min(set(info["lookback"]) - original)
    pool = set(cfg.audit.domain_control_token_ids[domain])
    other = 99999
    assert other not in pool
    ids = torch.tensor([[min(original), term_id, look_id, other, 151645, max(original)]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    batch = SimpleNamespace(batch={"responses": ids, "response_mask": mask},
                            non_tensor_batch={"domain": np.array([domain], dtype=object)})
    weights = audit.training_gradient_mask(batch)
    assert weights is not None
    actual = (weights.float() * mask.float()).cpu()
    expected = torch.tensor([[1.0, float(term_id in pool), float(look_id in pool), 0.0, 1.0, 0.0]])
    torch.testing.assert_close(actual, expected)
