"""control_token_other_weight and the v3-pool ignore/only OPD configs."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.phase_control import phase_token_weights
from mopd_verl.settings import load_config
from mopd_verl.verl_audit import MOPDAuditLogger

ROOT = Path(__file__).resolve().parents[1]
TAX = ROOT / "configs/token_selection/math_code/taxonomy"
Q4B = ROOT / "configs/qwen4b_suite/math_code"
Q1P7B = ROOT / "configs/qwen1p7b_suite/math_code"
NATIVE = [Q1P7B / f"opd_native_v3pool_{arm}_toklog_8gpu_8c_b256.yaml" for arm in ("ignore", "only")]
CONFIGS = (
    [TAX / f"mopd_math_code_v3pool_{arm}_{topo}.yaml" for arm in ("ignore", "only") for topo in ("8gpu_8c", "4gpu", "3gpu_3c")]
    + [Q4B / f"opd_v3pool_{arm}_{topo}.yaml" for arm in ("ignore", "only") for topo in ("8gpu_8c", "3gpu_3c")]
    + NATIVE
)


def _weights(control: float, other: float, normalize: bool = False) -> torch.Tensor:
    token_ids = torch.tensor([[10, 11, 12], [20, 21, 20]])
    return phase_token_weights(
        token_ids, torch.ones_like(token_ids), ["math", "code"],
        domain_token_ids={"math": [10], "code": [20]}, control_weight=control,
        phase_enabled=False, span_enabled=False, phase_gates={}, span_length=0,
        span_decay_tau=1.0, normalize_per_domain=normalize, other_weight=other,
    )


def test_other_weight_default_is_unchanged() -> None:
    torch.testing.assert_close(_weights(4.0, 1.0), torch.tensor([[4.0, 1.0, 1.0], [4.0, 1.0, 4.0]]))


def test_only_pool_and_ignore_pool_weights() -> None:
    torch.testing.assert_close(_weights(1.0, 0.0), torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]))
    torch.testing.assert_close(_weights(0.0, 1.0), torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]]))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_configs_load_with_v3_pools(path: Path) -> None:
    audit = load_config(path).audit
    only = "_only_" in path.name
    assert audit.control_token_loss_weighting_enabled
    assert not audit.control_token_online_selection_enabled
    assert not audit.control_token_normalize_per_domain
    assert audit.control_token_ids == []
    assert (audit.control_token_loss_weight, audit.control_token_other_weight) == ((1.0, 0.0) if only else (0.0, 1.0))
    assert len(set(audit.domain_control_token_ids["math"])) == 98
    assert len(set(audit.domain_control_token_ids["code"])) == 167


def test_other_weight_reaches_domain_gradient_config(tmp_path: Path) -> None:
    config = load_config(TAX / "mopd_math_code_v3pool_only_4gpu.yaml")
    logger = MOPDAuditLogger({"mopd_audit": {**asdict(config.audit), "output_dir": str(tmp_path)}})
    domain = DomainGradientConfig.from_meta(logger.full_gradient_meta("train", 1)["mopd_full_gradient"])
    assert domain.control_token_other_weight == 0.0
    assert domain.control_token_weight == 1.0


@pytest.mark.parametrize("change,message", [
    ({"control_token_other_weight": -1.0}, "finite and non-negative"),
    ({"control_token_other_weight": 0.0, "control_token_loss_weight": 0.0}, "cannot both be 0"),
    ({"control_token_other_weight": 0.0, "control_token_normalize_per_domain": True}, "supports only fixed weighting"),
])
def test_invalid_other_weight_combinations_fail(tmp_path: Path, change: dict, message: str) -> None:
    import yaml
    path = TAX / "mopd_math_code_v3pool_only_4gpu.yaml"
    tmp = tmp_path / "c.yaml"
    tmp.write_text(yaml.safe_dump({"extends": str(path), "audit": change}))
    with pytest.raises(ValueError, match=message):
        load_config(tmp)


@pytest.mark.parametrize("arm,path", [(arm, TAX / f"mopd_math_code_v3pool_{arm}_4gpu.yaml") for arm in ("only", "ignore")]
                         + [("ignore" if "_ignore_" in p.name else "only", p) for p in NATIVE], ids=lambda x: getattr(x, "name", x))
def test_training_gradient_mask_applies_pool_weights(tmp_path: Path, arm: str, path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib, sys
    from types import ModuleType
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
        for name in [n for n in sys.modules if n.startswith("mopd_verl.domain_gradient.")]:
            monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        from mopd_verl.domain_gradient.audit import DomainGradientAudit
    except ModuleNotFoundError as exc:
        pytest.skip(f"training stack unavailable: {exc}")
    config = load_config(path)
    logger = MOPDAuditLogger({"mopd_audit": {**asdict(config.audit), "output_dir": str(tmp_path)}})
    meta = logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
    actor = SimpleNamespace(actor_module=torch.nn.Linear(1, 1), actor_optimizer=SimpleNamespace(param_groups=[{}]),
                            config={"policy_loss": asdict(config.actor)})
    audit = DomainGradientAudit(actor, meta)
    m_pool, c_pool = config.audit.domain_control_token_ids["math"][0], config.audit.domain_control_token_ids["code"][0]
    other = 99999
    assert other not in config.audit.domain_control_token_ids["math"] + config.audit.domain_control_token_ids["code"]
    ids = torch.tensor([[m_pool, other, c_pool, 0], [c_pool, other, m_pool, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
    batch = SimpleNamespace(batch={"responses": ids, "response_mask": mask}, non_tensor_batch={"domain": np.array(["math", "code"], dtype=object)})
    w = audit.training_gradient_mask(batch)
    assert w is not None
    w = (w.float() * mask.float()).cpu()
    # math row: its own pool ID counts, the Code pool ID does not belong to the Math pool unless shared
    m_in_c = c_pool in config.audit.domain_control_token_ids["math"]; c_in_m = m_pool in config.audit.domain_control_token_ids["code"]
    pool_w, other_w = (1.0, 0.0) if arm == "only" else (0.0, 1.0)
    expected = torch.tensor([[pool_w, other_w, pool_w if m_in_c else other_w, 0.0],
                             [pool_w, other_w, pool_w if c_in_m else other_w, 0.0]])
    torch.testing.assert_close(w, expected)


SINGLE = (
    [(ROOT / f"configs/token_selection/{d}/{'taxonomy/' if d == 'math' else ''}v3pool_{arm}_top32kl_{t}.yaml", d, arm)
     for d in ("math", "code") for arm in ("ignore", "only") for t in ("3gpu_3c_b255",)]
    + [(ROOT / f"configs/qwen4b_suite/{d}/opd_v3pool_{arm}_{t}.yaml", d, arm)
       for d in ("math", "code") for arm in ("ignore", "only") for t in ("3gpu_3c", "8gpu_8c")]
    + [(ROOT / f"configs/token_selection/{d}/{'taxonomy/' if d == 'math' else ''}native_opd_v3pool_{arm}_toklog_8gpu_8c_b256.yaml", d, arm)
       for d in ("math", "code") for arm in ("ignore", "only")]
)


@pytest.mark.parametrize("path,domain,arm", SINGLE, ids=lambda x: getattr(x, "name", str(x)))
def test_single_domain_configs(path: Path, domain: str, arm: str) -> None:
    cfg = load_config(path)
    audit = cfg.audit
    assert audit.domains == [domain]
    assert set(audit.domain_control_token_ids) == {domain}
    assert len(set(audit.domain_control_token_ids[domain])) == {"math": 98, "code": 167}[domain]
    assert (audit.control_token_loss_weight, audit.control_token_other_weight) == ((1.0, 0.0) if arm == "only" else (0.0, 1.0))
    assert not audit.control_token_online_selection_enabled and not audit.control_token_normalize_per_domain
    assert 255 <= cfg.data.train_batch_size <= 258
    assert set(cfg.data.domain_train_files) == {domain}


@pytest.mark.parametrize("path", [ROOT / "configs/token_selection/code/uniform_top32kl_no_weighting_4gpu_3a1t_b255.yaml",
                                  ROOT / "configs/qwen4b_suite/code/opd_3gpu_3c.yaml"], ids=lambda p: p.name)
def test_code_only_references_are_equal_weight(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.audit.domains == ["code"]
    assert not cfg.audit.control_token_loss_weighting_enabled
    assert cfg.actor.distill_mode == "topk_renormalized_reverse_kl"


@pytest.mark.parametrize("path", sorted((ROOT / "configs/token_selection").rglob("*3gpu_3c*.yaml")), ids=lambda p: p.name)
def test_three_gpu_configs_are_colocated(path: Path) -> None:
    if path.name.startswith("_"):
        pytest.skip("intermediate file")
    cfg = load_config(path)
    if "v3pool" not in path.name and "toklog" not in path.name and "uniform_top32kl_no_weighting" not in path.name:
        pytest.skip("not part of the v3 / toklog family")
    assert not cfg.worker_placement.separate_ref_policy
    assert cfg.worker_placement.actor_rollout.n_gpus_per_node == 3
    assert cfg.data.train_batch_size % 3 == 0
    assert cfg.trainer.total_training_steps == 60
