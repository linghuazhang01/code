"""R1: Top-32 RKL OPD with full 结果结构 ∪ 推理阻力 up-weighted ×4, mean one (Math-only and Code-only, 1.7B)."""
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
STEM = "top32kl_result_brake_fixed4_toklog_3gpu_3c_b255.yaml"
LEAVES = {
    "math": ROOT / "configs/token_selection/math/taxonomy" / STEM,
    "code": ROOT / "configs/token_selection/code" / STEM,
}
REFERENCES = {
    domain: path.with_name("uniform_top32kl_toklog_3gpu_3c_b255.yaml") for domain, path in LEAVES.items()
}
CHANGED_AUDIT = {
    "control_token_loss_weighting_enabled",
    "control_token_normalize_per_domain",
    "domain_control_token_ids",
    "control_token_loss_weight",
    "control_token_other_weight",
    "control_token_phase_gate_enabled",
    "control_token_span_weighting_enabled",
    "control_token_speed_weighting_enabled",
    "control_token_online_selection_enabled",
    "control_token_ids",
    "output_dir",
}


def _pool(domain: str) -> set[int]:
    full = RECORD["domains"][domain]["full_classes"]
    return (set(full["result"]["ids"]) - {25521}) | set(full["brake"]["ids"])


@pytest.mark.parametrize("domain", sorted(LEAVES))
def test_pool_and_weights(domain: str) -> None:
    audit = load_config(LEAVES[domain]).audit
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
    assert 25521 not in ids  # byte fragment of " ✅", removed by the author 2026-11-17
    assert len(ids) == {"math": 75, "code": 77}[domain]


@pytest.mark.parametrize("domain", sorted(LEAVES))
def test_only_weighting_and_names_differ_from_reference(domain: str) -> None:
    leaf, ref = load_config(LEAVES[domain]), load_config(REFERENCES[domain])
    leaf_d, ref_d = asdict(leaf), asdict(ref)
    # region_dpo is off; it only mirrors the control-token IDs through with_control_token_fallback.
    assert not leaf.region_dpo.enabled
    leaf_d["region_dpo"].pop("domain_control_token_ids")
    ref_d["region_dpo"].pop("domain_control_token_ids")
    for section in leaf_d:
        if section in {"audit", "runtime", "trainer", "paper_eval", "huggingface_checkpoint"}:
            continue
        assert leaf_d[section] == ref_d[section], section
    diff = {k for k in leaf_d["audit"] if leaf_d["audit"][k] != ref_d["audit"][k]}
    assert diff <= CHANGED_AUDIT, diff - CHANGED_AUDIT
    run = f"q1p7b-{domain}-top32kl-resbrake-fixed4-toklog-3g3c-b255-s42"
    assert leaf.runtime.wandb_run_id == run
    assert leaf.trainer.experiment_name == run
    assert leaf.huggingface_checkpoint.path_prefix == f"checkpoints/{domain}/{run}"
    assert list(leaf.huggingface_checkpoint.steps) == [60]
    assert leaf.trainer.total_training_steps == 60
    assert leaf.trainer.save_freq == 1
    assert leaf.data.train_batch_size == 255
    assert leaf.rollout.gpu_memory_utilization == 0.6
    assert not leaf.worker_placement.separate_ref_policy
    for key in ("runtime", "trainer", "paper_eval", "huggingface_checkpoint"):
        changed = {k for k in leaf_d[key] if leaf_d[key][k] != ref_d[key][k]}
        assert changed <= {"wandb_run_id", "wandb_resume", "experiment_name", "default_local_dir",
                           "output_dir", "path_prefix"}, (key, changed)


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
