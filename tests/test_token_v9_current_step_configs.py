"""Token V9 current-step Math 5% / Code 1% profiles: pool, timing, topology and run namespaces."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.launch import build_command
from mopd_verl.settings import MOPDConfig, load_config
from mopd_verl.verl_audit import MOPDAuditLogger

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/token_selection/math_code/taxonomy"
GPUS = (3, 4, 8)
MATH_SIZE = 117
CODE_SIZE = 169
# Frozen against the ID inventories registered in Token.md section 0.5.
MATH_POOL_SHA256 = "1d63fc235cc80438413f37423d2a8d415d6b783d4e675b051521ab9b72c8445a"
CODE_POOL_SHA256 = "69c0149131a065b7c6c78272341eb90213ec3977b06465a843f35c2ccf03a909"
# '.\n\n' removed by user decision; '**' IDs are excluded from every candidate pool (V6-V9).
DROPPED_IDS = {382}
BOLD_IDS = {334, 1019, 3070, 32295, 56177, 95518, 97219}
# Code control-flow keywords that V9 adds from the former Code Structure inventory.
CODE_KEYWORDS = {
    "def": 750, " def": 707, " elif": 4409, " else": 770, " while": 1393, " except": 3650,
    " continue": 3060, " return": 470, " for": 369, " break": 1438, " raise": 4828,
}


def _pool_sha256(ids: set[int]) -> str:
    payload = json.dumps(sorted(ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path(gpus: int) -> Path:
    return CONFIGS / f"mopd_math_code_current_step_token_v9_toploss_m05_c01_fixed4_{gpus}gpu_colocated.yaml"


def _pools(config: MOPDConfig) -> tuple[set[int], set[int]]:
    ids = config.audit.domain_control_token_candidate_ids
    return set(ids["math"]), set(ids["code"])


def _runtime_metadata(config: MOPDConfig, tmp_path: Path) -> DomainGradientConfig:
    audit = {
        key.removeprefix("+mopd_audit."): yaml.safe_load(value)
        for key, value in (
            argument.split("=", 1)
            for argument in build_command(config)
            if argument.startswith("+mopd_audit.")
        )
    }
    audit["output_dir"] = str(tmp_path)
    meta = MOPDAuditLogger({"mopd_audit": audit}).full_gradient_meta("train", 1)
    return DomainGradientConfig.from_meta(meta["mopd_full_gradient"])


@pytest.mark.parametrize("gpus", GPUS)
def test_profile_contract(gpus: int, tmp_path: Path) -> None:
    config = load_config(_path(gpus))
    audit = config.audit

    assert audit.control_token_online_selection_timing == "current_step"
    assert audit.control_token_online_selection_unit == "token_id"
    assert audit.control_token_online_selection_mode == "top_loss"
    assert audit.control_token_online_budget_mode == "top_p"
    assert audit.control_token_online_audit_interval_steps == 1
    assert audit.control_token_online_window_steps == 1
    assert audit.control_token_tail_top_p == 0
    assert audit.control_token_loss_weight == 4.0
    assert audit.control_token_normalize_per_domain is True
    assert audit.control_token_online_top_p_by_domain == {"math": 0.05, "code": 0.01}

    # Same legacy selector as V6/V7/V8: no versioned taxonomy, no Code position gate.
    assert audit.token_taxonomy_version == "legacy"
    assert audit.structure_token_loss_weighting_enabled is False
    assert audit.code_cs_position_gate_enabled is None
    assert audit.versioned_cs_selection_mode_by_domain == {}

    math, code = _pools(config)
    assert len(math) == MATH_SIZE and len(code) == CODE_SIZE
    assert _pool_sha256(math) == MATH_POOL_SHA256
    assert _pool_sha256(code) == CODE_POOL_SHA256
    assert not (math | code) & (DROPPED_IDS | BOLD_IDS)
    assert set(CODE_KEYWORDS.values()) <= code

    assert config.data.train_batch_size == 528
    assert config.actor.ppo_mini_batch_size == 528
    assert config.trainer.total_training_steps == 60 <= 65
    assert config.huggingface_checkpoint.private is False
    assert config.huggingface_checkpoint.steps == (60,)
    assert config.runtime.slurm_allocation_gpus == gpus
    assert config.trainer.n_gpus_per_node == gpus
    assert config.worker_placement.separate_ref_policy is False
    assert config.worker_placement.actor_rollout.n_gpus_per_node == gpus
    assert f"actor_rollout_ref.ref.fsdp_config.fsdp_size={gpus}" in config.extra_overrides
    assert config.teacher_performance.enabled is True
    assert config.data.train_batch_size % gpus == 0

    domain = _runtime_metadata(config, tmp_path)
    assert domain.control_token_online_selection_timing == "current_step"
    assert domain.control_token_online_selection_unit == "token_id"


def test_pools_identical_across_topologies_and_namespaces_unique() -> None:
    pools = {_path(gpus): _pools(load_config(_path(gpus))) for gpus in GPUS}
    assert len({(frozenset(m), frozenset(c)) for m, c in pools.values()}) == 1
    seen: set[str] = set()
    for gpus in GPUS:
        config = load_config(_path(gpus))
        run_id = config.runtime.wandb_run_id
        assert run_id == f"q1p7b{gpus}g-mc-v9-ctl-current-m05c01-f4-b528-s60"
        values = (
            config.audit.output_dir,
            config.paper_eval.output_dir,
            config.huggingface_checkpoint.path_prefix,
            config.trainer.experiment_name,
            config.trainer.default_local_dir,
        )
        assert all(run_id in value for value in values)
        assert not seen & set(values)
        seen |= set(values)


def test_legacy_guard_accepts_v9_but_still_rejects_foreign_or_mixed_ids() -> None:
    from dataclasses import replace

    from mopd_verl.domain_gradient.frozen_taxonomy import (
        CONTROL_TOKEN_IDS,
        STRUCTURE_TOKEN_IDS,
        TOKEN_V9_CONTROL_IDS,
    )
    from mopd_verl.domain_gradient.occurrence_config import validate_occurrence_config

    audit = load_config(_path(4)).audit
    math, code = (set(v) for v in (audit.domain_control_token_candidate_ids[d] for d in ("math", "code")))
    assert math | code == TOKEN_V9_CONTROL_IDS
    validate_occurrence_config(audit)

    v9_only = min(TOKEN_V9_CONTROL_IDS - CONTROL_TOKEN_IDS - STRUCTURE_TOKEN_IDS)  # e.g. Wait/Let
    structure_only = min(STRUCTURE_TOKEN_IDS - TOKEN_V9_CONTROL_IDS)
    foreign = max(TOKEN_V9_CONTROL_IDS | CONTROL_TOKEN_IDS | STRUCTURE_TOKEN_IDS) + 1
    for bad in ([v9_only, structure_only], [foreign]):
        pools = {"math": tuple(bad), "code": tuple(sorted(code))}
        with pytest.raises(ValueError, match="C\\+S or Token V9"):
            validate_occurrence_config(replace(audit, domain_control_token_candidate_ids=pools))
