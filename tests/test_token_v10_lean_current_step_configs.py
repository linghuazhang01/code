"""Token V10-lean current-step Math 5% / Code 1% profiles: pool, timing, topology and run namespaces.

Math = the 390-ID Math pool of the best Math+Code run (mopd_math_code_toploss_m05_c01_code_structure_*);
Code = the 379-ID V10-lean pool (analysis-output/code-pool-v10-20261006/tables/code_pool_v10_ids.json).
Only the Code pool differs from the best run's pools; the selector mirrors the V9 current-step profiles.
"""

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
MATH_SIZE = 390
CODE_SIZE = 379
# Frozen against Token.md section 0.6 (sorted compact-JSON SHA256 of the ID arrays).
MATH_POOL_SHA256 = "3467edb63771921d3fc65fe81852ebe7617799c4df2acc941acb63d6259fb96c"
CODE_POOL_SHA256 = "967d3929b757a216e1369649b84b501e45bb817ed08e864a6050450540a8dfcc"
UNION_SHA256 = "f01ef060b1ce24b41b19cc3f173c5b768bc837038b508a80e0fddabfd6b50c85"
# '**' IDs are excluded from the Code pool by construction (V6-V10 convention); the Math pool is the
# best run's pool and is NOT subject to that rule.
BOLD_IDS = {334, 1019, 3070, 32295, 56177, 95518, 97219}
# Representative members of the new Code classes (decoded surface -> Qwen3 ID).
CODE_BOUNDARY = {"```": 73594, "<|im_end|>": 151645, "###": 14374}
CODE_SCAFFOLD = {" Approach": 53084, " Implementation": 30813, " Strategy": 27745, " Steps": 39861}
CODE_ALGOLEX = {" dp": 11329, " visited": 11994, " graph": 4771, " defaultdict": 42908, " heapq": 88522}
# Opener class and no-space AlgoLex variants are deliberately absent from the lean pool.
CODE_ABSENT = {"Here": 8420, "Wait": 14190, "dp": 9796}


def _pool_sha256(ids: set[int]) -> str:
    payload = json.dumps(sorted(ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path(gpus: int) -> Path:
    return CONFIGS / f"mopd_math_code_current_step_token_v10lean_toploss_m05_c01_fixed4_{gpus}gpu_colocated.yaml"


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

    # Same legacy selector as V6-V9: no versioned taxonomy, no Code position gate.
    assert audit.token_taxonomy_version == "legacy"
    assert audit.structure_token_loss_weighting_enabled is False
    assert audit.code_cs_position_gate_enabled is None
    assert audit.versioned_cs_selection_mode_by_domain == {}

    math, code = _pools(config)
    assert len(math) == MATH_SIZE and len(code) == CODE_SIZE
    assert _pool_sha256(math) == MATH_POOL_SHA256
    assert _pool_sha256(code) == CODE_POOL_SHA256
    assert _pool_sha256(math | code) == UNION_SHA256
    assert not code & BOLD_IDS
    assert set(CODE_BOUNDARY.values()) <= code
    assert set(CODE_SCAFFOLD.values()) <= code
    assert set(CODE_ALGOLEX.values()) <= code
    assert not set(CODE_ABSENT.values()) & code

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
    # The 4-GPU server run shares GPU 0 with another process, so it reserves less vLLM memory.
    assert config.rollout.gpu_memory_utilization == (0.6 if gpus == 4 else 0.75)

    domain = _runtime_metadata(config, tmp_path)
    assert domain.control_token_online_selection_timing == "current_step"
    assert domain.control_token_online_selection_unit == "token_id"


def test_math_pool_is_the_best_runs_math_pool() -> None:
    best = load_config(CONFIGS / "mopd_math_code_toploss_m05_c01_code_structure_fixed4_8gpu_6s2t.yaml")
    expected = set(best.audit.domain_control_token_candidate_ids["math"])
    math, _ = _pools(load_config(_path(4)))
    assert math == expected and len(expected) == 390


def test_pools_identical_across_topologies_and_namespaces_unique() -> None:
    pools = {_path(gpus): _pools(load_config(_path(gpus))) for gpus in GPUS}
    assert len({(frozenset(m), frozenset(c)) for m, c in pools.values()}) == 1
    seen: set[str] = set()
    for gpus in GPUS:
        config = load_config(_path(gpus))
        run_id = config.runtime.wandb_run_id
        assert run_id == f"q1p7b{gpus}g-mc-v10lean-m390c379-current-m05c01-f4-b528-s60"
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


def test_legacy_guard_accepts_v10_but_still_rejects_foreign_or_mixed_ids() -> None:
    from dataclasses import replace

    from mopd_verl.domain_gradient.frozen_taxonomy import (
        CONTROL_TOKEN_IDS,
        STRUCTURE_TOKEN_IDS,
        TOKEN_V9_CONTROL_IDS,
        TOKEN_V10_CANDIDATE_IDS,
    )
    from mopd_verl.domain_gradient.occurrence_config import validate_occurrence_config

    audit = load_config(_path(4)).audit
    math, code = (set(v) for v in (audit.domain_control_token_candidate_ids[d] for d in ("math", "code")))
    assert math | code == TOKEN_V10_CANDIDATE_IDS
    validate_occurrence_config(audit)

    taxonomy = CONTROL_TOKEN_IDS | STRUCTURE_TOKEN_IDS
    v10_only = min(TOKEN_V10_CANDIDATE_IDS - taxonomy - TOKEN_V9_CONTROL_IDS)  # e.g. ' Approach'
    v9_only = min(TOKEN_V9_CONTROL_IDS - taxonomy - TOKEN_V10_CANDIDATE_IDS)  # e.g. Wait/Let
    structure_only = min(STRUCTURE_TOKEN_IDS - TOKEN_V10_CANDIDATE_IDS)
    foreign = max(TOKEN_V10_CANDIDATE_IDS | TOKEN_V9_CONTROL_IDS | taxonomy) + 1
    for bad in ([v10_only, v9_only], [v10_only, structure_only], [foreign]):
        pools = {"math": tuple(sorted(math)), "code": tuple(sorted(code | set(bad)))}
        with pytest.raises(ValueError, match="C\\+S or Token V9"):
            validate_occurrence_config(replace(audit, domain_control_token_candidate_ids=pools))
