"""SAPD Math+Code profiles (paper/paper.md section 1.3): pools, selector, topology and run namespaces."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.frozen_taxonomy import CONTROL_TOKEN_IDS, STRUCTURE_TOKEN_IDS
from mopd_verl.launch import build_command
from mopd_verl.settings import MOPDConfig, load_config
from mopd_verl.verl_audit import MOPDAuditLogger

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/token_selection/math_code/taxonomy"
PREFIX = "mopd_math_code_toploss_m05_c01_math_control_code_progvocab_fixed4_"
# Topology -> (reference profile that differs only in the two pools and the run names,
#              actor GPUs, dedicated teacher GPUs, allocated GPUs, run id).
PROFILES = {
    "4gpu": (
        "mopd_math_code_toploss_m05_c01_code_structure_fixed4_4gpu.yaml",
        4, None, 4, "q1p7b4g-mc-tl-m05c01-mctrl-cprog-f4-b528-s60",
    ),
    "8gpu_6s2t": (
        "mopd_math_code_toploss_m05_c01_code_structure_fixed4_8gpu_6s2t.yaml",
        6, 2, 8, "q1p7b4g-mc-tl-m05c01-mctrl-cprog-8g6s2t-b528-s60",
    ),
    "3student_1teacher": (
        "mopd_math_code_toploss_m05_c01_code_structure_disc33_fixed4_3student_1teacher.yaml",
        3, 1, 4, "q1p7b3s1t-mc-tl-m05c01-mctrl-cprog-f4-b528-s60",
    ),
}
MATH_SIZE = 124
CODE_SIZE = 163
# Math Control (`pool == Control`) and Code programming vocabulary (`taxonomy == code_lexical_structure`)
# of analysis-output/c01-c02-selection-stepdiff-20260930/pruned_pool/{math,code}_candidates_{kept,dropped}.csv.
MATH_POOL_SHA256 = "ccb1adb8eda11b0935c20e77b714f1e8f1de0b79c8dcfe87458ca72bc9bc8bda"
CODE_POOL_SHA256 = "9a7120d1bb8768bb4e7491987e2722bcecdd1d26b2f633359910c93c1895a265"
CANDIDATES_ARG = "+mopd_audit.domain_control_token_candidate_ids="


def _pool_sha256(ids: set[int]) -> str:
    payload = json.dumps(sorted(ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path(name: str) -> Path:
    return CONFIGS / f"{PREFIX}{name}.yaml"


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


@pytest.mark.parametrize("name", PROFILES)
def test_profile_contract(name: str, tmp_path: Path) -> None:
    _, actor_gpus, teacher_gpus, allocated_gpus, run_id = PROFILES[name]
    config = load_config(_path(name))
    audit = config.audit

    # Next-Step TopLoss ID-mean ranking, whole-ID top-p budget, Fixed4 with per-domain mean-one.
    assert audit.control_token_online_selection_timing == "next_step"
    assert audit.control_token_online_selection_unit == "token_id"
    assert audit.control_token_online_selection_mode == "top_loss"
    assert audit.control_token_online_budget_mode == "top_p"
    assert audit.control_token_online_audit_interval_steps == 1
    assert audit.control_token_online_window_steps == 1
    assert audit.control_token_online_min_mean_occurrences_per_step == 20.0
    assert audit.control_token_online_strict_occurrence_gate is True
    assert audit.control_token_online_top_p_by_domain == {"math": 0.05, "code": 0.01}
    assert audit.control_token_loss_weight == 4.0
    assert audit.control_token_normalize_per_domain is True
    assert audit.token_taxonomy_version == "legacy"
    assert audit.structure_token_loss_weighting_enabled is False

    # One semantic category per domain: Math Control only, Code programming vocabulary only.
    math, code = _pools(config)
    assert len(math) == MATH_SIZE and len(code) == CODE_SIZE
    assert _pool_sha256(math) == MATH_POOL_SHA256
    assert _pool_sha256(code) == CODE_POOL_SHA256
    assert math <= CONTROL_TOKEN_IDS
    assert code <= STRUCTURE_TOKEN_IDS and not code & CONTROL_TOKEN_IDS
    assert 369 not in code  # " for" is a Control ID, not Code vocabulary

    assert config.data.train_batch_size == 528
    assert config.actor.ppo_mini_batch_size == 528
    assert config.trainer.total_training_steps == 60
    assert config.huggingface_checkpoint.private is False
    assert config.huggingface_checkpoint.steps == (60,)
    assert config.runtime.slurm_allocation_gpus == allocated_gpus
    assert config.trainer.n_gpus_per_node == actor_gpus
    assert config.worker_placement.actor_rollout.n_gpus_per_node == actor_gpus
    assert config.worker_placement.separate_ref_policy is (teacher_gpus is not None)
    assert config.worker_placement.ref_policy.n_gpus_per_node == teacher_gpus
    assert config.teacher_performance.enabled is True
    assert config.data.train_batch_size % actor_gpus == 0
    assert config.runtime.wandb_run_id == run_id

    domain = _runtime_metadata(config, tmp_path)
    assert domain.control_token_online_selection_timing == "next_step"
    assert domain.control_token_online_selection_unit == "token_id"


@pytest.mark.parametrize("name", PROFILES)
def test_only_pools_and_run_names_differ_from_reference(name: str) -> None:
    reference_name = PROFILES[name][0]
    config = load_config(_path(name))
    reference = load_config(CONFIGS / reference_name)
    new_name = config.trainer.experiment_name
    old_name = reference.trainer.experiment_name

    def strip(command: list[str], run_name: str) -> list[str]:
        return [
            argument.replace(run_name, "<run>")
            for argument in command
            if not argument.startswith(CANDIDATES_ARG)
        ]

    assert strip(build_command(config), new_name) == strip(build_command(reference), old_name)


def test_pools_identical_across_topologies_and_namespaces_unique() -> None:
    configs = {name: load_config(_path(name)) for name in PROFILES}
    assert len({tuple(map(frozenset, _pools(c))) for c in configs.values()}) == 1
    seen: set[str] = set()
    for config in configs.values():
        run_id = config.runtime.wandb_run_id
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

    ours = {_path(name) for name in PROFILES}
    run_ids = {c.runtime.wandb_run_id for c in configs.values()}
    for path in CONFIGS.glob("*.yaml"):
        if path in ours:
            continue
        runtime = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("runtime") or {}
        assert runtime.get("wandb_run_id") not in run_ids, path.name
