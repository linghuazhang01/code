"""Token V6/V7/V8 current-step profiles: pools, timing, topology and run namespaces."""

import hashlib
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path

import pytest
import yaml

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.launch import build_command
from mopd_verl.settings import MOPDConfig, load_config
from mopd_verl.verl_audit import MOPDAuditLogger

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/token_selection/math_code/taxonomy"
VERSIONS = (6, 7, 8)
GPUS = (3, 4, 8)
RUNS = {"c01": 0.01, "c02": 0.02}
CODE_CONTROL_21 = {
    323, 369, 389, 421, 438, 448, 504, 1156, 1172, 1221, 1249, 1393, 1416, 1752,
    1986, 2014, 2055, 2461, 4416, 8704, 13023,
}
# C23 = the 21 Control IDs chosen from >=2-baseline Rising plus the two Code Control IDs
# (` or`=476, ` Now`=4695) found only in a 1.7B Rising Top-200.
CODE_CONTROL_23 = CODE_CONTROL_21 | {476, 4695}
BOLD_IDS = {334, 1019, 3070, 32295, 56177, 95518, 97219}
MATH_SIZE = 274
CODE_SIZE = {6: 219, 7: 196, 8: 341}
# Frozen against the ID inventories registered in Token.md section 0.4.
MATH_POOL_SHA256 = "b0e6445ac181e03e46b8c88776157ecc41224598eeba26dbf6608d999749b665"
CODE_POOL_SHA256 = {
    6: "f888c81aeeeb47dc258b1fe90eef8971e9c76ad03864b7bce410ef18f1672750",
    7: "3ed9edc703988b707e68cfbd9bd7c3b7b5ea52d3d552c3a699bfff547b818fe8",
    8: "2c69c5c9b08c96cf8552750ac4f17b57133910368048d31abe2a916c0d6566ba",
}
RISING_FIXTURE = Path(__file__).parent / "fixtures/v678_rising_1p7b.json"


def _pool_sha256(ids: set[int]) -> str:
    payload = json.dumps(sorted(ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path(version: int, run: str, gpus: int, structure_only: bool) -> Path:
    middle = "code_structure_" if structure_only else ""
    return CONFIGS / (
        f"mopd_math_code_current_step_token_v{version}_toploss_m05_{run}_{middle}"
        f"fixed4_{gpus}gpu_colocated.yaml"
    )


def _all_profiles() -> list[tuple[int, str, int, bool]]:
    return list(product(VERSIONS, RUNS, GPUS, (False, True)))


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


def test_profile_matrix_is_complete() -> None:
    assert len(_all_profiles()) == 36
    assert all(_path(*profile).exists() for profile in _all_profiles())


@pytest.mark.parametrize("version,run,gpus,structure_only", _all_profiles())
def test_profile_contract(version: int, run: str, gpus: int, structure_only: bool, tmp_path: Path) -> None:
    config = load_config(_path(version, run, gpus, structure_only))
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
    assert audit.control_token_online_top_p_by_domain == {"math": 0.05, "code": RUNS[run]}

    # Legacy selector: no versioned taxonomy, no Code C/S position gate, no fenced-block logic.
    assert audit.token_taxonomy_version == "legacy"
    assert audit.structure_token_loss_weighting_enabled is False
    assert audit.code_cs_position_gate_enabled is None
    assert audit.versioned_cs_selection_mode_by_domain == {}

    math, code = _pools(config)
    assert len(math) == MATH_SIZE and not math & BOLD_IDS
    assert not code & BOLD_IDS
    dropped_control = structure_only and version == 6
    expected = CODE_SIZE[7] if dropped_control else CODE_SIZE[version]
    assert len(code) == expected
    assert _pool_sha256(math) == MATH_POOL_SHA256
    assert _pool_sha256(code) == CODE_POOL_SHA256[7 if dropped_control else version]
    if version == 6 and not structure_only:
        assert CODE_CONTROL_23 <= code
        assert len(code - _pools(load_config(_path(7, run, gpus, False)))[1]) == len(CODE_CONTROL_23)
    else:
        assert not code & CODE_CONTROL_23

    assert config.data.train_batch_size == 528
    assert config.actor.ppo_mini_batch_size == 528
    assert config.trainer.total_training_steps == 60 <= 65
    assert config.huggingface_checkpoint.private is False
    assert config.huggingface_checkpoint.steps == (60,)
    assert max(config.huggingface_checkpoint.steps) <= config.trainer.total_training_steps
    assert config.runtime.slurm_allocation_gpus == gpus
    assert config.trainer.n_gpus_per_node == gpus
    assert config.worker_placement.separate_ref_policy is False
    assert config.worker_placement.actor_rollout.n_gpus_per_node == gpus
    assert f"actor_rollout_ref.ref.fsdp_config.fsdp_size={gpus}" in config.extra_overrides
    assert config.data.train_batch_size % gpus == 0

    domain = _runtime_metadata(config, tmp_path)
    assert domain.control_token_online_selection_timing == "current_step"
    assert domain.control_token_online_selection_unit == "token_id"


def test_pool_relations_across_versions() -> None:
    v6 = _pools(load_config(_path(6, "c01", 4, False)))
    v6_struct = _pools(load_config(_path(6, "c01", 4, True)))
    v7 = _pools(load_config(_path(7, "c01", 4, False)))
    v8 = _pools(load_config(_path(8, "c01", 4, False)))
    assert v6[0] == v6_struct[0] == v7[0] == v8[0]
    assert v6[1] == v7[1] | CODE_CONTROL_23
    assert v6_struct[1] == v7[1]
    assert v7[1] < v8[1]
    assert not v6[1] & CODE_CONTROL_23 & v8[1]


def _rising_control_structure(domain: str) -> dict[str, set[int]]:
    """Control/Structure IDs in either 1.7B baseline's Rising Top-200, keyed by type."""
    rising = json.loads(RISING_FIXTURE.read_text(encoding="utf-8"))["tokens"][domain]
    return {kind: set(ids) for kind, ids in rising.items()}


def test_every_1p7b_rising_control_structure_token_is_in_the_pools() -> None:
    """Hit rate 100% on the Control/Structure part of the 1.7B Rising Top-200."""
    math_rising = _rising_control_structure("math")
    code_rising = _rising_control_structure("code")
    assert math_rising["Control"] and code_rising["Control"] and code_rising["Structure"]
    v6 = _pools(load_config(_path(6, "c01", 4, False)))
    v7 = _pools(load_config(_path(7, "c01", 4, False)))
    v8 = _pools(load_config(_path(8, "c01", 4, False)))
    everything_math = math_rising["Control"] | math_rising["Structure"]
    assert everything_math <= v6[0] == v7[0] == v8[0]
    # V6 carries Code Control, so all of the Code C/S part is covered.
    assert code_rising["Control"] | code_rising["Structure"] <= v6[1]
    # V7/V8 have no Code Control by design; their whole Code Structure part is covered.
    assert code_rising["Structure"] <= v7[1]
    assert code_rising["Structure"] <= v8[1]
    assert not code_rising["Control"] <= v7[1]


def test_run_namespaces_are_unique_and_structure_only_aliases_match() -> None:
    fields = {
        "runtime.wandb_run_id": lambda c: c.runtime.wandb_run_id,
        "audit.output_dir": lambda c: c.audit.output_dir,
        "paper_eval.output_dir": lambda c: c.paper_eval.output_dir,
        "huggingface_checkpoint.path_prefix": lambda c: c.huggingface_checkpoint.path_prefix,
        "trainer.experiment_name": lambda c: c.trainer.experiment_name,
        "trainer.default_local_dir": lambda c: c.trainer.default_local_dir,
    }
    seen = {name: {} for name in fields}
    for version, run, gpus, structure_only in _all_profiles():
        config = load_config(_path(version, run, gpus, structure_only))
        alias = structure_only and version != 6
        if alias:
            regular = load_config(_path(version, run, gpus, False))
            assert asdict(config) == asdict(regular)
            continue
        run_id = config.runtime.wandb_run_id
        assert run_id.startswith(f"q1p7b{gpus}g-mc-v{version}-")
        assert "-current-" in run_id and f"-m05{run}-" in run_id
        assert ("-cstruct-" in run_id) == (structure_only and version == 6)
        for name, getter in fields.items():
            value = getter(config)
            assert run_id in value
            assert value not in seen[name], (name, value)
            seen[name][value] = run_id
    assert all(len(values) == 24 for values in seen.values())
