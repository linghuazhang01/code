"""Contracts for the Qwen3-1.7B math-only controls of the list-12 ignore / only runs (author 2026-11-01).

configs/token_selection/math/taxonomy/native_opd_toklog_8gpu_8c_b256{,_s1042,_s2042}.yaml: equal weight, same topology as the
ignore / only runs, three training seeds.
configs/token_selection/math/taxonomy/native_opd_list12_matched_{ignore,only}_toklog_8gpu_8c_b256.yaml: same weights as the
list-12 runs on a frequency-matched pool that keeps the 8 list-12 termination IDs
(analysis-output/token-importance-controls-20261101/build_matched.py).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mopd_verl.launch import build_command
from mopd_verl.settings import load_config

from test_qwen1p7b_math_code_native_toklog import IDENTITY_KEYS, PER_TOKEN_AUDIT, _changed
from test_qwen1p7b_native_list12_pool import SHA, TOPOLOGY

DIRECTORY = Path(__file__).resolve().parents[1] / "configs/token_selection/math/taxonomy"
MATCHED_SHA = "8d39e9364b12052253b66d3485a1dcf51975a4eeab408a6bf8f2ef0a44f2fe0e"
TERMINATION = {4226, 13023, 16688, 21806, 25521, 31094, 73877, 151645}
SEEDS = [(42, ""), (1042, "_s1042"), (2042, "_s2042")]
SEED_KEYS = {"data.seed", "rollout.seed", "trainer.seed"}
ARMS = [("ignore", "l12matchign", 0.0, 1.0), ("only", "l12matchonly", 1.0, 0.0)]


def _naive(suffix: str) -> Path:
    return DIRECTORY / f"native_opd_toklog_8gpu_8c_b256{suffix}.yaml"


def _matched(arm: str) -> Path:
    return DIRECTORY / f"native_opd_list12_matched_{arm}_toklog_8gpu_8c_b256.yaml"


def _pool(path: Path) -> list[int]:
    return load_config(path).audit.domain_control_token_ids["math"]


def _common(cfg) -> None:
    assert cfg.model.student_path.endswith("Qwen3-1.7B")
    assert cfg.actor.distill_mode == "chosen_token_policy_gradient"
    assert cfg.actor.topk_distill_enabled is False and cfg.actor.token_baseline_method == "none"
    assert cfg.worker_placement.separate_ref_policy is False
    assert cfg.worker_placement.actor_rollout.n_gpus_per_node == cfg.trainer.n_gpus_per_node == cfg.runtime.slurm_allocation_gpus == 8
    assert "actor_rollout_ref.ref.fsdp_config.fsdp_size=8" in cfg.extra_overrides
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == 256
    assert cfg.rollout.gpu_memory_utilization == 0.6
    assert cfg.trainer.total_training_steps == 60 and cfg.trainer.save_freq == 1 and cfg.huggingface_checkpoint.steps == (60,)
    for key in PER_TOKEN_AUDIT:
        assert getattr(cfg.audit, key) is True, key
    run = cfg.trainer.experiment_name
    assert cfg.runtime.wandb_run_id == run and len(run) <= 64
    assert cfg.huggingface_checkpoint.path_prefix == f"checkpoints/math/{run}"
    assert cfg.trainer.default_local_dir == f"checkpoints/MOPD/{run}"
    assert cfg.audit.output_dir == f"audit/{run}"


@pytest.mark.parametrize("seed,suffix", SEEDS)
def test_naive_seed_contract(seed: int, suffix: str) -> None:
    cfg = load_config(_naive(suffix))
    _common(cfg)
    assert cfg.trainer.experiment_name == f"q1p7b-math-opdnative-toklog-8g8c-b256-s{seed}"
    assert cfg.audit.control_token_loss_weighting_enabled is False
    assert cfg.data.seed == cfg.rollout.seed == cfg.trainer.seed == seed
    command = build_command(cfg)
    assert f"data.seed={seed}" in command
    assert f"actor_rollout_ref.rollout.seed={seed}" in command
    assert f"+trainer.seed={seed}" in command


@pytest.mark.parametrize("seed,suffix", SEEDS)
def test_naive_differs_from_3gpu_base_only_in_topology_seed_and_names(seed: int, suffix: str) -> None:
    allowed = set(IDENTITY_KEYS) | TOPOLOGY | SEED_KEYS
    changed = _changed(_naive(suffix), DIRECTORY / "native_opd_toklog_3gpu_3c_b255.yaml")
    assert changed <= allowed, sorted(changed - allowed)


@pytest.mark.parametrize("arm,tag,pool_w,other_w", ARMS)
def test_matched_contract(arm: str, tag: str, pool_w: float, other_w: float) -> None:
    cfg = load_config(_matched(arm))
    _common(cfg)
    assert cfg.trainer.experiment_name == f"q1p7b-math-opdnative-{tag}-toklog-8g8c-b256-s42"
    audit = cfg.audit
    assert audit.control_token_loss_weighting_enabled is True and audit.control_token_online_selection_enabled is False
    assert audit.control_token_normalize_per_domain is False
    assert (audit.control_token_loss_weight, audit.control_token_other_weight) == (pool_w, other_w)
    ids = audit.domain_control_token_ids["math"]
    assert ids == sorted(set(ids)) and len(ids) == 78
    assert hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest() == MATCHED_SHA


@pytest.mark.parametrize("arm", [arm for arm, *_ in ARMS])
def test_matched_shares_only_termination_ids_with_list12(arm: str) -> None:
    list12 = _pool(DIRECTORY / f"native_opd_list12_{arm}_toklog_8gpu_8c_b256.yaml")
    assert hashlib.sha256(",".join(map(str, list12)).encode()).hexdigest() == SHA["math"]
    assert set(_pool(_matched(arm))) & set(list12) == TERMINATION


@pytest.mark.parametrize("arm", [arm for arm, *_ in ARMS])
def test_matched_differs_from_list12_only_in_pool_and_names(arm: str) -> None:
    allowed = set(IDENTITY_KEYS) | {"audit.domain_control_token_ids.math"}
    changed = _changed(_matched(arm), DIRECTORY / f"native_opd_list12_{arm}_toklog_8gpu_8c_b256.yaml")
    assert changed <= allowed, sorted(changed - allowed)


def test_control_run_names_are_unique() -> None:
    paths = [_naive(suffix) for _, suffix in SEEDS] + [_matched(arm) for arm, *_ in ARMS]
    runs = [load_config(path).trainer.experiment_name for path in paths]
    assert len(set(runs)) == len(runs) == 5
