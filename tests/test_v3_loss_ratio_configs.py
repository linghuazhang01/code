from __future__ import annotations

import hashlib
from dataclasses import asdict
from operator import attrgetter
from pathlib import Path

import pytest
import yaml

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


CONFIG_DIR = (
    Path(__file__).resolve().parents[1] / "configs" / "token_selection" / "math"
)
BASE_FILENAME = "a_next_step_expanded_pruned_v3_unified_i1_w1_k25_5gpu_4a1t_b256.yaml"
CASES = (
    ("0p01", 0.01),
    ("0p02", 0.02),
    ("0p05", 0.05),
    ("0p075", 0.075),
    ("0p1", 0.1),
)
EXPECTED_POOL_SHA256 = (
    "ab3b17d65320ef778dbe5c6f6f475012658735711c41314ced618f78b80a3bb0"
)


def _filename(slug: str) -> str:
    return (
        "top32kl_next_step_expanded_pruned_v3_unified_"
        f"topp{slug}_i1_w1_lossratio_5gpu_4a1t_b256.yaml"
    )


@pytest.mark.parametrize("slug,top_p", CASES)
def test_v3_loss_ratio_config_contract(slug: str, top_p: float) -> None:
    path = CONFIG_DIR / _filename(slug)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = load_config(path)
    base = load_config(CONFIG_DIR / BASE_FILENAME)
    run_id = f"q1p7b-math-top32kl-ns-v3-u-topp{slug}-lossratio-i1w1-5g4a1t-b256"
    overrides = {
        "audit": {
            "output_dir": f"audit/{run_id}",
            "control_token_online_selection_mode": "top_loss",
            "control_token_online_weight_mode": "loss_ratio",
            "control_token_loss_weight": 4.0,
            "control_token_online_budget_mode": "top_p",
            "control_token_online_top_p": top_p,
            "control_token_online_top_k_per_group": None,
            "control_token_adaptive_neighborhood_enabled": False,
        },
        "paper_eval": {"output_dir": f"eval_outputs/paper_suite/{run_id}"},
        "huggingface_checkpoint": {"path_prefix": f"checkpoints/math/{run_id}"},
        "trainer": {
            "experiment_name": run_id,
            "default_local_dir": f"checkpoints/MOPD/{run_id}",
        },
        "runtime": {"wandb_run_id": run_id},
    }
    assert raw == {"extends": BASE_FILENAME, **overrides}
    expected = asdict(base)
    for section, values in overrides.items():
        expected[section].update(values)
    assert asdict(config) == expected
    assert len(run_id) <= 64

    audit = config.audit
    candidates = audit.domain_control_token_candidate_ids
    assert candidates.keys() == {"math"}
    candidate_ids = candidates["math"]
    assert len(candidate_ids) == 115
    assert candidate_ids == sorted(set(candidate_ids))
    pool_bytes = ",".join(str(token_id) for token_id in candidate_ids).encode()
    assert hashlib.sha256(pool_bytes).hexdigest() == EXPECTED_POOL_SHA256
    assert audit.domain_control_token_candidate_groups == {}
    assert audit.execution_timing == "pre_update"
    assert audit.control_token_online_selection_enabled
    assert audit.control_token_loss_weighting_enabled
    assert audit.control_token_normalize_per_domain
    assert audit.control_token_online_audit_interval_steps == 1
    assert audit.control_token_online_window_steps == 1
    assert audit.control_token_online_strict_occurrence_gate
    assert audit.control_token_online_min_mean_occurrences_per_step == 20.0
    assert audit.control_token_online_top_k == 25  # Inactive for Top-P budgets.

    assert config.actor.distill_loss_builder == "topk_kl"
    assert config.actor.distill_mode == "topk_renormalized_reverse_kl"
    assert config.actor.topk_distill_k == 32
    assert config.actor.topk_distill_support_source == "teacher"
    assert config.actor.ppo_mini_batch_size == 256
    assert config.data.train_batch_size == 256
    assert config.data.seed == config.rollout.seed == config.trainer.seed == 42
    assert config.runtime.slurm_allocation_gpus == 5
    assert config.worker_placement.separate_ref_policy
    assert config.worker_placement.actor_rollout.n_gpus_per_node == 4
    assert config.worker_placement.ref_policy.n_gpus_per_node == 1
    assert config.trainer.n_gpus_per_node == 4
    assert config.runtime.wandb_resume == "never"
    assert "trainer.resume_mode=disable" in config.extra_overrides

    command = format_command(build_command(config))
    for key, value in (
        ("control_token_online_selection_mode", "top_loss"),
        ("control_token_online_weight_mode", "loss_ratio"),
        ("control_token_online_budget_mode", "top_p"),
        ("control_token_online_top_p", top_p),
        ("control_token_loss_weight", 4.0),
        ("control_token_adaptive_neighborhood_enabled", "false"),
    ):
        assert f"+mopd_audit.{key}={value}" in command


def test_v3_loss_ratio_paths_are_isolated_from_existing_profiles() -> None:
    paths = {CONFIG_DIR / _filename(slug) for slug, _ in CASES}
    configs = [load_config(path) for path in sorted(paths)]
    existing_paths = (
        set(CONFIG_DIR.glob("*expanded_pruned_v3*.yaml"))
        | set(CONFIG_DIR.glob("*full_taxonomy*lossratio*.yaml"))
    ) - paths
    existing = [load_config(path) for path in sorted(existing_paths)]

    for field in (
        "runtime.wandb_run_id",
        "trainer.experiment_name",
        "audit.output_dir",
        "paper_eval.output_dir",
        "huggingface_checkpoint.path_prefix",
        "trainer.default_local_dir",
    ):
        get_value = attrgetter(field)
        values = [get_value(config) for config in configs]
        existing_values = {get_value(config) for config in existing}
        assert len(values) == len(set(values)) == len(CASES), field
        assert set(values).isdisjoint(existing_values), field
