"""The occurrence-50 profiles differ from the best Math+Code profile only where intended."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from mopd_verl.launch import build_command
from mopd_verl.settings import load_config

TAXONOMY = Path("configs/token_selection/math_code/taxonomy")
BEST = TAXONOMY / "mopd_math_code_toploss_m05_c01_code_structure_fixed4_8gpu_8c.yaml"
OCC50_8G = TAXONOMY / "mopd_math_code_toploss_m05_c01_code_structure_occ50_fixed4_8gpu_8c.yaml"
NAME_8G = "q1p7b8g-mc-tl-m05c01-cstruct-occ50-f4-8c-b512-s60"
def _changed_keys(reference: dict, candidate: dict) -> dict:
    return {
        section: {key for key in reference[section] if reference[section][key] != candidate[section][key]}
        if isinstance(reference[section], dict)
        else None
        for section in reference
        if reference[section] != candidate[section]
    }


def _assert_naming(config: dict, name: str) -> None:
    assert config["runtime"]["wandb_run_id"] == name
    assert config["audit"]["output_dir"] == f"audit/{name}"
    assert config["paper_eval"]["output_dir"] == f"eval_outputs/paper_suite/{name}"
    assert config["huggingface_checkpoint"]["path_prefix"] == f"checkpoints/mopd/{name}"
    assert config["huggingface_checkpoint"]["private"] is False
    assert config["huggingface_checkpoint"]["steps"] == (60,)
    assert config["trainer"]["experiment_name"] == name
    assert config["trainer"]["default_local_dir"] == f"checkpoints/MOPD/{name}"


def test_8gpu_variant_raises_the_occurrence_gate() -> None:
    best, occ = asdict(load_config(BEST)), asdict(load_config(OCC50_8G))
    assert _changed_keys(best, occ) == {
        "runtime": {"wandb_run_id"},
        "audit": {"control_token_online_min_mean_occurrences_per_step", "output_dir"},
        "paper_eval": {"output_dir"},
        "huggingface_checkpoint": {"path_prefix"},
        "trainer": {"experiment_name", "default_local_dir"},
    }
    assert best["rollout"]["gpu_memory_utilization"] == 0.6
    assert occ["rollout"]["gpu_memory_utilization"] == 0.6
    assert best["audit"]["control_token_online_min_mean_occurrences_per_step"] == 20.0
    assert occ["audit"]["control_token_online_min_mean_occurrences_per_step"] == 50.0
    assert occ["audit"]["control_token_online_strict_occurrence_gate"] is True
    assert occ["audit"]["control_token_online_window_steps"] == 1
    _assert_naming(occ, NAME_8G)


def test_pools_selector_and_training_match_the_best_profile() -> None:
    best, occ = load_config(BEST), load_config(OCC50_8G)
    assert occ.audit.domain_control_token_candidate_ids == best.audit.domain_control_token_candidate_ids
    assert len(occ.audit.domain_control_token_candidate_ids["math"]) == 390
    assert len(occ.audit.domain_control_token_candidate_ids["code"]) == 551
    assert occ.audit.control_token_online_top_p_by_domain == {"math": 0.05, "code": 0.01}
    assert occ.audit.control_token_online_selection_mode == "top_loss"
    assert occ.audit.control_token_online_selection_timing == "next_step"
    assert occ.audit.control_token_loss_weight == 4.0 and occ.audit.control_token_normalize_per_domain
    assert occ.data.train_batch_size == occ.actor.ppo_mini_batch_size == 512
    assert occ.trainer.total_training_steps == 60
    assert occ.teacher_performance.enabled is True
    command = build_command(occ)
    assert "+mopd_audit.control_token_online_min_mean_occurrences_per_step=50.0" in command
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.6" in command
    assert f"trainer.experiment_name={NAME_8G}" in command


