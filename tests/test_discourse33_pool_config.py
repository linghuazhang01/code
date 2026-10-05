"""The discourse-33 profile differs from the best Math+Code profile only where intended."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from mopd_verl.verl_audit import MOPDAuditLogger

TAXONOMY = Path("configs/token_selection/math_code/taxonomy")
BEST = TAXONOMY / "mopd_math_code_toploss_m05_c01_code_structure_fixed4_8gpu_6s2t.yaml"
DISC33 = TAXONOMY / "mopd_math_code_toploss_m05_c01_code_structure_disc33_fixed4_3student_1teacher.yaml"
NAME = "q1p7b3s1t-mc-tl-m05c01-cstruct-disc33-f4-b528-s60"
# Reasoning-discourse words that pass the four-baseline occurrence rules (Math domain).
ADDED = {
    1246, 2087, 2704, 2797, 3151, 3170, 3330, 3555, 3654, 3681, 3838,
    4024, 4092, 4428, 4460, 4854, 5158, 5185, 6505, 7196, 8729, 9355,
    11416, 13824, 14037, 14068, 16688, 19198, 21390, 23085, 25538, 27699, 82610,
}


def test_only_the_math_pool_grows_by_the_33_words() -> None:
    best, disc = load_config(BEST), load_config(DISC33)
    best_pool = best.audit.domain_control_token_candidate_ids
    disc_pool = disc.audit.domain_control_token_candidate_ids
    assert len(ADDED) == 33 and not ADDED & set(best_pool["math"])
    assert set(disc_pool["math"]) == set(best_pool["math"]) | ADDED
    assert len(disc_pool["math"]) == 423
    assert list(disc_pool["code"]) == list(best_pool["code"])
    assert len(disc_pool["code"]) == 551


def test_selector_training_and_upload_settings_match_the_best_profile() -> None:
    best, disc = asdict(load_config(BEST)), asdict(load_config(DISC33))
    changed = {
        section: {key for key in best[section] if best[section][key] != disc[section][key]}
        if isinstance(best[section], dict)
        else None
        for section in best
        if best[section] != disc[section]
    }
    assert changed == {
        "worker_placement": {"actor_rollout", "ref_policy"},
        "audit": {"domain_control_token_candidate_ids", "output_dir"},
        "paper_eval": {"output_dir"},
        "huggingface_checkpoint": {"path_prefix"},
        "trainer": {"experiment_name", "n_gpus_per_node", "default_local_dir"},
        "runtime": {"wandb_run_id", "slurm_allocation_gpus", "cuda_visible_devices"},
        "extra_overrides": None,
    }
    assert disc["huggingface_checkpoint"]["private"] is False
    assert disc["huggingface_checkpoint"]["steps"] == (60,)
    assert disc["huggingface_checkpoint"]["path_prefix"] == f"checkpoints/mopd/{NAME}"
    trainer = disc["trainer"]
    assert trainer["experiment_name"] == NAME
    assert trainer["default_local_dir"] == f"checkpoints/MOPD/{NAME}"
    assert trainer["total_training_steps"] == 60 <= 65
    assert trainer["launch_reward_fn_async"] is True
    assert disc["data"]["train_batch_size"] == disc["actor"]["ppo_mini_batch_size"] == 528
    assert disc["audit"]["control_token_online_selection_timing"] == "next_step"
    assert disc["audit"]["control_token_online_top_p_by_domain"] == {"math": 0.05, "code": 0.01}
    assert disc["teacher_performance"]["enabled"] is True


def test_three_students_one_dedicated_teacher_on_physical_gpus() -> None:
    config = load_config(DISC33)
    placement = config.worker_placement
    assert placement.separate_ref_policy is True
    assert placement.actor_rollout.n_gpus_per_node == 3
    assert placement.ref_policy.n_gpus_per_node == 1
    assert config.trainer.n_gpus_per_node == 3
    assert config.runtime.cuda_visible_devices == "0,1,2,4"
    assert "actor_rollout_ref.ref.fsdp_config.fsdp_size=1" in config.extra_overrides
    assert 528 % 3 == 0
    command = build_command(config)
    assert f"trainer.experiment_name={NAME}" in command


def test_runtime_audit_config_accepts_the_expanded_pool(tmp_path: Path) -> None:
    config = load_config(DISC33)
    overrides = [arg for arg in build_command(config) if arg.startswith("+mopd_audit.")]
    runtime_audit = {
        key.removeprefix("+mopd_audit."): yaml.safe_load(value)
        for key, value in (arg.split("=", 1) for arg in overrides)
    }
    runtime_audit["output_dir"] = str(tmp_path / "audit")
    meta = MOPDAuditLogger({"mopd_audit": runtime_audit}).full_gradient_meta("train", 1)
    domain_config = DomainGradientConfig.from_meta(meta["mopd_full_gradient"])
    assert set(domain_config.effective_domain_candidate_map()["math"]) == set(
        config.audit.domain_control_token_candidate_ids["math"]
    )
