"""V6 3-student / 1-dedicated-teacher layout keeps the 4s3t recipe and enables teacher batching."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from mopd_verl.launch import build_overrides
from mopd_verl.settings import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs/token_selection/math_code/taxonomy"
SHARED = CONFIG_DIR / "mopd_math_code_current_step_token_v6_toploss_m05_c01_fixed4_4student_3teacher_shared.yaml"
DEDICATED = CONFIG_DIR / "mopd_math_code_current_step_token_v6_toploss_m05_c01_fixed4_3student_1teacher.yaml"
RUN_ID = "q1p7b3s1t-mc-v6-cs-current-m05c01-f4-b528-s60"


def test_three_student_one_teacher_placement() -> None:
    config = load_config(DEDICATED)
    overrides = build_overrides(config)

    assert config.runtime.cuda_visible_devices == "0,1,2,4"
    assert config.worker_placement.separate_ref_policy
    assert not config.worker_placement.share_ref_policy_gpus
    assert config.worker_placement.actor_rollout.n_gpus_per_node == 3
    assert config.worker_placement.ref_policy.n_gpus_per_node == 1
    assert config.worker_placement.ref_policy.gpu_ids is None
    assert config.trainer.n_gpus_per_node == 3
    assert config.data.train_batch_size % 3 == 0
    assert config.teacher_performance.enabled
    assert config.teacher_performance.moe_dispatch == "stable_sort"
    assert "actor_rollout_ref.ref.fsdp_config.fsdp_size=1" in overrides
    assert "+actor_rollout_ref.ref.teacher_performance.enabled=true" in overrides
    assert config.runtime.wandb_run_id == RUN_ID
    assert config.trainer.experiment_name == RUN_ID
    assert config.trainer.default_local_dir == f"checkpoints/MOPD/{RUN_ID}"
    assert config.audit.output_dir == f"audit/{RUN_ID}"
    assert config.trainer.total_training_steps <= 65
    assert not config.huggingface_checkpoint.private


def test_three_student_one_teacher_keeps_shared_recipe() -> None:
    shared, dedicated = asdict(load_config(SHARED)), asdict(load_config(DEDICATED))
    for section in ("data", "model", "actor", "rollout", "teacher_performance", "algorithm"):
        if section in shared:
            assert dedicated[section] == shared[section], section
    excluded = {"output_dir", "experiment_name", "default_local_dir", "n_gpus_per_node"}
    for section in ("audit", "trainer"):
        for key, value in shared[section].items():
            if key not in excluded:
                assert dedicated[section][key] == value, f"{section}.{key}"
