"""Six-GPU variants of the Math-only ablation runs only change the topology."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from mopd_verl.launch import build_command
from mopd_verl.settings import load_config


MATH = Path(__file__).resolve().parents[1] / "configs/token_selection/math"
FULL = MATH / "full_vocabulary"
TAXONOMY = MATH / "taxonomy"
REFERENCE_6GPU = (
    TAXONOMY / "top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_6gpu_4a2t_b256.yaml"
)
PAIRS = [
    (
        FULL
        / "mopd_qwen1p7b_30b_a3b_instruct_2507_6gpu_math_fullvocab_random_topp05_fixed4_4a2t_b256.yaml",
        FULL
        / "mopd_qwen1p7b_30b_a3b_instruct_2507_4gpu_math_fullvocab_random_topp05_fixed4_3a1t_b255.yaml",
    ),
    (
        TAXONOMY / "uniform_top32kl_no_weighting_6gpu_4a2t_b256.yaml",
        TAXONOMY / "uniform_top32kl_no_weighting_4gpu_3a1t_b255.yaml",
    ),
]


@pytest.mark.parametrize("six_gpu,four_gpu", PAIRS)
def test_six_gpu_variant_only_changes_topology_and_identities(
    six_gpu: Path, four_gpu: Path
) -> None:
    reference = asdict(load_config(REFERENCE_6GPU))
    base, cfg = load_config(four_gpu), load_config(six_gpu)
    run = cfg.runtime.wandb_run_id
    assert run != base.runtime.wandb_run_id and len(run) <= 64
    expected = asdict(base)
    expected["runtime"].update(slurm_allocation_gpus=6, wandb_run_id=run)
    expected["data"]["train_batch_size"] = 256
    expected["actor"]["ppo_mini_batch_size"] = 256
    expected["worker_placement"] = reference["worker_placement"]
    expected["extra_overrides"] = reference["extra_overrides"]
    expected["audit"]["output_dir"] = f"audit/{run}"
    expected["paper_eval"]["output_dir"] = f"eval_outputs/paper_suite/{run}"
    expected["huggingface_checkpoint"]["path_prefix"] = f"checkpoints/math/{run}"
    expected["trainer"].update(
        experiment_name=run,
        default_local_dir=f"checkpoints/MOPD/{run}",
        n_gpus_per_node=4,
    )
    assert asdict(cfg) == expected
    assert cfg.worker_placement.actor_rollout.n_gpus_per_node == 4
    assert cfg.worker_placement.ref_policy.n_gpus_per_node == 2
    assert cfg.trainer.save_freq == 1 and cfg.trainer.max_actor_ckpt_to_keep == 2
    assert cfg.teacher_performance.enabled
    command = build_command(cfg)
    assert "actor_rollout_ref.ref.fsdp_config.fsdp_size=2" in command
    assert "trainer.n_gpus_per_node=4" in command
    assert "trainer.total_training_steps=65" in command
    assert "trainer.huggingface_checkpoint.private=False" in command
