"""Exact config-delta and frozen-reference calibration contracts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import pytest

from mopd_verl.launch import build_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "token_selection" / "math"
PARENT = (
    "top32kl_next_step_full_taxonomy_split_topp0p05_i1_w1_lossratio_5gpu_4a1t_b256.yaml"
)
SCALED = (
    "top32kl_next_step_full_taxonomy_split_topp0p05_i1_w1_"
    "lossratio_alpha1p75_5gpu_4a1t_b256.yaml"
)
ALPHA = 1.75
REFERENCE_RUN = "q1p7b-math-top32kl-topp05-lossratio-5g4a1t-b256"
NEW_RUN = "q1p7b-math-top32kl-topp05-lossratio-a1p75-5g4a1t-b256"


def test_scaled_config_changes_only_alpha_and_output_identities() -> None:
    parent = load_config(CONFIG_DIR / PARENT)
    scaled = load_config(CONFIG_DIR / SCALED)
    expected = asdict(parent)
    overrides = {
        "audit": {
            "control_token_loss_ratio_alpha": ALPHA,
            "output_dir": f"audit/{NEW_RUN}",
        },
        "runtime": {"wandb_run_id": NEW_RUN},
        "paper_eval": {"output_dir": f"eval_outputs/paper_suite/{NEW_RUN}"},
        "huggingface_checkpoint": {"path_prefix": f"checkpoints/math/{NEW_RUN}"},
        "trainer": {
            "experiment_name": NEW_RUN,
            "default_local_dir": f"checkpoints/MOPD/{NEW_RUN}",
        },
    }
    for section, values in overrides.items():
        expected[section].update(values)
    assert asdict(scaled) == expected
    assert parent.audit.control_token_loss_ratio_alpha == 1.0
    assert parent.runtime.wandb_run_id == REFERENCE_RUN
    assert len(NEW_RUN) <= 64
    assert scaled.runtime.slurm_allocation_gpus == 5
    assert scaled.worker_placement.actor_rollout.n_gpus_per_node == 4
    assert scaled.worker_placement.ref_policy.n_gpus_per_node == 1
    assert scaled.audit.control_token_online_top_p == 0.05
    assert scaled.audit.control_token_online_weight_mode == "loss_ratio"
    assert scaled.audit.control_token_online_selection_mode == "top_loss"
    assert scaled.audit.control_token_loss_weight == 4.0
    command = build_command(scaled)
    assert f"+mopd_audit.control_token_loss_ratio_alpha={ALPHA}" in command
    assert scaled.data.train_batch_size == scaled.actor.ppo_mini_batch_size == 256


def test_alpha_matches_frozen_reference_through_applied_step_60() -> None:
    path = ROOT / "analysis-output/taxonomy-lossratio-diagnosis-20260905/evidence.json"
    if not path.is_file():
        pytest.skip("Frozen experiment archive is not included in this checkout.")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    runs = [run for run in evidence["runs"] if run["run_id"] == REFERENCE_RUN]
    assert len(runs) == 1
    rows = sorted(
        [row for row in runs[0]["rows"] if 1 <= row["training/global_step"] < 60],
        key=lambda row: row["training/global_step"],
    )
    assert [row["training/global_step"] for row in rows] == list(range(1, 60))
    weights = [row["math/token_weight/loss_ratio_selected_raw_weight"] for row in rows]
    assert all(math.isfinite(weight) and 1.0 < weight < 4.0 for weight in weights)
    calibrated_alpha = 1.510087729609387
    assert calibrated_alpha == pytest.approx(4.0 / mean(weights), rel=1e-14)
    assert mean([ALPHA * weight for weight in weights]) == pytest.approx(
        4.635492271571985
    )
    assert max([ALPHA * weight for weight in weights]) > 5.0


@pytest.mark.parametrize("gpus,actors,batch", [(6, 4, 256), (7, 5, 255), (8, 6, 258)])
def test_scaled_resource_overlays_change_only_topology_and_identities(
    gpus: int, actors: int, batch: int
) -> None:
    base = load_config(CONFIG_DIR / SCALED)
    name = (
        "top32kl_next_step_full_taxonomy_split_topp0p05_i1_w1_"
        f"lossratio_alpha1p75_{gpus}gpu_{actors}a2t_b{batch}.yaml"
    )
    config = load_config(CONFIG_DIR / name)
    run = f"q1p7b-math-top32kl-topp05-lossratio-a1p75-{gpus}g{actors}a2t-b{batch}"
    expected = asdict(base)
    expected["runtime"].update(slurm_allocation_gpus=gpus, wandb_run_id=run)
    expected["data"]["train_batch_size"] = batch
    expected["actor"]["ppo_mini_batch_size"] = batch
    expected["worker_placement"]["actor_rollout"]["n_gpus_per_node"] = actors
    expected["worker_placement"]["ref_policy"]["n_gpus_per_node"] = 2
    expected["trainer"].update(
        n_gpus_per_node=actors,
        experiment_name=run,
        default_local_dir=f"checkpoints/MOPD/{run}",
    )
    expected["audit"]["output_dir"] = f"audit/{run}"
    expected["paper_eval"]["output_dir"] = f"eval_outputs/paper_suite/{run}"
    expected["huggingface_checkpoint"]["path_prefix"] = f"checkpoints/math/{run}"
    expected["extra_overrides"].append("actor_rollout_ref.ref.fsdp_config.fsdp_size=2")
    assert asdict(config) == expected
    assert batch % actors == 0 and gpus == actors + 2 and len(run) <= 64
    command = build_command(config)
    assert "+mopd_audit.control_token_loss_ratio_alpha=1.75" in command
    assert command.count("actor_rollout_ref.ref.fsdp_config.fsdp_size=2") == 1
