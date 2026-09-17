"""Regression contracts for the four m05/c01/s01 mixed-KL variants."""

from pathlib import Path

import pytest
import yaml

from mopd_verl.config_profiles import load_raw_config
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config


@pytest.mark.parametrize("gpus", [4, 8])
@pytest.mark.parametrize("science", ["teacherconf", "toploss"])
def test_variant_contract(gpus: int, science: str) -> None:
    root = Path(__file__).resolve().parents[1]
    suffix = "_colocated" if gpus == 4 else ""
    name = (
        f"mopd_qwen1p7b_30b_{gpus}gpu_m05_c01_s01_sci_{science}"
        f"_cfkl_srkl_fixed4_b528{suffix}.yaml"
    )
    path = root / "configs/token_selection/math_code_science/taxonomy" / name
    raw = load_raw_config(path)
    assert raw["data"]["train_batch_size"] == 528
    assert raw["actor"]["ppo_mini_batch_size"] == 528
    assert raw["actor"]["topk_distill_k"] == 32
    assert raw["trainer"]["total_training_steps"] == 60
    assert raw["trainer"]["save_freq"] == 5
    assert raw["huggingface_checkpoint"]["steps"] == [60]
    assert raw["huggingface_checkpoint"]["private"] is False
    placement = raw["worker_placement"]
    assert placement["separate_ref_policy"] == (gpus == 8)
    assert placement["actor_rollout"]["n_gpus_per_node"] == (6 if gpus == 8 else 4)
    if gpus == 8:
        assert placement["ref_policy"]["n_gpus_per_node"] == 2
    audit = raw["audit"]
    assert audit["control_token_online_top_p_by_domain"] == {
        "math": .05, "code": .01, "science": .01,
    }
    expected = "top_loss_teacher_confidence" if science == "teacherconf" else "top_loss"
    assert audit["control_token_online_selection_mode_by_domain"] == {
        "math": "top_loss", "code": "top_loss", "science": expected,
    }
    routes = {"default": {"control": "forward", "structure": "reverse", "other": "reverse"}}
    assert raw["actor"]["topk_distill_loss_by_domain"] == routes
    command = build_command(load_config(path))
    argument = next(arg for arg in command if arg.startswith(
        "++actor_rollout_ref.actor.policy_loss.topk_distill_loss_by_domain="
    ))
    assert yaml.safe_load(argument.split("=", 1)[1]) == routes
