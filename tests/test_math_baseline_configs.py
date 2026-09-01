from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mopd_verl.config_profiles import load_raw_config
from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config
from mopd_verl.token_baselines import token_baseline_method

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "baselines" / "math"
QWEN4B_CONFIG_DIR = CONFIG_DIR / "qwen4b"
GPU_PROFILES = {
    4: (3, 255),
    5: (4, 256),
    6: (5, 255),
    7: (6, 258),
    8: (7, 259),
}
METHODS = ("opd", "fire_opd", "tip_topk32", "eopd")
MODEL_VARIANTS = {
    "qwen1p7b": (CONFIG_DIR, "../mopd/models/Qwen3-1.7B"),
    "qwen4b": (QWEN4B_CONFIG_DIR, "../mopd/models/Qwen3-4B"),
}
HF_CHECKPOINT_STEPS = (55, 60, 65, 70)
HF_REPO_ID = "icemoon28/opd-checkpoints"
HF_MODEL_SAVE_OVERRIDE = (
    "actor_rollout_ref.actor.checkpoint.save_contents="
    "[model,optimizer,extra,hf_model]"
)
LEGACY_MATH_CONFIGS = (
    "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math.yaml",
    "mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math.yaml",
)
TARGETED_MATH_CONFIGS = ("exopd_4gpu_b255.yaml",)
SCALE_SPECIFIC_CONFIG_FIELDS = {
    "audit.output_dir",
    "huggingface_checkpoint.path_prefix",
    "model.student_path",
    "paper_eval.output_dir",
    "trainer.default_local_dir",
    "trainer.experiment_name",
}
MATH_TRAIN_FILE = (
    "../mopd/code/data/G-OPD-Training-Data/DeepMath-103K/"
    "train_filtered_level6.parquet"
)
MATH_VAL_FILES = [
    "../mopd/code/data/eval_data/math/AIME24/test.parquet",
    "../mopd/code/data/eval_data/math/AIME25/test.parquet",
    "../mopd/code/data/eval_data/math/HMMT25Feb/test.parquet",
    "../mopd/code/data/eval_data/math/HMMT25Nov/test.parquet",
]


def _config_path(
    model_variant: str,
    method: str,
    gpu_count: int,
    batch_size: int,
) -> Path:
    config_dir, _ = MODEL_VARIANTS[model_variant]
    return config_dir / f"{method}_{gpu_count}gpu_b{batch_size}.yaml"


def _baseline_config_paths() -> list[Path]:
    paths: list[Path] = []
    for config_dir, _ in MODEL_VARIANTS.values():
        paths.extend(
            sorted(
                path
                for path in config_dir.glob("*.yaml")
                if not path.name.startswith("_")
            )
        )
    return paths


def _all_math_config_paths() -> list[Path]:
    paths = _baseline_config_paths()
    paths.extend(ROOT / "configs" / filename for filename in LEGACY_MATH_CONFIGS)
    return paths


def _config_differences(
    left: Any,
    right: Any,
    prefix: str = "",
) -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: set[str] = set()
        for key in left.keys() | right.keys():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.add(child_prefix)
                continue
            differences.update(
                _config_differences(left[key], right[key], child_prefix)
            )
        return differences

    return set() if left == right else {prefix}


@pytest.mark.parametrize("model_variant", MODEL_VARIANTS)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("gpu_count", "actor_gpus", "batch_size"),
    [
        (gpu_count, actor_gpus, batch_size)
        for gpu_count, (actor_gpus, batch_size) in GPU_PROFILES.items()
    ],
)
def test_math_baseline_resource_and_data_contract(
    model_variant: str,
    method: str,
    gpu_count: int,
    actor_gpus: int,
    batch_size: int,
) -> None:
    config = load_config(
        _config_path(model_variant, method, gpu_count, batch_size)
    )
    command = format_command(build_command(config))
    _, expected_student_path = MODEL_VARIANTS[model_variant]

    assert config.model.student_path == expected_student_path
    assert config.trainer.experiment_name.startswith(
        f"{model_variant}-30b-math-"
    )
    assert config.data.domain_train_files == {"math": [MATH_TRAIN_FILE]}
    assert config.data.domain_sampling_weights == {"math": 1.0}
    assert config.data.val_files == MATH_VAL_FILES
    assert config.audit.domains == ["math"]
    assert config.paper_eval.datasets == [
        "aime24",
        "aime25",
        "hmmt25_feb",
        "hmmt25_nov",
    ]

    assert config.runtime.slurm_allocation_gpus == gpu_count
    assert config.worker_placement.separate_ref_policy
    assert config.worker_placement.actor_rollout.n_gpus_per_node == actor_gpus
    assert config.worker_placement.ref_policy.n_gpus_per_node == 1
    assert config.trainer.n_gpus_per_node == actor_gpus
    assert config.rollout.tensor_model_parallel_size == 1
    assert config.actor.fsdp_size == 1

    assert config.data.train_batch_size == batch_size
    assert config.actor.ppo_mini_batch_size == batch_size
    assert batch_size % actor_gpus == 0
    assert abs(batch_size - 256) <= 3
    assert f"data.train_batch_size={batch_size}" in command
    assert config.trainer.total_training_steps == 70
    assert config.trainer.max_actor_ckpt_to_keep == 4

    huggingface = config.huggingface_checkpoint
    assert huggingface.enabled
    assert huggingface.steps == HF_CHECKPOINT_STEPS
    assert huggingface.repo_id == HF_REPO_ID
    assert not huggingface.private
    assert huggingface.token_env_var == "HF_TOKEN"
    assert huggingface.path_prefix == (
        f"checkpoints/math/{config.trainer.experiment_name}"
    )
    assert "trainer.huggingface_checkpoint.enabled=True" in command
    assert "trainer.huggingface_checkpoint.private=False" in command
    assert HF_MODEL_SAVE_OVERRIDE in command


@pytest.mark.parametrize("model_variant", MODEL_VARIANTS)
@pytest.mark.parametrize("gpu_count", GPU_PROFILES)
def test_math_baseline_method_contracts(
    model_variant: str,
    gpu_count: int,
) -> None:
    _, batch_size = GPU_PROFILES[gpu_count]
    configs = {
        method: load_config(
            _config_path(model_variant, method, gpu_count, batch_size)
        )
        for method in METHODS
    }

    opd = configs["opd"]
    assert opd.actor.distill_loss_builder == "policy_gradient"
    assert opd.actor.distill_mode == "chosen_token_policy_gradient"
    assert token_baseline_method(opd.actor) == "none"
    assert not opd.actor.topk_distill_enabled

    fire = configs["fire_opd"]
    assert fire.actor.distill_loss_builder == "policy_gradient"
    assert token_baseline_method(fire.actor) == "fire_opd"
    assert fire.actor.fire_opd_filter_trajectories
    assert fire.actor.fire_opd_trajectory_drop_ratio == 0.2
    assert fire.actor.loss_agg_mode == "seq-mean-token-mean"

    tip = configs["tip_topk32"]
    assert tip.actor.distill_loss_builder == "topk_kl"
    assert token_baseline_method(tip.actor) == "tip_topk32"
    assert tip.actor.token_baseline_retention_ratio == 0.5
    assert tip.actor.topk_distill_enabled
    assert tip.actor.topk_distill_k == 32

    eopd = configs["eopd"]
    assert eopd.actor.distill_loss_builder == "eopd"
    assert eopd.actor.eopd_entropy_threshold == 0.8
    assert eopd.actor.eopd_forward_kl_weight == 1.0
    assert eopd.actor.eopd_topk_k == 16
    assert eopd.rollout_correction.rollout_is is None


def test_math_exopd_four_gpu_contract() -> None:
    config = load_config(CONFIG_DIR / "exopd_4gpu_b255.yaml")
    command = format_command(build_command(config))

    assert config.data.domain_train_files == {"math": [MATH_TRAIN_FILE]}
    assert config.data.train_batch_size == 255
    assert config.actor.ppo_mini_batch_size == 255
    assert config.runtime.slurm_allocation_gpus == 4
    assert config.worker_placement.separate_ref_policy
    assert config.worker_placement.actor_rollout.n_gpus_per_node == 3
    assert config.worker_placement.ref_policy.n_gpus_per_node == 1
    assert config.trainer.n_gpus_per_node == 3
    assert config.rollout.tensor_model_parallel_size == 1
    assert config.model.gopd_reference_path == "../mopd/models/Qwen3-1.7B"
    assert config.actor.distill_loss_builder == "exopd"
    assert config.actor.distill_mode == "chosen_token_policy_gradient"
    assert config.actor.lambda_vals == 1.25
    assert not config.actor.topk_distill_enabled
    assert config.huggingface_checkpoint.steps == HF_CHECKPOINT_STEPS
    assert config.huggingface_checkpoint.path_prefix.endswith(
        "qwen1p7b-30b-math-exopd-lambda1p25-4gpu-b255"
    )
    assert "policy_loss.distill_loss_builder=exopd" in command
    assert "policy_loss.lambda_vals=1.25" in command


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("gpu_count", "batch_size"),
    [
        (gpu_count, batch_size)
        for gpu_count, (_, batch_size) in GPU_PROFILES.items()
    ],
)
def test_qwen4b_overlays_only_change_scale_specific_fields(
    method: str,
    gpu_count: int,
    batch_size: int,
) -> None:
    qwen1p7b = load_raw_config(
        _config_path("qwen1p7b", method, gpu_count, batch_size)
    )
    qwen4b = load_raw_config(
        _config_path("qwen4b", method, gpu_count, batch_size)
    )

    assert _config_differences(qwen1p7b, qwen4b) == SCALE_SPECIFIC_CONFIG_FIELDS


def test_all_math_run_names_and_output_dirs_are_unique() -> None:
    experiment_names: set[str] = set()
    checkpoint_dirs: set[str] = set()
    audit_dirs: set[str] = set()
    paper_eval_dirs: set[str] = set()
    huggingface_path_prefixes: set[str] = set()

    config_paths = _all_math_config_paths()
    expected_baselines = (
        len(MODEL_VARIANTS) * len(METHODS) * len(GPU_PROFILES)
        + len(TARGETED_MATH_CONFIGS)
    )
    assert len(config_paths) == expected_baselines + len(LEGACY_MATH_CONFIGS)

    for path in config_paths:
        config = load_config(path)
        experiment_names.add(config.trainer.experiment_name)
        checkpoint_dirs.add(config.trainer.default_local_dir)
        audit_dirs.add(config.audit.output_dir)
        paper_eval_dirs.add(config.paper_eval.output_dir)
        huggingface_path_prefixes.add(config.huggingface_checkpoint.path_prefix)

    assert len(experiment_names) == len(config_paths)
    assert len(checkpoint_dirs) == len(config_paths)
    assert len(audit_dirs) == len(config_paths)
    assert len(paper_eval_dirs) == len(config_paths)
    assert len(huggingface_path_prefixes) == len(config_paths)


@pytest.mark.parametrize(
    "filename",
    LEGACY_MATH_CONFIGS,
)
def test_legacy_math_configs_upload_selected_checkpoints(filename: str) -> None:
    config = load_config(ROOT / "configs" / filename)
    huggingface = config.huggingface_checkpoint

    assert set(config.data.domain_train_files) == {"math"}
    assert config.trainer.total_training_steps == 70
    assert config.trainer.max_actor_ckpt_to_keep == 4
    assert huggingface.enabled
    assert huggingface.steps == HF_CHECKPOINT_STEPS
    assert huggingface.repo_id == HF_REPO_ID
    assert not huggingface.private
    assert huggingface.token_env_var == "HF_TOKEN"
    assert huggingface.path_prefix == (
        f"checkpoints/math/{config.trainer.experiment_name}"
    )
    assert HF_MODEL_SAVE_OVERRIDE in config.extra_overrides
