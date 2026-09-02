from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pytest

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "token_selection" / "math"
POOL_MEMBERSHIP = (
    ROOT
    / "analysis-output"
    / "four-baseline-global-token-taxonomy"
    / "tables"
    / "candidate-pool-membership.csv"
)
DOMAIN_SUBSETS = (
    ROOT
    / "analysis-output"
    / "four-baseline-global-token-taxonomy"
    / "tables"
    / "domain-subsets.csv"
)
MATH_VALIDATION_FILES = [
    "../mopd/code/data/eval_data/math/AIME24/test.parquet",
    "../mopd/code/data/eval_data/math/AIME25/test.parquet",
    "../mopd/code/data/eval_data/math/HMMT25Feb/test.parquet",
    "../mopd/code/data/eval_data/math/HMMT25Nov/test.parquet",
]


@dataclass(frozen=True)
class SelectorCase:
    filename: str
    pool: str
    interval: int
    window: int
    top_k: int
    candidate_count: int


@dataclass(frozen=True)
class TaxonomyTopPCase:
    filename: str
    top_p: float
    total_gpus: int
    actor_gpus: int
    batch_size: int


CASES = (
    SelectorCase(
        "a_next_step_expanded_pruned_v2_i1_w1_k29_4gpu_b255.yaml",
        "ExpandedPruned-V2",
        1,
        1,
        29,
        89,
    ),
    SelectorCase(
        "a_next_window_expanded_pruned_v2_i6_w6_k30_4gpu_b255.yaml",
        "ExpandedPruned-V2",
        6,
        6,
        30,
        89,
    ),
    SelectorCase(
        "a_next_step_robust190_i1_w1_k27_4gpu_b255.yaml",
        "Robust190",
        1,
        1,
        27,
        66,
    ),
    SelectorCase(
        "a_next_window_robust190_i6_w6_k30_4gpu_b255.yaml",
        "Robust190",
        6,
        6,
        30,
        66,
    ),
    SelectorCase(
        "a_next_step_control44_i1_w1_k8_4gpu_b255.yaml",
        "Control-44",
        1,
        1,
        8,
        40,
    ),
    SelectorCase(
        "a_next_window_control44_i7_w7_k8_4gpu_b255.yaml",
        "Control-44",
        7,
        7,
        8,
        40,
    ),
)


TAXONOMY_TOP_P_CASES = (
    TaxonomyTopPCase(
        "top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_6gpu_4a2t_b256.yaml",
        0.1,
        6,
        4,
        256,
    ),
    TaxonomyTopPCase(
        "top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_7gpu_5a2t_b255.yaml",
        0.1,
        7,
        5,
        255,
    ),
    TaxonomyTopPCase(
        "top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_8gpu_6a2t_b258.yaml",
        0.1,
        8,
        6,
        258,
    ),
    TaxonomyTopPCase(
        "top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_6gpu_4a2t_b256.yaml",
        0.2,
        6,
        4,
        256,
    ),
    TaxonomyTopPCase(
        "top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_7gpu_5a2t_b255.yaml",
        0.2,
        7,
        5,
        255,
    ),
    TaxonomyTopPCase(
        "top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_8gpu_6a2t_b258.yaml",
        0.2,
        8,
        6,
        258,
    ),
)


def test_all_math_token_selector_configs_use_periodic_validation() -> None:
    config_paths = sorted(
        path
        for path in CONFIG_DIR.glob("*.yaml")
        if not path.name.startswith("_")
    )

    assert config_paths
    for path in config_paths:
        config = load_config(path)

        assert config.data.val_files == MATH_VALIDATION_FILES, path.name
        assert config.trainer.test_freq == 20, path.name
        assert not config.trainer.val_before_train, path.name


def _effective_math_pool(pool: str) -> list[int]:
    with POOL_MEMBERSHIP.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return sorted(
            {
                int(row["token_id"])
                for row in rows
                if row["pool"] == pool
                and row["domain"] == "math"
                and row["is_effective"] == "True"
            }
        )


def _math_taxonomy(token_type: str) -> list[int]:
    with DOMAIN_SUBSETS.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return sorted(
            {
                int(row["token_id"])
                for row in rows
                if row["domain"] == "math" and row["token_type"] == token_type
            }
        )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.filename)
def test_math_token_selector_config_contract(case: SelectorCase) -> None:
    config = load_config(CONFIG_DIR / case.filename)
    command = format_command(build_command(config))
    candidates = config.audit.domain_control_token_candidate_ids

    assert config.data.domain_train_files.keys() == {"math"}
    assert config.data.domain_sampling_weights == {"math": 1.0}
    assert config.data.train_batch_size == 255
    assert config.actor.ppo_mini_batch_size == 255
    assert config.runtime.slurm_allocation_gpus == 4
    assert config.worker_placement.separate_ref_policy
    assert config.worker_placement.actor_rollout.n_gpus_per_node == 3
    assert config.worker_placement.ref_policy.n_gpus_per_node == 1
    assert config.trainer.n_gpus_per_node == 3

    assert config.actor.distill_loss_builder == "topk_kl"
    assert config.actor.topk_distill_enabled
    assert config.actor.topk_distill_k == 32
    assert config.audit.control_token_online_selection_enabled
    assert config.audit.control_token_online_selection_mode == "top_logp_diff"
    assert config.audit.control_token_online_strict_occurrence_gate
    assert config.audit.control_token_online_min_mean_occurrences_per_step == 20.0
    assert config.audit.control_token_online_audit_interval_steps == case.interval
    assert config.audit.control_token_online_window_steps == case.window
    assert config.audit.control_token_online_top_k == case.top_k
    assert len(candidates["math"]) == case.candidate_count
    assert candidates["math"] == _effective_math_pool(case.pool)

    assert "+mopd_audit.control_token_online_selection_mode=top_logp_diff" in command
    assert "+mopd_audit.control_token_online_strict_occurrence_gate=true" in command


def test_math_token_selector_supports_top_p_budget(tmp_path: Path) -> None:
    base = CONFIG_DIR / "a_next_step_control44_i1_w1_k8_4gpu_b255.yaml"
    config_path = tmp_path / "top_p_selector.yaml"
    config_path.write_text(
        "\n".join(
            (
                f"extends: {base}",
                "audit:",
                "  control_token_online_budget_mode: top_p",
                "  control_token_online_top_p: 0.8",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)
    command = format_command(build_command(config))

    assert config.audit.control_token_online_budget_mode == "top_p"
    assert config.audit.control_token_online_top_p == 0.8
    assert "+mopd_audit.control_token_online_budget_mode=top_p" in command
    assert "+mopd_audit.control_token_online_top_p=0.8" in command


@pytest.mark.parametrize(
    (
        "filename",
        "interval",
        "window",
        "top_k",
        "top_k_per_group",
        "selection_mode",
    ),
    (
        (
            "a_next_step_full_taxonomy_split_i3_w1_k21_per_type_4gpu_b255.yaml",
            3,
            1,
            42,
            21,
            "top_logp_diff",
        ),
        (
            "a_next_window_full_taxonomy_split_i6_w6_k32_per_type_4gpu_b255.yaml",
            6,
            6,
            64,
            32,
            "top_logp_diff",
        ),
        (
            "top32kl_next_step_full_taxonomy_split_i1_w1_k21_per_type_4gpu_b255.yaml",
            1,
            1,
            42,
            21,
            "top_loss",
        ),
    ),
)
def test_math_full_taxonomy_selector_config_contract(
    filename: str,
    interval: int,
    window: int,
    top_k: int,
    top_k_per_group: int,
    selection_mode: str,
) -> None:
    config = load_config(CONFIG_DIR / filename)
    command = format_command(build_command(config))
    groups = config.audit.domain_control_token_candidate_groups

    assert config.data.domain_train_files.keys() == {"math"}
    assert config.data.train_batch_size == 255
    assert config.runtime.slurm_allocation_gpus == 4
    assert config.audit.control_token_online_selection_enabled
    assert config.audit.control_token_online_selection_mode == selection_mode
    assert config.audit.control_token_online_strict_occurrence_gate
    assert config.audit.control_token_online_audit_interval_steps == interval
    assert config.audit.control_token_online_window_steps == window
    assert config.audit.control_token_online_top_k == top_k
    assert config.audit.control_token_online_top_k_per_group == top_k_per_group
    assert groups["math"]["Control"] == _math_taxonomy("Control")
    assert groups["math"]["Structure"] == _math_taxonomy("Structure")
    assert len(groups["math"]["Control"]) == 124
    assert len(groups["math"]["Structure"]) == 266

    assert "+mopd_audit.domain_control_token_candidate_groups=" in command
    assert (
        f"+mopd_audit.control_token_online_selection_mode={selection_mode}"
        in command
    )
    assert (
        f"+mopd_audit.control_token_online_top_k_per_group={top_k_per_group}"
        in command
    )


def test_math_full_taxonomy_top_p_five_gpu_contract() -> None:
    config = load_config(
        CONFIG_DIR
        / "top32kl_next_step_full_taxonomy_split_topp0p05_i1_w1_5gpu_b256.yaml"
    )
    command = format_command(build_command(config))
    groups = config.audit.domain_control_token_candidate_groups

    assert config.data.train_batch_size == 256
    assert config.actor.ppo_mini_batch_size == 256
    assert config.runtime.slurm_allocation_gpus == 5
    assert config.worker_placement.actor_rollout.n_gpus_per_node == 4
    assert config.worker_placement.ref_policy.n_gpus_per_node == 1
    assert config.audit.control_token_online_budget_mode == "top_p"
    assert config.audit.control_token_online_top_p == 0.05
    assert config.audit.control_token_online_top_k_per_group is None
    assert len(groups["math"]["Control"]) == 124
    assert len(groups["math"]["Structure"]) == 266
    assert "+mopd_audit.control_token_online_budget_mode=top_p" in command
    assert "+mopd_audit.control_token_online_top_p=0.05" in command


@pytest.mark.parametrize(
    "case",
    TAXONOMY_TOP_P_CASES,
    ids=lambda case: case.filename,
)
def test_math_full_taxonomy_top_p_dual_teacher_contract(
    case: TaxonomyTopPCase,
) -> None:
    config = load_config(CONFIG_DIR / case.filename)
    command = format_command(build_command(config))
    groups = config.audit.domain_control_token_candidate_groups

    assert config.data.domain_train_files.keys() == {"math"}
    assert config.data.train_batch_size == case.batch_size
    assert config.actor.ppo_mini_batch_size == case.batch_size
    assert case.batch_size % case.actor_gpus == 0
    assert config.runtime.slurm_allocation_gpus == case.total_gpus
    assert config.worker_placement.separate_ref_policy
    assert (
        config.worker_placement.actor_rollout.n_gpus_per_node
        == case.actor_gpus
    )
    assert config.worker_placement.ref_policy.n_gpus_per_node == 2
    assert config.trainer.n_gpus_per_node == case.actor_gpus

    assert config.audit.control_token_online_selection_mode == "top_loss"
    assert config.audit.control_token_online_audit_interval_steps == 1
    assert config.audit.control_token_online_window_steps == 1
    assert config.audit.control_token_online_budget_mode == "top_p"
    assert config.audit.control_token_online_top_p == case.top_p
    assert config.audit.control_token_online_top_k_per_group is None
    assert len(groups["math"]["Control"]) == 124
    assert len(groups["math"]["Structure"]) == 266

    assert (
        "+actor_rollout_ref.worker_placement.ref_policy.n_gpus_per_node=2"
        in command
    )
    assert "actor_rollout_ref.ref.fsdp_config.fsdp_size=2" in command
    assert "+mopd_audit.control_token_online_budget_mode=top_p" in command
    assert f"+mopd_audit.control_token_online_top_p={case.top_p}" in command
    assert "+mopd_audit.control_token_online_top_k_per_group=null" in command
