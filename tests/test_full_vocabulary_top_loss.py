"""Full-tokenizer-vocabulary Top-Loss selector coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.control_full_vocabulary import (
    global_full_vocabulary_loss_statistics,
)
from mopd_verl.domain_gradient.control_top_loss import (
    OnlineControlSelectionState,
    initial_online_control_selection_state,
    update_online_control_selection,
)
from mopd_verl.domain_gradient.control_top_loss_runtime import (
    global_candidate_loss_statistics_with_valid_counts,
)
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from mopd_verl.verl_audit import MOPDAuditLogger


CONFIG_PATH = Path(__file__).resolve().parents[1] / (
    "configs/token_selection/math_code_science/full_vocabulary/"
    "mopd_qwen1p7b_30b_a3b_instruct_2507_4gpu_math_code_science_"
    "fullvocab_toploss_topp05_fixed4_b528_colocated.yaml"
)
TAXONOMY_CONFIG_PATH = Path(__file__).resolve().parents[1] / (
    "configs/token_selection/math_code_science/taxonomy/"
    "mopd_qwen1p7b_30b_a3b_instruct_2507_4gpu_math_code_science_"
    "toploss_topp05_fixed4_b528_colocated.yaml"
)
MATH_CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "configs/token_selection/math/full_vocabulary"
)
MATH_CONFIG_FILENAMES = (
    "mopd_qwen1p7b_30b_a3b_instruct_2507_4gpu_math_fullvocab_"
    "toploss_topp05_i1_w1_fixed4_b256_colocated.yaml",
    "mopd_qwen1p7b_30b_a3b_instruct_2507_8gpu_math_fullvocab_"
    "toploss_topp05_i1_w1_fixed4_6a2t_b258.yaml",
)


def test_full_vocabulary_statistics_cover_all_valid_observed_token_ids() -> None:
    statistics = global_full_vocabulary_loss_statistics(
        (torch.tensor([[0, 7, 99], [7, 8, 127]], dtype=torch.long),),
        (torch.tensor([[-2.0, 3.0, 100.0], [4.0, -5.0, 6.0]]),),
        (torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),),
        (("math", "code"),),
        domains=("math", "code"),
        vocab_size=128,
    )

    assert statistics.by_domain == {
        "math": {0: (2.0, 1), 7: (3.0, 1)},
        "code": {7: (4.0, 1), 8: (5.0, 1), 127: (6.0, 1)},
    }
    assert statistics.valid_token_counts == {"math": 2, "code": 3}
    assert statistics.valid_score_sums == {"math": 5.0, "code": 15.0}


def test_full_vocabulary_statistics_reject_out_of_tokenizer_range() -> None:
    with pytest.raises(ValueError, match="token IDs in"):
        global_full_vocabulary_loss_statistics(
            (torch.tensor([[128]], dtype=torch.long),),
            (torch.tensor([[1.0]]),),
            (torch.tensor([[1]], dtype=torch.bool),),
            (("math",),),
            domains=("math",),
            vocab_size=128,
        )


def test_runtime_full_scope_ignores_configured_pool_and_keeps_token_zero() -> None:
    result = global_candidate_loss_statistics_with_valid_counts(
        (torch.tensor([[0, 7, 99]], dtype=torch.long),),
        (torch.tensor([[1.0, -2.0, 3.0]]),),
        (torch.ones(1, 3, dtype=torch.bool),),
        (("math",),),
        domains=("math",),
        domain_candidate_token_ids={"math": ()},
        candidate_scope="full_vocabulary",
        candidate_vocab_size=128,
    )

    assert result.by_domain["math"] == {
        0: (1.0, 1),
        7: (2.0, 1),
        99: (3.0, 1),
    }
    assert result.valid_token_counts == {"math": 3}


def test_top_p_five_percent_uses_full_valid_occurrence_denominator() -> None:
    state = initial_online_control_selection_state(
        ("math",),
        {"math": ()},
        audit_interval_steps=1,
        window_steps=1,
        min_mean_occurrences_per_step=20.0,
        strict_occurrence_gate=True,
        top_k=30,
        budget_mode="top_p",
        top_p=0.05,
        candidate_scope="full_vocabulary",
        candidate_vocab_size=128,
    )

    outcome, next_state = update_online_control_selection(
        state,
        {
            "math": {
                6: (2_000.0, 20),
                7: (105.0, 21),
                8: (84.0, 21),
            }
        },
        step=1,
        valid_token_counts={"math": 100},
    )

    result = outcome.domain_results[0]
    assert result.target_occurrence_count == 5
    assert result.eligible_token_count == 2
    assert result.selected_occurrence_count == 21
    assert result.selected_occurrence_fraction == pytest.approx(0.21)
    assert next_state.active_map() == {"math": (7,)}
    assert OnlineControlSelectionState.from_mapping(next_state.as_dict()) == next_state


def test_full_scope_rejects_configured_ids_and_paired_selector() -> None:
    common = {
        "audit_interval_steps": 1,
        "window_steps": 1,
        "min_mean_occurrences_per_step": 0.0,
        "top_k": 1,
        "candidate_scope": "full_vocabulary",
        "candidate_vocab_size": 128,
    }
    with pytest.raises(ValueError, match="cannot be combined"):
        initial_online_control_selection_state(("math",), (7,), **common)
    with pytest.raises(ValueError, match="only top_loss and top_speed"):
        initial_online_control_selection_state(
            ("math",),
            (),
            selection_mode="top_kl_student_entropy",
            **common,
        )


def test_full_scope_window_candidate_map_and_schema_eleven_migration() -> None:
    state = initial_online_control_selection_state(
        ("math",),
        {"math": ()},
        audit_interval_steps=2,
        window_steps=2,
        min_mean_occurrences_per_step=0.0,
        top_k=1,
        candidate_scope="full_vocabulary",
        candidate_vocab_size=128,
    )
    _, state = update_online_control_selection(
        state,
        {"math": {7: (1.0, 1)}},
        step=1,
    )
    _, state = update_online_control_selection(
        state,
        {"math": {8: (2.0, 1)}},
        step=2,
    )
    assert state.window_candidate_map() == {"math": (7, 8)}

    configured = initial_online_control_selection_state(
        ("math",),
        {"math": (7,)},
        audit_interval_steps=1,
        window_steps=1,
        min_mean_occurrences_per_step=0.0,
        top_k=1,
    ).as_dict()
    configured["schema_version"] = 11
    configured.pop("candidate_scope")
    configured.pop("candidate_vocab_size")
    migrated = OnlineControlSelectionState.from_mapping(configured)
    assert migrated.candidate_scope == "configured"
    assert migrated.candidate_vocab_size is None


def test_logger_infers_tokenizer_vocab_and_sidecar_accepts_empty_pool() -> None:
    class Tokenizer:
        def __len__(self) -> int:
            return 128

    logger = MOPDAuditLogger(
        {
            "mopd_audit": {
                "enabled": True,
                "domains": ["math"],
                "control_token_loss_weighting_enabled": True,
                "control_token_online_selection_enabled": True,
                "control_token_online_candidate_scope": "full_vocabulary",
            }
        },
        tokenizer=Tokenizer(),
    )
    meta = logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
    config = DomainGradientConfig.from_meta(meta)

    assert logger.control_token_online_candidate_vocab_size == 128
    assert logger.control_token_online_candidate_vocab_size_source == "tokenizer"
    assert config.control_token_online_candidate_scope == "full_vocabulary"
    assert config.control_token_online_candidate_vocab_size == 128
    assert config.effective_domain_candidate_map() == {"math": ()}


def test_full_vocabulary_experiment_profile_and_launcher_contract() -> None:
    config = load_config(CONFIG_PATH)
    command = build_command(config)

    assert config.audit.control_token_online_candidate_scope == "full_vocabulary"
    assert config.audit.control_token_online_candidate_vocab_size is None
    assert config.audit.control_token_online_top_p == pytest.approx(0.05)
    assert config.audit.control_token_online_strict_occurrence_gate
    assert config.audit.control_token_online_min_mean_occurrences_per_step == 20.0
    assert not config.audit.domain_control_token_candidate_groups
    assert "+mopd_audit.control_token_online_candidate_scope=full_vocabulary" in command
    assert "+mopd_audit.control_token_online_candidate_vocab_size=null" in command


@pytest.mark.parametrize(
    ("filename", "total_gpus", "actor_gpus", "teacher_gpus", "batch_size"),
    (
        (
            MATH_CONFIG_FILENAMES[0],
            4,
            4,
            None,
            256,
        ),
        (
            MATH_CONFIG_FILENAMES[1],
            8,
            6,
            2,
            258,
        ),
    ),
)
def test_math_only_full_vocabulary_profile_and_launcher_contract(
    filename: str,
    total_gpus: int,
    actor_gpus: int,
    teacher_gpus: int | None,
    batch_size: int,
) -> None:
    config = load_config(MATH_CONFIG_DIR / filename)
    command = build_command(config)

    assert config.audit.domains == ["math"]
    assert config.audit.control_token_online_candidate_scope == "full_vocabulary"
    assert config.audit.control_token_online_candidate_vocab_size is None
    assert config.audit.control_token_online_audit_interval_steps == 1
    assert config.audit.control_token_online_window_steps == 1
    assert config.audit.control_token_online_budget_mode == "top_p"
    assert config.audit.control_token_online_top_p == pytest.approx(0.05)
    assert config.audit.control_token_online_selection_mode == "top_loss"
    assert config.audit.control_token_online_weight_mode == "fixed"
    assert config.audit.control_token_loss_weight == pytest.approx(4.0)
    assert not config.audit.control_token_candidate_ids
    assert not config.audit.domain_control_token_candidate_ids
    assert not config.audit.domain_control_token_candidate_groups
    assert config.runtime.slurm_allocation_gpus == total_gpus
    assert config.worker_placement.actor_rollout.n_gpus_per_node == actor_gpus
    assert config.worker_placement.ref_policy.n_gpus_per_node == teacher_gpus
    assert config.data.train_batch_size == batch_size
    run_id = config.runtime.wandb_run_id
    assert run_id is not None
    assert config.audit.output_dir == f"audit/{run_id}"
    assert config.paper_eval.output_dir == f"eval_outputs/paper_suite/{run_id}"
    assert config.huggingface_checkpoint.path_prefix == (
        f"checkpoints/math/{run_id}"
    )
    assert config.trainer.experiment_name == run_id
    assert config.trainer.default_local_dir == f"checkpoints/MOPD/{run_id}"
    if teacher_gpus is None:
        assert not config.worker_placement.separate_ref_policy
        assert not config.teacher_performance.enabled
    else:
        assert config.worker_placement.separate_ref_policy
        assert config.teacher_performance.enabled
        assert "actor_rollout_ref.ref.fsdp_config.fsdp_size=2" in command
    assert "+mopd_audit.control_token_online_candidate_scope=full_vocabulary" in command
    assert "+mopd_audit.control_token_online_candidate_vocab_size=null" in command


def test_math_only_profiles_use_disjoint_output_identifiers() -> None:
    configs = [
        load_config(MATH_CONFIG_DIR / filename)
        for filename in MATH_CONFIG_FILENAMES
    ]
    identifier_groups = (
        [config.runtime.wandb_run_id for config in configs],
        [config.audit.output_dir for config in configs],
        [config.paper_eval.output_dir for config in configs],
        [config.huggingface_checkpoint.path_prefix for config in configs],
        [config.trainer.experiment_name for config in configs],
        [config.trainer.default_local_dir for config in configs],
    )

    for identifiers in identifier_groups:
        assert None not in identifiers
        assert len(set(identifiers)) == len(identifiers)


def test_experiment_matches_taxonomy_control_outside_candidate_universe() -> None:
    full = load_config(CONFIG_PATH)
    taxonomy = load_config(TAXONOMY_CONFIG_PATH)

    assert full.data == taxonomy.data
    assert full.actor == taxonomy.actor
    assert full.rollout == taxonomy.rollout
    assert full.rollout_correction == taxonomy.rollout_correction
    assert full.worker_placement == taxonomy.worker_placement
    assert full.teacher_performance == taxonomy.teacher_performance
    assert full.audit.control_token_online_top_p == (
        taxonomy.audit.control_token_online_top_p
    )
    assert full.audit.control_token_online_budget_mode == (
        taxonomy.audit.control_token_online_budget_mode
    )
    assert full.audit.control_token_online_selection_mode == (
        taxonomy.audit.control_token_online_selection_mode
    )
    assert full.audit.control_token_online_weight_mode == (
        taxonomy.audit.control_token_online_weight_mode
    )
    assert full.audit.control_token_loss_weight == (
        taxonomy.audit.control_token_loss_weight
    )
