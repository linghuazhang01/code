from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    ROOT
    / "configs"
    / "mopd_qwen0p6b_30b_a3b_instruct_2507_4gpu_math_code_science_"
    "topk32_baseline_b528.yaml"
)
BASELINE_PATH = (
    ROOT
    / "configs"
    / "mopd_qwen0p6b_30b_a3b_instruct_2507_4gpu_math_code_wildsci_"
    "topk32_baseline_b528.yaml"
)
DYNAMIC_PATH = (
    ROOT
    / "configs"
    / "mopd_qwen0p6b_30b_a3b_instruct_2507_4gpu_math_code_wildsci_"
    "topk32_control_online_toploss_i3_w3_f20_k30_w4_b528.yaml"
)
EXPECTED_CANDIDATE_SHA256 = {
    "math": "6d7af9672dae50a9bd806a94083343b177ad795db592d0b820cdd5cd16d02c93",
    "code": "58414b18ddad6401877d5b495edce9c50128505dbff29c822c8ea7fccede6e5a",
    "science": "a196af65df6687e1c79a51dc7969b49636cae10cabaaf44d12e92153865bfb06",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _different_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[str] = set()
        for key in set(left) | set(right):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths.update(_different_paths(left[key], right[key], child))
        return paths
    return set() if left == right else {prefix}


def test_wildsci_baseline_only_changes_data_and_run_identity() -> None:
    source = _load_yaml(SOURCE_PATH)
    baseline = _load_yaml(BASELINE_PATH)

    assert _different_paths(source, baseline) == {
        "runtime.wandb_run_id",
        "data.domain_train_files.science",
        "rollout.gpu_memory_utilization",
        "audit.output_dir",
        "paper_eval.output_dir",
        "trainer.experiment_name",
        "trainer.default_local_dir",
    }


def test_wildsci_profiles_preserve_running_four_gpu_contract() -> None:
    for path in (BASELINE_PATH, DYNAMIC_PATH):
        config = load_config(path)
        assert config.runtime.slurm_allocation_gpus == 4
        assert config.worker_placement.actor_rollout.n_gpus_per_node == 1
        assert config.worker_placement.ref_policy.n_gpus_per_node == 3
        assert config.trainer.n_gpus_per_node == 1
        assert config.data.train_batch_size == 528
        assert config.actor.ppo_mini_batch_size == 528
        assert config.rollout.gpu_memory_utilization == 0.8
        assert config.data.domain_sampling_weights == {
            "math": 1.0,
            "code": 1.0,
            "science": 1.0,
        }
        assert config.data.domain_train_files["science"] == [
            "../mopd/code/data/G-OPD-Training-Data/WildSci/train.parquet"
        ]


def test_baseline_disables_all_reweighting() -> None:
    audit = load_config(BASELINE_PATH).audit

    assert not audit.control_token_loss_weighting_enabled
    assert not audit.control_token_online_selection_enabled
    assert not audit.control_token_speed_weighting_enabled
    assert not audit.dynamic_domain_loss_weighting_enabled
    assert not audit.all_domain_shared_token_loss_weighting_enabled


def test_dynamic_profile_enables_only_online_control_weighting() -> None:
    baseline = load_config(BASELINE_PATH)
    config = load_config(DYNAMIC_PATH)
    audit = config.audit

    assert config.model == baseline.model
    assert config.actor == baseline.actor
    assert config.rollout == baseline.rollout
    assert config.worker_placement == baseline.worker_placement
    assert config.data == baseline.data
    assert audit.loss_variance_signal == "opd_loss_token"
    assert audit.control_token_loss_weighting_enabled
    assert audit.control_token_online_selection_enabled
    assert audit.control_token_online_selection_mode == "top_loss"
    assert audit.control_token_normalize_per_domain
    assert audit.control_token_loss_weight == 4.0
    assert audit.control_token_online_audit_interval_steps == 3
    assert audit.control_token_online_window_steps == 3
    assert audit.control_token_online_min_mean_occurrences_per_step == 20.0
    assert audit.control_token_online_top_k == 30
    assert not audit.control_token_speed_weighting_enabled
    assert not audit.control_token_phase_gate_enabled
    assert not audit.control_token_span_weighting_enabled
    assert not audit.dynamic_domain_loss_weighting_enabled
    assert not audit.all_domain_shared_token_loss_weighting_enabled


def test_dynamic_profile_has_an_exact_change_allowlist() -> None:
    baseline = _load_yaml(BASELINE_PATH)
    dynamic = _load_yaml(DYNAMIC_PATH)

    assert _different_paths(baseline, dynamic) == {
        "runtime.wandb_run_id",
        "audit.loss_variance_signal",
        "audit.output_dir",
        "audit.control_token_loss_weighting_enabled",
        "audit.control_token_loss_weight",
        "audit.control_token_normalize_per_domain",
        "audit.control_token_online_selection_enabled",
        "audit.control_token_online_audit_interval_steps",
        "audit.control_token_online_window_steps",
        "audit.control_token_online_min_mean_occurrences_per_step",
        "audit.control_token_online_top_k",
        "audit.control_token_online_selection_mode",
        "audit.domain_control_token_candidate_ids.math",
        "audit.domain_control_token_candidate_ids.code",
        "audit.domain_control_token_candidate_ids.science",
        "paper_eval.output_dir",
        "trainer.experiment_name",
        "trainer.default_local_dir",
    }


def test_dynamic_profile_freezes_structural_codelex_candidates() -> None:
    audit = load_config(DYNAMIC_PATH).audit
    candidates = audit.domain_control_token_candidate_ids

    assert audit.control_token_ids == []
    assert audit.domain_control_token_ids == {}
    assert audit.control_token_candidate_ids == []
    assert {domain: len(ids) for domain, ids in candidates.items()} == {
        "math": 55,
        "code": 56,
        "science": 39,
    }
    for domain, token_ids in candidates.items():
        assert token_ids == sorted(set(token_ids))
        digest = hashlib.sha256(
            ",".join(str(token_id) for token_id in token_ids).encode("utf-8")
        ).hexdigest()
        assert digest == EXPECTED_CANDIDATE_SHA256[domain]


def test_dynamic_launcher_command_contains_online_selector_contract() -> None:
    rendered = format_command(build_command(load_config(DYNAMIC_PATH)))

    for fragment in (
        "+mopd_audit.control_token_loss_weighting_enabled=true",
        "+mopd_audit.control_token_online_selection_enabled=true",
        "+mopd_audit.control_token_online_audit_interval_steps=3",
        "+mopd_audit.control_token_online_window_steps=3",
        "+mopd_audit.control_token_online_min_mean_occurrences_per_step=20.0",
        "+mopd_audit.control_token_online_top_k=30",
        "+mopd_audit.control_token_online_selection_mode=top_loss",
        "+mopd_audit.domain_control_token_candidate_ids=",
    ):
        assert fragment in rendered
