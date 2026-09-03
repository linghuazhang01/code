from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "token_selection" / "math"
EXPECTED_MATH_V3_POOL_SIZE = 115
EXPECTED_MATH_V3_POOL_SHA256 = (
    "ab3b17d65320ef778dbe5c6f6f475012658735711c41314ced618f78b80a3bb0"
)


@dataclass(frozen=True)
class V3SelectorCase:
    target: str
    interval: int
    window: int
    top_k: int
    total_gpus: int
    actor_gpus: int
    ref_gpus: int
    batch_size: int

    @property
    def filename(self) -> str:
        target_name = "next_step" if self.target == "ns" else "next_window"
        return (
            f"a_{target_name}_expanded_pruned_v3_unified_"
            f"i{self.interval}_w{self.window}_k{self.top_k}_"
            f"{self.total_gpus}gpu_{self.actor_gpus}a{self.ref_gpus}t_"
            f"b{self.batch_size}.yaml"
        )


TOPOLOGIES = (
    (4, 3, 1, 255),
    (5, 4, 1, 256),
    (6, 4, 2, 256),
    (7, 5, 2, 255),
    (8, 6, 2, 258),
)
CASES = tuple(
    V3SelectorCase(target, interval, window, top_k, *topology)
    for target, interval, window, top_k in (
        ("ns", 1, 1, 25),
        ("nw", 7, 7, 41),
    )
    for topology in TOPOLOGIES
)


def _pool_sha256(token_ids: list[int]) -> str:
    normalized = ",".join(str(token_id) for token_id in token_ids)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.filename)
def test_v3_math_selector_config_contract(case: V3SelectorCase) -> None:
    path = CONFIG_DIR / case.filename
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = load_config(path)
    command = format_command(build_command(config))
    candidates = config.audit.domain_control_token_candidate_ids
    candidate_ids = candidates["math"]

    assert "extends" not in raw
    assert candidates.keys() == {"math"}
    assert len(candidate_ids) == EXPECTED_MATH_V3_POOL_SIZE
    assert candidate_ids == sorted(set(candidate_ids))
    assert _pool_sha256(candidate_ids) == EXPECTED_MATH_V3_POOL_SHA256
    assert config.audit.domain_control_token_candidate_groups == {}
    assert config.audit.control_token_online_selection_enabled
    assert config.audit.control_token_online_selection_mode == "top_logp_diff"
    assert config.audit.control_token_online_weight_mode == "fixed"
    assert config.audit.control_token_online_budget_mode == "top_k"
    assert config.audit.control_token_online_top_k == case.top_k
    assert config.audit.control_token_online_top_k_per_group is None
    assert config.audit.control_token_online_top_p == 1.0
    assert config.audit.control_token_online_audit_interval_steps == case.interval
    assert config.audit.control_token_online_window_steps == case.window
    assert config.audit.control_token_online_strict_occurrence_gate
    assert config.audit.control_token_online_min_mean_occurrences_per_step == 20.0

    assert config.data.domain_train_files.keys() == {"math"}
    assert config.data.train_batch_size == case.batch_size
    assert config.actor.ppo_mini_batch_size == case.batch_size
    assert case.batch_size % case.actor_gpus == 0
    assert config.runtime.slurm_allocation_gpus == case.total_gpus
    assert case.actor_gpus + case.ref_gpus == case.total_gpus
    assert config.worker_placement.separate_ref_policy
    assert config.worker_placement.actor_rollout.n_gpus_per_node == case.actor_gpus
    assert config.worker_placement.ref_policy.n_gpus_per_node == case.ref_gpus
    assert config.trainer.n_gpus_per_node == case.actor_gpus
    assert config.runtime.wandb_run_id == config.trainer.experiment_name
    assert len(config.runtime.wandb_run_id or "") <= 64

    fsdp_override = "actor_rollout_ref.ref.fsdp_config.fsdp_size=2"
    assert (fsdp_override in config.extra_overrides) == (case.ref_gpus == 2)
    assert "+mopd_audit.control_token_online_budget_mode=top_k" in command
    assert f"+mopd_audit.control_token_online_top_k={case.top_k}" in command
    assert "+mopd_audit.domain_control_token_candidate_ids=" in command
    assert "custom_reward_function.name=compute_score_batched" in command
    assert "+custom_reward_function.reward_kwargs.max_workers=32" in command
    assert (
        "+custom_reward_function.reward_kwargs.batch_timeout_seconds=120.0"
        in command
    )
    assert "reward_model.reward_manager=batch" in command


def test_v3_math_selector_run_id_and_output_paths_are_unique() -> None:
    configs = [load_config(CONFIG_DIR / case.filename) for case in CASES]

    run_ids = [config.runtime.wandb_run_id for config in configs]
    audit_dirs = [config.audit.output_dir for config in configs]
    checkpoint_dirs = [config.trainer.default_local_dir for config in configs]
    assert len(run_ids) == len(set(run_ids)) == len(CASES)
    assert len(audit_dirs) == len(set(audit_dirs)) == len(CASES)
    assert len(checkpoint_dirs) == len(set(checkpoint_dirs)) == len(CASES)
