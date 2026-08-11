from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mopd_verl.domain_sampling import allocate_domain_batch_counts
from mopd_verl.launch import build_overrides, main
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT
    / "configs"
    / (
        "mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_if_"
        "topk32_dynamic_budget.yaml"
    )
)
DOMAINS = {"math", "code", "science", "if"}


def test_dynamic_budget_profile_has_required_runtime_contracts() -> None:
    config = load_config(PROFILE)
    overrides = build_overrides(config)

    assert config.domain_budgeting.enabled
    assert not config.domain_budgeting.teacher_scores_calibrated
    assert config.domain_budgeting.gap_normalization_floor == pytest.approx(0.05)
    assert config.actor.loss_agg_mode == "seq-mean-token-mean"
    assert config.actor.ppo_epochs == 1
    assert config.actor.ppo_mini_batch_size == config.data.train_batch_size
    assert config.data.dataloader_num_workers == 0
    assert config.trainer.val_before_train
    assert config.trainer.test_freq > 0
    assert set(config.data.domain_train_files) == DOMAINS
    assert set(config.data.domain_sampling_weights) == DOMAINS
    assert set(config.model.domain_teacher_paths) == DOMAINS
    assert set(config.audit.domains) == DOMAINS
    assert set(config.domain_budgeting.domains) == DOMAINS
    assert set(config.domain_budgeting.teacher_scores) == DOMAINS
    assert set(config.domain_budgeting.validation_metric_keys) == DOMAINS
    assert config.data.domain_train_files["if"] == [
        "data/G-OPD-Training-Data/IF/train.parquet"
    ]
    assert "data/eval_data/if/IFBench/test.parquet" in config.data.val_files
    assert config.domain_budgeting.validation_metric_keys["if"] == [
        "val-core/m2rl_ifbench/reward/mean@1"
    ]
    assert allocate_domain_batch_counts(
        config.data.train_batch_size,
        config.data.domain_sampling_weights,
        domains=config.domain_budgeting.domains,
        min_samples_per_domain=config.domain_budgeting.min_samples_per_domain,
    ) == {domain: 126 for domain in config.domain_budgeting.domains}
    assert "+mopd_domain_budgeting.enabled=true" in overrides
    assert "+mopd_domain_budgeting.teacher_scores_calibrated=false" in overrides
    assert "+mopd_domain_budgeting.gap_normalization_floor=0.05" in overrides
    assert "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean" in overrides
    domain_files_override = next(
        item for item in overrides if item.startswith("+data.domain_train_files=")
    )
    teacher_paths_override = next(
        item
        for item in overrides
        if item.startswith("+actor_rollout_ref.ref.model.teacher_paths=")
    )
    assert "if: ['data/G-OPD-Training-Data/IF/train.parquet']" in domain_files_override
    assert "if: '../models/Qwen3-30B-A3B-Instruct-2507'" in teacher_paths_override


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("actor", "loss_agg_mode"), "token-mean", "seq-mean-token-mean"),
        (("data", "dataloader_num_workers"), 2, "dataloader_num_workers=0"),
        (
            ("audit", "dynamic_domain_loss_weighting_enabled"),
            True,
            "cannot be enabled together",
        ),
        (("rollout", "n"), 2, "rollout.n=1"),
        (
            ("actor", "teacher_prefix_enabled"),
            True,
            "does not support teacher-prefix",
        ),
        (
            ("actor", "topk_distill_loss_weight"),
            0.0,
            "positive top-k OPD objective",
        ),
    ],
)
def test_invalid_dynamic_budget_contracts_fail_fast(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    payload = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    payload[path[0]][path[1]] = value
    bad_profile = tmp_path / "bad.yaml"
    bad_profile.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(bad_profile)


def test_dynamic_budget_requires_every_domain_teacher_alias(tmp_path: Path) -> None:
    payload = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    payload["model"]["if_teacher_path"] = None
    del payload["model"]["domain_teacher_paths"]["if"]
    bad_profile = tmp_path / "missing-if-teacher.yaml"
    bad_profile.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must cover every dynamic domain"):
        load_config(bad_profile)


def test_unknown_dynamic_budget_key_fails_fast(tmp_path: Path) -> None:
    payload = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    payload["domain_budgeting"]["gap_normalizaton_floor"] = 0.1
    bad_profile = tmp_path / "unknown-budget-key.yaml"
    bad_profile.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="gap_normalizaton_floor"):
        load_config(bad_profile)


def test_placeholder_teacher_scores_allow_dry_run_but_block_training() -> None:
    assert main(["--config", str(PROFILE), "--dry-run"]) == 0
    with pytest.raises(ValueError, match="placeholder teacher scores"):
        main(["--config", str(PROFILE)])


def test_dynamic_resume_requires_controller_and_dataloader_state() -> None:
    trainer_source = (
        ROOT / "third_party" / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    ).read_text(encoding="utf-8")

    assert "cannot resume without dataloader state" in trainer_source
    assert "cannot resume without controller" in trainer_source


def test_runtime_rechecks_hydra_overridable_budget_contracts() -> None:
    trainer_source = (
        ROOT / "third_party" / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    ).read_text(encoding="utf-8")

    for contract in (
        "runtime requires rollout.n=1",
        "loss_agg_mode=seq-mean-token-mean",
        "data.dataloader_num_workers=0",
        "runtime requires val_before_train=true",
        "runtime is missing configured teacher domains",
        "supports only FSDP/FSDP2",
        "runtime requires a top-k OPD objective",
        "does not support teacher-prefix training",
        "sampler={sampler_batch_size}",
    ):
        assert contract in trainer_source
