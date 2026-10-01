"""Lagged V4 Control selection retains position gates and avoids a prepass."""

from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml
import test_current_step_runtime as runtime

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.control_top_loss import (
    initial_online_control_selection_state,
    update_online_control_selection,
)
from mopd_verl.domain_gradient.occurrence_config import uses_versioned_next_step_selection
from mopd_verl.domain_gradient.structure_positions import attach_structure_position_mask
from mopd_verl.domain_gradient.versioned_next_step import next_step_candidate_statistics
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from mopd_verl.verl_audit import MOPDAuditLogger
from test_structure_positions import _Tokenizer
from test_next_step_config_migration import with_post_snapshot_defaults


CONFIG = Path("configs/token_selection/math_code/taxonomy/") / (
    "mopd_math_code_next_step_token_v4_toploss_m05_c01_fixed4_4gpu_colocated.yaml"
)


def _logger(tmp_path: Path, **audit_overrides: object) -> MOPDAuditLogger:
    config = load_config(CONFIG)
    return MOPDAuditLogger({"mopd_audit": {
        **asdict(config.audit), "output_dir": str(tmp_path), **audit_overrides,
    }})


def _domain(tmp_path: Path) -> DomainGradientConfig:
    return DomainGradientConfig.from_meta(
        _logger(tmp_path).full_gradient_meta("train", 1)["mopd_full_gradient"]
    )


def _actor() -> SimpleNamespace:
    return SimpleNamespace(
        actor_module=torch.nn.Linear(1, 1),
        actor_optimizer=SimpleNamespace(param_groups=[{}]),
        config={"policy_loss": asdict(load_config(CONFIG).actor)},
    )


def _positioned_batch(domain: str) -> SimpleNamespace:
    batch = runtime._batch(
        [[704, 333, 9000, 704, 333, 9001, 9002, 16688, 9003]], [domain]
    )
    attach_structure_position_mask(batch, _Tokenizer({
        704: "so", 333: "if", 9000: "\n```python\n", 9001: "\n```\n",
        9002: "##", 16688: " conclusion", 9003: "\nplain",
    }), asdict(load_config(CONFIG).audit))
    return batch


def test_planned_config_changes_only_timing_and_output_namespaces() -> None:
    new = load_config(CONFIG)
    baseline = Path("tests/fixtures/next_step_migration/current-step-resolved.json.gz")
    source = str(CONFIG).replace("next_step", "current_step")
    with gzip.open(baseline, "rt") as handle:
        before = with_post_snapshot_defaults(json.load(handle)[source])
    after = json.loads(json.dumps(asdict(new)))
    assert after["audit"]["code_cs_position_gate_enabled"] is None
    after["audit"].pop("code_cs_position_gate_enabled")
    changed = {
        f"{section}.{name}"
        for section in before
        if isinstance(before[section], dict)
        for name in before[section]
        if before[section][name] != after[section][name]
    }
    assert changed == {
        "audit.control_token_online_selection_timing", "audit.output_dir",
        "runtime.wandb_run_id", "paper_eval.output_dir",
        "huggingface_checkpoint.path_prefix", "trainer.experiment_name",
        "trainer.default_local_dir",
    }
    matched = deepcopy(after)
    for field in changed:
        section, name = field.split(".", 1)
        matched[section][name] = before[section][name]
    # New optional selector controls default to the historical position rule.
    assert matched["audit"]["versioned_cs_selection_mode_by_domain"] == {}
    matched["audit"].pop("versioned_cs_selection_mode_by_domain")
    assert matched == before
    assert "+mopd_audit.control_token_online_selection_timing=next_step" in build_command(new)
    assert new.data.train_batch_size == new.actor.ppo_mini_batch_size == 528
    assert new.audit.control_token_online_window_steps == 1
    assert new.audit.control_token_online_audit_interval_steps == 1
    assert new.audit.control_token_online_top_p_by_domain == {"math": .05, "code": .01}
    assert "-next-" in new.trainer.experiment_name


@pytest.mark.parametrize("alias", ["token_v4", "v4", "TOKEN_V4", "token-v4"])
def test_version_aliases_keep_the_same_next_step_contract(tmp_path: Path, alias: str) -> None:
    path = tmp_path / "alias.yaml"
    content = {
        "extends": str(CONFIG.resolve()),
        "audit": {"token_taxonomy_version": alias},
    }
    path.write_text(yaml.safe_dump(content))
    assert uses_versioned_next_step_selection(load_config(path).audit)
    content["audit"]["control_token_online_window_steps"] = 2
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(ValueError, match="window_steps"):
        load_config(path)


@pytest.mark.parametrize("domain", ["math", "code"])
def test_two_step_control_lag_structure_cold_start_and_checkpoint_restore(
    tmp_path: Path, domain: str,
) -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.audit import DomainGradientAudit

        logger = _logger(tmp_path, control_token_online_min_mean_occurrences_per_step=0)
        actor = _actor()
        first = DomainGradientAudit(actor, logger.full_gradient_meta("train", 1)["mopd_full_gradient"])
        batch = _positioned_batch(domain)
        initial = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 4., 1.]])
        initial /= initial.mean()
        torch.testing.assert_close(first.training_gradient_mask(batch), initial)
        original_ids = batch.batch["responses"].clone()
        raw = torch.tensor([[8., 1., 0., 8., 1e4 if domain == "code" else 1., 0., 0., 1e6, 0.]])
        # Deliberately contradictory configured losses must not drive selection.
        configured = torch.tensor([[1., 1e6, 0., 1., 1e6, 0., 0., 0., 0.]])
        with runtime._patch_forward(side_effect=AssertionError("unexpected scoring forward")), patch(
            "mopd_verl.domain_gradient.audit.build_actor_micro_batch_loss",
            side_effect=AssertionError("unexpected replay forward"),
        ):
            first.run_before_training([batch], [1.], on_policy=True, temperature=1.)
            first.observe_completed_step(
                [batch], [configured], [batch.batch["response_mask"]],
                selector_token_loss_batches=[raw],
                selector_token_loss_mask_batches=[batch.batch["response_mask"]],
            )
        assert first._applied_online_control_token_ids[domain] == ()
        assert first._online_control_selection_state.active_map()[domain] == (704,)
        torch.testing.assert_close(batch.batch["responses"], original_ids)
        # Reconstruct with only serialized optimizer state, as checkpoint resume does.
        restored = _actor()
        restored.actor_optimizer.param_groups[0].update(actor.actor_optimizer.param_groups[0])
        second = DomainGradientAudit(restored, logger.full_gradient_meta("train", 2)["mopd_full_gradient"])
        expected = torch.tensor([[4., 1., 1., 1. if domain == "code" else 4., 1., 1., 1., 4., 1.]])
        expected /= expected.mean()
        weights = second.training_gradient_mask(batch)
        torch.testing.assert_close(weights, expected)
        probe = torch.ones_like(weights, requires_grad=True)
        (probe * weights).mean().backward()
        torch.testing.assert_close(probe.grad, expected / expected.numel())


@pytest.mark.parametrize("outside_count", [20, 21])
def test_strict_source_frequency_excludes_fences_but_budget_counts_all_tokens(
    tmp_path: Path, outside_count: int,
) -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        config = _domain(tmp_path)
        # 50 occurrences in a fence cannot rescue the 20-occurrence candidate.
        batch = runtime._batch([[704] * outside_count + [333] * 21 + [704] * 50 + [99999] * 10], ["code"])
        size = batch.batch["responses"].shape[1]
        batch.batch["mopd_control_position_mask"] = torch.tensor(
            [[1] * (outside_count + 21) + [0] * 50 + [1] * 10]
        )
        losses = torch.tensor([[10.] * outside_count + [1.] * 21 + [1e9] * 50 + [0.] * 10])
        stats = next_step_candidate_statistics(config, [batch], [losses], [torch.ones(1, size)])
        assert stats.by_domain["code"][704] == (10. * outside_count, outside_count)
        assert stats.valid_token_counts["code"] == size
        state = initial_online_control_selection_state(
            config.domains, config.effective_domain_candidate_map(),
            audit_interval_steps=1, window_steps=1,
            min_mean_occurrences_per_step=20, strict_occurrence_gate=True,
            top_k=30, budget_mode="top_p", top_p=.01,
        )
        _, updated = update_online_control_selection(
            state, stats.by_domain, step=1,
            valid_token_counts=stats.valid_token_counts,
            valid_score_sums=stats.valid_score_sums,
        )
        assert updated.active_map()["code"] == ((333,) if outside_count == 20 else (704,))


@pytest.mark.parametrize("mask_name", ["mopd_control_position_mask", "mopd_structure_position_mask"])
def test_missing_position_masks_fail_closed(tmp_path: Path, mask_name: str) -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.audit import DomainGradientAudit

        audit = DomainGradientAudit(_actor(), _logger(tmp_path).full_gradient_meta("train", 1)["mopd_full_gradient"])
        batch = _positioned_batch("code")
        del batch.batch[mask_name]
        with pytest.raises(ValueError, match=mask_name):
            audit.training_gradient_mask(batch)


def test_raw_loss_required_without_configured_loss_fallback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="production raw RKL"):
        next_step_candidate_statistics(_domain(tmp_path), [_positioned_batch("code")], None, None)


@pytest.mark.parametrize("section,field,value", [
    ("audit", "control_token_online_selection_unit", "occurrence"),
    ("audit", "control_token_online_selection_mode", "top_speed"),
    ("audit", "control_token_online_window_steps", 2),
    ("audit", "control_token_online_weight_mode", "loss_ratio"),
    ("audit", "structure_token_loss_weighting_enabled", False),
    ("actor", "ppo_epochs", 2),
    ("actor", "ppo_mini_batch_size", 264),
    ("actor", "topk_distill_support_source", "student"),
])
def test_unsupported_next_step_combinations_fail_closed(
    tmp_path: Path, section: str, field: str, value: object,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump({"extends": str(CONFIG.resolve()), section: {field: value}}))
    with pytest.raises(ValueError):
        load_config(path)


def _distributed_statistics_worker(rank: int, rendezvous: str, run_dir: str) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
            config = _domain(Path(run_dir))
            count, loss = (12, 9.) if rank == 0 else (9, 3.)
            batch = runtime._batch([[704] * count + [333] * 12 + [704] * 10], ["code"])
            batch.batch["mopd_control_position_mask"] = torch.tensor(
                [[1] * (count + 12) + [0] * 10]
            )
            raw = torch.tensor([[loss] * count + [1.] * 12 + [1e9] * 10])
            stats = next_step_candidate_statistics(
                config, [batch], [raw], [batch.batch["response_mask"]],
            )
            assert stats.by_domain["code"][704] == (135., 21)
            assert stats.by_domain["code"][333] == (24., 24)
            assert stats.valid_token_counts == {"math": 0, "code": 65}
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_source_statistics_are_global_and_position_gated(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _distributed_statistics_worker,
        args=("file://" + str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
    )
