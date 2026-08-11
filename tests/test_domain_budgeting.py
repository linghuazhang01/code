from __future__ import annotations

import json
from pathlib import Path

import pytest

from mopd_verl.domain_budgeting import DynamicDomainBudgetController
from mopd_verl.domain_budgeting_math import (
    apply_probability_floor,
    power_weighted_distribution,
)
from mopd_verl.domain_sampling import (
    DomainBatchSampler,
    allocate_domain_batch_counts,
)


def _config(tmp_path: Path, **overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "enabled": True,
        "domains": ["math", "code"],
        "domain_priors": {"math": 1.0, "code": 1.0},
        "teacher_scores": {"math": 1.0, "code": 1.0},
        "teacher_scores_calibrated": True,
        "validation_metric_keys": {
            "math": ["val/math"],
            "code": ["val/code"],
        },
        "gap_ema_beta": 0.0,
        "gap_alpha": 1.0,
        "gap_epsilon": 1e-12,
        "gap_normalization_floor": 0.05,
        "exploration_mass": 0.0,
        "variance_log_ema_beta": 0.0,
        "variance_epsilon": 1e-12,
        "variance_min_samples": 2,
        "variance_update_freq_steps": 1,
        "min_samples_per_domain": 2,
        "min_sampling_probability": 0.0,
        "output_dir": str(tmp_path),
    }
    config.update(overrides)
    return config


def test_time_normalized_gap_updates_target_contribution(tmp_path: Path) -> None:
    controller = DynamicDomainBudgetController(_config(tmp_path))

    controller.observe_validation({"val/math": 0.5, "val/code": 0.5}, step=0)
    assert controller.state.normalized_gaps == pytest.approx({"math": 1.0, "code": 1.0})
    assert controller.state.target_contributions == pytest.approx(
        {"math": 0.5, "code": 0.5}
    )

    update_metrics = controller.observe_validation(
        {"val/math": 0.75, "val/code": 0.5}, step=10
    )
    assert controller.state.normalized_gaps == pytest.approx({"math": 0.5, "code": 1.0})
    assert controller.state.target_contributions["math"] == pytest.approx(1.0 / 3.0)
    assert controller.state.target_contributions["code"] == pytest.approx(2.0 / 3.0)
    assert "domain_budgeting/math/next_q" in update_metrics
    assert "domain_budgeting/math/q" not in update_metrics


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"teacher_scores_calibrated": False}, "calibrated teacher scores"),
        (
            {"teacher_scores": {"math": float("nan"), "code": 1.0}},
            "Teacher scores must be finite",
        ),
        ({"variance_update_freq_steps": 0}, "at least two samples"),
    ],
)
def test_runtime_config_rechecks_hydra_overrides(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DynamicDomainBudgetController(_config(tmp_path, **overrides))


def test_time_normalization_uses_relative_remaining_gap(tmp_path: Path) -> None:
    controller = DynamicDomainBudgetController(_config(tmp_path))

    controller.observe_validation({"val/math": 0.1, "val/code": 0.9}, step=0)
    assert controller.state.initial_gaps == pytest.approx({"math": 0.9, "code": 0.1})
    assert controller.state.normalized_gaps == pytest.approx({"math": 1.0, "code": 1.0})
    assert controller.state.target_contributions == pytest.approx(
        {"math": 0.5, "code": 0.5}
    )

    controller.observe_validation({"val/math": 0.55, "val/code": 0.91}, step=1)
    assert controller.state.normalized_gaps == pytest.approx({"math": 0.5, "code": 0.9})
    assert controller.state.target_contributions == pytest.approx(
        {"math": 5.0 / 14.0, "code": 9.0 / 14.0}
    )


def test_time_normalization_floor_stabilizes_zero_initial_gap(
    tmp_path: Path,
) -> None:
    controller = DynamicDomainBudgetController(_config(tmp_path))
    controller.observe_validation({"val/math": 1.0, "val/code": 0.5}, step=0)

    controller.observe_validation({"val/math": 0.99, "val/code": 0.5}, step=1)

    assert controller.state.normalized_gaps == pytest.approx({"math": 0.2, "code": 1.0})
    assert controller.state.normalized_gaps["math"] < controller.max_normalized_gap


def test_validation_update_is_idempotent_after_checkpoint_resume(
    tmp_path: Path,
) -> None:
    controller = DynamicDomainBudgetController(_config(tmp_path, gap_ema_beta=0.9))
    metrics = {"val/math": 0.5, "val/code": 0.25}
    controller.observe_validation(metrics, step=10)
    state = controller.state

    controller.observe_validation(metrics, step=10)

    assert controller.state == state


def test_variance_changes_sampling_without_changing_q(tmp_path: Path) -> None:
    controller = DynamicDomainBudgetController(_config(tmp_path))
    controller.observe_validation({"val/math": 0.5, "val/code": 0.5}, step=0)
    initial_q = dict(controller.state.target_contributions)

    controller.observe_sequence_losses(
        ["math", "math", "code", "code"],
        [0.0, 2.0, 0.9, 1.1],
        step=1,
    )

    assert controller.state.target_contributions == initial_q
    assert controller.desired_sampling["math"] > controller.desired_sampling["code"]


def test_variance_update_is_idempotent_and_monotonic(tmp_path: Path) -> None:
    controller = DynamicDomainBudgetController(_config(tmp_path))
    labels = ["math", "math", "code", "code"]
    losses = [0.0, 2.0, 0.9, 1.1]
    controller.observe_sequence_losses(labels, losses, step=2)
    state = controller.state

    controller.observe_sequence_losses(labels, losses, step=2)
    assert controller.state == state
    with pytest.raises(ValueError, match="monotonically increasing"):
        controller.observe_sequence_losses(labels, losses, step=1)


def test_observed_batch_share_gives_exact_q_p_lambda_closure(tmp_path: Path) -> None:
    controller = DynamicDomainBudgetController(_config(tmp_path))
    controller.observe_validation({"val/math": 0.5, "val/code": 0.5}, step=0)
    labels = ["math"] * 8 + ["code"] * 2

    scales, metrics = controller.loss_scales_for_batch(labels, step=1)

    assert scales[:8] == pytest.approx([0.625] * 8)
    assert scales[8:] == pytest.approx([2.5] * 2)
    for domain in ("math", "code"):
        assert metrics[f"domain_budgeting/{domain}/closure_error"] < 1e-12
        observed = controller.state.observed_sampling[domain]
        scale = controller.state.loss_scales[domain]
        target = controller.state.target_contributions[domain]
        assert observed * scale == pytest.approx(target)


def test_empty_responses_use_active_domain_share_for_exact_closure(
    tmp_path: Path,
) -> None:
    controller = DynamicDomainBudgetController(_config(tmp_path))
    labels = ["math", "math", "code", "code"]

    scales, metrics = controller.loss_scales_for_batch(
        labels,
        step=1,
        active_mask=[False, True, True, True],
    )

    assert scales[:2] == pytest.approx([1.5, 1.5])
    assert scales[2:] == pytest.approx([0.75, 0.75])
    assert controller.state.observed_sampling == pytest.approx(
        {"math": 0.5, "code": 0.5}
    )
    assert controller.state.effective_sampling == pytest.approx(
        {"math": 1.0 / 3.0, "code": 2.0 / 3.0}
    )
    assert metrics["domain_budgeting/math/empty_response_rate"] == pytest.approx(0.5)
    for domain in ("math", "code"):
        assert metrics[f"domain_budgeting/{domain}/closure_error"] < 1e-12


def test_four_domain_batch_including_if_preserves_closure(tmp_path: Path) -> None:
    domains = ["math", "code", "science", "if"]
    controller = DynamicDomainBudgetController(
        _config(
            tmp_path,
            domains=domains,
            domain_priors={domain: 1.0 for domain in domains},
            teacher_scores={domain: 1.0 for domain in domains},
            validation_metric_keys={domain: [f"val/{domain}"] for domain in domains},
        )
    )
    controller.observe_validation(
        {f"val/{domain}": 0.5 for domain in domains},
        step=0,
    )

    labels = ["math"] * 4 + ["code"] * 2 + ["science", "if"]
    _, metrics = controller.loss_scales_for_batch(labels, step=1)

    for domain in domains:
        assert metrics[f"domain_budgeting/{domain}/closure_error"] < 1e-12
        assert (
            controller.state.observed_sampling[domain]
            * controller.state.loss_scales[domain]
        ) == pytest.approx(controller.state.target_contributions[domain])


def test_missing_domain_in_actor_batch_fails_fast(tmp_path: Path) -> None:
    controller = DynamicDomainBudgetController(_config(tmp_path))
    with pytest.raises(ValueError, match="missing"):
        controller.loss_scales_for_batch(["math", "math"], step=1)


def test_missing_active_domain_in_actor_batch_fails_fast(tmp_path: Path) -> None:
    controller = DynamicDomainBudgetController(_config(tmp_path))

    with pytest.raises(ValueError, match="missing active responses"):
        controller.loss_scales_for_batch(
            ["math", "math", "code", "code"],
            step=1,
            active_mask=[False, False, True, True],
        )


def test_fixed_probe_requires_every_configured_metric(tmp_path: Path) -> None:
    controller = DynamicDomainBudgetController(
        _config(
            tmp_path,
            validation_metric_keys={
                "math": ["val/math-a", "val/math-b"],
                "code": ["val/code"],
            },
        )
    )

    with pytest.raises(KeyError, match="val/math-b"):
        controller.observe_validation({"val/math-a": 0.5, "val/code": 0.5}, step=0)


def test_state_round_trip_and_json_persistence(tmp_path: Path) -> None:
    controller = DynamicDomainBudgetController(_config(tmp_path))
    controller.observe_validation({"val/math": 0.4, "val/code": 0.6}, step=0)
    controller.observe_sequence_losses(
        ["math", "math", "code", "code"], [0.0, 1.0, 0.4, 0.6], step=1
    )
    restored = DynamicDomainBudgetController(_config(tmp_path / "restored"))

    restored.load_state_dict(controller.state_dict())

    assert restored.state == controller.state
    payload = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert payload["controller_spec"]["domains"] == ["math", "code"]


def test_checkpoint_rejects_semantic_config_change(tmp_path: Path) -> None:
    controller = DynamicDomainBudgetController(_config(tmp_path, gap_alpha=1.0))
    restored = DynamicDomainBudgetController(
        _config(tmp_path / "restored", gap_alpha=2.0)
    )

    with pytest.raises(ValueError, match="gap_alpha"):
        restored.load_state_dict(controller.state_dict())


def test_corrupt_restored_state_fails_fast(tmp_path: Path) -> None:
    controller = DynamicDomainBudgetController(_config(tmp_path))
    payload = controller.state_dict()
    payload["state"]["desired_sampling"]["math"] = float("nan")

    with pytest.raises(ValueError, match="desired_sampling.math"):
        controller.load_state_dict(payload)


@pytest.mark.parametrize(
    ("missing_mapping", "message"),
    [
        ("gap_ema", "capability mappings"),
        ("log_variance_ema", "must cover the same domains"),
    ],
)
def test_inconsistent_restored_state_fails_fast(
    tmp_path: Path,
    missing_mapping: str,
    message: str,
) -> None:
    controller = DynamicDomainBudgetController(_config(tmp_path))
    controller.observe_validation({"val/math": 0.4, "val/code": 0.6}, step=0)
    controller.observe_sequence_losses(
        ["math", "math", "code", "code"],
        [0.0, 1.0, 0.4, 0.6],
        step=1,
    )
    payload = controller.state_dict()
    del payload["state"][missing_mapping]["code"]

    with pytest.raises(ValueError, match=message):
        controller.load_state_dict(payload)


def test_runtime_sampler_update_reserves_each_domain() -> None:
    labels = ["math"] * 10 + ["code"] * 10 + ["science"] * 10
    sampler = DomainBatchSampler(
        labels,
        {"math": 1.0, "code": 1.0, "science": 1.0},
        batch_size=12,
        replacement=True,
        seed=7,
    )

    actual = sampler.update_target_weights(
        {"math": 0.98, "code": 0.01, "science": 0.01},
        min_samples_per_domain=2,
    )

    assert sum(sampler.batch_counts.values()) == 12
    assert min(sampler.batch_counts.values()) >= 2
    assert sum(actual.values()) == pytest.approx(1.0)
    assert sampler.allocation_version == 1

    sampler.update_target_weights(
        {"math": 0.98, "code": 0.01, "science": 0.01},
        min_samples_per_domain=2,
    )
    assert sampler.allocation_version == 1


def test_minimum_quota_does_not_distort_feasible_target_counts() -> None:
    assert allocate_domain_batch_counts(
        100,
        {"math": 0.4, "code": 0.3, "science": 0.2, "if": 0.1},
        min_samples_per_domain=2,
    ) == {"math": 40, "code": 30, "science": 20, "if": 10}


def test_probability_floor_is_inactive_when_already_satisfied() -> None:
    weights = {"math": 0.4, "code": 0.3, "science": 0.2, "if": 0.1}

    assert apply_probability_floor(weights, 0.05) == pytest.approx(weights)
    floored = apply_probability_floor(
        {"math": 0.97, "code": 0.01, "science": 0.01, "if": 0.01},
        0.05,
    )
    assert sum(floored.values()) == pytest.approx(1.0)
    assert min(floored.values()) == pytest.approx(0.05)


def test_gap_power_distribution_stays_finite_for_large_alpha() -> None:
    distribution = power_weighted_distribution(
        {"math": 0.5, "code": 0.5},
        {"math": 1e-6, "code": 1.0},
        alpha=1_000.0,
    )

    assert sum(distribution.values()) == pytest.approx(1.0)
    assert all(value > 0.0 for value in distribution.values())
    torch = pytest.importorskip("torch")
    assert (torch.tensor(list(distribution.values()), dtype=torch.float32) > 0).all()


def test_sampler_rng_progress_and_checkpoint_resume() -> None:
    labels = ["math"] * 20 + ["code"] * 20
    sampler = DomainBatchSampler(
        labels,
        {"math": 0.5, "code": 0.5},
        batch_size=8,
        replacement=True,
        seed=17,
    )
    first = next(iter(sampler))
    second = next(iter(sampler))
    assert second != first

    payload = sampler.state_dict()
    expected_next = next(iter(sampler))
    restored = DomainBatchSampler(
        labels,
        {"math": 0.5, "code": 0.5},
        batch_size=8,
        replacement=True,
        seed=17,
    )
    restored.load_state_dict(payload)

    assert next(iter(restored)) == expected_next
    assert restored.batches_yielded == sampler.batches_yielded


def test_stateful_dataloader_resume_preserves_dynamic_sampler_state() -> None:
    torchdata = pytest.importorskip("torchdata.stateful_dataloader")
    labels = ["math"] * 20 + ["code"] * 20
    sampler = DomainBatchSampler(
        labels,
        {"math": 0.5, "code": 0.5},
        batch_size=8,
        replacement=True,
        seed=17,
    )
    loader = torchdata.StatefulDataLoader(
        list(range(40)),
        batch_sampler=sampler,
        num_workers=0,
    )
    iterator = iter(loader)
    next(iterator)
    sampler.update_target_weights(
        {"math": 0.75, "code": 0.25},
        min_samples_per_domain=1,
    )
    next(iterator)
    checkpoint = loader.state_dict()
    expected_next = next(iterator).tolist()

    restored_sampler = DomainBatchSampler(
        labels,
        {"math": 0.5, "code": 0.5},
        batch_size=8,
        replacement=True,
        seed=17,
    )
    restored_loader = torchdata.StatefulDataLoader(
        list(range(40)),
        batch_sampler=restored_sampler,
        num_workers=0,
    )
    restored_loader.load_state_dict(checkpoint)

    assert next(iter(restored_loader)).tolist() == expected_next
    assert restored_sampler.batch_counts == {"math": 6, "code": 2}
    assert restored_sampler.allocation_version == 1


def test_batch_count_allocation_rejects_impossible_minimum() -> None:
    with pytest.raises(ValueError, match="cannot provide"):
        allocate_domain_batch_counts(
            5,
            {"math": 1.0, "code": 1.0, "science": 1.0},
            min_samples_per_domain=2,
        )


def test_domain_loss_scale_composes_with_gradient_mask() -> None:
    torch = pytest.importorskip("torch")
    from mopd_verl.full_gradient.loss_support import (
        domain_loss_gradient_mask,
        gate_tensor_gradient,
    )

    response_mask = torch.ones((2, 3), dtype=torch.float32)
    audit_mask = torch.tensor([[1.0, 0.0, 1.0], [0.5, 1.0, 0.0]])
    combined = domain_loss_gradient_mask(
        torch.tensor([0.5, 2.0]), response_mask, audit_mask
    )
    torch.testing.assert_close(
        combined,
        torch.tensor([[0.5, 0.0, 0.5], [1.0, 2.0, 0.0]]),
    )

    values = torch.ones((2, 3), requires_grad=True)
    gated = gate_tensor_gradient(values, combined)
    assert torch.equal(gated, values)
    gated.sum().backward()
    torch.testing.assert_close(values.grad, combined)
