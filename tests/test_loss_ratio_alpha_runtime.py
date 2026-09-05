"""Runtime transport, logging, mean-one and resume contracts for alpha."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
import yaml

import test_loss_ratio_alpha as alpha_contracts
from mopd_verl.audit_io import step_jsonl_dir
from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.control_top_loss import OnlineControlSelectionState
from mopd_verl.launch import build_command
from mopd_verl.verl_audit import MOPDAuditLogger


@pytest.mark.parametrize("alpha", [0.25, 1.0, 2.0])
def test_settings_launcher_logger_and_actor_config_preserve_alpha(
    alpha: float,
    tmp_path: Path,
) -> None:
    config = alpha_contracts._settings(tmp_path, alpha)
    assert config.audit.control_token_loss_ratio_alpha == alpha
    overrides = [arg for arg in build_command(config) if arg.startswith("+mopd_audit.")]
    runtime_audit = {
        key.removeprefix("+mopd_audit."): yaml.safe_load(value)
        for key, value in (arg.split("=", 1) for arg in overrides)
    }
    assert runtime_audit["control_token_loss_ratio_alpha"] == alpha
    runtime_audit["output_dir"] = str(tmp_path / "audit")
    logger = MOPDAuditLogger({"mopd_audit": runtime_audit})
    meta = logger.full_gradient_meta("train", 2)["mopd_full_gradient"]
    assert meta["control_token_loss_ratio_alpha"] == alpha
    assert DomainGradientConfig.from_meta(meta).control_token_loss_ratio_alpha == alpha


@pytest.fixture
def audit_runtime() -> Iterator[tuple[Any, Any]]:
    """Reuse only the stub context; do not inherit or collect its test class."""
    import torch
    import test_domain_gradient_optimization_contracts as contracts

    with contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.audit import DomainGradientAudit

        yield torch, DomainGradientAudit


@pytest.mark.parametrize("raw_weight,all_selected", [(3e38, False), (3e-20, True)])
def test_mean_one_remains_finite_for_extreme_representable_weights(
    raw_weight: float,
    all_selected: bool,
) -> None:
    import torch
    from mopd_verl.domain_gradient.phase_control import online_token_score_weights

    ids = torch.tensor([[10, 10, 10, 10] if all_selected else [10, 10, 99, 99]])
    actual = online_token_score_weights(
        ids,
        torch.ones_like(ids),
        ["math"],
        domain_token_weights={"math": {10: raw_weight}},
        normalize_per_domain=True,
    )
    raw = torch.where(ids == 10, torch.tensor(raw_weight, dtype=torch.float64), 1.0)
    expected = (raw / raw.mean()).float()
    assert torch.isfinite(actual).all() and (actual > 0).all()
    torch.testing.assert_close(actual, expected)
    assert actual.mean().item() == pytest.approx(1.0)


@pytest.mark.parametrize("alpha", [0.25, 1.0, 2.0])
def test_audit_next_step_mean_one_logging_and_resume(
    alpha: float,
    tmp_path: Path,
    audit_runtime: tuple[Any, Any],
) -> None:
    torch, audit_type = audit_runtime
    actor = SimpleNamespace(actor_optimizer=SimpleNamespace(param_groups=[{}]))
    meta = {**alpha_contracts._meta(alpha), "output_dir": str(tmp_path)}
    batch = SimpleNamespace(
        batch={
            "response_mask": torch.tensor([[1, 1, 1, 1, 0]]),
            "responses": torch.tensor([[10, 20, 99, 99, 10]]),
        },
        non_tensor_batch={"domain": ["math"]},
    )
    losses = torch.tensor([[3.0, 1.0, 1.0, 1.0, 999.0]])
    valid = batch.batch["response_mask"].bool()
    audit = audit_type(actor, {**meta, "step": 1})
    metrics = audit.observe_completed_step((batch,), (losses,), (valid,))
    record = json.loads(
        (step_jsonl_dir(tmp_path, 1) / "online_control_selection.jsonl").read_text()
    )
    assert record["loss_ratio_alpha"] == alpha
    record = record["domains"]["math"]
    assert record["raw_selected_to_other_loss_ratio"] == pytest.approx(3.0)
    assert record["selected_unscaled_loss_ratio_weight"] == pytest.approx(3.0)
    assert record["selected_raw_loss_ratio_weight"] == pytest.approx(3 * alpha)
    assert metrics["global/token_weight/loss_ratio_alpha"] == alpha
    assert metrics["math/token_weight/loss_ratio_selected_unscaled_weight"] == 3.0
    assert metrics["math/token_weight/loss_ratio_selected_raw_weight"] == 3 * alpha
    payload = json.loads(
        json.dumps(
            actor.actor_optimizer.param_groups[0]["mopd_online_control_selection_state"]
        )
    )
    assert payload["loss_ratio_alpha"] == alpha
    resumed_actor = SimpleNamespace(
        actor_optimizer=SimpleNamespace(
            param_groups=[
                {"mopd_online_control_selection_state": payload},
            ]
        )
    )
    resumed = audit_type(resumed_actor, {**meta, "step": 2})
    effective = resumed.training_gradient_mask(batch)
    expected = torch.tensor([3 * alpha, 1.0, 1.0, 1.0])
    expected /= (3 * alpha + 3) / 4
    torch.testing.assert_close(effective[valid], expected)
    assert effective[valid].mean().item() == pytest.approx(1.0)
    resumed.observe_completed_step((batch,), (losses,), (valid,))
    saved = resumed_actor.actor_optimizer.param_groups[0][
        "mopd_online_control_selection_state"
    ]
    assert saved["loss_ratio_alpha"] == alpha
    assert OnlineControlSelectionState.from_mapping(saved).active_weight_map() == {
        "math": {10: 3 * alpha},
    }
    # Check both in-memory reuse and restoration into a fresh actor.
    for mismatch_actor in (
        actor,
        SimpleNamespace(
            actor_optimizer=SimpleNamespace(
                param_groups=[{"mopd_online_control_selection_state": payload}],
            )
        ),
    ):
        with pytest.raises(ValueError, match="(?i)(match|alpha|configuration)"):
            audit_type(
                mismatch_actor,
                {**meta, "step": 2, "control_token_loss_ratio_alpha": alpha * 2},
            )
