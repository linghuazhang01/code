"""R2 resolved-config parity and actual Fixed4 audit consumer contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import test_domain_gradient_optimization_contracts as contracts
from test_loss_ratio_alpha import _meta
from mopd_verl.audit_io import step_jsonl_dir
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "configs/token_selection/math/taxonomy/top32kl_next_step_full_taxonomy_split_topp0p05_i1_w1_5gpu_b256.yaml"
)
MODE = "top_q_loss_entropy"


def test_q_audit_uses_configured_loss_and_applies_fixed4_next_step(
    tmp_path: Path,
) -> None:
    with contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.audit import DomainGradientAudit

        actor = SimpleNamespace(actor_optimizer=SimpleNamespace(param_groups=[{}]))
        meta = {
            **_meta(1, "fixed"),
            "control_token_online_selection_mode": MODE,
            "output_dir": str(tmp_path),
        }
        ids = torch.tensor([[10, 20, 99, 99]])
        mask = torch.ones_like(ids).bool()
        batch = SimpleNamespace(
            batch={
                "responses": ids,
                "response_mask": mask,
                "student_entropy": torch.tensor([[1.0, 8.0, 1.0, 1.0]]),
            },
            non_tensor_batch={"domain": ["math"]},
        )
        # Q should favor ID20 despite ID10's larger configured loss.
        loss = torch.tensor([[4.0, 2.0, 1.0, 1.0]])
        audit = DomainGradientAudit(actor, {**meta, "step": 1})
        torch.testing.assert_close(
            audit.training_gradient_mask(batch), torch.ones_like(loss)
        )
        metrics = audit.observe_completed_step(
            (batch,),
            (loss,),
            (mask,),
            selector_token_loss_batches=(torch.tensor([[100.0, 0.0, 0.0, 0.0]]),),
            selector_token_loss_mask_batches=(mask,),
        )
        assert metrics["math/token_weight/q_all_valid_mean_abs_loss"] == 2
        assert metrics["math/token_weight/q_all_valid_mean_student_entropy"] == 2.75
        record = json.loads(
            (step_jsonl_dir(tmp_path, 1) / "online_control_selection.jsonl").read_text()
        )
        assert record["q_normalization_stats"]["math"]["q_all_valid_mean_abs_loss"] == 2
        state = actor.actor_optimizer.param_groups[0][
            "mopd_online_control_selection_state"
        ]
        assert dict(state["active_token_ids"]) == {"math": (20,)}
        assert state["weight_mode"] == "fixed" and state["loss_ratio_alpha"] == 1
        resumed_actor = SimpleNamespace(
            actor_optimizer=SimpleNamespace(
                param_groups=[
                    {
                        "mopd_online_control_selection_state": json.loads(
                            json.dumps(state)
                        )
                    }
                ]
            )
        )
        resumed = DomainGradientAudit(resumed_actor, {**meta, "step": 2})
        torch.testing.assert_close(
            resumed.training_gradient_mask(batch),
            torch.tensor([[1.0, 4.0, 1.0, 1.0]]) / 1.75,
        )
        with pytest.raises(ValueError, match="match"):
            DomainGradientAudit(
                resumed_actor,
                {**meta, "step": 2, "control_token_online_selection_mode": "top_loss"},
            )


@pytest.mark.parametrize(
    "field,value",
    [
        ("control_token_online_window_steps", 2),
        ("control_token_online_audit_interval_steps", 2),
        ("control_token_online_weight_mode", "paired"),
    ],
)
def test_q_rejects_unimplemented_modes(
    field: str, value: object, tmp_path: Path
) -> None:
    path = tmp_path / "invalid.yaml"
    audit = {"control_token_online_selection_mode": MODE, field: value}
    path.write_text(yaml.safe_dump({"extends": str(BASE), "audit": audit}))
    with pytest.raises(ValueError, match="Q selection"):
        load_config(path)
