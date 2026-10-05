"""Per-step TopLoss statistics for every candidate, for offline selection replay."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
import torch

import test_current_step_runtime as current_step
import test_loss_ratio_alpha as alpha_contracts
from mopd_verl.audit_io import step_jsonl_dir
from mopd_verl.domain_gradient.current_step_selection import (
    CurrentStepSelection,
    current_step_selection_record,
)
from mopd_verl.domain_gradient.occurrence import Position


@pytest.fixture
def audit_type() -> Iterator[Any]:
    """Load the audit module under the small verl stub used by the contracts."""
    import test_domain_gradient_optimization_contracts as contracts

    with contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.audit import DomainGradientAudit

        yield DomainGradientAudit


def _by_id(domain_record: dict[str, Any]) -> dict[int, tuple[float, int]]:
    return {
        token_id: (loss_sum, count)
        for token_id, loss_sum, count in zip(
            domain_record["token_ids"],
            domain_record["abs_loss_sums"],
            domain_record["counts"],
            strict=True,
        )
    }


def test_next_step_record_lists_every_observed_candidate(
    tmp_path: Path,
    audit_type: Any,
) -> None:
    actor = SimpleNamespace(actor_optimizer=SimpleNamespace(param_groups=[{}]))
    meta = {**alpha_contracts._meta(1.0, "fixed"), "output_dir": str(tmp_path)}
    batch = SimpleNamespace(
        batch={
            "response_mask": torch.tensor([[1, 1, 1, 1, 0]]),
            "responses": torch.tensor([[10, 20, 20, 99, 10]]),
        },
        non_tensor_batch={"domain": ["math"]},
    )
    # The masked final ID 10 must not contribute its loss.
    losses = torch.tensor([[-3.0, 1.0, 0.5, 7.0, 999.0]])
    valid = batch.batch["response_mask"].bool()
    audit = audit_type(actor, {**meta, "step": 1})
    audit.observe_completed_step((batch,), (losses,), (valid,))
    record = json.loads(
        (step_jsonl_dir(tmp_path, 1) / "online_control_selection.jsonl").read_text()
    )
    statistics = record["candidate_step_statistics"]
    assert statistics["step"] == 1
    assert statistics["score"] == "abs_configured_token_loss"
    math_record = statistics["domains"]["math"]
    assert math_record["valid_token_count"] == 4
    # Only configured candidates appear; 99 is outside the pool.
    assert _by_id(math_record) == {10: (3.0, 1), 20: (1.5, 2)}


def test_current_step_record_pools_ranks_without_changing_selection() -> None:
    selection = CurrentStepSelection(
        head={"math": frozenset({(1,)})},
        tail={"math": frozenset({(2,)})},
        metrics={
            "math/current_step/head/budget_count": 2.0,
            "math/current_step/head/selected_count": 4.0,
            "math/current_step/tail/budget_count": 1.0,
            "math/current_step/tail/selected_count": 1.0,
        },
    )
    # Compacted token-ID positions carry mean |loss| and an occurrence count.
    ranks = [
        ({"math": 10}, [Position("math", 0, 0, 0, 1, 2.0, None, 3)], None),
        (
            {"math": 6},
            [
                Position("math", 0, 0, 0, 1, 4.0, None, 1),
                Position("math", 0, 0, 0, 2, 0.5, None, 1),
                Position("math", 0, 0, 0, 7, 9.0, None, 5),
            ],
            None,
        ),
    ]
    record = current_step_selection_record(
        ranks,
        {"math": (1, 2)},
        selection,
        step=12,
        unit="token_id",
        head_fractions={"math": 0.05},
        tail_fractions={"math": 0.0},
        minimum=20,
        strict=True,
    )
    assert record["step"] == 12
    assert record["selection_timing"] == "current_step"
    math_record = record["domains"]["math"]
    assert math_record["valid_token_count"] == 16
    # ID 7 is not a candidate, so it is excluded from the statistics.
    assert _by_id(math_record) == {1: (10.0, 4), 2: (0.5, 1)}
    assert math_record["head_token_ids"] == [1]
    assert math_record["tail_token_ids"] == [2]
    assert math_record["head_budget_count"] == 2.0
    assert math_record["head_selected_count"] == 4.0


def test_current_step_record_uses_absolute_raw_occurrence_losses() -> None:
    selection = CurrentStepSelection(
        head={"math": frozenset({(0, 0, 0, 1)})},
        tail={"math": frozenset()},
        metrics={},
    )
    ranks = [
        (
            {"math": 2},
            [
                Position("math", 0, 0, 0, 5, -2.0),
                Position("math", 0, 0, 1, 5, 3.0),
            ],
            None,
        )
    ]
    record = current_step_selection_record(
        ranks,
        {"math": (5,)},
        selection,
        step=3,
        unit="occurrence",
        head_fractions={"math": 0.5},
        tail_fractions={"math": 0.0},
        minimum=0,
        strict=False,
    )
    math_record = record["domains"]["math"]
    assert _by_id(math_record) == {5: (5.0, 2)}
    assert math_record["head_position_count"] == 1
    assert "head_token_ids" not in math_record


def test_prepass_attaches_a_record_matching_the_installed_selection() -> None:
    with current_step.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(
        torch
    ):
        audit = current_step._audit()
        batch = current_step._batch([[0, 1, 2]])
        current_step._prepare(audit, [batch], [current_step._result(batch, [[9.0, 1.0, 4.0]])])
        record = audit._current_step_selection_record
        assert record["step"] == audit.config.step
        math_record = record["domains"]["math"]
        assert _by_id(math_record) == {0: (9.0, 1), 1: (1.0, 1), 2: (4.0, 1)}
        assert math_record["valid_token_count"] == 3
        # Head is the highest |loss| ID; the tail is the lowest one.
        assert math_record["head_token_ids"] == [0]
        assert math_record["tail_token_ids"] == [1]


def test_audit_writes_the_record_once_and_skips_invalid_steps(
    tmp_path: Path,
    audit_type: Any,
) -> None:
    record = {"step": 5, "domains": {}}
    fake = SimpleNamespace(
        config=SimpleNamespace(output_dir=str(tmp_path)),
        _current_step_selection_record=record,
    )
    audit_type._write_current_step_selection_record(fake)
    path = step_jsonl_dir(tmp_path, 5) / "current_step_selection.jsonl"
    assert json.loads(path.read_text()) == record
    assert fake._current_step_selection_record is None
    # A second call has nothing left to write.
    audit_type._write_current_step_selection_record(fake)
    assert len(path.read_text().splitlines()) == 1
    fake._current_step_selection_record = {"step": -1, "error": "boom"}
    audit_type._write_current_step_selection_record(fake)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["step_000005"]
