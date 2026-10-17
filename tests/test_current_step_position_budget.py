"""Position-level SAPD budget (S9) and its ID-level control (S9-ID): runtime and config contracts."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

import test_current_step_runtime as runtime
import test_occurrence_selection as existing
from mopd_verl.domain_gradient.current_step_selection import (
    CurrentStepSelection,
    select_current_step,
)
from mopd_verl.domain_gradient.occurrence import Position, occurrence_mask
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from test_sapd_progvocab_math_code_configs import (
    CODE_POOL_SHA256,
    MATH_POOL_SHA256,
    _pool_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "configs/token_selection/math_code/taxonomy"
SUITE = ROOT / "configs/qwen4b_suite/math_code"
PAPER = "mopd_math_code_toploss_m05_c01_math_control_code_progvocab_fixed4_{}.yaml"
NEW = "mopd_math_code_current_step_toploss_m05_c01_math_control_code_progvocab_{}_fixed4_{}.yaml"
UNITS = {"position": "occurrence", "idunit": "token_id", "sapd_pos": "occurrence", "sapd_csid": "token_id"}
# (new config, reference config it must equal except for timing/unit and run names, unit)
PROFILES = [
    (TAXONOMY / NEW.format(arm, topo), TAXONOMY / PAPER.format(topo), UNITS[arm])
    for arm in ("position", "idunit")
    for topo in ("8gpu_8c", "4gpu")
] + [
    (SUITE / f"{arm}_{topo}.yaml", SUITE / f"sapd_{topo}.yaml", UNITS[arm])
    for arm in ("sapd_pos", "sapd_csid")
    for topo in ("3gpu_3c", "8gpu_8c")
]
SELECTION_ARGS = (
    "+mopd_audit.control_token_online_selection_timing=",
    "+mopd_audit.control_token_online_selection_unit=",
)


def _positions(rank_tokens: list[tuple[int, float]], domain: str = "math") -> list[Position]:
    return [
        Position(domain, 0, 0, column, token, loss)
        for column, (token, loss) in enumerate(rank_tokens)
    ]


def _select(ranks, *, unit: str, minimum: float, head: float = .05) -> CurrentStepSelection:
    return select_current_step(
        ranks, {"math": [300, 301, 302]}, {"math": head}, {"math": 0.},
        minimum, True, unit=unit, head_modes={"math": "top_loss"}, tail_mode="bottom_loss",
    )


# 3b. Occurrence gate: per ID, per step, summed over ranks; strict ">"; minimum 0 disables it.
def test_occurrence_gate_counts_each_id_over_all_ranks_and_zero_minimum_disables_it() -> None:
    # ID300: 12 + 9 = 21 occurrences (eligible); ID301: 10 + 10 = 20 (not > 20); ID302: 3.
    rank0 = [(300, 5.)] * 12 + [(301, 9.)] * 10 + [(302, 50.)] * 3
    rank1 = [(300, 1.)] * 9 + [(301, 9.)] * 10
    ranks = [({"math": 500}, _positions(rank0), None), ({"math": 500}, _positions(rank1), None)]
    gated = _select(ranks, unit="occurrence", minimum=20, head=.5)
    assert gated.metrics["math/current_step/eligible_count"] == 21
    assert gated.metrics["math/current_step/eligible_unit_count"] == 21
    # Budget 500 exceeds the 21 eligible positions: shortfall, every eligible ID300 position taken.
    assert len(gated.head["math"]) == 21
    assert gated.metrics["math/current_step/head/shortfall"] == 500 - 21
    open_gate = _select(ranks, unit="occurrence", minimum=0, head=.5)
    assert open_gate.metrics["math/current_step/eligible_count"] == 44
    assert len(open_gate.head["math"]) == 44


# 3c. Budget: B = ceil(p x all valid domain tokens across ranks); positions stop exactly at B.
def test_occurrence_budget_uses_all_valid_tokens_across_ranks_and_never_overshoots() -> None:
    losses0 = [(300, 10. - i / 100) for i in range(30)] + [(301, 3.)] * 30
    losses1 = [(300, 9. - i / 100) for i in range(30)] + [(301, 2.)] * 30
    ranks = [({"math": 401}, _positions(losses0), None), ({"math": 399}, _positions(losses1), None)]
    budget = math.ceil(.05 * (401 + 399))  # 40, not 5% of either rank or of the candidates
    positional = _select(ranks, unit="occurrence", minimum=20)
    assert positional.metrics["math/current_step/valid_count"] == 800
    assert positional.metrics["math/current_step/head/budget_count"] == budget
    assert positional.metrics["math/current_step/head/selected_count"] == budget
    assert positional.metrics["math/current_step/head/overshoot"] == 0
    # Highest losses first: all 30 rank-0 ID300 positions, then the 10 best rank-1 ones.
    expected = {(0, 0, 0, c) for c in range(30)} | {(1, 0, 0, c) for c in range(10)}
    assert positional.head["math"] == expected
    # The ID-level control takes whole IDs and overshoots the same budget.
    by_id = _select(ranks, unit="token_id", minimum=20)
    assert by_id.head["math"] == {(300,)}
    assert by_id.metrics["math/current_step/head/selected_count"] == 60
    assert by_id.metrics["math/current_step/head/overshoot"] == 60 - budget


# 3a. Mean-one scope: identical to the paper SAPD next-step path (phase_token_weights).
@pytest.mark.parametrize("unit", ["token_id", "occurrence"])
def test_mean_one_scope_matches_paper_phase_token_weights(unit: str) -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.current_step_weights import install_current_step_masks
        from mopd_verl.domain_gradient.phase_control import phase_token_weights

        ids = [[0, 1, 0, 2], [1, 0, 2, 2], [0, 0, 1, 2]]
        mask = [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]]
        labels = ["math", "code", "math"]
        batch = runtime._batch(ids, labels, mask)
        audit = runtime._audit(control_token_online_selection_unit=unit)
        audit._occurrence_masks = {}
        valid = batch.batch["response_mask"].bool()
        if unit == "token_id":
            head = {"math": frozenset({(0,)}), "code": frozenset({(2,)})}
        else:
            # Every valid occurrence of the same IDs, written as positions of rank 0, micro-batch 0,
            # plus positions of another rank / micro-batch that must be ignored here.
            head = {
                domain: frozenset(
                    (0, 0, row, column)
                    for row, label in enumerate(labels) if label == domain
                    for column, token in enumerate(ids[row])
                    if token == chosen and mask[row][column]
                ) | {(1, 0, 0, 0), (0, 1, 1, 1)}
                for domain, chosen in (("math", 0), ("code", 2))
            }
        selection = CurrentStepSelection(
            head=head, tail={"math": frozenset(), "code": frozenset()}, metrics={},
        )
        install_current_step_masks(audit, [(batch, valid, labels)], selection, rank=0)
        paper = phase_token_weights(
            torch.tensor(ids), valid, labels,
            domain_token_ids={"math": torch.tensor([0]), "code": torch.tensor([2])},
            control_weight=4.0, phase_enabled=False, span_enabled=False, phase_gates={},
            span_length=1, span_decay_tau=1.0, normalize_per_domain=True,
        )
        ours = occurrence_mask(audit, batch)
        torch.testing.assert_close(ours[valid], paper[valid])
        # Two Math rows share one denominator inside the micro-batch: (4+1+4+4+4)/5 = 17/5.
        torch.testing.assert_close(ours[0, 0], torch.tensor(4 / (17 / 5)))
        assert torch.all(ours[~valid] == 0)


def test_indexed_position_marking_equals_full_scan() -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.current_step_weights import _local_positions, _mark

        selected = frozenset({(0, 0, 0, 1), (0, 1, 1, 2), (1, 0, 0, 0), (0, 0, 1, 3), (2, 1, 0, 0)})
        ids = torch.zeros(2, 4, dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]).bool()
        for rank in range(3):
            local = _local_positions(selected, rank)
            for batch_index in range(2):
                expected = torch.zeros_like(mask)
                for owner, batch, row, column in selected:
                    if owner == rank and batch == batch_index:
                        expected[row, column] = True
                expected &= mask
                for index in (None, local):
                    got = _mark(selected, ids, mask, unit="occurrence", rank=rank,
                                batch_index=batch_index, local=index)
                    assert torch.equal(got, expected)


# Configs: only the selection timing/unit and the run names differ from the reference SAPD file.
@pytest.mark.parametrize(("path", "reference", "unit"), PROFILES, ids=lambda v: getattr(v, "stem", v))
def test_profile_differs_from_reference_only_in_timing_unit_and_names(
    path: Path, reference: Path, unit: str,
) -> None:
    config, base = load_config(path), load_config(reference)
    audit = config.audit
    assert audit.control_token_online_selection_timing == "current_step"
    assert audit.control_token_online_selection_unit == unit
    assert base.audit.control_token_online_selection_timing == "next_step"
    assert base.audit.control_token_online_selection_unit == "token_id"
    assert audit.control_token_online_min_mean_occurrences_per_step == 20.0
    assert audit.control_token_online_strict_occurrence_gate is True
    assert audit.control_token_online_budget_mode == "top_p"
    assert audit.control_token_online_top_p_by_domain == {"math": 0.05, "code": 0.01}
    assert audit.control_token_loss_weight == 4.0
    assert audit.control_token_normalize_per_domain is True
    ids = audit.domain_control_token_candidate_ids
    assert _pool_sha256(set(ids["math"])) == MATH_POOL_SHA256 and len(ids["math"]) == 124
    assert _pool_sha256(set(ids["code"])) == CODE_POOL_SHA256 and len(ids["code"]) == 163
    assert config.actor.ppo_mini_batch_size == config.data.train_batch_size == 528

    def strip(command: list[str], run: str) -> list[str]:
        return [
            argument.replace(run, "<run>")
            for argument in command
            if not argument.startswith(SELECTION_ARGS)
        ]

    assert strip(build_command(config), config.trainer.experiment_name) == strip(
        build_command(base), base.trainer.experiment_name
    )
    run = config.runtime.wandb_run_id
    assert run == config.trainer.experiment_name
    for value in (
        audit.output_dir, config.paper_eval.output_dir,
        config.huggingface_checkpoint.path_prefix, config.trainer.default_local_dir,
    ):
        assert value.endswith(run)


def test_gate_can_be_disabled_through_the_threshold(tmp_path: Path) -> None:
    source = PROFILES[0][0]
    override = tmp_path / "ungated.yaml"
    override.write_text(yaml.safe_dump({
        "extends": str(source),
        "audit": {"control_token_online_min_mean_occurrences_per_step": 0.0},
    }))
    config = load_config(override)
    assert config.audit.control_token_online_min_mean_occurrences_per_step == 0.0
    assert "+mopd_audit.control_token_online_min_mean_occurrences_per_step=0.0" in build_command(config)


def test_run_names_are_unique_across_all_configs() -> None:
    ours = {path.resolve() for path, _, _ in PROFILES}
    run_ids = {load_config(path).runtime.wandb_run_id for path in ours}
    assert len(run_ids) == len(PROFILES)
    for path in (ROOT / "configs").rglob("*.yaml"):
        if path.resolve() in ours:
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        runtime_section = raw.get("runtime") if isinstance(raw, dict) else None
        if isinstance(runtime_section, dict):
            assert runtime_section.get("wandb_run_id") not in run_ids, path
