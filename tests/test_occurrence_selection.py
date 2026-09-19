"""Position selection budgets and production prepass integration."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.occurrence import Position, occurrence_mask, select_positions
from mopd_verl.domain_gradient.occurrence_config import validate_occurrence_config
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from mopd_verl.verl_audit import MOPDAuditLogger
import test_domain_gradient_optimization_contracts as contracts


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "configs/token_selection/math_code_science/taxonomy/mopd_qwen1p7b_30b_4gpu_m05_c01_s01_cs_occurrence_rkl32_fixed4_b528_colocated.yaml"


def config() -> DomainGradientConfig:
    loaded = load_config(OVERLAY)
    logger = MOPDAuditLogger({"mopd_audit": asdict(loaded.audit)})
    return DomainGradientConfig.from_meta(logger.full_gradient_meta("train", 1)["mopd_full_gradient"])


def test_propagation_overlay_and_default() -> None:
    loaded = load_config(OVERLAY)
    assert loaded.audit.control_token_online_selection_unit == "occurrence"
    assert config().control_token_online_selection_unit == "occurrence"
    command = build_command(loaded)
    assert "+mopd_audit.control_token_online_selection_unit=occurrence" in command
    base = load_config(OVERLAY.with_name(OVERLAY.name.replace("occurrence", "toploss")))
    assert base.audit.control_token_online_selection_unit == "token_id"
    assert loaded.runtime.wandb_run_id != base.runtime.wandb_run_id
    assert loaded.trainer.default_local_dir != base.trainer.default_local_dir
    assert loaded.worker_placement.actor_rollout.n_gpus_per_node == 4
    logger = MOPDAuditLogger({"mopd_audit": asdict(loaded.audit)})
    assert logger.full_gradient_meta("eval", 1)["mopd_full_gradient"]["control_token_online_selection_unit"] == "token_id"


def test_full_dp_budget_gate_ties_scarcity_and_normalization() -> None:
    ranks = []
    for rank in range(2):
        positions = [Position(d, 0, row, i, 300, float(rank * 100 + i))
                     for row, d in enumerate(("math", "code", "science")) for i in range(25)]
        ranks.append(({d: 50 for d in ("math", "code", "science")}, positions, None))
    selected, means, metrics = select_positions(
        ranks, {d: [300] for d in ("math", "code", "science")},
        {"math": .05, "code": .01, "science": .01}, 20, True,
    )
    # All winners are on rank 1, not a separate quota on each rank.
    assert all(owner == 1 for owner, *_ in selected)
    assert len(selected) == 7
    for domain, expected in (("math", 5), ("code", 1), ("science", 1)):
        assert metrics[f"{domain}/occurrence/selected_count"] == expected
        assert (4 * expected + 100 - expected) / means[domain] == pytest.approx(100)
    tied = [({"math": 2}, [Position("math", 0, 0, i, 300, 1) for i in range(2)], None)] * 2
    selected, _, _ = select_positions(tied, {"math": [300]}, {"math": .25}, 0, False)
    assert selected == {(0, 0, 0, 0)}
    _, means, metrics = select_positions(tied, {"math": [300]}, {"math": 1}, 4, True)
    assert means["math"] == 1
    assert metrics["math/occurrence/selected_count"] == 0
    _, _, metrics = select_positions(tied, {"math": [300]}, {"math": 1}, 4, False)
    assert metrics["math/occurrence/selected_count"] == 4


@pytest.mark.parametrize("field,value", [
    ("control_token_online_selection_unit", "bad"),
    ("control_token_online_window_steps", 2),
    ("control_token_online_weight_mode", "loss_ratio"),
    ("control_token_online_candidate_scope", "full_vocabulary"),
    ("control_token_adaptive_neighborhood_enabled", True),
    ("control_token_online_top_k_per_group", 1),
])
def test_unsupported_options_fail_closed(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        validate_occurrence_config(replace(config(), **{field: value}))


def test_raw_rkl_prepass_masks_identity_and_rng() -> None:
    with contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.audit import DomainGradientAudit
        from mopd_verl.domain_gradient.occurrence import prepare_occurrence_masks

        cfg = replace(config(), control_token_online_min_mean_occurrences_per_step=0,
                      control_token_online_strict_occurrence_gate=False,
                      control_token_online_top_p_by_domain=(("math", .25), ("code", .01), ("science", .01)))
        actor = SimpleNamespace(actor_module=torch.nn.Linear(1, 1), config={"policy_loss": asdict(load_config(OVERLAY).actor)})
        audit = SimpleNamespace(config=cfg, actor=actor)
        def batch() -> SimpleNamespace:
            return SimpleNamespace(batch={"responses": torch.tensor([[300, 300, 99999]]),
                                          "response_mask": torch.tensor([[1, 1, 0]])},
                                   non_tensor_batch={"domain": ["math"]})
        first, second = batch(), batch()
        results = [SimpleNamespace(selector_token_loss=torch.tensor([[score, 2., float("nan")]]),
                                   selector_token_loss_mask=torch.tensor([[1, 1, 0]]),
                                   configured_token_loss=torch.tensor([[0., 10000., 0.]]))
                   for score in (9., 3.)]
        def forward(*args: object, **kwargs: object) -> SimpleNamespace:
            assert not torch.is_grad_enabled()
            torch.rand(3)
            return results.pop(0)
        rng = torch.get_rng_state().clone()
        with patch("mopd_verl.full_gradient.actor_loss.build_actor_micro_batch_loss", side_effect=forward):
            metrics = prepare_occurrence_masks(audit, [first, second], [0.5, 0.5], on_policy=True, temperature=1)
        assert torch.equal(rng, torch.get_rng_state())
        assert metrics["math/occurrence/valid_count"] == 4
        assert metrics["math/occurrence/selected_count"] == 1
        a, b = occurrence_mask(audit, first), occurrence_mask(audit, second)
        assert a[0, 0] == 4 * a[0, 1]
        assert b[0, 0] == b[0, 1]
        assert a[0, 2] == b[0, 2] == 0
        assert (a.sum() + b.sum()).item() == pytest.approx(4)
        assert DomainGradientAudit.training_gradient_mask(audit, first) is a
        with pytest.raises(RuntimeError, match="current-batch"):
            occurrence_mask(audit, batch())
        # A new prepass invalidates the old batch, even at the same step.
        with patch("mopd_verl.full_gradient.actor_loss.build_actor_micro_batch_loss", return_value=SimpleNamespace(
            selector_token_loss=torch.tensor([[1., 9., 0.]]), selector_token_loss_mask=torch.tensor([[1, 1, 0]]))):
            prepare_occurrence_masks(audit, [second], [1], on_policy=True, temperature=1)
        with pytest.raises(RuntimeError):
            occurrence_mask(audit, first)
        assert occurrence_mask(audit, second)[0, 1] == 4 * occurrence_mask(audit, second)[0, 0]


@pytest.mark.parametrize("raw,valid", [([float("nan"), 1], [1, 1]), ([float("inf"), 1], [1, 1]), ([1, 1], [1, 0])])
def test_nonfinite_and_inconsistent_validity_fail_closed(raw: list, valid: list) -> None:
    with contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.occurrence import prepare_occurrence_masks

        audit = SimpleNamespace(config=config(), actor=SimpleNamespace(actor_module=torch.nn.Linear(1, 1),
                                config={"policy_loss": asdict(load_config(OVERLAY).actor)}))
        batch = SimpleNamespace(batch={"responses": torch.tensor([[99999, 300]]), "response_mask": torch.ones(1, 2)},
                                non_tensor_batch={"domain": ["math"]})
        with patch("mopd_verl.full_gradient.actor_loss.build_actor_micro_batch_loss", return_value=SimpleNamespace(
            selector_token_loss=torch.tensor([raw]), selector_token_loss_mask=torch.tensor([valid]))):
            with pytest.raises(ValueError, match="occurrence scoring failed"):
                prepare_occurrence_masks(audit, [batch], [1], on_policy=True, temperature=1)
        assert audit._occurrence_masks == {}


def test_cache_survives_replay_and_post_step_metrics_then_retires() -> None:
    with contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.audit import DomainGradientAudit

        audit = DomainGradientAudit.__new__(DomainGradientAudit)
        audit.config = replace(config(), control_token_online_min_mean_occurrences_per_step=0)
        audit.actor = SimpleNamespace(actor_module=torch.nn.Linear(1, 1),
                                      config={"policy_loss": asdict(load_config(OVERLAY).actor)})
        batch = SimpleNamespace(batch={"responses": torch.tensor([[300, 300]]), "response_mask": torch.ones(1, 2)},
                                non_tensor_batch={"domain": ["math"]})
        result = SimpleNamespace(selector_token_loss=torch.tensor([[2., 1.]]), selector_token_loss_mask=torch.ones(1, 2))
        def replay(*args: object, **kwargs: object) -> dict:
            # Audit-state restoration must leave production positional weights intact.
            from mopd_verl.domain_gradient.state import AuditState
            before = audit.training_gradient_mask(batch).clone()
            AuditState.capture(audit.actor).restore()
            assert torch.equal(before, audit.training_gradient_mask(batch))
            return {}
        def source_metrics(*args: object, **kwargs: object) -> dict:
            assert torch.equal(args[2][0], audit.training_gradient_mask(batch))
            return {"source_checked": 1.0}
        with patch("mopd_verl.full_gradient.actor_loss.build_actor_micro_batch_loss", return_value=result), \
             patch.object(audit, "_run_before_training_replay", side_effect=replay):
            audit.run_before_training([batch], [1], on_policy=True, temperature=1)
        with patch("mopd_verl.domain_gradient.audit.amplified_token_source_metrics", side_effect=source_metrics):
            assert audit.observe_completed_step([batch], [result.selector_token_loss], [torch.ones(1, 2)]) == {"source_checked": 1.0}
        with pytest.raises(RuntimeError):
            audit.training_gradient_mask(batch)


def _dp_worker(rank: int, rendezvous: str) -> None:
    torch.distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        with contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
            from mopd_verl.domain_gradient.occurrence import prepare_occurrence_masks

            audit = SimpleNamespace(config=config(), actor=SimpleNamespace(actor_module=torch.nn.Linear(1, 1),
                                    config={"policy_loss": asdict(load_config(OVERLAY).actor)}))
            # Each rank has 21 copies of a C+S ID and 29 Other positions.
            batch = SimpleNamespace(batch={"responses": torch.tensor([[300] * 21 + [99999] * 29]),
                                    "response_mask": torch.ones(1, 50)}, non_tensor_batch={"domain": ["math"]})
            raw = torch.arange(50).float().unsqueeze(0) + rank * 100
            with patch("mopd_verl.full_gradient.actor_loss.build_actor_micro_batch_loss", return_value=SimpleNamespace(
                selector_token_loss=raw, selector_token_loss_mask=torch.ones(1, 50))):
                metrics = prepare_occurrence_masks(audit, [batch], [1], on_policy=True, temperature=1)
            weights = occurrence_mask(audit, batch)
            assert metrics["math/occurrence/budget_count"] == 5
            assert int((weights > 1).sum()) == (5 if rank else 0)
            total = weights.sum()
            torch.distributed.all_reduce(total)
            assert total.item() == pytest.approx(100, abs=1e-5)
    finally:
        torch.distributed.destroy_process_group()


def test_real_two_rank_gloo_budget(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(_dp_worker, args=("file://" + str(tmp_path / "rendezvous"),), nprocs=2)


@pytest.mark.parametrize("field,value", [("topk_distill_loss_by_domain", {"math": {"control": "forward"}}),
                                          ("token_baseline_method", "tip_topk32"), ("region_dpo_enabled", True)])
def test_mixed_rkl_and_baselines_rejected(field: str, value: object) -> None:
    policy = asdict(load_config(OVERLAY).actor)
    policy[field] = value
    with pytest.raises(ValueError):
        validate_occurrence_config(config(), policy)
