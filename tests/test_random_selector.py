"""Contracts for the random-order ablation of the Next-Step ranking signal."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import test_domain_gradient_optimization_contracts as contracts
from test_loss_ratio_alpha import _meta
from mopd_verl.audit_io import step_jsonl_dir
from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.control_selection_scoring import random_selection_rank
from mopd_verl.domain_gradient.control_top_loss import (
    OnlineControlSelectionState,
    initial_online_control_selection_state,
    update_online_control_selection,
)
from mopd_verl.domain_gradient.control_top_loss_runtime import (
    global_candidate_loss_statistics_with_valid_counts as aggregate,
)
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from mopd_verl.verl_audit import MOPDAuditLogger


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "configs/token_selection/math/taxonomy"
CONFIG = (
    TAXONOMY
    / "random_next_step_full_taxonomy_unified_topp0p05_i1_w1_fixed4_5gpu_4a1t_b256.yaml"
)
BASE = TAXONOMY / "top32kl_next_step_full_taxonomy_split_topp0p05_i1_w1_5gpu_b256.yaml"
CONFIG_4GPU = (
    TAXONOMY
    / "random_next_step_full_taxonomy_unified_topp0p05_i1_w1_fixed4_4gpu_3a1t_b255.yaml"
)
BASE_4GPU = (
    TAXONOMY
    / "top32kl_next_step_full_taxonomy_split_topp0p05_i1_w1_4gpu_3a1t_b255_r20260911.yaml"
)
CONFIG_FULL_VOCAB = (
    ROOT
    / "configs/token_selection/math/full_vocabulary"
    / "mopd_qwen1p7b_30b_a3b_instruct_2507_4gpu_math_fullvocab_random_topp05_fixed4_3a1t_b255.yaml"
)
MODE = "random"
# Eight eligible IDs of 30 occurrences each plus one ID below the >20 gate.
IDS = tuple(range(100, 108))
RARE = 200
VALID = 1000


def _state(mode: str = MODE, top_p: float = 0.1) -> OnlineControlSelectionState:
    return initial_online_control_selection_state(
        ("math",),
        {"math": (*IDS, RARE)},
        audit_interval_steps=1,
        window_steps=1,
        min_mean_occurrences_per_step=20.0,
        strict_occurrence_gate=True,
        top_k=1,
        budget_mode="top_p",
        top_p=top_p,
        selection_mode=mode,
    )


def _stats(losses: dict[int, float]) -> dict[str, dict[int, tuple[float, int]]]:
    rows = {token_id: (losses[token_id] * 30, 30) for token_id in IDS}
    rows[RARE] = (1e6, 20)
    return {"math": rows}


def _select(
    state: OnlineControlSelectionState, losses: dict[int, float], step: int
) -> tuple[tuple[int, ...], OnlineControlSelectionState]:
    outcome, state = update_online_control_selection(
        state, _stats(losses), step=step, valid_token_counts={"math": VALID}
    )
    assert outcome.audit_triggered
    return state.active_map()["math"], state


def test_random_rank_is_deterministic_and_keyed_by_domain_step_and_id() -> None:
    assert random_selection_rank("math", 3, 11) == random_selection_rank("math", 3, 11)
    keys = {
        random_selection_rank(domain, step, token_id)
        for domain in ("math", "code")
        for step in (3, 4)
        for token_id in (11, 12)
    }
    assert len(keys) == 8


def test_random_order_ignores_loss_but_keeps_pool_gate_and_budget() -> None:
    rising = {token_id: float(index + 1) for index, token_id in enumerate(IDS)}
    falling = {token_id: float(len(IDS) - index) for index, token_id in enumerate(IDS)}
    selected, state = _select(_state(), rising, step=5)
    assert _select(_state(), falling, step=5)[0] == selected
    expected = tuple(
        sorted(IDS, key=lambda token_id: random_selection_rank("math", 5, token_id))
    )[:4]
    # ceil(0.1 * 1000) = 100 occurrences -> the first four whole IDs (4 * 30).
    assert selected == expected and RARE not in selected
    top_loss, _ = _select(_state("top_loss"), rising, step=5)
    assert top_loss == IDS[:3:-1] and len(top_loss) == len(selected)

    outcome, _ = update_online_control_selection(
        _state(), _stats(rising), step=5, valid_token_counts={"math": VALID}
    )
    (result,) = outcome.domain_results
    assert result.eligible_token_count == len(IDS)
    assert result.target_occurrence_count == 100
    assert result.selected_occurrence_count == 120 and result.top_p_target_reached
    assert [item.mean_abs_loss for item in result.selected_tokens] == [
        rising[token_id] for token_id in selected
    ]
    assert state.selection_mode == MODE


def test_random_order_is_redrawn_each_step_and_replays_after_restore() -> None:
    losses = dict.fromkeys(IDS, 1.0)
    state = _state()
    orders = []
    first = set()
    for step in range(1, 201):
        selected, state = _select(state, losses, step)
        orders.append(selected)
        first.add(selected[0])
    assert len(set(orders)) > 150 and first == set(IDS)

    restored = OnlineControlSelectionState.from_mapping(
        json.loads(json.dumps(state.as_dict()))
    )
    assert restored.selection_mode == MODE
    assert _select(restored, losses, 201)[0] == _select(state, losses, 201)[0]


def test_random_reuses_the_top_loss_statistics_wire_format() -> None:
    batches = {
        "token_id_batches": [torch.tensor([[100, 101, 999]]), torch.tensor([[100]])],
        "loss_batches": [torch.tensor([[1.0, -2.0, 8.0]]), torch.tensor([[3.0]])],
        "mask_batches": [torch.ones(1, 3, dtype=torch.bool)] * 1
        + [torch.ones(1, 1, dtype=torch.bool)],
        "label_batches": [("math",), ("math",)],
    }
    results = [
        aggregate(
            **batches,
            domains=("math",),
            domain_candidate_token_ids={"math": (100, 101)},
            selection_mode=mode,
        )
        for mode in (MODE, "top_loss")
    ]
    assert results[0] == results[1]
    assert results[0].by_domain == {"math": {100: (4.0, 2), 101: (2.0, 1)}}
    assert results[0].valid_token_counts == {"math": 4}


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"weight_mode": "loss_ratio"}, "top-loss selection"),
        ({"weight_mode": "paired"}, "paired-signal"),
    ],
)
def test_random_rejects_score_dependent_weight_modes(
    kwargs: dict[str, str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        initial_online_control_selection_state(
            ("math",),
            {"math": IDS},
            audit_interval_steps=1,
            window_steps=1,
            min_mean_occurrences_per_step=20.0,
            top_k=1,
            selection_mode=MODE,
            **kwargs,
        )


def test_random_order_covers_every_observed_id_under_full_vocabulary_scope() -> None:
    # Token 7 has 30 occurrences, token 9 has 25 and token 3 stays below the gate.
    ids = torch.tensor([[7] * 30 + [9] * 25 + [3] * 20 + [5] * 25])
    mask = torch.ones_like(ids).bool()
    mask[0, -25:] = False
    statistics = aggregate(
        [ids],
        [torch.ones(ids.shape)],
        [mask],
        [("math",)],
        domains=("math",),
        selection_mode=MODE,
        candidate_scope="full_vocabulary",
        candidate_vocab_size=16,
    )
    assert statistics.by_domain == {"math": {3: (20.0, 20), 7: (30.0, 30), 9: (25.0, 25)}}
    assert statistics.valid_token_counts == {"math": 75}
    state = initial_online_control_selection_state(
        ("math",),
        {"math": ()},
        audit_interval_steps=1,
        window_steps=1,
        min_mean_occurrences_per_step=20.0,
        strict_occurrence_gate=True,
        top_k=1,
        budget_mode="top_p",
        top_p=0.05,
        selection_mode=MODE,
        candidate_scope="full_vocabulary",
        candidate_vocab_size=16,
    )
    for step in range(1, 41):
        outcome, state = update_online_control_selection(
            state,
            statistics.by_domain,
            step=step,
            valid_token_counts=statistics.valid_token_counts,
        )
        expected = min((7, 9), key=lambda token_id: random_selection_rank("math", step, token_id))
        assert state.active_map() == {"math": (expected,)}
        assert outcome.domain_results[0].eligible_token_count == 2


def test_full_vocabulary_config_only_drops_the_candidate_pool() -> None:
    pool, full = load_config(CONFIG_4GPU), load_config(CONFIG_FULL_VOCAB)
    run = full.runtime.wandb_run_id
    assert run != pool.runtime.wandb_run_id and len(run) <= 64
    expected = asdict(pool)
    expected["audit"].update(
        control_token_online_candidate_scope="full_vocabulary",
        domain_control_token_candidate_groups={},
        output_dir=f"audit/{run}",
    )
    expected["runtime"]["wandb_run_id"] = run
    expected["paper_eval"]["output_dir"] = f"eval_outputs/paper_suite/{run}"
    expected["huggingface_checkpoint"]["path_prefix"] = f"checkpoints/math/{run}"
    expected["trainer"].update(
        experiment_name=run, default_local_dir=f"checkpoints/MOPD/{run}"
    )
    assert asdict(full) == expected
    assert full.trainer.save_freq == 1 and full.trainer.max_actor_ckpt_to_keep == 2
    assert full.audit.control_token_online_selection_mode == MODE
    assert full.audit.control_token_online_candidate_vocab_size is None
    assert not full.audit.control_token_candidate_ids
    assert not full.audit.domain_control_token_candidate_ids
    command = build_command(full)
    assert "+mopd_audit.control_token_online_candidate_scope=full_vocabulary" in command
    assert "+mopd_audit.control_token_online_selection_mode=random" in command
    assert "trainer.total_training_steps=65" in command


def test_random_rejects_current_step_timing(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "extends": str(CONFIG),
                "audit": {"control_token_online_selection_timing": "current_step"},
            }
        )
    )
    with pytest.raises(ValueError, match="current-step"):
        load_config(path)


@pytest.mark.parametrize(
    "config_path,base_path,batch,gpus,save_freq,kept",
    [(CONFIG, BASE, 256, 5, 5, 4), (CONFIG_4GPU, BASE_4GPU, 255, 4, 1, 2)],
)
def test_ablation_config_only_changes_selector_and_identities(
    config_path: Path,
    base_path: Path,
    batch: int,
    gpus: int,
    save_freq: int,
    kept: int,
    tmp_path: Path,
) -> None:
    base, cfg = load_config(base_path), load_config(config_path)
    run = cfg.runtime.wandb_run_id
    assert run != base.runtime.wandb_run_id and len(run) <= 64
    expected = asdict(base)
    expected["audit"].update(
        control_token_online_selection_mode=MODE, output_dir=f"audit/{run}"
    )
    expected["runtime"]["wandb_run_id"] = run
    expected["paper_eval"]["output_dir"] = f"eval_outputs/paper_suite/{run}"
    expected["huggingface_checkpoint"]["path_prefix"] = f"checkpoints/math/{run}"
    expected["trainer"].update(
        experiment_name=run,
        default_local_dir=f"checkpoints/MOPD/{run}",
        save_freq=save_freq,
        max_actor_ckpt_to_keep=kept,
    )
    assert asdict(cfg) == expected
    assert cfg.audit.control_token_online_budget_mode == "top_p"
    assert cfg.audit.control_token_online_top_p == 0.05
    assert cfg.audit.control_token_online_weight_mode == "fixed"
    assert cfg.audit.control_token_loss_weight == 4
    assert cfg.audit.control_token_normalize_per_domain
    assert cfg.audit.control_token_online_selection_timing == "next_step"
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == batch
    assert cfg.runtime.slurm_allocation_gpus == gpus
    assert cfg.worker_placement.actor_rollout.n_gpus_per_node == gpus - 1
    assert cfg.worker_placement.ref_policy.n_gpus_per_node == 1
    assert cfg.trainer.total_training_steps <= 65
    assert not cfg.huggingface_checkpoint.private
    command = build_command(cfg)
    overrides = [a for a in command if a.startswith("+mopd_audit.")]
    audit_cfg = {
        key.removeprefix("+mopd_audit."): yaml.safe_load(value)
        for key, value in (a.split("=", 1) for a in overrides)
    }
    audit_cfg["output_dir"] = str(tmp_path)
    logger = MOPDAuditLogger({"mopd_audit": audit_cfg})
    meta = logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
    assert (
        DomainGradientConfig.from_meta(meta).control_token_online_selection_mode == MODE
    )


def test_random_audit_applies_fixed4_next_step_and_survives_resume(
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
            batch={"responses": ids, "response_mask": mask},
            non_tensor_batch={"domain": ["math"]},
        )
        chosen = min((10, 20), key=lambda token_id: random_selection_rank("math", 1, token_id))
        # The larger loss sits on the ID that the random order does not pick.
        loss = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        loss[0, (10, 20).index(chosen) ^ 1] = 9.0
        audit = DomainGradientAudit(actor, {**meta, "step": 1})
        torch.testing.assert_close(
            audit.training_gradient_mask(batch), torch.ones_like(loss)
        )
        audit.observe_completed_step((batch,), (loss,), (mask,))
        record = json.loads(
            (step_jsonl_dir(tmp_path, 1) / "online_control_selection.jsonl").read_text()
        )
        assert record["selection_mode"] == MODE
        state = actor.actor_optimizer.param_groups[0][
            "mopd_online_control_selection_state"
        ]
        assert dict(state["active_token_ids"]) == {"math": (chosen,)}
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
        raw = torch.ones_like(loss)
        raw[0, (10, 20).index(chosen)] = 4.0
        torch.testing.assert_close(resumed.training_gradient_mask(batch), raw / 1.75)
        with pytest.raises(ValueError, match="match"):
            DomainGradientAudit(
                resumed_actor,
                {**meta, "step": 2, "control_token_online_selection_mode": "top_loss"},
            )
