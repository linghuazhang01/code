"""Contracts for the Math-only equal-weight Top-32 reverse-KL reference run."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

import test_domain_gradient_optimization_contracts as contracts
from test_loss_ratio_alpha import _meta
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config


TAXONOMY = Path(__file__).resolve().parents[1] / "configs/token_selection/math/taxonomy"
CONFIG = TAXONOMY / "uniform_top32kl_no_weighting_4gpu_3a1t_b255.yaml"
BASE = (
    TAXONOMY
    / "top32kl_next_step_full_taxonomy_split_topp0p05_i1_w1_4gpu_3a1t_b255_r20260911.yaml"
)
OFF = {
    "control_token_loss_weighting_enabled": False,
    "control_token_online_selection_enabled": False,
}


def test_uniform_config_only_switches_token_weighting_off() -> None:
    base, cfg = load_config(BASE), load_config(CONFIG)
    run = cfg.runtime.wandb_run_id
    assert run != base.runtime.wandb_run_id and len(run) <= 64
    expected = asdict(base)
    expected["audit"].update(OFF, output_dir=f"audit/{run}")
    expected["runtime"]["wandb_run_id"] = run
    expected["paper_eval"]["output_dir"] = f"eval_outputs/paper_suite/{run}"
    expected["huggingface_checkpoint"]["path_prefix"] = f"checkpoints/math/{run}"
    expected["trainer"].update(
        experiment_name=run,
        default_local_dir=f"checkpoints/MOPD/{run}",
        save_freq=1,
        max_actor_ckpt_to_keep=2,
    )
    assert asdict(cfg) == expected
    assert cfg.actor.distill_mode == "topk_renormalized_reverse_kl"
    assert cfg.actor.topk_distill_k == 32 and cfg.actor.loss_agg_mode == "token-mean"
    assert cfg.data.train_batch_size == cfg.actor.ppo_mini_batch_size == 255
    assert cfg.trainer.total_training_steps <= 65
    command = build_command(cfg)
    assert "+mopd_audit.control_token_loss_weighting_enabled=false" in command
    assert "+mopd_audit.control_token_online_selection_enabled=false" in command


def test_switched_off_weighting_applies_no_gradient_gate(tmp_path: Path) -> None:
    with contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.audit import DomainGradientAudit

        actor = SimpleNamespace(actor_optimizer=SimpleNamespace(param_groups=[{}]))
        ids = torch.tensor([[10, 20, 99, 99]])
        mask = torch.ones_like(ids).bool()
        batch = SimpleNamespace(
            batch={"responses": ids, "response_mask": mask},
            non_tensor_batch={"domain": ["math"]},
        )
        # Candidate ID 10 carries the largest loss; TopLoss Fixed4 would gate it.
        loss = torch.tensor([[9.0, 1.0, 1.0, 1.0]])
        meta = {**_meta(1, "fixed"), **OFF, "output_dir": str(tmp_path)}
        for step in (1, 2, 3):
            audit = DomainGradientAudit(actor, {**meta, "step": step})
            assert audit.training_gradient_mask(batch) is None
            audit.observe_completed_step((batch,), (loss,), (mask,))
        assert not actor.actor_optimizer.param_groups[0]
