"""Numerical, gradient, validation and launcher contracts for category KL routing."""

from dataclasses import replace
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from mopd_verl.token_category_loss import routed_topk_loss, validate_token_category_loss
from mopd_verl.topk_distill import configured_distill_loss_name, topk_distill_loss_matrix


def config() -> dict:
    return {
        "distill_mode": "topk_renormalized_reverse_kl",
        "topk_distill_k": 32,
        "topk_distill_tail_bucket": False,
        "topk_distill_loss_by_domain": {
            "default": {"control": "forward", "structure": "reverse"},
            "code": {"structure": "forward", "control": "reverse"},
        },
    }


def test_routing_values_and_weighted_gradients() -> None:
    torch.manual_seed(7)
    student = torch.randn(3, 3, 32, requires_grad=True)
    teacher = torch.randn(3, 3, 32, requires_grad=True)
    ids = torch.tensor([[258, 0, 100000]] * 3)
    actual = routed_topk_loss(
        student=student, teacher=teacher, config=OmegaConf.create(config()),
        token_ids=ids, domains=["math", "code", "science"],
    )
    log_p = student.log_softmax(-1)
    log_q = teacher.detach().log_softmax(-1)
    fkl = (log_q.exp() * (log_q - log_p)).sum(-1)
    rkl = (log_p.exp() * (log_p - log_q)).sum(-1)
    mask = torch.tensor([[True, False, False], [False, True, False], [True, False, False]])
    expected = torch.where(mask, fkl, rkl)
    torch.testing.assert_close(actual, expected)
    weights = torch.tensor([[4., 1., 0.], [1., 4., 1.], [4., 1., 1.]])
    grad = torch.autograd.grad((actual * weights).sum(), student, retain_graph=True)[0]
    oracle = torch.autograd.grad((expected * weights).sum(), student)[0]
    torch.testing.assert_close(grad, oracle)
    assert teacher.grad is None
    assert torch.count_nonzero(grad[0, 2]) == 0


def test_disabled_is_identical_to_existing_loss() -> None:
    student, teacher = torch.randn(2, 3, 32), torch.randn(2, 3, 32)
    cfg = config()
    cfg["topk_distill_loss_by_domain"] = {}
    actual = routed_topk_loss(student=student, teacher=teacher, config=cfg,
                             token_ids=None, domains=[])
    expected = topk_distill_loss_matrix(
        student_topk_log_probs=student, teacher_topk_log_probs=teacher,
        mode=cfg["distill_mode"], include_tail=False, temperature=1.,
    )
    assert torch.equal(actual, expected)
    assert configured_distill_loss_name(cfg) == cfg["distill_mode"]
    assert configured_distill_loss_name(config()) == "topk32_domain_category_renormalized_kl"


@pytest.mark.parametrize("key,value", [
    ("topk_distill_k", 16),
    ("topk_distill_support_source", "student"),
    ("teacher_prefix_enabled", True),
    ("distill_mode", "chosen_token_reverse_kl"),
    ("topk_distill_loss_by_domain", {"math": {"connective": "forward"}}),
    ("topk_distill_loss_by_domain", {"typo": {"control": "forward"}}),
    ("topk_distill_loss_by_domain", {"math": {"control": "typo"}}),
])
def test_invalid_config_fails(key: str, value: object) -> None:
    cfg = {**config(), key: value}
    with pytest.raises(ValueError):
        validate_token_category_loss(cfg)


def test_missing_domains_and_ids_fail() -> None:
    scores = torch.randn(1, 3, 32)
    for ids, domains in [(None, ["math"]), (torch.zeros(1, 3), ["unknown"])]:
        with pytest.raises(ValueError):
            routed_topk_loss(student=scores, teacher=scores, config=config(),
                             token_ids=ids, domains=domains)


def test_launcher_preserves_nested_routing() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/mopd_qwen1p7b_nonthinking_goosereason4b_instruct_8gpu_math_code_science_topk32_reverse_kl_baseline.yaml")
    routes = config()["topk_distill_loss_by_domain"]
    updated = replace(base, actor=replace(base.actor, topk_distill_loss_by_domain=routes))
    arg = next(a for a in build_command(updated)
               if a.startswith("++actor_rollout_ref.actor.policy_loss.topk_distill_loss_by_domain="))
    assert yaml.safe_load(arg.split("=", 1)[1]) == routes


def test_verl_schema_preserves_routing() -> None:
    root = Path(__file__).resolve().parents[1]
    source = ast.parse((root / "third_party/verl/verl/workers/config/actor.py").read_text())
    policy = next(node for node in source.body
                  if isinstance(node, ast.ClassDef) and node.name == "PolicyLossConfig")
    assert any(isinstance(node, ast.AnnAssign)
               and node.target.id == "topk_distill_loss_by_domain" for node in policy.body)
    actor_yaml = yaml.safe_load((root / "third_party/verl/verl/trainer/config/actor/actor.yaml").read_text())
    assert actor_yaml["policy_loss"]["topk_distill_loss_by_domain"] == {}


def test_shared_actor_routes_before_importance_weights() -> None:
    import test_domain_gradient_optimization_contracts as contracts

    with contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.full_gradient.actor_loss import build_actor_micro_batch_loss

        torch.manual_seed(12)
        student = torch.randn(3, 3, 32, requires_grad=True)
        teacher = torch.randn(3, 3, 32).log_softmax(-1)
        ids = torch.tensor([[258, 0, 100000]] * 3)
        mask = torch.tensor([[1., 1., 0.], [1., 1., 1.], [1., 1., 1.]])
        weights = torch.tensor([[2., .5, 1.], [1., 2., 1.], [.5, 1., 2.]])
        domains = ["math", "code", "science"]
        cfg = {**config(), "multi_teacher_distill": True}
        batch = {"responses": ids, "response_mask": mask, "rollout_is_weights": weights}
        for domain in domains:
            batch[f"{domain}_teacher_topk_ids"] = torch.arange(32).expand(3, 3, 32)
            batch[f"{domain}_teacher_topk_logprobs"] = teacher
        micro = SimpleNamespace(
            batch=batch, non_tensor_batch={"domain": domains, "opd_teacher": domains},
            meta_info={"temperature": 1.},
        )
        micro.to = lambda device: micro
        actor = SimpleNamespace(
            config={"policy_loss": cfg, "use_kl_loss": False,
                    "entropy_coeff": 0., "loss_agg_mode": "token-mean"},
            _forward_micro_batch=lambda *args, **kwargs: (None, torch.zeros(3, 3), None, None, student),
        )
        result = build_actor_micro_batch_loss(
            actor, micro, loss_scale_factor=1., on_policy=True,
            return_configured_token_loss=True,
        )
        raw = routed_topk_loss(student=student, teacher=teacher, config=cfg,
                              token_ids=ids, domains=domains)
        torch.testing.assert_close(result.configured_token_loss, raw.detach() * weights * mask)
        torch.testing.assert_close(result.loss, (raw * weights * mask).sum() / mask.sum())
        result.loss.backward()
        assert torch.isfinite(student.grad).all()
        assert torch.count_nonzero(student.grad[0, 2]) == 0
