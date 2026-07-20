"""Single actor loss used by production training and gradient audit replay."""

from __future__ import annotations

from typing import Any

import torch

from mopd_verl.full_gradient.config import _cfg_get
from mopd_verl.full_gradient.loss_support import (
    ActorMicroBatchLossResult,
    active_kl_loss,
    actor_advantages,
    gate_tensor_gradient as _gate_tensor_gradient,
    masked_mean,
    policy_gradient_rewards,
    selected_teacher_log_prob,
    selected_topk_support,
    topk_runtime_config,
)
from mopd_verl.topk_distill import (
    DISTILL_LOSS_BUILDER_POLICY_GRADIENT,
    TOPK_RENORMALIZED_FORWARD_KL,
    TOPK_SUPPORT_SOURCE_STUDENT,
    chosen_token_forward_kl_matrix,
    distill_loss_builder,
    resolved_topk_distill_mode,
    select_teacher_log_prob_tensor,
    teacher_prefix_forward_weight,
    teacher_prefix_masks,
    topk_distill_bucket_metrics,
    topk_distill_include_tail,
    topk_distill_logprob_chunk_size,
    topk_distill_loss_matrix,
    topk_distill_support_source,
    topk_distill_temperature,
    topk_distill_weight,
    topk_teacher_student_cross_entropy_matrix,
    uses_topk_distill_loss,
)
from verl import DataProto
from verl.utils.device import get_device_id


def build_actor_micro_batch_loss(
    actor: Any,
    micro_batch: DataProto,
    *,
    loss_scale_factor: float,
    on_policy: bool,
    gradient_mask_override: torch.Tensor | None = None,
    include_metrics: bool = False,
    return_teacher_student_cross_entropy: bool = False,
    temperature: float | None = None,
) -> ActorMicroBatchLossResult:
    from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty

    micro_batch = micro_batch.to(get_device_id())
    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
    metrics: dict[str, Any] = {}
    response_mask = model_inputs["response_mask"]
    entropy_coeff = float(_cfg_get(actor.config, "entropy_coeff", 0.0) or 0.0)
    forward_temperature = (
        float(temperature) if temperature is not None else float(micro_batch.meta_info.get("temperature", 1.0))
    )
    forward_kwargs = {
        "temperature": forward_temperature,
        "calculate_entropy": entropy_coeff != 0,
    }
    policy_loss_cfg = _cfg_get(actor.config, "policy_loss", {})
    builder_name = distill_loss_builder(policy_loss_cfg)
    topk_distill_active = uses_topk_distill_loss(policy_loss_cfg)
    use_renormalized_support, effective_topk_logprob_mode = topk_runtime_config(
        policy_loss_cfg
    )
    use_renormalized_support = topk_distill_active and use_renormalized_support
    kl_loss_active, kl_coef = active_kl_loss(actor.config)
    needs_log_probs = not topk_distill_active or kl_loss_active
    forward_kwargs["calculate_log_probs"] = needs_log_probs
    topk_support_ids = None
    teacher_support_log_probs = None
    topk_support_source_value = topk_distill_support_source(policy_loss_cfg)
    if topk_distill_active:
        topk_support_ids, teacher_support_log_probs = selected_topk_support(
            model_inputs,
            policy_loss_cfg,
        )
        forward_kwargs["gather_topk_ids"] = topk_support_ids
        forward_kwargs["normalize_gathered_topk"] = not use_renormalized_support
        forward_kwargs["topk_logprob_chunk_size"] = topk_distill_logprob_chunk_size(policy_loss_cfg)
        forward_kwargs["topk_logprob_mode"] = effective_topk_logprob_mode
        forward_kwargs["return_extra"] = True
    forward_output = actor._forward_micro_batch(model_inputs, **forward_kwargs)
    if topk_distill_active:
        entropy, log_prob, _topk_ids, _topk_log_probs, student_topk_log_probs = forward_output
    else:
        entropy, log_prob = forward_output
    teacher_student_cross_entropy = None
    if return_teacher_student_cross_entropy:
        if not topk_distill_active:
            raise ValueError(
                "Teacher-student cross entropy reuse requires an active top-k "
                "distillation loss."
            )
        # Record the exact train-mode distribution used by this optimizer step.
        teacher_student_cross_entropy = (
            topk_teacher_student_cross_entropy_matrix(
                student_topk_log_probs=student_topk_log_probs.detach(),
                teacher_topk_log_probs=teacher_support_log_probs.detach(),
                include_tail=topk_distill_include_tail(policy_loss_cfg),
                temperature=topk_distill_temperature(policy_loss_cfg),
            ).detach()
        )
    if gradient_mask_override is not None:
        if gradient_mask_override.shape != response_mask.shape:
            raise ValueError(
                "Gradient mask must have the same shape as response_mask: "
                f"{tuple(gradient_mask_override.shape)} != "
                f"{tuple(response_mask.shape)}."
            )
        gradient_mask = gradient_mask_override.to(
            device=response_mask.device,
            dtype=torch.float32,
        )
        entropy = _gate_tensor_gradient(entropy, gradient_mask)
        log_prob = _gate_tensor_gradient(log_prob, gradient_mask)
        if topk_distill_active:
            student_topk_log_probs = _gate_tensor_gradient(
                student_topk_log_probs,
                gradient_mask,
            )
    prefix_loss_mask, suffix_loss_mask, teacher_prefix_active = teacher_prefix_masks(
        model_inputs,
        response_mask,
        policy_loss_cfg,
    )
    distill_response_mask = suffix_loss_mask if teacher_prefix_active else response_mask
    loss_token_mask = (
        (prefix_loss_mask + suffix_loss_mask).clamp(max=1.0)
        if teacher_prefix_active
        else response_mask
    )
    if bool(_cfg_get(actor.config, "use_rollout_log_probs", False)):
        old_log_prob = model_inputs["old_log_probs"]
    elif on_policy:
        old_log_prob = log_prob.detach()
    else:
        old_log_prob = model_inputs["old_log_probs"]

    if topk_distill_active:
        policy_loss = log_prob.new_zeros(())
        pg_loss = policy_loss
    else:
        if builder_name == DISTILL_LOSS_BUILDER_POLICY_GRADIENT:
            advantages = policy_gradient_rewards(
                model_inputs,
                policy_loss_cfg,
                old_log_prob,
            )
        else:
            advantages = actor_advantages(actor, model_inputs, old_log_prob)
        loss_mode = str(_cfg_get(_cfg_get(actor.config, "policy_loss", {}), "loss_mode", "vanilla"))
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=distill_response_mask,
            loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
            config=actor.config,
            rollout_is_weights=model_inputs.get("rollout_is_weights", None),
        )
        policy_loss = pg_loss
        if include_metrics:
            metrics.update(pg_metrics)
            if builder_name == DISTILL_LOSS_BUILDER_POLICY_GRADIENT:
                metrics["actor/chosen_token_pg_reward_mean"] = masked_mean(
                    advantages,
                    distill_response_mask,
                )
            rollout_log_prob = model_inputs.get("rollout_log_probs", None)
            if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                metrics.update(
                    compute_rollout_corr_metrics_from_logprobs(
                        log_prob=log_prob,
                        rollout_log_prob=rollout_log_prob,
                        response_mask=distill_response_mask,
                    )
                )
    if entropy_coeff != 0 and entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=loss_token_mask,
            loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
        )
        policy_loss = policy_loss - entropy_loss * entropy_coeff
    if topk_distill_active:
        topk_loss_mat = topk_distill_loss_matrix(
            student_topk_log_probs=student_topk_log_probs,
            teacher_topk_log_probs=teacher_support_log_probs,
            mode=resolved_topk_distill_mode(policy_loss_cfg),
            include_tail=topk_distill_include_tail(policy_loss_cfg),
            temperature=topk_distill_temperature(policy_loss_cfg),
        )
        topk_loss = agg_loss(
            loss_mat=topk_loss_mat,
            loss_mask=distill_response_mask,
            loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
        )
        topk_weight = topk_distill_weight(policy_loss_cfg)
        policy_loss = policy_loss + topk_loss * topk_weight
        if include_metrics:
            metrics["actor/topk_distill_loss"] = topk_loss.detach().item() * float(loss_scale_factor)
            metrics["actor/topk_distill_weight"] = topk_weight
            metrics["actor/topk_distill_support_is_student"] = float(
                topk_support_source_value == TOPK_SUPPORT_SOURCE_STUDENT
            )
            for key, value in topk_distill_bucket_metrics(
                student_topk_log_probs=student_topk_log_probs,
                teacher_topk_log_probs=teacher_support_log_probs,
                response_mask=distill_response_mask,
                student_values_are_log_probs=not use_renormalized_support,
                support_source=topk_support_source_value,
            ).items():
                metrics[f"actor/{key}"] = value
    if teacher_prefix_active:
        prefix_weight = teacher_prefix_forward_weight(policy_loss_cfg)
        if topk_distill_active:
            prefix_loss_mat = topk_distill_loss_matrix(
                student_topk_log_probs=student_topk_log_probs,
                teacher_topk_log_probs=teacher_support_log_probs,
                mode=TOPK_RENORMALIZED_FORWARD_KL,
                include_tail=False,
                temperature=topk_distill_temperature(policy_loss_cfg),
            )
        else:
            teacher_log_prob = select_teacher_log_prob_tensor(model_inputs, policy_loss_cfg)
            prefix_loss_mat = chosen_token_forward_kl_matrix(
                student_log_probs=log_prob,
                teacher_log_probs=teacher_log_prob,
            )
        prefix_loss = agg_loss(
            loss_mat=prefix_loss_mat,
            loss_mask=prefix_loss_mask,
            loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
        )
        policy_loss = policy_loss + prefix_loss * prefix_weight
        if include_metrics:
            metrics["actor/teacher_prefix_forward_kl_loss"] = (
                prefix_loss.detach().item() * float(loss_scale_factor)
            )
            metrics["actor/teacher_prefix_forward_kl_weight"] = prefix_weight
            metrics["actor/teacher_prefix_token_count"] = (
                prefix_loss_mask.detach().sum().item() * float(loss_scale_factor)
            )
            metrics["actor/student_suffix_token_count"] = (
                suffix_loss_mask.detach().sum().item() * float(loss_scale_factor)
            )
    if kl_loss_active and (
        "math_teacher_log_prob" in model_inputs or "ref_log_prob" in model_inputs
    ):
        reference_log_prob = selected_teacher_log_prob(
            model_inputs,
            policy_loss_cfg,
        )
        kld = kl_penalty(
            logprob=log_prob,
            ref_logprob=reference_log_prob,
            kl_penalty=str(_cfg_get(actor.config, "kl_loss_type", "kl")),
        )
        kl_loss = agg_loss(
            loss_mat=kld,
            loss_mask=distill_response_mask,
            loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
        )
        policy_loss = policy_loss + kl_loss * kl_coef
        if include_metrics:
            metrics["actor/kl_loss"] = kl_loss.detach().item() * float(
                loss_scale_factor
            )
            metrics["actor/kl_coef"] = kl_coef
    if include_metrics:
        metrics["actor/pg_loss"] = pg_loss.detach().item() * float(loss_scale_factor)
    return ActorMicroBatchLossResult(
        loss=policy_loss * float(loss_scale_factor),
        metrics=metrics,
        teacher_student_cross_entropy=teacher_student_cross_entropy,
    )
