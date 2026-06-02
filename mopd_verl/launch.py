"""Build and run verl commands for two-domain Multi-Teacher OPD."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from mopd_verl.settings import MOPDConfig, load_config


def _bool(value: bool) -> str:
    return "True" if value else "False"


def _hydra_list(values: Sequence[str]) -> str:
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"[{quoted}]"


def _hydra_scalar(value: object) -> str:
    if value is None:
        return "null"
    return str(value)


def _audit_overrides(config: MOPDConfig) -> list[str]:
    audit = config.audit
    if not audit.enabled:
        return []

    return [
        f"+mopd_audit.enabled={str(audit.enabled).lower()}",
        f"+mopd_audit.output_dir={audit.output_dir}",
        f"+mopd_audit.domains={_hydra_list(audit.domains)}",
        f"+mopd_audit.tensorboard_prefix={audit.tensorboard_prefix}",
        f"+mopd_audit.tensorboard_layout={audit.tensorboard_layout}",
        f"+mopd_audit.tensorboard_prune_mode={audit.tensorboard_prune_mode}",
        f"+mopd_audit.loss_variance_signal={audit.loss_variance_signal}",
        f"+mopd_audit.max_samples_per_domain={audit.max_samples_per_domain}",
        f"+mopd_audit.high_variance_cv_threshold={audit.high_variance_cv_threshold}",
        f"+mopd_audit.log_sample_level={str(audit.log_sample_level).lower()}",
        f"+mopd_audit.log_validation_metrics={str(audit.log_validation_metrics).lower()}",
        f"+mopd_audit.tier2_window_size={audit.tier2_window_size}",
        f"+mopd_audit.calibration_bins={audit.calibration_bins}",
        f"+mopd_audit.validation_anchor_enabled={str(audit.validation_anchor_enabled).lower()}",
        f"+mopd_audit.validation_anchor_refresh_steps={audit.validation_anchor_refresh_steps}",
        f"+mopd_audit.full_gradient_enabled={str(audit.full_gradient_enabled).lower()}",
        f"+mopd_audit.full_gradient_freq_steps={audit.full_gradient_freq_steps}",
        "+mopd_audit.full_gradient_train_max_samples_per_domain="
        f"{_hydra_scalar(audit.full_gradient_train_max_samples_per_domain)}",
        "+mopd_audit.full_gradient_validation_max_samples_per_domain="
        f"{_hydra_scalar(audit.full_gradient_validation_max_samples_per_domain)}",
        f"+mopd_audit.full_gradient_micro_batch_size_per_gpu={audit.full_gradient_micro_batch_size_per_gpu}",
        f"+mopd_audit.full_gradient_storage_dtype={audit.full_gradient_storage_dtype}",
    ]


def _paper_eval_overrides(config: MOPDConfig) -> list[str]:
    paper_eval = config.paper_eval
    if not paper_eval.enabled:
        return []

    return [
        f"+paper_eval.enabled={str(paper_eval.enabled).lower()}",
        f"+paper_eval.script_path={_hydra_scalar(paper_eval.script_path)}",
        f"+paper_eval.model_path={_hydra_scalar(paper_eval.model_path)}",
        f"+paper_eval.output_dir={paper_eval.output_dir}",
        f"+paper_eval.datasets={_hydra_list(paper_eval.datasets)}",
        f"+paper_eval.run_on_initial_validation={str(paper_eval.run_on_initial_validation).lower()}",
        f"+paper_eval.evaluate_current_checkpoint={str(paper_eval.evaluate_current_checkpoint).lower()}",
        f"+paper_eval.fail_on_error={str(paper_eval.fail_on_error).lower()}",
        f"+paper_eval.timeout_seconds={paper_eval.timeout_seconds}",
    ]


def build_overrides(config: MOPDConfig) -> list[str]:
    data = config.data
    model = config.model
    actor = config.actor
    rollout = config.rollout
    rollout_correction = config.rollout_correction
    trainer = config.trainer
    ray_init = config.ray_kwargs.ray_init

    ray_overrides = []
    if ray_init.include_dashboard is not None:
        ray_overrides.append(f"+ray_kwargs.ray_init.include_dashboard={_bool(ray_init.include_dashboard)}")
    if ray_init.num_cpus is not None:
        ray_overrides.append(f"ray_kwargs.ray_init.num_cpus={ray_init.num_cpus}")

    vllm_engine_overrides = []
    if rollout.num_gpu_blocks_override is not None:
        vllm_engine_overrides.append(
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override="
            f"{rollout.num_gpu_blocks_override}"
        )

    model_overrides = [f"actor_rollout_ref.model.path={model.student_path}"]
    if model.student_base_path is not None:
        model_overrides.append(f"+actor_rollout_ref.model.base_model_path={model.student_base_path}")
    model_overrides.extend(
        [
            f"+actor_rollout_ref.ref.model.path={model.math_teacher_path}",
            f"+actor_rollout_ref.ref.model.base_model_path={model.code_teacher_path}",
        ]
    )

    return [
        "algorithm.adv_estimator=grpo",
        f"algorithm.rollout_correction.rollout_is={rollout_correction.rollout_is}",
        f"algorithm.rollout_correction.rollout_is_threshold={rollout_correction.rollout_is_threshold}",
        f"algorithm.rollout_correction.rollout_rs={_hydra_scalar(rollout_correction.rollout_rs)}",
        f"algorithm.rollout_correction.bypass_mode={str(rollout_correction.bypass_mode).lower()}",
        f"actor_rollout_ref.rollout.calculate_log_probs={_bool(rollout.calculate_log_probs)}",
        f"data.train_files={_hydra_list(data.train_files)}",
        f"data.val_files={_hydra_list(data.val_files)}",
        f"data.train_batch_size={data.train_batch_size}",
        f"data.val_batch_size={_hydra_scalar(data.val_batch_size)}",
        f"data.max_prompt_length={data.max_prompt_length}",
        f"data.max_response_length={data.max_response_length}",
        f"data.filter_overlong_prompts={_bool(data.filter_overlong_prompts)}",
        f"data.truncation={data.truncation}",
        f"data.shuffle={_bool(data.shuffle)}",
        f"data.validation_shuffle={_bool(data.validation_shuffle)}",
        f"data.seed={data.seed}",
        f"data.return_raw_chat={_bool(data.return_raw_chat)}",
        f"+data.apply_chat_template_kwargs.enable_thinking={_bool(data.enable_thinking)}",
        *model_overrides,
        f"actor_rollout_ref.actor.optim.lr={actor.learning_rate}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps_ratio={actor.lr_warmup_steps_ratio}",
        "actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages={_bool(actor.only_reverse_kl_advantages)}",
        f"actor_rollout_ref.actor.policy_loss.lambda_vals={actor.lambda_vals}",
        f"actor_rollout_ref.actor.policy_loss.multi_teacher_distill={str(actor.multi_teacher_distill).lower()}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={actor.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={actor.ppo_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.use_kl_loss={_bool(actor.use_kl_loss)}",
        f"actor_rollout_ref.actor.kl_loss_coef={actor.kl_loss_coef}",
        f"actor_rollout_ref.actor.kl_loss_type={actor.kl_loss_type}",
        f"actor_rollout_ref.actor.entropy_coeff={actor.entropy_coeff}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={actor.ppo_max_token_len_per_gpu}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={_bool(actor.gradient_checkpointing)}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={_bool(actor.param_offload)}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={_bool(actor.optimizer_offload)}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={rollout.log_prob_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout.tensor_model_parallel_size}",
        f"actor_rollout_ref.rollout.name={rollout.name}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.enforce_eager={_bool(rollout.enforce_eager)}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={_bool(rollout.enable_chunked_prefill)}",
        f"actor_rollout_ref.rollout.n={rollout.n}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={rollout.max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.max_num_seqs={rollout.max_num_seqs}",
        f"actor_rollout_ref.rollout.temperature={rollout.temperature}",
        f"actor_rollout_ref.rollout.top_p={rollout.top_p}",
        f"actor_rollout_ref.rollout.val_kwargs.do_sample={_bool(rollout.val_do_sample)}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={rollout.val_temperature}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={rollout.val_top_p}",
        f"actor_rollout_ref.rollout.val_kwargs.n={rollout.val_n}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "reward_model.reward_manager=naive",
        f"trainer.critic_warmup={trainer.critic_warmup}",
        f"trainer.val_before_train={_bool(trainer.val_before_train)}",
        f"trainer.logger={trainer.logger}",
        f"trainer.log_val_generations={trainer.log_val_generations}",
        f"trainer.project_name={trainer.project_name}",
        f"trainer.experiment_name={trainer.experiment_name}",
        f"trainer.n_gpus_per_node={trainer.n_gpus_per_node}",
        f"trainer.nnodes={trainer.nnodes}",
        f"trainer.save_freq={trainer.save_freq}",
        f"trainer.default_local_dir={trainer.default_local_dir}",
        f"trainer.test_freq={trainer.test_freq}",
        f"trainer.total_epochs={trainer.total_epochs}",
        f"trainer.total_training_steps={_hydra_scalar(trainer.total_training_steps)}",
        f"+trainer.max_actor_ckpt_to_keep={_hydra_scalar(trainer.max_actor_ckpt_to_keep)}",
        f"+trainer.max_critic_ckpt_to_keep={_hydra_scalar(trainer.max_critic_ckpt_to_keep)}",
    ] + ray_overrides + vllm_engine_overrides + _audit_overrides(config) + _paper_eval_overrides(config)


def build_command(config: MOPDConfig, extra_args: Sequence[str] | None = None) -> list[str]:
    command = [config.runtime.python_bin, "-m", config.runtime.verl_module]
    command.extend(build_overrides(config))
    command.extend(extra_args or [])
    return command


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: Sequence[str], config: MOPDConfig) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("WANDB_MODE", config.runtime.wandb_mode)
    env.setdefault("USED_MODEL", config.runtime.used_model)
    return subprocess.call(list(command), env=env)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "mopd_math_code.yaml"),
        help="Path to a MOPD YAML config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the verl command without executing it.")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra Hydra overrides after '--'.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    extra_args = list(args.extra)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    command = build_command(config, extra_args)
    if args.dry_run:
        sys.stdout.write(format_command(command) + "\n")
        return 0
    return run_command(command, config)


if __name__ == "__main__":
    raise SystemExit(main())
