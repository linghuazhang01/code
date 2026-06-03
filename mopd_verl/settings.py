"""Typed configuration for the math+code MOPD verl launcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PAPER_EVAL_DATASETS = [
    "aime24",
    "aime25",
    "hmmt25_feb",
    "hmmt25_nov",
    "humaneval_plus",
    "mbpp_plus",
    "lcb",
]


@dataclass(frozen=True)
class DataConfig:
    train_files: list[str]
    val_files: list[str]
    train_batch_size: int = 1024
    val_batch_size: int | None = None
    max_prompt_length: int = 2048
    max_response_length: int = 16384
    filter_overlong_prompts: bool = True
    truncation: str = "error"
    shuffle: bool = True
    validation_shuffle: bool = False
    seed: int = 42
    return_raw_chat: bool = True
    enable_thinking: bool = False


@dataclass(frozen=True)
class ModelConfig:
    student_path: str
    student_base_path: str | None
    math_teacher_path: str
    code_teacher_path: str


@dataclass(frozen=True)
class ActorConfig:
    learning_rate: str = "1e-5"
    lr_warmup_steps_ratio: float = 0.0
    only_reverse_kl_advantages: bool = True
    lambda_vals: float = 1.25
    multi_teacher_distill: bool = True
    ppo_mini_batch_size: int = 1024
    ppo_micro_batch_size_per_gpu: int = 1
    use_kl_loss: bool = True
    kl_loss_coef: int = 0
    kl_loss_type: str = "low_var_kl"
    entropy_coeff: int = 0
    ppo_max_token_len_per_gpu: int = 32768
    gradient_checkpointing: bool = True
    param_offload: bool = False
    optimizer_offload: bool = False


@dataclass(frozen=True)
class RolloutConfig:
    calculate_log_probs: bool = True
    log_prob_micro_batch_size_per_gpu: int = 4
    tensor_model_parallel_size: int = 4
    name: str = "vllm"
    gpu_memory_utilization: float = 0.6
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False
    n: int = 1
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 1024
    num_gpu_blocks_override: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    val_n: int = 1
    val_do_sample: bool = False
    val_temperature: float = 1.0
    val_top_p: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class RolloutCorrectionConfig:
    rollout_is: str = "token"
    rollout_is_threshold: float = 5.0
    rollout_rs: str | None = "null"
    bypass_mode: bool = False


@dataclass(frozen=True)
class AuditConfig:
    enabled: bool = False
    output_dir: str = "mopd_audit"
    domains: list[str] = field(default_factory=lambda: ["math", "code"])
    tensorboard_prefix: str = "mopd"
    tensorboard_layout: str = "domain_category"
    tensorboard_prune_mode: str = "none"
    loss_variance_signal: str = "opd_loss_token"
    max_samples_per_domain: int = 32
    high_variance_cv_threshold: float = 1.0
    log_sample_level: bool = True
    log_validation_metrics: bool = True
    tier2_window_size: int = 20
    calibration_bins: int = 10
    validation_anchor_enabled: bool = True
    validation_anchor_on_step0: bool = False
    validation_anchor_refresh_steps: int = 0
    full_gradient_enabled: bool = False
    full_gradient_freq_steps: int = 1
    full_gradient_train_max_samples_per_domain: int | None = None
    full_gradient_validation_max_samples_per_domain: int | None = None
    full_gradient_validation_files: list[str] = field(default_factory=list)
    full_gradient_validation_batch_size: int | None = None
    full_gradient_micro_batch_size_per_gpu: int = 1
    full_gradient_storage_dtype: str = "float32"


@dataclass(frozen=True)
class PaperEvalConfig:
    enabled: bool = False
    script_path: str | None = None
    model_path: str | None = None
    output_dir: str = "paper_eval"
    datasets: list[str] = field(default_factory=lambda: list(DEFAULT_PAPER_EVAL_DATASETS))
    run_on_initial_validation: bool = True
    evaluate_current_checkpoint: bool = True
    fail_on_error: bool = False
    timeout_seconds: int = 0


@dataclass(frozen=True)
class TrainerConfig:
    project_name: str = "on-policy-distillation"
    experiment_name: str = "Qwen3-4B-Non-Thinking-Multi-Teacher-Distill-ExOPD"
    logger: str = '["console","wandb"]'
    n_gpus_per_node: int = 8
    nnodes: int = 1
    save_freq: int = 50
    default_local_dir: str = "/G-OPD-checkpoints/Qwen3-4B-Non-Thinking-Multi-Teacher-Distill-ExOPD"
    test_freq: int = 10
    total_epochs: int = 3
    total_training_steps: int | None = None
    max_actor_ckpt_to_keep: int | None = None
    max_critic_ckpt_to_keep: int | None = None
    critic_warmup: int = 0
    val_before_train: bool = True
    log_val_generations: int = 10


@dataclass(frozen=True)
class RayInitConfig:
    include_dashboard: bool | None = None
    num_cpus: int | None = None


@dataclass(frozen=True)
class RayKwargsConfig:
    ray_init: RayInitConfig = field(default_factory=RayInitConfig)


@dataclass(frozen=True)
class RuntimeConfig:
    python_bin: str = "python3"
    verl_module: str = "verl.trainer.main_ppo"
    wandb_mode: str = "online"
    used_model: str = "no_api"


@dataclass(frozen=True)
class MOPDConfig:
    data: DataConfig
    model: ModelConfig
    actor: ActorConfig = field(default_factory=ActorConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    paper_eval: PaperEvalConfig = field(default_factory=PaperEvalConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    ray_kwargs: RayKwargsConfig = field(default_factory=RayKwargsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _expect_mapping(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    return value


def _string_list(value: Any, key: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(f"Expected '{key}' to be a string or a list of strings.")


def _optional_string(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"Expected '{key}' to be a string or null.")


def load_config(path: str | Path) -> MOPDConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _expect_mapping(raw, "root")

    data_raw = _expect_mapping(root.get("data", {}), "data")
    model_raw = _expect_mapping(root.get("model", {}), "model")
    paper_eval_raw = _expect_mapping(root.get("paper_eval", {}), "paper_eval")

    data = DataConfig(
        train_files=_string_list(data_raw.get("train_files"), "data.train_files"),
        val_files=_string_list(data_raw.get("val_files"), "data.val_files"),
        train_batch_size=int(data_raw.get("train_batch_size", DataConfig.train_batch_size)),
        val_batch_size=(
            None if data_raw.get("val_batch_size") is None else int(data_raw["val_batch_size"])
        ),
        max_prompt_length=int(data_raw.get("max_prompt_length", DataConfig.max_prompt_length)),
        max_response_length=int(data_raw.get("max_response_length", DataConfig.max_response_length)),
        filter_overlong_prompts=bool(data_raw.get("filter_overlong_prompts", True)),
        truncation=str(data_raw.get("truncation", DataConfig.truncation)),
        shuffle=bool(data_raw.get("shuffle", True)),
        validation_shuffle=bool(data_raw.get("validation_shuffle", DataConfig.validation_shuffle)),
        seed=int(data_raw.get("seed", DataConfig.seed)),
        return_raw_chat=bool(data_raw.get("return_raw_chat", True)),
        enable_thinking=bool(data_raw.get("enable_thinking", False)),
    )
    model = ModelConfig(
        student_path=str(model_raw["student_path"]),
        student_base_path=(
            None
            if model_raw.get("student_base_path", model_raw["student_path"]) is None
            else str(model_raw.get("student_base_path", model_raw["student_path"]))
        ),
        math_teacher_path=str(model_raw["math_teacher_path"]),
        code_teacher_path=str(model_raw["code_teacher_path"]),
    )

    return MOPDConfig(
        data=data,
        model=model,
        actor=ActorConfig(**_expect_mapping(root.get("actor", {}), "actor")),
        rollout=RolloutConfig(**_expect_mapping(root.get("rollout", {}), "rollout")),
        rollout_correction=RolloutCorrectionConfig(
            **_expect_mapping(root.get("rollout_correction", {}), "rollout_correction")
        ),
        audit=AuditConfig(**_expect_mapping(root.get("audit", {}), "audit")),
        paper_eval=PaperEvalConfig(
            enabled=bool(paper_eval_raw.get("enabled", PaperEvalConfig.enabled)),
            script_path=_optional_string(paper_eval_raw.get("script_path"), "paper_eval.script_path"),
            model_path=_optional_string(paper_eval_raw.get("model_path"), "paper_eval.model_path"),
            output_dir=str(paper_eval_raw.get("output_dir", PaperEvalConfig.output_dir)),
            datasets=_string_list(
                paper_eval_raw.get("datasets", DEFAULT_PAPER_EVAL_DATASETS),
                "paper_eval.datasets",
            ),
            run_on_initial_validation=bool(
                paper_eval_raw.get("run_on_initial_validation", PaperEvalConfig.run_on_initial_validation)
            ),
            evaluate_current_checkpoint=bool(
                paper_eval_raw.get("evaluate_current_checkpoint", PaperEvalConfig.evaluate_current_checkpoint)
            ),
            fail_on_error=bool(paper_eval_raw.get("fail_on_error", PaperEvalConfig.fail_on_error)),
            timeout_seconds=int(paper_eval_raw.get("timeout_seconds", PaperEvalConfig.timeout_seconds)),
        ),
        trainer=TrainerConfig(**_expect_mapping(root.get("trainer", {}), "trainer")),
        ray_kwargs=RayKwargsConfig(
            ray_init=RayInitConfig(**_expect_mapping(root.get("ray_kwargs", {}).get("ray_init", {}), "ray_init"))
        ),
        runtime=RuntimeConfig(**_expect_mapping(root.get("runtime", {}), "runtime")),
    )
