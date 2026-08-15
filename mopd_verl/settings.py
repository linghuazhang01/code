"""Typed configuration for the multi-domain MOPD verl launcher."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mopd_verl.config_profiles import load_raw_config
from mopd_verl.domain_budgeting_config import (
    DomainBudgetingConfig,
    parse_domain_budgeting_config,
    validate_domain_budgeting_config,
)

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
    domain_train_files: dict[str, list[str]] = field(default_factory=dict)
    domain_sampling_weights: dict[str, float] = field(default_factory=dict)
    domain_sampling_replacement: bool = True
    load_parquet_direct: bool = False
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
    need_tools_kwargs: bool = False
    dataloader_num_workers: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    student_path: str
    student_base_path: str | None
    math_teacher_path: str
    code_teacher_path: str
    if_teacher_path: str | None
    domain_teacher_paths: dict[str, str]
    reasoning_teacher_path: str | None
    primary_teacher_path: str
    secondary_teacher_path: str | None
    teacher_model_device: str = "cpu"
    attn_implementation: str = "flash_attention_2"


@dataclass(frozen=True)
class ActorConfig:
    learning_rate: str = "1e-5"
    lr_warmup_steps_ratio: float = 0.0
    only_reverse_kl_advantages: bool = True
    lambda_vals: float = 1.25
    multi_teacher_distill: bool = True
    distill_loss_builder: str = "auto"
    distill_mode: str = "chosen_token_reverse_kl"
    eopd_entropy_threshold: float = 0.8
    eopd_forward_kl_weight: float = 1.0
    eopd_topk_k: int = 16
    topk_distill_enabled: bool = False
    topk_distill_kl_direction: str = "reverse"
    topk_distill_k: int = 8
    topk_distill_support_source: str = "teacher"
    topk_distill_tail_bucket: bool = True
    topk_distill_temperature: float = 1.0
    topk_distill_loss_weight: float = 1.0
    topk_distill_logprob_chunk_size: int = 16
    topk_distill_logprob_mode: str = "sparse"
    teacher_prefix_enabled: bool = False
    teacher_prefix_loss_region: str = "suffix_only"
    teacher_prefix_forward_kl_weight: float = 1.0
    ppo_mini_batch_size: int = 1024
    ppo_micro_batch_size_per_gpu: int = 1
    ppo_epochs: int = 1
    use_dynamic_bsz: bool = False
    use_kl_loss: bool = True
    kl_loss_coef: int = 0
    kl_loss_type: str = "low_var_kl"
    entropy_coeff: int = 0
    ppo_max_token_len_per_gpu: int = 32768
    gradient_checkpointing: bool = True
    param_offload: bool = False
    optimizer_offload: bool = False
    fsdp_size: int | None = None
    loss_agg_mode: str = "token-mean"


@dataclass(frozen=True)
class RolloutConfig:
    calculate_log_probs: bool = True
    log_prob_micro_batch_size_per_gpu: int = 4
    tensor_model_parallel_size: int = 4
    name: str = "vllm"
    mode: str = "sync"
    gpu_memory_utilization: float = 0.6
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False
    n: int = 1
    max_num_batched_tokens: int = 32768
    max_model_len: int | None = None
    max_num_seqs: int = 1024
    num_gpu_blocks_override: int | None = None
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    teacher_prefix_sampling_enabled: bool = False
    teacher_prefix_length: int = 1024
    teacher_prefix_dataset_key: str = "prefix"
    val_n: int = 1
    val_do_sample: bool = False
    val_temperature: float = 1.0
    val_top_p: float = 1.0
    seed: int = 42
    multi_turn_enable: bool = False
    multi_turn_tool_config_path: str | None = None
    multi_turn_max_assistant_turns: int | None = None
    multi_turn_max_user_turns: int | None = None
    multi_turn_max_parallel_calls: int = 1
    multi_turn_max_tool_response_length: int = 256
    multi_turn_tool_response_truncate_side: str = "middle"
    multi_turn_format: str = "hermes"
    multi_turn_tokenization_sanity_check_mode: str = "strict"


@dataclass(frozen=True)
class RolloutCorrectionConfig:
    rollout_is: str = "token"
    rollout_is_threshold: float = 5.0
    rollout_rs: str | None = "null"
    bypass_mode: bool = False


@dataclass(frozen=True)
class WorkerPoolPlacementConfig:
    process_on_nodes: list[int] | None = None
    n_gpus_per_node: int | None = None
    nnodes: int | None = None


@dataclass(frozen=True)
class WorkerPlacementConfig:
    separate_ref_policy: bool = False
    actor_rollout: WorkerPoolPlacementConfig = field(default_factory=WorkerPoolPlacementConfig)
    ref_policy: WorkerPoolPlacementConfig = field(default_factory=WorkerPoolPlacementConfig)


@dataclass(frozen=True)
class AuditConfig:
    enabled: bool = False
    output_dir: str = "mopd_audit"
    domains: list[str] = field(default_factory=lambda: ["math", "code"])
    tensorboard_prefix: str = "mopd"
    tensorboard_layout: str = "domain_category"
    tensorboard_prune_mode: str = "none"
    loss_variance_signal: str = "opd_loss_token"
    max_samples_per_domain: int | None = None
    high_variance_cv_threshold: float = 1.0
    log_sample_level: bool = True
    log_sample_level_freq_steps: int = 1
    response_level_enabled: bool = False
    response_level_freq_steps: int = 1
    response_level_compression: str = "gzip"
    log_validation_metrics: bool = True
    log_validation_metrics_freq_steps: int = 1
    tier2_window_size: int = 20
    calibration_bins: int = 10
    full_gradient_enabled: bool = False
    full_gradient_freq_steps: int = 1
    full_grad_training_parity_freq_steps: int = 1
    full_grad_training_parity_rel_l2_threshold: float = 1e-5
    full_gradient_train_max_samples_per_domain: int | None = None
    full_gradient_micro_batch_size_per_gpu: int = 1
    full_gradient_storage_dtype: str = "float32"
    execution_timing: str = "pre_update"
    full_gradient_direct_recompute_enabled: bool = True
    sequence_masked_target_enabled: bool = False
    sequence_masked_target_use_as_primary: bool = False
    sequence_replay_skip_non_target_domains: bool = False
    sequence_masked_target_closure_rel_l2_threshold: float = 0.02
    sample_gradient_enabled: bool = False
    sample_gradient_freq_steps: int = 1
    sample_gradient_norm_enabled: bool = True
    sample_gradient_cos_enabled: bool = False
    sample_gradient_cos_freq_steps: int = 1
    sample_gradient_backward_recompute_enabled: bool = True
    sample_gradient_backward_sync_enabled: bool = True
    sample_gradient_log_sample_level: bool = True
    sample_gradient_log_sample_level_freq_steps: int = 1
    full_gradient_offload_domain_gradients: bool = True
    token_gap_enabled: bool = True
    token_gap_freq_steps: int = 1
    token_gap_vocab_vector_enabled: bool = False
    token_gap_vocab_vector_freq_steps: int = 1
    token_gap_vocab_size: int | None = None
    vocab_per_occurrence_mean_vector_enabled: bool = True
    logp_vocab_per_occurrence_mean_vector_enabled: bool | None = None
    logp_abs_vocab_per_occurrence_mean_vector_enabled: bool | None = None
    entropy_vocab_per_occurrence_mean_vector_enabled: bool | None = None
    entropy_enabled: bool = True
    entropy_freq_steps: int = 1
    entropy_vocab_vector_enabled: bool = False
    entropy_vocab_vector_freq_steps: int = 1
    topk_teacher_student_cross_entropy_vocab_enabled: bool = False
    topk_teacher_student_cross_entropy_vocab_freq_steps: int = 1
    topk_teacher_student_cross_entropy_k: int = 32
    topk_teacher_student_cross_entropy_include_tail: bool = False
    topk_teacher_student_cross_entropy_temperature: float = 1.0
    logp_vector_enabled: bool = False
    logp_vector_freq_steps: int = 1
    logp_abs_vector_enabled: bool = False
    logp_abs_vector_freq_steps: int = 1
    token_gradient_enabled: bool = False
    token_gradient_freq_steps: int = 10
    token_gradient_tail_enabled: bool = True
    token_gradient_tail_fraction: float = 0.10
    token_gradient_tail_min_tokens: int = 1
    token_gradient_gap_selection_enabled: bool = True
    token_gradient_gap_abs_selection_enabled: bool = True
    token_gradient_loss_abs_selection_enabled: bool = True
    token_gradient_top_k: int | None = 100
    token_gradient_top_p_enabled: bool = False
    token_gradient_top_p: float = 0.10
    token_gradient_log_tokens_jsonl_enabled: bool = True
    token_gradient_strict_grad_restore: bool = False
    token_gradient_backward_recompute_enabled: bool = True
    token_gradient_backward_sync_enabled: bool = True
    dynamic_domain_loss_weighting_enabled: bool = False
    dynamic_domain_loss_weighting_freq_steps: int = 10
    dynamic_domain_loss_weighting_signal_source: str = "gradient_norm"
    dynamic_domain_loss_weighting_ema_beta: float = 0.90
    dynamic_domain_loss_weighting_weight_ema_beta: float = 0.90
    dynamic_domain_loss_weighting_alpha: float = 0.50
    dynamic_domain_loss_weighting_min: float = 1.0 / 3.0
    dynamic_domain_loss_weighting_max: float = 3.0
    control_token_loss_weighting_enabled: bool = False
    control_token_loss_weight: float = 1.0
    control_token_ids: list[int] = field(default_factory=list)
    domain_control_token_ids: dict[str, list[int]] = field(default_factory=dict)
    control_token_normalize_per_domain: bool = False
    control_token_phase_gate_enabled: bool = False
    control_token_span_weighting_enabled: bool = False
    control_token_phase_gate_window_steps: int = 5
    control_token_phase_gate_ema_beta: float = 0.90
    control_token_phase_gate_temperature: float = 0.10
    control_token_phase_gate_initial: float = 0.80
    control_token_span_length: int = 16
    control_token_span_decay_tau: float = 8.0
    control_token_speed_weighting_enabled: bool = False
    control_token_speed_window_steps: int = 5
    control_token_speed_ema_beta: float = 0.80
    control_token_speed_update_interval_steps: int = 2
    control_token_speed_initial_weight: float = 3.0
    control_token_speed_min_occurrences: int = 128
    control_token_speed_weight_knots: list[list[float]] = field(
        default_factory=lambda: [
            [-0.0025, 0.0],
            [0.0, 0.2],
            [0.005, 2.0],
            [0.010, 3.0],
            [0.015, 4.0],
        ]
    )
    all_domain_shared_token_loss_weighting_enabled: bool = False
    all_domain_shared_token_loss_weight: float = 1.0
    all_domain_shared_token_selection_mode: str = "per_step_mean_abs_loss"
    all_domain_shared_token_top_k: int | None = 100
    gradient_fingerprint_enabled: bool = False
    gradient_fingerprint_freq_steps: int = 1


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
    seed: int = 42
    logger: str = '["console","wandb"]'
    n_gpus_per_node: int = 8
    nnodes: int = 1
    save_freq: int = 50
    default_local_dir: str = "checkpoints/Qwen3-4B-Non-Thinking-Multi-Teacher-Distill-ExOPD"
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
    wandb_entity: str | None = None
    wandb_run_id: str | None = None
    wandb_resume: str | None = None
    env_file: str | None = None
    used_model: str = "no_api"
    slurm_allocation_gpus: int | None = None
    cuda_visible_devices: str | None = None

    def __post_init__(self) -> None:
        if self.wandb_run_id is not None:
            if not self.wandb_run_id.strip():
                raise ValueError("runtime.wandb_run_id must be non-empty.")
            if len(self.wandb_run_id) > 64:
                raise ValueError(
                    "runtime.wandb_run_id must not exceed 64 characters."
                )
        if self.wandb_resume not in {None, "allow", "must", "never", "auto"}:
            raise ValueError(
                "runtime.wandb_resume must be one of: allow, must, never, "
                "auto, or null."
            )
        if self.wandb_resume == "must" and self.wandb_run_id is None:
            raise ValueError(
                "runtime.wandb_run_id is required when "
                "runtime.wandb_resume=must."
            )
        if (
            self.slurm_allocation_gpus is not None
            and self.slurm_allocation_gpus <= 0
        ):
            raise ValueError("runtime.slurm_allocation_gpus must be positive.")
        if self.cuda_visible_devices is not None:
            gpu_ids = [part.strip() for part in self.cuda_visible_devices.split(",")]
            if not gpu_ids or any(not part.isdigit() for part in gpu_ids):
                raise ValueError(
                    "runtime.cuda_visible_devices must be comma-separated "
                    "non-negative integer GPU IDs."
                )
            if len(gpu_ids) != len(set(gpu_ids)):
                raise ValueError(
                    "runtime.cuda_visible_devices must not contain duplicate GPU IDs."
                )


@dataclass(frozen=True)
class MOPDConfig:
    data: DataConfig
    model: ModelConfig
    actor: ActorConfig = field(default_factory=ActorConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)
    worker_placement: WorkerPlacementConfig = field(default_factory=WorkerPlacementConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    domain_budgeting: DomainBudgetingConfig = field(default_factory=DomainBudgetingConfig)
    paper_eval: PaperEvalConfig = field(default_factory=PaperEvalConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    ray_kwargs: RayKwargsConfig = field(default_factory=RayKwargsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    extra_overrides: list[str] = field(default_factory=list)


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


def _float_mapping(value: Any, key: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    output: dict[str, float] = {}
    for item_key, item_value in value.items():
        numeric = float(item_value)
        if numeric <= 0:
            raise ValueError(f"Expected '{key}.{item_key}' to be positive.")
        output[str(item_key)] = numeric
    return output


def _string_mapping(value: Any, key: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    output: dict[str, str] = {}
    for item_key, item_value in value.items():
        if item_value is None:
            continue
        output[str(item_key)] = str(item_value)
    return output


def _string_list_mapping(value: Any, key: str) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    output: dict[str, list[str]] = {}
    for item_key, item_value in value.items():
        output[str(item_key)] = _string_list(item_value, f"{key}.{item_key}")
    return output


def _optional_positive_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"Expected '{key}' to be positive.")
    return numeric


def _optional_positive_int_list(value: Any, key: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Expected '{key}' to be a list of positive integers or null.")
    if not value:
        raise ValueError(f"Expected '{key}' to be non-empty when provided.")
    output: list[int] = []
    for index, item in enumerate(value):
        numeric = _optional_positive_int(item, f"{key}[{index}]")
        if numeric is None:
            raise ValueError(f"Expected '{key}[{index}]' to be a positive integer.")
        output.append(numeric)
    return output


def _worker_pool_placement(value: Any, key: str) -> WorkerPoolPlacementConfig:
    raw = _expect_mapping(value or {}, key)
    return WorkerPoolPlacementConfig(
        process_on_nodes=_optional_positive_int_list(raw.get("process_on_nodes"), f"{key}.process_on_nodes"),
        n_gpus_per_node=_optional_positive_int(raw.get("n_gpus_per_node"), f"{key}.n_gpus_per_node"),
        nnodes=_optional_positive_int(raw.get("nnodes"), f"{key}.nnodes"),
    )


def _worker_placement(value: Any) -> WorkerPlacementConfig:
    raw = _expect_mapping(value or {}, "worker_placement")
    return WorkerPlacementConfig(
        separate_ref_policy=bool(raw.get("separate_ref_policy", WorkerPlacementConfig.separate_ref_policy)),
        actor_rollout=_worker_pool_placement(raw.get("actor_rollout"), "worker_placement.actor_rollout"),
        ref_policy=_worker_pool_placement(raw.get("ref_policy"), "worker_placement.ref_policy"),
    )


def _optional_string(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"Expected '{key}' to be a string or null.")


def _same_model_path(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(os.path.normpath(str(right)))


def load_config(path: str | Path) -> MOPDConfig:
    root = load_raw_config(path)

    data_raw = _expect_mapping(root.get("data", {}), "data")
    model_raw = _expect_mapping(root.get("model", {}), "model")
    paper_eval_raw = _expect_mapping(root.get("paper_eval", {}), "paper_eval")
    domain_budgeting_raw = _expect_mapping(
        root.get("domain_budgeting", {}), "domain_budgeting"
    )
    domain_train_files = _string_list_mapping(data_raw.get("domain_train_files"), "data.domain_train_files")
    train_files = (
        _string_list(data_raw.get("train_files"), "data.train_files")
        if data_raw.get("train_files") is not None
        else [file_path for files in domain_train_files.values() for file_path in files]
    )
    if not train_files:
        raise ValueError("Expected 'data.train_files' or 'data.domain_train_files' to contain at least one file.")

    data = DataConfig(
        train_files=train_files,
        val_files=_string_list(data_raw.get("val_files"), "data.val_files"),
        domain_train_files=domain_train_files,
        domain_sampling_weights=_float_mapping(
            data_raw.get("domain_sampling_weights"), "data.domain_sampling_weights"
        ),
        domain_sampling_replacement=bool(
            data_raw.get("domain_sampling_replacement", DataConfig.domain_sampling_replacement)
        ),
        load_parquet_direct=bool(data_raw.get("load_parquet_direct", DataConfig.load_parquet_direct)),
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
        need_tools_kwargs=bool(data_raw.get("need_tools_kwargs", False)),
        dataloader_num_workers=(
            None
            if data_raw.get("dataloader_num_workers") is None
            else int(data_raw["dataloader_num_workers"])
        ),
    )
    primary_teacher_raw = model_raw.get(
        "primary_teacher_path",
        model_raw.get("reasoning_teacher_path", model_raw.get("math_teacher_path")),
    )
    if primary_teacher_raw is None:
        raise ValueError(
            "Expected 'model.primary_teacher_path', 'model.reasoning_teacher_path', "
            "or 'model.math_teacher_path'."
        )
    secondary_teacher_raw = model_raw.get(
        "secondary_teacher_path",
        model_raw.get("code_teacher_path", primary_teacher_raw),
    )
    code_teacher_raw = model_raw.get(
        "code_teacher_path",
        secondary_teacher_raw if secondary_teacher_raw is not None else primary_teacher_raw,
    )
    if _same_model_path(secondary_teacher_raw, primary_teacher_raw):
        secondary_teacher_raw = (
            None if _same_model_path(code_teacher_raw, primary_teacher_raw) else code_teacher_raw
        )
    teacher_model_device = str(model_raw.get("teacher_model_device", "cpu")).lower()
    if teacher_model_device == "cuda":
        teacher_model_device = "gpu"
    if teacher_model_device not in {"cpu", "gpu"}:
        raise ValueError("Expected model.teacher_model_device to be one of: 'cpu', 'gpu', or 'cuda'.")
    attn_implementation_raw = model_raw.get(
        "attn_implementation",
        ModelConfig.attn_implementation,
    )
    if not isinstance(attn_implementation_raw, str) or not attn_implementation_raw.strip():
        raise ValueError("Expected model.attn_implementation to be a non-empty string.")
    attn_implementation = attn_implementation_raw.strip()
    math_teacher_path = str(model_raw.get("math_teacher_path", primary_teacher_raw))
    code_teacher_path = str(code_teacher_raw)
    if_teacher_raw = model_raw.get("if_teacher_path")
    if_teacher_path = None if if_teacher_raw is None else str(if_teacher_raw)
    domain_teacher_paths = _string_mapping(
        model_raw.get("domain_teacher_paths", model_raw.get("teacher_paths")),
        "model.domain_teacher_paths",
    )
    if not domain_teacher_paths:
        domain_teacher_paths = {
            "math": math_teacher_path,
            "code": code_teacher_path,
        }
        if if_teacher_path is not None:
            domain_teacher_paths["if"] = if_teacher_path
    else:
        domain_teacher_paths.setdefault("math", math_teacher_path)
        domain_teacher_paths.setdefault("code", code_teacher_path)
        if if_teacher_path is not None:
            domain_teacher_paths.setdefault("if", if_teacher_path)
    model = ModelConfig(
        student_path=str(model_raw["student_path"]),
        student_base_path=(
            None
            if model_raw.get("student_base_path", model_raw["student_path"]) is None
            else str(model_raw.get("student_base_path", model_raw["student_path"]))
        ),
        math_teacher_path=math_teacher_path,
        code_teacher_path=code_teacher_path,
        if_teacher_path=if_teacher_path,
        domain_teacher_paths=domain_teacher_paths,
        reasoning_teacher_path=(
            None
            if model_raw.get("reasoning_teacher_path") is None
            else str(model_raw["reasoning_teacher_path"])
        ),
        primary_teacher_path=str(primary_teacher_raw),
        secondary_teacher_path=(None if secondary_teacher_raw is None else str(secondary_teacher_raw)),
        teacher_model_device=teacher_model_device,
        attn_implementation=attn_implementation,
    )
    actor = ActorConfig(**_expect_mapping(root.get("actor", {}), "actor"))
    rollout = RolloutConfig(**_expect_mapping(root.get("rollout", {}), "rollout"))
    trainer = TrainerConfig(**_expect_mapping(root.get("trainer", {}), "trainer"))
    audit = AuditConfig(**_expect_mapping(root.get("audit", {}), "audit"))
    domain_budgeting = parse_domain_budgeting_config(domain_budgeting_raw)
    normalized_loss_builder = actor.distill_loss_builder.strip().lower()
    if normalized_loss_builder in {"eopd", "entropy_aware", "entropy_aware_opd"}:
        if actor.eopd_topk_k < 1:
            raise ValueError("actor.eopd_topk_k must be positive for EOPD.")
        if (
            not math.isfinite(actor.eopd_entropy_threshold)
            or actor.eopd_entropy_threshold < 0.0
        ):
            raise ValueError(
                "actor.eopd_entropy_threshold must be finite and non-negative."
            )
        if (
            not math.isfinite(actor.eopd_forward_kl_weight)
            or actor.eopd_forward_kl_weight < 0.0
        ):
            raise ValueError(
                "actor.eopd_forward_kl_weight must be finite and non-negative."
            )
    if not 0.0 < audit.token_gradient_tail_fraction <= 1.0:
        raise ValueError(
            "audit.token_gradient_tail_fraction must be in (0, 1]."
        )
    if audit.token_gradient_tail_min_tokens < 1:
        raise ValueError(
            "audit.token_gradient_tail_min_tokens must be at least 1."
        )
    if not 0.0 <= audit.token_gradient_top_p <= 1.0:
        raise ValueError(
            "audit.token_gradient_top_p must be in [0, 1]."
        )
    if audit.token_gradient_enabled and (
        audit.token_gradient_tail_enabled
        or audit.token_gradient_top_p_enabled
    ) and not audit.token_gradient_loss_abs_selection_enabled:
        raise ValueError(
            "Loss-ranked token-gradient statistics require "
            "audit.token_gradient_loss_abs_selection_enabled=true."
        )
    if not 0.0 <= audit.dynamic_domain_loss_weighting_ema_beta < 1.0:
        raise ValueError(
            "audit.dynamic_domain_loss_weighting_ema_beta must be in [0, 1)."
        )
    if not 0.0 <= audit.dynamic_domain_loss_weighting_weight_ema_beta < 1.0:
        raise ValueError(
            "audit.dynamic_domain_loss_weighting_weight_ema_beta must be "
            "in [0, 1)."
        )
    if audit.dynamic_domain_loss_weighting_signal_source not in {
        "gradient_norm",
        "domain_gradient_projection_share",
    }:
        raise ValueError(
            "audit.dynamic_domain_loss_weighting_signal_source must be "
            "'gradient_norm' or 'domain_gradient_projection_share'."
        )
    if audit.dynamic_domain_loss_weighting_alpha < 0.0:
        raise ValueError(
            "audit.dynamic_domain_loss_weighting_alpha must be non-negative."
        )
    if (
        audit.dynamic_domain_loss_weighting_min <= 0.0
        or audit.dynamic_domain_loss_weighting_min > 1.0
        or audit.dynamic_domain_loss_weighting_max < 1.0
    ):
        raise ValueError(
            "Dynamic domain loss weight bounds must be positive and contain "
            "1.0."
        )
    if (
        not math.isfinite(audit.control_token_loss_weight)
        or audit.control_token_loss_weight < 0.0
    ):
        raise ValueError(
            "audit.control_token_loss_weight must be finite and non-negative."
        )
    if (
        audit.control_token_loss_weighting_enabled
        and not audit.control_token_ids
        and not audit.domain_control_token_ids
    ):
        raise ValueError(
            "audit.control_token_ids or audit.domain_control_token_ids must "
            "be non-empty when control-token loss weighting is enabled."
        )
    if audit.control_token_ids and audit.domain_control_token_ids:
        raise ValueError(
            "Configure either audit.control_token_ids or "
            "audit.domain_control_token_ids, not both."
        )
    unknown_control_domains = set(audit.domain_control_token_ids) - set(
        audit.domains
    )
    if unknown_control_domains:
        raise ValueError(
            "audit.domain_control_token_ids contains unknown domains: "
            + ", ".join(sorted(unknown_control_domains))
        )
    if any(not token_ids for token_ids in audit.domain_control_token_ids.values()):
        raise ValueError(
            "audit.domain_control_token_ids entries must be non-empty."
        )
    if not 0.0 <= audit.control_token_phase_gate_ema_beta < 1.0:
        raise ValueError(
            "audit.control_token_phase_gate_ema_beta must be in [0, 1)."
        )
    if audit.control_token_phase_gate_window_steps < 1:
        raise ValueError(
            "audit.control_token_phase_gate_window_steps must be positive."
        )
    if audit.control_token_phase_gate_temperature <= 0.0:
        raise ValueError(
            "audit.control_token_phase_gate_temperature must be positive."
        )
    if not 0.0 <= audit.control_token_phase_gate_initial <= 1.0:
        raise ValueError(
            "audit.control_token_phase_gate_initial must be in [0, 1]."
        )
    if audit.control_token_span_length < 0:
        raise ValueError(
            "audit.control_token_span_length must be non-negative."
        )
    if audit.control_token_span_decay_tau <= 0.0:
        raise ValueError(
            "audit.control_token_span_decay_tau must be positive."
        )
    if audit.control_token_speed_window_steps < 1:
        raise ValueError(
            "audit.control_token_speed_window_steps must be positive."
        )
    if not 0.0 <= audit.control_token_speed_ema_beta < 1.0:
        raise ValueError(
            "audit.control_token_speed_ema_beta must be in [0, 1)."
        )
    if audit.control_token_speed_update_interval_steps < 1:
        raise ValueError(
            "audit.control_token_speed_update_interval_steps must be positive."
        )
    if audit.control_token_speed_initial_weight < 0.0:
        raise ValueError(
            "audit.control_token_speed_initial_weight must be non-negative."
        )
    if audit.control_token_speed_min_occurrences < 1:
        raise ValueError(
            "audit.control_token_speed_min_occurrences must be positive."
        )
    if len(audit.control_token_speed_weight_knots) < 2:
        raise ValueError(
            "audit.control_token_speed_weight_knots requires at least two knots."
        )
    prior_speed: float | None = None
    for knot in audit.control_token_speed_weight_knots:
        if len(knot) != 2:
            raise ValueError(
                "Each audit.control_token_speed_weight_knots entry must contain "
                "exactly [speed, weight]."
            )
        speed, weight = float(knot[0]), float(knot[1])
        if not math.isfinite(speed) or not math.isfinite(weight):
            raise ValueError(
                "audit.control_token_speed_weight_knots must be finite."
            )
        if weight < 0.0:
            raise ValueError(
                "audit.control_token_speed_weight_knots weights must be "
                "non-negative."
            )
        if prior_speed is not None and speed <= prior_speed:
            raise ValueError(
                "audit.control_token_speed_weight_knots speeds must be "
                "strictly increasing."
            )
        prior_speed = speed
    if audit.control_token_speed_weighting_enabled and (
        not audit.control_token_loss_weighting_enabled
        or not audit.domain_control_token_ids
    ):
        raise ValueError(
            "Control-token speed weighting requires enabled domain-specific "
            "control-token loss weighting."
        )
    if (
        audit.control_token_speed_weighting_enabled
        and audit.control_token_phase_gate_enabled
    ):
        raise ValueError(
            "Control-token speed weighting and phase gating are mutually "
            "exclusive."
        )
    if audit.control_token_phase_gate_enabled and (
        not audit.control_token_loss_weighting_enabled
        or not audit.domain_control_token_ids
        or audit.control_token_span_length < 1
    ):
        raise ValueError(
            "Control-token phase gating requires enabled domain-specific "
            "control weighting and a positive span length."
        )
    if (
        audit.control_token_span_weighting_enabled
        and not audit.control_token_phase_gate_enabled
    ):
        raise ValueError(
            "Successor-span weighting requires control-token phase gating."
        )
    if audit.all_domain_shared_token_loss_weight < 1.0:
        raise ValueError(
            "audit.all_domain_shared_token_loss_weight must be at least 1.0."
        )
    if audit.all_domain_shared_token_selection_mode not in {
        "per_step_mean_abs_loss",
        "cumulative_abs_loss",
    }:
        raise ValueError(
            "audit.all_domain_shared_token_selection_mode must be "
            "'per_step_mean_abs_loss' or 'cumulative_abs_loss'."
        )
    if (
        audit.all_domain_shared_token_top_k is not None
        and audit.all_domain_shared_token_top_k < 1
    ):
        raise ValueError(
            "audit.all_domain_shared_token_top_k must be null or at least 1."
        )
    if (
        audit.all_domain_shared_token_loss_weighting_enabled
        and len(audit.domains) < 2
    ):
        raise ValueError(
            "All-domain shared-token loss weighting requires at least two "
            "audit domains."
        )
    retired_gradient_modes = [
        name
        for name, enabled in (
            ("audit.sample_gradient_enabled", audit.sample_gradient_enabled),
        )
        if enabled
    ]
    if retired_gradient_modes:
        raise ValueError(
            "The clean domain-gradient rebuild retired nested sample backward "
            f"replay. Disable: {', '.join(retired_gradient_modes)}."
        )

    if data.dataloader_num_workers is not None and data.dataloader_num_workers < 0:
        raise ValueError("data.dataloader_num_workers must be non-negative.")
    validate_domain_budgeting_config(
        domain_budgeting,
        data=data,
        model=model,
        actor=actor,
        rollout=rollout,
        audit=audit,
        trainer=trainer,
    )

    return MOPDConfig(
        data=data,
        model=model,
        actor=actor,
        rollout=rollout,
        rollout_correction=RolloutCorrectionConfig(
            **_expect_mapping(root.get("rollout_correction", {}), "rollout_correction")
        ),
        worker_placement=_worker_placement(root.get("worker_placement", {})),
        audit=audit,
        domain_budgeting=domain_budgeting,
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
        trainer=trainer,
        ray_kwargs=RayKwargsConfig(
            ray_init=RayInitConfig(**_expect_mapping(root.get("ray_kwargs", {}).get("ray_init", {}), "ray_init"))
        ),
        runtime=RuntimeConfig(**_expect_mapping(root.get("runtime", {}), "runtime")),
        extra_overrides=_string_list(root.get("extra_overrides", []), "extra_overrides"),
    )
