# OPD / MOPD 配置说明

本文档说明当前保留的正式配置矩阵。训练入口仍是 `scripts/run_mopd.sh` 或 `scripts/start_remote_mopd_training.sh`；YAML 会由 `mopd_verl/launch.py` 转换成 `verl.trainer.main_ppo` 的 Hydra overrides。

## 配置文件总览

| 配置 | 适用场景 | 主要特点 |
| --- | --- | --- |
| `configs/mopd_formal_audit_all_2gpu.yaml` | 2 卡正式诊断训练 | 4B student，math/code 4B teachers，teacher top-k distillation，所有 audit 开启 |
| `configs/mopd_formal_audit_all_4gpu.yaml` | 4 卡正式诊断训练 | 同 objective，batch 按卡数放大 |
| `configs/mopd_formal_audit_all_8gpu.yaml` | 8 卡正式诊断训练 | TP=4，8 卡 batch |
| `configs/mopd_formal_audit_off_2gpu.yaml` | 2 卡无 audit 训练 | 同样的模型、数据和 objective，关闭所有 MOPD audit 输出 |
| `configs/mopd_formal_audit_off_4gpu.yaml` | 4 卡无 audit 训练 | 同 objective，batch 按卡数放大 |
| `configs/mopd_formal_audit_off_8gpu.yaml` | 8 卡无 audit 训练 | TP=4，8 卡 batch |
| `configs/mopd_formal_audit_all_smoke.yaml` | 指标 smoke 测试 | 2 卡 one-step，保持正式 response 长度，所有 audit 与 full-vocab vector 开启 |

卡数 scaling：

- `data.max_prompt_length=2048`
- `data.max_response_length=16384`

| GPU 数 | 配置后缀 | `trainer.n_gpus_per_node` | `rollout.tensor_model_parallel_size` | `data.train_batch_size` | `actor.ppo_mini_batch_size` | `ray_kwargs.ray_init.num_cpus` |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | `_2gpu` | 2 | 2 | 256 | 256 | 8 |
| 4 | `_4gpu` | 4 | 4 | 512 | 512 | 16 |
| 8 | `_8gpu` | 8 | 4 | 1024 | 1024 | 32 |

指标 smoke profile 使用独立设置：`trainer.n_gpus_per_node=2`、`rollout.tensor_model_parallel_size=2`、`data.train_batch_size=32`、`actor.ppo_mini_batch_size=32`、`trainer.total_training_steps=1`；response 长度保持正式配置的 `data.max_response_length=16384`。

## 模型与数据

```yaml
model:
  student_path: ../models/Qwen3-4B
  math_teacher_path: ../models/Qwen3-4B-Non-Thinking-RL-Math-Step500
  code_teacher_path: ../models/Qwen3-4B-Non-Thinking-RL-Code-Step300
```

```yaml
data:
  domain_train_files:
    math:
      - data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet
    code:
      - data/G-OPD-Training-Data/Eurus/code_train.parquet
  domain_sampling_weights:
    math: 0.5
    code: 0.5
```

## 蒸馏目标

当前保留配置默认使用 teacher top-k local-support distillation：

```yaml
actor:
  distill_mode: chosen_token_reverse_kl
  topk_distill_enabled: true
  topk_distill_support_source: teacher
  topk_distill_kl_direction: reverse
  topk_distill_k: 5
  topk_distill_tail_bucket: false
```

保留 teacher top-k 是因为真实训练中的 `teacher_student_cross_entropy` 指标来自该 local-support CE 路径；关闭 top-k 时 CE vector 不一定可用。

## Audit All

`configs/mopd_formal_audit_all_*gpu.yaml` 打开所有 audit family：

```yaml
audit:
  enabled: true
  output_dir: audit/formal_audit_all_<gpu>
  log_sample_level: true
  log_validation_metrics: true
  full_gradient_enabled: true
  sample_gradient_enabled: true
  sample_gradient_norm_enabled: true
  sample_gradient_cos_enabled: true
  token_gap_enabled: true
  token_gap_vocab_vector_enabled: true
  entropy_enabled: true
  entropy_vocab_vector_enabled: true
  token_conflict_enabled: true
  token_gradient_enabled: true
  token_gradient_gap_selection_enabled: true
  token_gradient_gap_abs_selection_enabled: true
  token_gradient_loss_abs_selection_enabled: true
  token_gradient_top_k: 100
  token_gradient_top_p: 0.10
```

它还设置：

```yaml
actor:
  use_dynamic_bsz: true
rollout:
  gpu_memory_utilization: 0.6
```

这给 full/sample/token gradient audit 留出更多 actor backward 显存余量。

## 指标 Smoke

`configs/mopd_formal_audit_all_smoke.yaml` 用于快速验证 TensorBoard scalar、JSONL audit 文件、full-vocab token gap vector 和 entropy vector 的记录逻辑。它保持 all-audit 开关：

```yaml
audit:
  enabled: true
  output_dir: audit/formal_audit_all_smoke
  full_gradient_enabled: true
  sample_gradient_enabled: true
  sample_gradient_cos_enabled: true
  token_gap_vocab_vector_enabled: true
  token_gap_vocab_size: null
  entropy_vocab_vector_enabled: true
  token_gradient_enabled: true
  token_gradient_gap_selection_enabled: true
  token_gradient_gap_abs_selection_enabled: true
  token_gradient_loss_abs_selection_enabled: true
  token_gradient_top_k: 100
  token_gradient_top_p: 0.10
```

其中 `token_gap_vocab_size: null` 表示使用 tokenizer 的完整词表维度，不是压缩到小词表的假 smoke。

## Audit Off

`configs/mopd_formal_audit_off_*gpu.yaml` 保持同样的训练 objective，但关闭所有 audit：

```yaml
audit:
  enabled: false
  output_dir: audit/formal_audit_off_<gpu>
  log_sample_level: false
  log_validation_metrics: false
  full_gradient_enabled: false
  sample_gradient_enabled: false
  sample_gradient_norm_enabled: false
  sample_gradient_cos_enabled: false
  token_gap_enabled: false
  token_gap_vocab_vector_enabled: false
  entropy_enabled: false
  entropy_vocab_vector_enabled: false
  token_conflict_enabled: false
  token_gradient_enabled: false
  token_gradient_gap_selection_enabled: true
  token_gradient_gap_abs_selection_enabled: true
  token_gradient_loss_abs_selection_enabled: true
  token_gradient_top_k: 100
  token_gradient_top_p: 0.10
```

## 常用启动

```bash
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_all_2gpu.yaml \
  --run-id mopd_audit_all_2gpu_$(date +%Y%m%d_%H%M%S)
```

```bash
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_off_2gpu.yaml \
  --run-id mopd_audit_off_2gpu_$(date +%Y%m%d_%H%M%S)
```

```bash
GPU_IDS=0,1,2,3 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_all_4gpu.yaml \
  --run-id mopd_audit_all_4gpu_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_all_8gpu.yaml \
  --run-id mopd_audit_all_8gpu_$(date +%Y%m%d_%H%M%S)
```

指标 smoke 测试直接使用维护中的 smoke YAML：

```bash
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_all_smoke.yaml \
  --run-id mopd_metrics_smoke_$(date +%Y%m%d_%H%M%S)
```

详细 metric 口径见 [metrics_zh.md](metrics_zh.md)。
