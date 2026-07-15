# OPD / MOPD 配置说明

本文档说明当前保留的正式配置矩阵。训练入口仍是 `scripts/run_mopd.sh` 或 `scripts/run_local_mopd_training.sh`；YAML 会由 `mopd_verl/launch.py` 转换成 `verl.trainer.main_ppo` 的 Hydra overrides。

## 配置文件总览

| 配置 | 适用场景 | 主要特点 |
| --- | --- | --- |
| `configs/mopd_formal_audit_all_2gpu.yaml` | 2 卡正式诊断训练 | 4B student，math/code 4B teachers，teacher top-k distillation，domain gradient 与 observation metrics |
| `configs/mopd_formal_audit_all_4gpu.yaml` | 4 卡正式诊断训练 | 同 objective，batch 按卡数放大 |
| `configs/mopd_formal_audit_all_6gpu.yaml` | 6 卡正式诊断训练 | TP=2，6 卡 batch，沿用 audit-off 实测显存安全 profile |
| `configs/mopd_formal_audit_all_8gpu.yaml` | 8 卡 OPD 正式诊断训练 | 6 student + 2 teacher 分离部署，policy-gradient objective，audit-only CE/logp vectors |
| `configs/mopd_formal_audit_loss_only_2gpu.yaml` | 2 卡兼容配置 | 保留旧 loss-only 命名；nested token backward 已关闭 |
| `configs/mopd_formal_audit_loss_only_4gpu.yaml` | 4 卡兼容配置 | 同 objective，batch 按卡数放大 |
| `configs/mopd_formal_audit_loss_only_6gpu.yaml` | 6 卡兼容配置 | TP=2，6 卡 batch，`fsdp_size=2` domain-gradient audit |
| `configs/mopd_formal_audit_loss_only_8gpu.yaml` | 8 卡兼容配置 | TP=4，8 卡 batch |
| `configs/mopd_formal_audit_off_2gpu.yaml` | 2 卡无 audit 训练 | 同样的模型、数据和 objective，关闭所有 MOPD audit 输出 |
| `configs/mopd_formal_audit_off_4gpu.yaml` | 4 卡无 audit 训练 | 同 objective，batch 按卡数放大 |
| `configs/mopd_formal_audit_off_6gpu.yaml` | 6 卡无 audit 训练 | TP=2，vLLM memory 0.6，max_num_seqs 24 |
| `configs/mopd_formal_audit_off_8gpu.yaml` | 8 卡无 audit 训练 | TP=4，8 卡 batch |
| `configs/mopd_formal_audit_all_smoke.yaml` | 指标 smoke 测试 | 2 卡 one-step，domain gradient 与 full-vocab observation vectors |
| `configs/mopd_formal_audit_loss_only_smoke.yaml` | 兼容 smoke 测试 | 2 卡 one-step，保留旧 loss-only 输出命名 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math.yaml` | 6 卡单域训练 | 4 actor + 2 teacher，math-only，真正的 HYBRID_SHARD |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_code.yaml` | 6 卡单域训练 | 4 actor + 2 teacher，code-only，真正的 HYBRID_SHARD |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_if.yaml` | 6 卡单域训练 | 4 actor + 2 teacher，IF-only，IFBench validation |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_science.yaml` | 6 卡单域训练 | 4 actor + 2 teacher，science-only，GPQA validation |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code.yaml` | 6 卡双域训练 | 4 actor + 2 teacher，math/code 等权采样 |

五个 FSDP/domain-gradient 回归配置统一位于 `test_grad_configs/`，不再在
`configs/` 下保留副本。

卡数 scaling：

- `data.max_prompt_length=2048`
- `data.max_response_length=16384`，其中 6 卡 compatibility profile 为了降低 audit 峰值改为 `10240`，并显式设置 `rollout.max_model_len=12288`

| GPU 数 | 配置后缀 | `trainer.n_gpus_per_node` | `rollout.tensor_model_parallel_size` | `data.train_batch_size` | `actor.ppo_mini_batch_size` | `ray_kwargs.ray_init.num_cpus` |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | `_2gpu` | 2 | 2 | 256 | 256 | 8 |
| 4 | `_4gpu` | 4 | 4 | 512 | 512 | 16 |
| 6 | `_6gpu` | 6 | 2 | 768 | 768 | 24 |
| 8 | `_8gpu` | 8（标准 profile）/ 6（OPD split profile） | 4（标准 profile）/ 2（OPD split profile） | 1024（标准 profile）/ 768（OPD split profile） | 1024（标准 profile）/ 768（OPD split profile） | 32 |

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

除 8 卡 OPD split profile 外，formal 配置默认使用 teacher top-k local-support distillation：

```yaml
actor:
  distill_mode: chosen_token_reverse_kl
  topk_distill_enabled: true
  topk_distill_support_source: teacher
  topk_distill_kl_direction: reverse
  topk_distill_k: 32
  topk_distill_tail_bucket: false
```

`topk_distill_enabled` 只控制训练 objective 是否使用 teacher top-k distillation。Policy-gradient 配置可以保持该开关关闭，同时通过 audit 的 `topk_teacher_student_cross_entropy_vocab_enabled` 独立收集 teacher/student cross-entropy vocab vector，不改变训练 loss。

8 卡 OPD split profile 使用：

```yaml
actor:
  distill_loss_builder: policy_gradient
  distill_mode: chosen_token_policy_gradient
  topk_distill_enabled: false
  topk_distill_loss_weight: 0.0
```

## Audit All

`configs/mopd_formal_audit_all_*gpu.yaml` 保留 domain-gradient audit 与
无需额外 backward 的 observation metrics。为避免重复同步、污染 `.grad`
以及极高的重放成本，nested sample/token backward 已从当前实现退役：

```yaml
audit:
  enabled: true
  output_dir: audit/formal_audit_all_<gpu>
  log_sample_level: true
  log_validation_metrics: true
  full_gradient_enabled: true
  sample_gradient_enabled: false
  sample_gradient_norm_enabled: true
  sample_gradient_cos_enabled: true
  token_gap_enabled: true
  token_gap_vocab_vector_enabled: true
  vocab_per_occurrence_mean_vector_enabled: true
  logp_vocab_per_occurrence_mean_vector_enabled: null
  logp_abs_vocab_per_occurrence_mean_vector_enabled: null
  entropy_vocab_per_occurrence_mean_vector_enabled: null
  entropy_enabled: true
  entropy_vocab_vector_enabled: true
  topk_teacher_student_cross_entropy_vocab_enabled: true
  logp_vector_enabled: true
  logp_abs_vector_enabled: true
  token_conflict_enabled: true
  token_gradient_enabled: false
  token_gradient_gap_selection_enabled: true
  token_gradient_gap_abs_selection_enabled: true
  token_gradient_loss_abs_selection_enabled: true
  token_gradient_top_k: 100
  token_gradient_top_p: 0.10
```

`logp_vector_enabled` 显式输出 signed gap
`teacher_logp - old_student_logp`，`logp_abs_vector_enabled` 输出其绝对值。
`vocab_per_occurrence_mean_vector_enabled` 是 legacy global 开关，并继续直接控制
token-gap family。`logp_vocab_per_occurrence_mean_vector_enabled`、
`logp_abs_vocab_per_occurrence_mean_vector_enabled` 和
`entropy_vocab_per_occurrence_mean_vector_enabled` 可以分别覆盖 logp、logp_abs 与
entropy family；值为 `null` 或省略时回退 global，显式 `true/false` 时独立生效。
entropy override 同时控制 student entropy 与 teacher-student cross-entropy 的 mean
vector。所有 `*_mean_vector_vocab` 都对每个 token id 使用
`sum / occurrence_count`，未出现 token 的维度保持 0。该统计是当前 step、当前
domain 内的 conditional mean，不是 `count / total_count` 的 token-frequency
probability。

它还设置：

```yaml
actor:
  use_dynamic_bsz: false
rollout:
  gpu_memory_utilization: 0.6
```

这让 full/domain-gradient audit 的统计路径保持固定 micro-batch，避免
dynamic batching 影响 domain-gradient 对比。`sample_gradient_norm_enabled`
等旧字段可能仍保留在 YAML 中，但在 `sample_gradient_enabled: false` 时不会
触发 sample backward；token selector 字段同理。

## Audit Loss Only

`configs/mopd_formal_audit_loss_only_*gpu.yaml` 现在是兼容旧实验名和输出目录
的 aliases。当前实现关闭 nested sample/token backward，因此 selector 字段
不会产生 token gradient；6 卡 profile 继续用 `fsdp_size: 2` 统计 domain
gradient：

```yaml
audit:
  enabled: true
  output_dir: audit/formal_audit_loss_only_<gpu>
  full_gradient_enabled: true
  sample_gradient_enabled: false
  sample_gradient_norm_enabled: true
  sample_gradient_cos_enabled: true
  token_gap_enabled: true
  token_gap_vocab_vector_enabled: true
  vocab_per_occurrence_mean_vector_enabled: true
  entropy_enabled: true
  entropy_vocab_vector_enabled: true
  topk_teacher_student_cross_entropy_vocab_enabled: true
  logp_vector_enabled: true
  logp_abs_vector_enabled: true
  token_conflict_enabled: true
  token_gradient_enabled: false
  token_gradient_gap_selection_enabled: false
  token_gradient_gap_abs_selection_enabled: false
  token_gradient_loss_abs_selection_enabled: true
  token_gradient_top_k: 100
  # 6gpu fsdp=2 profile uses 0.15; other loss-only formal profiles use 0.10.
  token_gradient_top_p: 0.10
```

因此当前配置不会生成 `token_grad_metrics.jsonl`，也不会做 sample/token
级别的额外 backward。`sequence_masked_target_*` 只服务于 domain-gradient
target：

```yaml
audit:
  sequence_masked_target_enabled: true
  sequence_masked_target_use_as_primary: true
```

这一路径不要求每个 worker 拥有完整 local params。不要再用
`token_gradient_top_p` 作为 closure sanity check；当前 closure 应查看
domain-sum、audit-total 和 training-total 的 cosine/relative-L2 指标。

## 指标 Smoke

`configs/mopd_formal_audit_all_smoke.yaml` 用于快速验证 TensorBoard scalar、
domain-gradient JSONL、full-vocab token gap vector 和 entropy vector 的记录逻辑：

```yaml
audit:
  enabled: true
  output_dir: audit/formal_audit_all_smoke
  full_gradient_enabled: true
  sample_gradient_enabled: false
  sample_gradient_cos_enabled: true
  token_gap_vocab_vector_enabled: true
  token_gap_vocab_size: null
  vocab_per_occurrence_mean_vector_enabled: true
  entropy_vocab_vector_enabled: true
  topk_teacher_student_cross_entropy_vocab_enabled: true
  logp_vector_enabled: true
  logp_abs_vector_enabled: true
  token_gradient_enabled: false
  token_gradient_gap_selection_enabled: true
  token_gradient_gap_abs_selection_enabled: true
  token_gradient_loss_abs_selection_enabled: true
  token_gradient_top_k: 100
  token_gradient_top_p: 0.10
```

其中 `token_gap_vocab_size: null` 表示使用 tokenizer 的完整词表维度，不是压缩到小词表的假 smoke。

`configs/mopd_formal_audit_loss_only_smoke.yaml` 使用同样的 one-step smoke
设置并保留旧 selector 元数据，但不会执行 token backward：

```yaml
audit:
  output_dir: audit/formal_audit_loss_only_smoke
  token_gradient_enabled: false
  token_gradient_gap_selection_enabled: false
  token_gradient_gap_abs_selection_enabled: false
  token_gradient_loss_abs_selection_enabled: true
```

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
  topk_teacher_student_cross_entropy_vocab_enabled: false
  logp_abs_vector_enabled: false
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
GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_2gpu.yaml \
  --run-id mopd_audit_all_2gpu_$(date +%Y%m%d_%H%M%S)
```

```bash
GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_off_2gpu.yaml \
  --run-id mopd_audit_off_2gpu_$(date +%Y%m%d_%H%M%S)
```

```bash
GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_loss_only_2gpu.yaml \
  --run-id mopd_audit_loss_only_2gpu_$(date +%Y%m%d_%H%M%S)
```

```bash
GPU_IDS=0,1,2,3 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_4gpu.yaml \
  --run-id mopd_audit_all_4gpu_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1,2,3,4,5 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_6gpu.yaml \
  --run-id mopd_audit_all_6gpu_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_8gpu.yaml \
  --run-id mopd_audit_all_8gpu_$(date +%Y%m%d_%H%M%S)
```

指标 smoke 测试直接使用维护中的 smoke YAML：

```bash
GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_smoke.yaml \
  --run-id mopd_metrics_smoke_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_loss_only_smoke.yaml \
  --run-id mopd_metrics_loss_only_smoke_$(date +%Y%m%d_%H%M%S)
```

详细 metric 口径见 [metrics_zh.md](metrics_zh.md)。
