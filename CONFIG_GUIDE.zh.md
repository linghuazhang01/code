# OPD / MOPD 配置说明

本文档集中说明本仓库的训练配置文件、关键字段和常用 override。训练入口仍是
`scripts/run_mopd.sh` 或 `scripts/start_remote_mopd_training.sh`；YAML 配置先由
`mopd_verl/launch.py` 转换为 `verl.trainer.main_ppo` 的 Hydra overrides。

## 配置文件总览

| 配置 | 适用场景 | 主要特点 |
| --- | --- | --- |
| `configs/mopd_formal_single_a800.yaml` | 单张 A800 正式训练 | 16K response，single-A800 profile，standard audit |
| `configs/mopd_formal_dual_a800.yaml` | 双 A800 诊断训练 | TP=2，full-gradient audit，sample grad norm |
| `configs/mopd_formal_4gpu_a800.yaml` | 4 卡 A800 扩展 | 从 dual-A800 线性扩展，约 128 prompts/GPU |
| `configs/mopd_formal_8gpu_a800.yaml` | 8 卡 A800 扩展 | 两组 TP=4 rollout group |
| `configs/mopd_formal_single_h200.yaml` | 单张 H200 | 更高 vLLM/actor 显存余量，optimizer 常驻 GPU |
| `configs/mopd_formal_dual_a800_pg_loss.yaml` | 双卡蒸馏目标对照 | chosen-token OPD / PG-style loss |
| `configs/mopd_formal_dual_a800_teacher_topk.yaml` | 双卡蒸馏目标对照 | teacher top-k LSM，reverse-KL，`k=5` |
| `configs/mopd_formal_dual_a800_student_topk.yaml` | 双卡蒸馏目标对照 | student top-k LSM，reverse-KL，`k=5` |
| `configs/mopd_general_reasoner.yaml` | General-Reasoner teacher | WebInstruct / reasoning teacher 路径 |
| `configs/mopd_math_code.yaml` | paper-style 两教师训练 | math/code teacher 基础配置 |
| `configs/mopd_audit_smoke.yaml` | 本地/远端 smoke | 小模型/小 batch 的 one-step audit 检查 |

## 启动方式

远端双卡示例：

```bash
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_dual_a800.yaml \
  --run-id mopd_dual_a800_$(date +%Y%m%d_%H%M%S)
```

本地或已登录远端的直接启动：

```bash
scripts/run_mopd.sh configs/mopd_formal_dual_a800.yaml -- \
  trainer.total_training_steps=1
```

所有临时覆盖项都放在 `--` 后面，使用 VERL/Hydra 路径，例如
`actor_rollout_ref.actor.ppo_mini_batch_size=32`，而不是 YAML 里的短路径
`actor.ppo_mini_batch_size=32`。

## 数据与模型

`data` 段控制训练/验证数据和序列长度：

- `data.domain_train_files`: 按 domain 配置训练 parquet。
- `data.domain_sampling_weights`: math/code 采样比例。
- `data.train_batch_size`: rollout batch size。
- `data.val_batch_size`: validation batch size。
- `data.max_prompt_length`: prompt 截断上限。
- `data.max_response_length`: response 生成上限。
- `data.return_raw_chat`: 保留原始 chat，供 ref retokenization 使用。

`model` 段控制 student 和 teacher 路径：

- `model.student_path`: actor/student 模型。
- `model.math_teacher_path`: math teacher / primary teacher。
- `model.code_teacher_path`: code teacher / secondary teacher。
- `model.student_base_path`: 可选 actor base model。

## Actor / OPD 基础配置

`actor` 段控制训练 loss、batch、FSDP 和 optimizer：

- `actor.learning_rate`: actor learning rate。
- `actor.only_reverse_kl_advantages`: 是否用 reverse-KL 信号替换 advantage。
- `actor.lambda_vals`: G-OPD / ExOPD 校正系数。
- `actor.multi_teacher_distill`: 是否按 sample domain 选择 math/code teacher。
- `actor.ppo_mini_batch_size`: actor update mini-batch size。
- `actor.ppo_micro_batch_size_per_gpu`: 单 GPU backward micro-batch。
- `actor.use_dynamic_bsz`: 按 token 数动态分 micro-batch，重型 top-k/audit 推荐打开。
- `actor.gradient_checkpointing`: actor gradient checkpointing。
- `actor.param_offload` / `actor.optimizer_offload`: FSDP 参数/optimizer offload。
- `actor.fsdp_size`: actor FSDP 分组大小；dual-A800 audit profile 用 `1` 保持 replicated audit coordinates。

## 蒸馏目标配置

当前支持三类训练 objective。

### 1. PG / chosen-token OPD

配置示例：

```yaml
actor:
  distill_mode: chosen_token_reverse_kl
  topk_distill_enabled: false
```

这条路径只对 student rollout 中实际采样到的 token 计算 chosen-token
`teacher_logp - student_logp` / reverse-KL 信号。训练时会有 `actor/pg_loss`。

对应配置：

```text
configs/mopd_formal_dual_a800_pg_loss.yaml
```

### 2. Teacher Top-k LSM

配置示例：

```yaml
actor:
  distill_mode: chosen_token_reverse_kl
  topk_distill_enabled: true
  topk_distill_support_source: teacher
  topk_distill_kl_direction: reverse
  topk_distill_k: 5
  topk_distill_tail_bucket: false
```

这条路径用 teacher 在每个 prefix 下概率最高的 `k` 个 token 作为 local
support。teacher 与当前 student 都在同一 support 内重归一化，再计算 KL。

对应配置：

```text
configs/mopd_formal_dual_a800_teacher_topk.yaml
```

### 3. Student Top-k LSM

配置示例：

```yaml
actor:
  distill_mode: chosen_token_reverse_kl
  topk_distill_enabled: true
  topk_distill_support_source: student
  topk_distill_kl_direction: reverse
  topk_distill_k: 5
  topk_distill_tail_bucket: false
```

这条路径先在 old actor/student 的分布上选 top-k token，得到
`student_topk_ids`。随后 teacher/ref 在这些 ids 上 gather logprob，当前
student 也在同一 support 上计算 logprob，然后做 local-support KL。它要求
recompute old logprob，不能和 rollout-correction bypass mode 同时使用。

对应配置：

```text
configs/mopd_formal_dual_a800_student_topk.yaml
```

## Top-k 相关字段

| 字段 | 含义 |
| --- | --- |
| `actor.topk_distill_enabled` | 是否开启 top-k / local-support distillation |
| `actor.topk_distill_support_source` | support 来源：`teacher` 或 `student` |
| `actor.topk_distill_kl_direction` | KL 方向：`reverse` 或 `forward` |
| `actor.topk_distill_k` | 每个 response position 的 support token 数 |
| `actor.topk_distill_tail_bucket` | 是否使用旧的 non-top-k tail bucket |
| `actor.topk_distill_temperature` | support 分布温度 |
| `actor.topk_distill_loss_weight` | top-k distill loss 权重 |
| `actor.topk_distill_logprob_mode` | 当前建议保持 `sparse` |
| `actor.topk_distill_logprob_chunk_size` | sparse selected-logits 计算 chunk size |

默认实现会在 top-k 模式下跳过 PG loss，因此 `actor/pg_loss=0.0` 是预期的；
训练目标主要看 `actor/topk_distill_loss`。

## Rollout 配置

`rollout` 段控制 vLLM rollout：

- `rollout.tensor_model_parallel_size`: vLLM TP size。
- `rollout.gpu_memory_utilization`: vLLM KV cache 显存比例。
- `rollout.max_num_batched_tokens`: vLLM batch token 上限。
- `rollout.max_num_seqs`: vLLM 并发序列数。
- `rollout.temperature` / `rollout.top_p`: 训练 rollout 采样参数。
- `rollout.val_do_sample` / `rollout.val_temperature` / `rollout.val_top_p`: validation rollout 参数。

重型 top-k/audit 运行时，若 actor backward 显存紧张，可以先降低
`rollout.gpu_memory_utilization`，或者缩短 `data.max_response_length`。

## Teacher Prefix 配置

可选 dataset teacher-prefix / roll-in 支持：

- `rollout.teacher_prefix_sampling_enabled`: 是否开启 teacher prefix。
- `rollout.teacher_prefix_length`: prefix token 数，默认可用 `1024`。
- `rollout.teacher_prefix_dataset_key`: 数据集中 prefix 字段名，默认 `prefix`。
- `actor.teacher_prefix_enabled`: launcher 会根据 rollout 配置自动打开。
- `actor.teacher_prefix_loss_region`: 默认 `suffix_only`，prefix 只作为上下文；若设置为 `prefix_and_suffix`，prefix 也用 forward-KL 训练。

开启后不会在训练中临时调用 teacher/ref policy 采样 prefix；trainer 只读取数据集里的 `rollout.teacher_prefix_dataset_key` 字段，tokenize 后截断到 `rollout.teacher_prefix_length`，拼到 prompt 后让 student 继续生成 suffix。

## Audit 配置

`audit` 段控制诊断指标，不直接改变训练 objective。

### 轻量指标

- `audit.token_gap_enabled`: 记录 `teacher_logp - student_logp` 和绝对值分布；默认写 response-token occurrence vector 到 `token_gap_vectors.jsonl`，并在有 token id 时记录全词表 domain-pair cosine scalar。
- `audit.token_gap_vocab_vector_enabled`: 默认 `false`。开启后额外写全词表 dense vector 到 `token_gap_vocab_vectors.jsonl`，第 `v` 维对应 token id `v`。正式长跑建议低频开启，因为文件体积会明显增加。
- `audit.token_gap_vocab_size`: 可选。通常 trainer 会从 tokenizer 自动推断 vocab size；无 tokenizer 的测试或离线场景可手动设置。
- `audit.entropy_enabled`: 记录 teacher entropy、student entropy、teacher-student cross entropy。
- `audit.token_conflict_enabled`: 记录 token-level teacher/student disagreement 与 top-token rows。

这些指标支持对应的 `*_freq_steps`：

```yaml
audit:
  token_gap_freq_steps: 1
  token_gap_vocab_vector_freq_steps: 10
  entropy_freq_steps: 1
  token_conflict_freq_steps: 1
```

### Gradient audit

- `audit.full_gradient_enabled`: full/domain gradient。
- `audit.sample_gradient_enabled`: sample gradient。
- `audit.sample_gradient_norm_enabled`: sample grad norm。
- `audit.sample_gradient_cos_enabled`: sample-to-domain cosine。
- `audit.token_gradient_enabled`: token gradient audit。
- `audit.token_gradient_top_p`: token-gradient top-p mass 比例，默认 `0.10`。

重型 gradient audit 都可以用 `*_freq_steps` 降低开销。token-gradient 当前会基于
本 step 全局所有 valid response token 的 `gap_abs` 分布选 `top100_gap_abs` 和
top-p mass token 集合。

## 常用 Override

### 小 batch smoke

```bash
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_dual_a800_student_topk.yaml \
  --run-id student_topk_smoke \
  -- \
  data.train_batch_size=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  data.val_batch_size=32 \
  data.max_response_length=1024 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  trainer.total_training_steps=1 \
  trainer.total_epochs=1 \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.val_before_train=false
```

### 切换 teacher / student top-k

```bash
actor_rollout_ref.actor.policy_loss.topk_distill_enabled=true \
actor_rollout_ref.actor.policy_loss.topk_distill_support_source=student \
actor_rollout_ref.actor.policy_loss.topk_distill_kl_direction=reverse \
actor_rollout_ref.actor.policy_loss.topk_distill_k=5
```

把 `topk_distill_support_source` 改成 `teacher` 即切回 teacher top-k。

### 关闭重型 audit

```bash
mopd_audit.log_sample_level=false \
mopd_audit.log_validation_metrics=false \
mopd_audit.full_gradient_enabled=false \
mopd_audit.sample_gradient_enabled=false \
mopd_audit.token_gradient_enabled=false
```

若还要关闭 gap / entropy / token-conflict：

```bash
mopd_audit.token_gap_enabled=false \
mopd_audit.entropy_enabled=false \
mopd_audit.token_conflict_enabled=false
```

### Audit all

```bash
mopd_audit.log_sample_level=true \
mopd_audit.log_validation_metrics=true \
mopd_audit.full_gradient_enabled=true \
mopd_audit.sample_gradient_enabled=true \
mopd_audit.sample_gradient_norm_enabled=true \
mopd_audit.sample_gradient_cos_enabled=true \
mopd_audit.token_gradient_enabled=true \
mopd_audit.full_gradient_freq_steps=1 \
mopd_audit.sample_gradient_freq_steps=1 \
mopd_audit.sample_gradient_cos_freq_steps=1 \
mopd_audit.token_gradient_freq_steps=1
```

## 最近验证状态

已在双 A800 远端完成 `configs/mopd_formal_dual_a800_student_topk.yaml` 的
1-step smoke：

- `data.train_batch_size=32`
- `data.max_response_length=1024`
- `actor.topk_distill_support_source=student`
- `actor.topk_distill_k=5`
- 完成 `global_step=1/1`
- `actor/topk_distill_loss=0.0100997`
- `actor/pg_loss=0.0`
- `timing_s/update_actor=8.22`
- `perf/max_memory_allocated_gb=87.21`

本地日志位于：

```text
temp/remote_logs/codex_student_topk_smoke_20260618_141930.log
temp/remote_logs/codex_student_topk_smoke_20260618_141930_gpu.csv
```
