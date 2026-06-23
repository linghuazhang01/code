# Multi-Teacher OPD Math + Code Training

本目录是当前 OPD/MOPD 训练入口。训练 runtime 从本仓库的 `third_party/verl` 导入，不再依赖远端额外的独立 `G-OPD` checkout。

## 路径约定

- 代码目录：`OPD-code/`
- 数据目录：`OPD-code/data/G-OPD-Training-Data/`
- vendored verl：`OPD-code/third_party/verl/`
- 模型目录：`OPD-code/../models/`
- 日志目录：`OPD-code/logs/`
- checkpoint 目录：`OPD-code/checkpoints/`
- audit 目录：`OPD-code/audit/`

## 配置文件

现在保留三个正式 MOPD 版本，每个版本提供 2/4/8 卡配置；另外保留指标 smoke profile：

| 配置 | 用途 |
| --- | --- |
| `configs/mopd_formal_audit_all_2gpu.yaml` | 2 卡正式 4B math/code 训练，启用全部 MOPD audit。 |
| `configs/mopd_formal_audit_all_4gpu.yaml` | 4 卡 all-audit 训练，同 objective，放大全局 batch。 |
| `configs/mopd_formal_audit_all_8gpu.yaml` | 8 卡 all-audit 训练，TP=4，并保留两个 rollout data-parallel group。 |
| `configs/mopd_formal_audit_loss_only_2gpu.yaml` | 2 卡 all-audit 训练，但 token-gradient selection 只用 loss magnitude。 |
| `configs/mopd_formal_audit_loss_only_4gpu.yaml` | 4 卡 loss-only token-gradient audit 训练。 |
| `configs/mopd_formal_audit_loss_only_8gpu.yaml` | 8 卡 loss-only token-gradient audit 训练。 |
| `configs/mopd_formal_audit_off_2gpu.yaml` | 2 卡同模型/数据/objective，关闭全部 audit。 |
| `configs/mopd_formal_audit_off_4gpu.yaml` | 4 卡 audit-off 训练。 |
| `configs/mopd_formal_audit_off_8gpu.yaml` | 8 卡 audit-off 训练。 |
| `configs/mopd_formal_audit_all_smoke.yaml` | 2 卡 one-step 指标 smoke，打开全部 audit 输出和 full-vocab vectors。 |
| `configs/mopd_formal_audit_loss_only_smoke.yaml` | 2 卡 one-step smoke，token-gradient selection 只用 loss magnitude。 |

所有配置共同使用：

- student: `../models/Qwen3-4B`
- math teacher: `../models/Qwen3-4B-Non-Thinking-RL-Math-Step500`
- code teacher: `../models/Qwen3-4B-Non-Thinking-RL-Code-Step300`
- train files: `DeepMath-103K/train_filtered_level6.parquet` 与 `Eurus/code_train.parquet`
- teacher top-k local-support distillation，`topk_distill_k=32`

卡数 scaling：

| GPU 数 | 配置后缀 | train/mini batch | rollout TP | Ray CPUs |
| --- | --- | --- | --- | --- |
| 2 | `_2gpu` | 256 | 2 | 8 |
| 4 | `_4gpu` | 512 | 4 | 16 |
| 8 | `_8gpu` | 1024 | 4 | 32 |

`mopd_formal_audit_all_*gpu.yaml` 额外打开：

- sample-level 与 validation audit rows
- full-gradient audit
- sample-gradient norm 与 sample-to-domain cosine
- token gap occurrence vector 与 full-vocab vector
- teacher/student entropy occurrence vector 与 full-vocab vector
- token conflict attribution
- token-gradient audit，支持 domain-level signed-gap、gap-abs 与 loss top-k/top-p selection

`mopd_formal_audit_loss_only_*gpu.yaml` 保持与 all-audit profile 相同的 audit surface，包括 full/sample gradients、token gap vectors、entropy vectors、token conflict 和 token-gradient 记录。唯一差异是 token-gradient 候选 token selection：`token_gradient_gap_selection_enabled=false`、`token_gradient_gap_abs_selection_enabled=false`、`token_gradient_loss_abs_selection_enabled=true`。

`mopd_formal_audit_off_*gpu.yaml` 设置 `audit.enabled=false`，并显式关闭所有 audit 子开关。

smoke profile 作为指标测试 profile 保留并纳入测试。它们使用 `data.train_batch_size=32`、`actor.ppo_mini_batch_size=32`、`trainer.total_training_steps=1`，但保持正式 `data.max_response_length=16384`，并保留 full-vocab token gap 与 entropy vectors。

## 启动

远端已同步 checkout 中启动：

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_all_2gpu.yaml \
  --run-id mopd_audit_all_2gpu_$(date +%Y%m%d_%H%M%S)
```

启动 audit-off 版本：

```bash
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_off_2gpu.yaml \
  --run-id mopd_audit_off_2gpu_$(date +%Y%m%d_%H%M%S)
```

启动 loss-only token-gradient audit 版本：

```bash
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_loss_only_2gpu.yaml \
  --run-id mopd_audit_loss_only_2gpu_$(date +%Y%m%d_%H%M%S)
```

4/8 卡使用对应 GPU 列表与 YAML：

```bash
GPU_IDS=0,1,2,3 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_all_4gpu.yaml \
  --run-id mopd_audit_all_4gpu_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_all_8gpu.yaml \
  --run-id mopd_audit_all_8gpu_$(date +%Y%m%d_%H%M%S)
```

本地 dry-run：

```bash
scripts/run_mopd.sh configs/mopd_formal_audit_all_2gpu.yaml --dry-run
```

指标 smoke：

```bash
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_all_smoke.yaml \
  --run-id mopd_metrics_smoke_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_loss_only_smoke.yaml \
  --run-id mopd_metrics_loss_only_smoke_$(date +%Y%m%d_%H%M%S)
```

只同步不启动：

```bash
ASSUME_YES=1 bash scripts/sync_and_start_remote_mopd.sh --sync-only
```

同步并启动：

```bash
ASSUME_YES=1 bash scripts/sync_and_start_remote_mopd.sh \
  configs/mopd_formal_audit_all_2gpu.yaml \
  --run-id mopd_audit_all_2gpu_$(date +%Y%m%d_%H%M%S)
```

## Audit 文件

使用 `mopd_formal_audit_all_*gpu.yaml` 或 `mopd_formal_audit_loss_only_*gpu.yaml` 时，JSONL audit 文件写入对应目录，例如 `audit/formal_audit_all_2gpu/` 或 `audit/formal_audit_loss_only_2gpu/`。

重点文件包括：

- `domain_step_metrics.jsonl`
- `loss_variance_domain_step.jsonl`
- `loss_variance_sample.jsonl`
- `token_gap_vectors.jsonl`
- `token_gap_vocab_vectors.jsonl`
- `entropy_distribution_vectors.jsonl`
- `entropy_vocab_vectors.jsonl`
- `token_conflict_attribution.jsonl`
- `token_grad_metrics.jsonl`
- `sample_grad_metrics.jsonl`
- `validation_probe.jsonl`
- `validation_gain_variance.jsonl`
- `training_cost.jsonl`
- `audit_errors.jsonl`

full-vocab vector 文件使用 token-id 坐标：第 `v` 维对应 tokenizer token id `v`。`token_gap_vocab_vectors.jsonl` 保存 signed/absolute log-prob gap 的 sum 和 mean vector；`entropy_vocab_vectors.jsonl` 保存 `student_entropy` 与 `teacher_student_cross_entropy` 的 sum 和 mean vector。

详细 metric 定义见 [metrics_zh.md](metrics_zh.md)。配置字段和常用 override 见 [CONFIG_GUIDE.zh.md](CONFIG_GUIDE.zh.md)。
