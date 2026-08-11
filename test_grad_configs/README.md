# MOPD GPU regression configs

本目录只保留两类 GPU integration profiles：

1. 当前研究目标：四领域 capability-gap / variance-aware dynamic budgeting；
2. 底层训练可靠性：FSDP gradient synchronization 与 audit isolation。

旧的 audit-based dynamic loss weighting、teacher-prefix feature coverage、
Science-only forward-KL 和 Top-p token-gradient profiles 已移除。它们不是当前
`q = p * lambda` 方法的一部分，相应算子语义继续由 unit/contract tests 覆盖。

## 模型与部署

- Student：`/root/autodl-tmp/models/Qwen3-0.6B`
- Shared teacher：`/root/autodl-tmp/models/Qwen3-8B`
- 3 GPU smoke：2 张 actor/rollout GPU + 1 张 teacher/ref GPU

## 当前动态预算 smoke

配置：
`mopd_dynamic_budget_qwen0p6b_8b_aw2_fsdp2_b16_4step_3gpu_smoke.yaml`

它在 Math、Code、Science、Instruction Following 四个 domain 上执行 4 steps：

- fixed-probe capability gap 及 time-series EMA 决定 objective contribution `q_d`；
- sequence-mean OPD loss variance 决定下一批数据比例 `p_d`；
- 当前 batch 的有效样本比例决定 `lambda_d = q_d / p_active,d`；
- batch size 为 16，每个 domain 至少 2 条样本；
- 每步 validation 和 variance update，step 2/4 保存 checkpoint；
- 关闭旧的 `audit.dynamic_domain_loss_weighting_enabled`，避免两套 controller
  同时修改 loss。

直接 dry-run：

```bash
bash test_grad_configs/start.sh \
  test_grad_configs/mopd_dynamic_budget_qwen0p6b_8b_aw2_fsdp2_b16_4step_3gpu_smoke.yaml \
  --dry-run
```

GPU smoke 中的 `teacher_scores: 1.0` 是 reward ceiling，只用于验证端到端
plumbing，因此该配置可直接执行。它不能用于报告模型质量或论文结果。正式实验必须
使用 `configs/` 下的四领域 dynamic-budget profile，并填入同一 fixed probe set 上
实测的 teacher scores。

## FSDP reliability matrix

Matrix：`mopd_grad_reliability_qwen0p6b_8b_matrix.yaml`，使用
`<matrix.yaml>::<profile>` 引用。

| Profile | Actor world | `fsdp_size` | Audit | GPU | 用途 |
| --- | ---: | ---: | --- | ---: | --- |
| `aw2_fsdp1_audit_on` | 2 | 1 | on | 3 | `NO_SHARD` replica gradient synchronization |
| `aw2_fsdp1_audit_off` | 2 | 1 | off | 3 | 对照 audit isolation |
| `aw2_fsdp2_audit_on` | 2 | 2 | on | 3 | `FULL_SHARD` gradient regression |
| `aw4_fsdp2_audit_on` | 4 | 2 | on | 5 | `HYBRID_SHARD` topology regression |
| `aw4_fsdp2_audit_off` | 4 | 2 | off | 5 | 对照 audit isolation |

## 执行约束

- 从代码仓库根目录启动。
- Dynamic budgeting 必须保持 `seq-mean-token-mean`、单 PPO epoch、完整
  mini-batch、replacement sampling、`dataloader_num_workers=0`，并关闭 entropy/KL
  auxiliary loss。
- 日志、checkpoint、audit JSONL 与 TensorBoard event 不应提交到本目录。
- 配置变更后运行 `pytest -q tests/test_dynamic_budget_smoke_profile.py
  tests/test_config_profiles.py`。

## SHA-256

```text
7a9cf82873ec25a4adbd02c58337d2b2bc7fcfcc79d3e7145bf5fcbdfddcfd7b  mopd_dynamic_budget_qwen0p6b_8b_aw2_fsdp2_b16_4step_3gpu_smoke.yaml
866d082c0895a29bae3496b1c276b24308a0e59d2416f95ad7c652a291c7b6e5  mopd_grad_reliability_qwen0p6b_8b_matrix.yaml
```
