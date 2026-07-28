# Domain-gradient experiment configs

这里集中保存 GPU domain-gradient regression 与专项 smoke 配置。目录从原先
15 份 expanded YAML 收敛为 5 个 YAML：两个 named-profile matrix 和三个独立
integration configs，共保留 12 个 canonical experiment profiles。

Matrix profile 使用 `<matrix.yaml>::<profile>` 引用。普通单配置 YAML 的启动方式
保持不变。

## 模型配置

- Student：`/root/autodl-tmp/models/Qwen3-0.6B`
- Teacher：`/root/autodl-tmp/models/Qwen3-8B`
- Math、Code、Science 的 teacher routing 均指向同一个 Qwen3-8B
  checkpoint；Teacher 与 Student 不再共享模型参数。

## 实验矩阵

Matrix：`mopd_grad_reliability_qwen0p6b_8b_matrix.yaml`

| Profile | Actor world | `fsdp_size` | FSDP1 strategy | Replica count | Audit | 所需 GPU | 用途 |
|---|---:|---:|---|---:|---|---:|---|
| `aw2_fsdp1_audit_on` | 2 | 1 | `NO_SHARD` replication | 2 | on | 3 | 验证 singleton shard group、跨 replica gradient 同步及完整 audit |
| `aw2_fsdp1_audit_off` | 2 | 1 | `NO_SHARD` replication | 2 | off | 3 | 与上一 profile 做训练隔离性 A/B |
| `aw2_fsdp2_audit_on` | 2 | 2 | `FULL_SHARD` | 1 | on | 3 | 验证标准 sharded gradient、reshard 与完整 audit |
| `aw4_fsdp2_audit_on` | 4 | 2 | `HYBRID_SHARD` | 2 | on | 5 | 验证真实两维 shard/replicate topology 与完整 audit |
| `aw4_fsdp2_audit_off` | 4 | 2 | `HYBRID_SHARD` | 2 | off | 5 | 与上一 profile 做训练隔离性 A/B |

每个 profile 均使用 4 个 training steps、batch size 16；audit-on profile 在 step 2/4
计算 full gradient，并使用 BF16 保存 gradient vector。

## 动态权重专项 smoke

| 配置 | Actor world | `fsdp_size` | 所需 GPU | Steps | 用途 |
|---|---:|---:|---:|---:|---|
| `mopd_dynamic_weight_qwen0p6b_8b_aw2_fsdpsize2_tail_topp1_b16_4step_smoke.yaml` | 2 | 2 | 3 | 4 | 在 math、code、science 三个 domain 上验证 `[1/3, 3]` bounded applied-weight EMA，以及每个 domain 内按 `abs(configured token loss)` 排序的低 loss 15% mass、Top-p=1 gradient；Top-k replay 关闭，并记录 token JSONL |
| `mopd_feature_coverage_qwen0p6b_8b_aw2_fsdpsize2_top_partial_prefix_ppo2_b8_2step_smoke.yaml` | 2 | 2 | 3 | 2 | 验证 Top-only、partial Top-p=0.5、teacher-prefix/suffix active mask，以及两次 PPO epoch 的 configured token-loss 平均 |
| `mopd_topk32_reweight_qwen0p6b_8b_aw4_fsdpsize2_topp0p1_b24_5step_5gpu_smoke.yaml` | 4 | 2 | 5 | 5 | 按正式 Top-32 reweight 配置验证 4+1 GPU placement、rollout TP=2、step 4 的 Top-p=0.1/full-gradient/bounded-EMA 更新，以及 step 5 继续应用已更新权重 |

## Domain/token weighting 专项 smoke

Matrix：`mopd_domain_weighting_qwen0p6b_8b_matrix.yaml`。全部 profile 使用
2 张 actor GPU（FSDP size 2）和 1 张 teacher GPU：

| Profile | Steps | 用途 |
|---|---:|---|
| `gradnorm` | 2 | 用 Gradient Norm 更新 domain loss 权重 |
| `projection` | 2 | 用 Domain Gradient Projection Share 更新 domain loss 权重 |
| `projection_control_perstep` | 2 | 同时验证 Projection Share、与旧报告对齐的 44-ID Control 集合和 per-step shared Top-50 weighting |
| `control44_cumulative` | 3 | 验证报告对齐的 Control-44 2×、cumulative shared Top-500 3×，以及 ordinary / Control-only / Shared-only / overlap 的 1× / 2× / 3× / 6× metrics |

原 Control-only、per-step shared-only 和 cumulative shared-only 三个 standalone
profiles 已移出 canonical GPU smoke 集合；它们的单功能语义由 unit/contract
tests 覆盖，两个 combined profiles 负责 GPU integration coverage。

## 文件与 SHA-256

```text
8ef86b6fdf4c201468508f1172bd7b3923de524850816c5be003a7ad3bb252e3  mopd_domain_weighting_qwen0p6b_8b_matrix.yaml
d44e0f4fd3abd4ceeb70816df7a7f756d4727de817518d1678fba35cca6b8525  mopd_dynamic_weight_qwen0p6b_8b_aw2_fsdpsize2_tail_topp1_b16_4step_smoke.yaml
ff0781b127134e9d0c0e1bd956e9a8eea6bd965c998285306681d1a64c9b2215  mopd_feature_coverage_qwen0p6b_8b_aw2_fsdpsize2_top_partial_prefix_ppo2_b8_2step_smoke.yaml
866d082c0895a29bae3496b1c276b24308a0e59d2416f95ad7c652a291c7b6e5  mopd_grad_reliability_qwen0p6b_8b_matrix.yaml
8da93f9e2a451e18748d1b368e3a27863b8e0c4c861810c2a214a0b6125add38  mopd_topk32_reweight_qwen0p6b_8b_aw4_fsdpsize2_topp0p1_b24_5step_5gpu_smoke.yaml
```

以上 hash 对应本目录的 canonical 文件。后续修改配置时，必须同时更新 hash、对应
实验矩阵与 profile contract 测试。

## 执行注意事项

- 3 GPU 配置：2 张 actor GPU + 1 张 teacher GPU。
- 5 GPU 配置：4 张 actor GPU + 1 张 teacher GPU。
- Matrix profile 从 `code/` 根目录启动，例如：
  `bash scripts/run_local_mopd_training.sh test_grad_configs/mopd_domain_weighting_qwen0p6b_8b_matrix.yaml::gradnorm`。
- 普通 YAML 继续使用：
  `bash scripts/run_local_mopd_training.sh test_grad_configs/<config>.yaml`。
- 磁盘空间不足或只验证训练/audit 时，可通过 launcher 的 Hydra override 设置
  `trainer.save_freq=-1`；不要直接改变本目录中的 golden config。
- audit on/off A/B 必须使用相同代码、seed、数据和 rollout 设置，并顺序执行。
- 日志、checkpoint、audit JSONL 与 TensorBoard event 不应提交到本目录。

`world=2, fsdp_size=2, audit-off` 不属于当前五个 GPU regression experiments，已随
重复配置清理一起退役；需要该 A/B 时应基于本目录的 FULL_SHARD audit-on profile
显式派生，而不是长期维护第六份隐藏 control。
