# Domain-gradient experiment configs

这里集中保存 GPU domain-gradient regression 与专项 smoke 配置。五份 paired
reliability experiments 使用 Qwen3-0.6B student / Qwen3-0.6B teacher；动态权重、token attribution
等专项检查也在本目录维护 canonical profile，不在 `configs/` 下保留第二份副本。
launcher、测试和 regression prompt 均直接引用本目录。

## 实验矩阵

| ID | Actor world | `fsdp_size` | FSDP1 strategy | Replica count | Audit | 所需 GPU | 用途 |
|---:|---:|---:|---|---:|---|---:|---|
| 1 | 2 | 1 | `NO_SHARD` replication | 2 | on | 3 | 验证 singleton shard group、跨 replica gradient 同步及完整 audit |
| 2 | 2 | 1 | `NO_SHARD` replication | 2 | off | 3 | 与 ID 1 做训练隔离性 A/B |
| 3 | 2 | 2 | `FULL_SHARD` | 1 | on | 3 | 验证标准 sharded gradient、reshard 与完整 audit |
| 4 | 4 | 2 | `HYBRID_SHARD` | 2 | on | 5 | 验证真实两维 shard/replicate topology 与完整 audit |
| 5 | 4 | 2 | `HYBRID_SHARD` | 2 | off | 5 | 与 ID 4 做训练隔离性 A/B |

每份配置均使用 4 个 training steps、batch size 16；audit-on 配置在 step 2/4
计算 full gradient，并使用 BF16 保存 gradient vector。

## 动态权重专项 smoke

| 配置 | Actor world | `fsdp_size` | 所需 GPU | Steps | 用途 |
|---|---:|---:|---:|---:|---|
| `mopd_dynamic_weight_qwen0p6b_0p6b_aw2_fsdpsize2_tail_topp1_b16_4step_smoke.yaml` | 2 | 2 | 3 | 4 | 在 math、code、science 三个 domain 上验证 `[1/3, 3]` bounded applied-weight EMA，以及每个 domain 内按 `abs(configured token loss)` 排序的低 loss 15% mass、Top-p=1 gradient；Top-k replay 关闭，并记录 token JSONL |
| `mopd_feature_coverage_qwen0p6b_0p6b_aw2_fsdpsize2_top_partial_prefix_ppo2_b8_2step_smoke.yaml` | 2 | 2 | 3 | 2 | 验证 Top-only、partial Top-p=0.5、teacher-prefix/suffix active mask，以及两次 PPO epoch 的 configured token-loss 平均 |

## 文件与 SHA-256

```text
d7c26962c3cf311a50350336b9b74e2bb19ac75c7014484a28ed523d84d7be93  mopd_grad_reliability_qwen0p6b_0p6b_aw2_fsdpsize1_audit_freq2_b16_4step_smoke.yaml
cf7775e904ac2c9c7f7be78f7dadb144ad2bd1edcd25a1b11ecde4fadb75ebce  mopd_grad_reliability_qwen0p6b_0p6b_aw2_fsdpsize1_audit_off_b16_4step_smoke.yaml
57536e39715a02692b6d4c14f79370baa80563f437f81a3ae4fad956655e32e6  mopd_grad_reliability_qwen0p6b_0p6b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml
6eec10dd3181c981651524f19408b6a472e989f51fe0f98f86d8234fa7eaaa63  mopd_grad_reliability_qwen0p6b_0p6b_aw4_fsdpsize2_audit_freq2_b16_4step_smoke.yaml
ae4d46cbf8c44251ad8e3fc4e0ffde78257d44ffb8b72632a1b2d755264cd4b4  mopd_grad_reliability_qwen0p6b_0p6b_aw4_fsdpsize2_audit_off_b16_4step_smoke.yaml
caa8760ce8c1644a270c41ef322ac7584b826a9babd974e8ec643641a1f5e114  mopd_dynamic_weight_qwen0p6b_0p6b_aw2_fsdpsize2_tail_topp1_b16_4step_smoke.yaml
cd71a11a76e3171213afe5c1ea475e741836c88198ec6c94c29d9ccfff0cab62  mopd_feature_coverage_qwen0p6b_0p6b_aw2_fsdpsize2_top_partial_prefix_ppo2_b8_2step_smoke.yaml
```

以上 hash 对应本目录的 canonical 文件。后续修改配置时，必须同时更新 hash、对应
实验矩阵与 profile contract 测试。

## 执行注意事项

- 3 GPU 配置：2 张 actor GPU + 1 张 teacher GPU。
- 5 GPU 配置：4 张 actor GPU + 1 张 teacher GPU。
- 从 `code/` 根目录启动，例如：
  `bash scripts/run_local_mopd_training.sh test_grad_configs/<config>.yaml`。
- 磁盘空间不足或只验证训练/audit 时，可通过 launcher 的 Hydra override 设置
  `trainer.save_freq=-1`；不要直接改变本目录中的 golden config。
- audit on/off A/B 必须使用相同代码、seed、数据和 rollout 设置，并顺序执行。
- 日志、checkpoint、audit JSONL 与 TensorBoard event 不应提交到本目录。

`world=2, fsdp_size=2, audit-off` 不属于当前五个 GPU regression experiments，已随
重复配置清理一起退役；需要该 A/B 时应基于本目录的 FULL_SHARD audit-on profile
显式派生，而不是长期维护第六份隐藏 control。
