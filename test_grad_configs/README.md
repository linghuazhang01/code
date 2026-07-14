# Domain-gradient experiment configs

这里集中保存当前约定的五份 4B student / 8B teacher domain-gradient regression
实验配置。本目录是这五个 GPU regression experiments 的唯一配置来源；不要在
`configs/` 下维护第二份副本。launcher、测试和 regression prompt 均直接引用本目录。

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

## 文件与 SHA-256

```text
0c3552fce4ed9ce15ce4e3a205f714217e60c816f61286bf6726dc6d9f864924  mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize1_audit_freq2_b16_4step_smoke.yaml
3b1e65482d1836833e903f6fc03d63d98a94654d328482e3434bdc0ee2fcec8d  mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize1_audit_off_b16_4step_smoke.yaml
10c9ff9da9764f91356d1216b36139ce4ce6de9f3728d521bd43e1579fb4e32d  mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml
86231492c7ce51499cf7ea2933760639cb8793b622d329e5d9acebe530152c09  mopd_grad_reliability_qwen4b_8b_aw4_fsdpsize2_audit_freq2_b16_4step_smoke.yaml
0fb474da1abac463c0a383f2ba5525dd7c07e6b000a15be7615f0e24c0448444  mopd_grad_reliability_qwen4b_8b_aw4_fsdpsize2_audit_off_b16_4step_smoke.yaml
```

以上 hash 对应本目录的 canonical 文件。后续修改配置时，必须同时更新 hash、实验
矩阵与 `tests/test_grad_reliability_profiles.py` 中的 contract。

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
