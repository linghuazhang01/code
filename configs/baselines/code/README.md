# Code-only baselines：八卡共置

共 4 个训练入口（2026-10-19）；`_common.yaml` 仅作为继承基配置。单 seed=42，Qwen3-1.7B student，Qwen3-30B-A3B-Instruct-2507 teacher。每个入口是 `../math_code/` 同名八卡入口的 Code-only 版本：method 块、rollout、optimizer、teacher_performance、batched reward 完全相同，只改训练数据（Eurus `code_train`）、validation/eval 集合（HumanEval+、MBPP+、LCB）、teacher 映射、audit domain、batch 和 run 名。

| 方法 | 八卡共置（actor、vLLM rollout、FSDP-8 teacher 共用 8 卡） | Math+Code 对应入口 |
|---|---|---|
| TIP Top-K32, ρ=0.5 | `tip_topk32_rho50_8gpu_8c_b256.yaml` | `../math_code/tip_topk32_rho50_8gpu_8c_b512.yaml` |
| EOPD Native | `eopd_native_8gpu_8c_b256.yaml` | `../math_code/eopd_native_8gpu_8c_b512.yaml` |
| ExOPD, λ=1.25 | `exopd_native_lambda1p25_8gpu_8c_b256.yaml` | `../math_code/exopd_native_lambda1p25_8gpu_8c_b512.yaml` |
| FiRE-OPD Native | `fire_opd_native_8gpu_8c_b256.yaml` | `../math_code/fire_opd_native_8gpu_8c_b512.yaml` |

- global train batch = PPO mini batch = **256** = 8 × 32；60 steps，每步保存，HF 仅上传 Step60，路径 `checkpoints/code/<experiment_name>`。
- vLLM `gpu_memory_utilization=0.6`；`worker_placement.separate_ref_policy: false`，`ref.fsdp_config.fsdp_size=8`。
- 只做过 launcher dry-run，没有 GPU smoke test，也没有启动训练。

```bash
bash scripts/run_mopd.sh configs/baselines/code/tip_topk32_rho50_8gpu_8c_b256.yaml --dry-run
```
