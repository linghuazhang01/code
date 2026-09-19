# 配置审计记录

审计范围：本目录六个 baseline 的四卡共置、八卡 6+2 配置及共享配置；2026-09-20。没有审计或提交工作区其他评测代码改动。

## 结论

未发现本次配置的阻塞问题。全部 12 个入口通过 typed config 加载与 launcher `--dry-run`。逐字段对照 canonical `configs/baselines/opd_baselines.yaml` 的 actor 设置，除明确统一为 516 的 PPO mini batch 外均保持一致。

核对了 Math/Code 域集合、等权采样、batch 整除、GPU placement、teacher FSDP=4/2、vLLM=0.75、Step60、输出目录唯一性。前一轮独立 reviewer 已确认当前 teacher 实现支持 dedicated multi-rank batching。文件凭据模式检查未发现密钥。

## 回归测试

运行：

```bash
uv run --no-project --with pyyaml --with pytest python -m pytest -q \
  tests/test_config_profiles.py tests/test_opd_baseline_profiles.py \
  tests/test_math_baseline_configs.py tests/test_teacher_performance_config.py
```

结果：**100 passed，3 failed**。三项失败均来自旧 `test_math_baseline_configs.py`，引用下列缺失文件：

- `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math.yaml`
- `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math.yaml`

这两个文件在审计时 Git HEAD `0ad5844` 中也不存在；相关测试和配置加载代码没有工作区改动。失败属于现有测试/历史文件缺口，本次未修复，也不将测试套件表述为全绿。

## 未验证范围

未启动训练、加载权重或执行 GPU smoke test。目标服务器的数据/模型路径、显存峰值、吞吐仍需实际运行确认；参见 README 中的部署与布局说明。
