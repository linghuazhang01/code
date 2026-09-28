# V4/V5 C+S 共享 TopP 选词

## 目标

对每个 domain 单独选择 `position_fixed`、`top_loss` 或
`top_teacher_confidence`。只要不是 `position_fixed`，冻结的 Control 和 Structure 就按
同一信号共同排名，并使用同一份 `control_token_online_top_p_by_domain` 预算。

## 实现约束

1. 旧配置缺省为 `position_fixed`：Control 使用上一 step 的 TopLoss，Structure 沿用固定位置规则。
2. 动态模式使用 step `t` 正式训练已产生的 raw RKL 或 teacher chosen-token log-probability，
   在成功更新后选出 step `t+1` 使用的 token IDs；不增加 student 评分 forward。
3. 每个 domain 只有一个 C∪S selector 状态和一个 TopP 分母（来源 step 全部 valid response tokens）。
   首步动态 C/S 均为空。动态 Code C/S 的位置资格由
   [Code 位置开关计划](CODE_POSITION_GATE_PLAN.md) 定义：关闭时均不限制位置，开启时
   分别使用已有的 Control/Structure mask。旧的 Control-only candidate groups 不覆盖 S，
   与共享模式同时启用时在配置校验阶段明确拒绝。
4. C/S 的配置权重分别生效，随后只进行一次 microbatch/domain mean-one normalization。
5. 保留冻结 V4/V5 membership、跨 rank 汇总、checkpoint 恢复和旧配置默认行为。

## 验证与交付

- 检查配置加载、Hydra 传递、运行时 metadata 和 V4/V5 候选池。
- 测试共享预算、跨类别排名、首步/下一步生效、Code 位置限制、跨 rank 归约和缺失信号报错。
- 提供独立的新配置示例；不覆盖或重启现有四卡训练。
