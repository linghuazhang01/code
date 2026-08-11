# Control-vs-Rollout Phase Weighting 实验方案

**Problem**: 现有 phase gate 使用 successor span 作为 control token 的比较基线，与原始 Top-100 分析使用的全 rollout token population 不一致。
**Method Thesis**: 用每个 domain 的 control-token gap 相对全 rollout occurrence-mean gap 的超额程度控制 marker 权重；A2 仅负责把释放的权重定向到 successor span。
**Date**: 2026-08-09

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: control-vs-rollout gate 能表达 domain-specific phase | 直接对应原始 finding | 三个 domain 的 relative gap 与 gate 同步下降/回升，且 A1 优于固定加权 | B1, B2 |
| C2: successor-span transfer 改善 Stable 最终性能 | 解释如何把早期增益转成后期增益 | A2 的最终离线性能稳定优于 A1，同时总 loss mass 不变 | B2, B3 |

## 方法定义

对 domain \(d\) 的有效 response token：

\[
g_i=\left|\log p_T(y_i)-\log p_S(y_i)\right|.
\]

在最近 \(W\) 个 step 内用 sum/count 重建 occurrence-weighted mean：

\[
C_d=\frac{\sum m_i^{ctrl}g_i}{\sum m_i^{ctrl}},\qquad
A_d=\frac{\sum m_i^{resp}g_i}{\sum m_i^{resp}}.
\]

其中 \(A_d\) 使用同一 domain 的全部有效 rollout response token，包括 control token；这与用户指定的比较总体一致。对 \(C_d,A_d\) 分别做 EMA：

\[
E_d=\frac{\operatorname{EMA}(C_d)-\operatorname{EMA}(A_d)}{\operatorname{EMA}(A_d)+\epsilon}.
\]

使用零中心 gate，保证 control 与平均 token 相同时不再获得特殊权重：

\[
\pi_d=\max\left(0,\;2\sigma(E_d/T)-1\right).
\]

### A1: Stage-aware control weighting

\[
w_i=1+(\lambda-1)\pi_d\quad\text{for control tokens},
\]

其他 token 的原始权重为 1。每个 response row 最后做 mean-one normalization，保持总 token-loss mass 不变。

### A2: A1 + successor-span transfer

control marker 沿用 A1；marker 后第 \(j\) 个 token 使用：

\[
w_{i+j}=1+(\lambda-1)(1-\pi_d)e^{-j/\tau},\qquad 1\le j\le K.
\]

span 不参与 gate 判断，只是释放权重的接收位置。重叠 span 取最大 targeting score，最后仍做 per-row mean-one normalization。

## 推荐初始参数

| 参数 | 值 | 说明 |
|---|---:|---|
| \(\lambda\) | 2.0 | 与既有 control weighting 对齐 |
| \(W\) | 5 steps | occurrence sum/count 滑动窗口 |
| EMA beta | 0.90 | 平滑 batch composition 噪声 |
| \(T\) | 0.25 | relative-gap gate 温度，量纲无关 |
| initial gate | 0.80 | 窗口未填满时的 early prior |
| \(K\) | 16 | successor span 长度 |
| \(\tau\) | 8.0 | span 距离衰减 |
| \(\epsilon\) | \(10^{-8}\) | 防止除零 |

## Config 语义

| Variant | Master | Phase gate (A1) | Span transfer (A2) |
|---|---:|---:|---:|
| Baseline | false | false | false |
| Fixed control | true | false | false |
| A1 | true | true | false |
| A2 | true | true | true |

不增加 hard-coded Rising/Stable step。每个 domain 独立维护窗口、EMA 和 gate；无 control occurrence 时保持上一 gate。状态随 optimizer checkpoint 保存。

## 必须记录的诊断量

- `{domain}/phase_control/control_gap`
- `{domain}/phase_control/rollout_gap`
- `{domain}/phase_control/control_gap_ema`
- `{domain}/phase_control/rollout_gap_ema`
- `{domain}/phase_control/relative_gap_excess`
- `{domain}/phase_control/gate`
- `{domain}/phase_control/control_occurrence_count`
- `{domain}/phase_control/rollout_occurrence_count`
- `{domain}/phase_control/span_gap`（只作诊断）
- `{domain}/token_weight/control_effective_weight_mean`
- `{domain}/token_weight/span_effective_weight_mean`

## Experiment Blocks

### B1: 统计与权重正确性 sanity
- Claim tested: gate 确实使用 control-vs-rollout，而非 span。
- Compared systems: synthetic unit test + 1-step dry-run。
- Success criterion: distributed sum/count 与离线直接计算一致；当 \(C=A\) 时 gate=0；mean-one normalization 后每行平均权重为 1。
- Priority: MUST-RUN。

### B2: 最小因果对照
- Compared systems: Baseline、Fixed control、A1、A2。
- Setup: 相同初始 checkpoint、数据顺序、seed、训练步数与 EOPD 配置；训练期间不做 validation。
- Decisive metrics: post-hoc 相同 benchmark 的最终性能；每个 domain 的 gate/relative-gap trajectory。
- Success criterion: A1 至少保留固定加权的 early behavior且改善或不损害最终性能；A2 的 Stable 最终性能优于 A1。
- Priority: MUST-RUN。

### B3: 机制诊断
- Claim tested: A2 的收益来自对 successor span 的定向分配，而非整体 loss scale 增大。
- Evidence: 所有 variant 的 mean effective weight 固定为 1；A2 中 control weight 下降时 span effective weight 上升。
- Priority: MUST-RUN。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 单元与离线重放 | synthetic + archived-step replay | gate/weight 全部符合公式 | CPU | mask/occurrence 口径错误 |
| M1 | 小步 smoke | A1、A2 各 2–5 steps | 三域指标有限且 checkpoint 可恢复 | 低 | 某域 control occurrence 稀疏 |
| M2 | 核心比较 | Baseline、Fixed、A1、A2 | 无 validation 泄漏，配置仅开关不同 | 高 | run-to-run variance |
| M3 | 完成后评估 | 四个 terminal checkpoints | A2 稳定优于 A1 | 中 | 只看单 seed 过度解释 |

## 风险与约束

- 当前 signal 是 relative residual gap，而不是 gap 的单步导数；这是为了降低噪声。下降速度保留为离线诊断，不进入第一版 control loop。
- 全 rollout mean 会被高频 token 主导；这是 occurrence-level finding 的一致口径，不在第一版引入 type-balanced 修正。
- control token 稀疏时不更新 gate，避免用单个 occurrence 驱动 phase。
- 训练期间保持 `val_before_train=false`、`test_freq=-1`、`log_validation_metrics=false`；性能比较只在训练完成后统一离线执行。

## Final Checklist

- [x] Gate reference 与原始 rollout token population 对齐
- [x] A1 与 A2 的作用可独立解释
- [x] 总 loss mass 受 mean-one normalization 控制
- [x] 不引入 hard stage boundary
- [ ] 用户确认公式与推荐参数
- [ ] 确认后再修改代码
