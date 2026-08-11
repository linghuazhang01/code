# Logic/Control token 动态速度加权

该实验按 domain 独立统计 Logic/Control token 的 teacher–student absolute log-p gap，并根据 gap 的下降速度动态调整这些 token 的训练权重。

## 计算

对 domain `d` 在 optimizer step `t` 出现的 Control token：

\[
G_d(t)=\operatorname{mean}\left|\log p_T-\log p_S\right|.
\]

先做 EMA：

\[
E_d(t)=\beta E_d(t-1)+(1-\beta)G_d(t).
\]

再用精确的五步端点计算平均优化速度：

\[
v_d(t)=\frac{E_d(t-5)-E_d(t)}{5}.
\]

`v_d(t)>0` 表示 gap 正在缩小，`v_d(t)<0` 表示 gap 正在扩大。速度通过 config 中的折线锚点线性插值为 raw Control weight，超出两端时 clamp：

| Speed | Raw weight |
|---:|---:|
| -0.0025 | 0.0 |
| 0.0000 | 0.2 |
| 0.0050 | 2.0 |
| 0.0100 | 3.0 |
| 0.0150 | 4.0 |

Math、Code、Science 分别维护 EMA、speed、weight 和历史，不共享 controller state。训练时再按 domain 做 mean-one normalization，避免仅因加权改变该 domain 的总 loss scale。

## 时序

为避免当前 batch 的统计结果反过来影响同一个 batch，controller 使用 one-step lag：

1. step `t` 使用 checkpoint/state 中已有的 raw weight；
2. 统计 step `t` 的 gap 并计算 speed；
3. 如果达到更新频率，则得到 `control_weight_next`；
4. 新权重从 step `t+1` 开始使用。

权重每两步最多更新一次，但 speed 在满足 observation 条件时每步计算。

## Config 字段

| 字段 | 含义 |
|---|---|
| `control_token_speed_weighting_enabled` | 开关 |
| `domain_control_token_ids` | 各 domain 固定 Logic/Control token universe |
| `control_token_normalize_per_domain` | 是否按 domain 做 mean-one normalization |
| `control_token_speed_window_steps` | speed 的端点间隔，当前为 5 |
| `control_token_speed_ema_beta` | gap EMA 的 `beta`，当前为 0.80 |
| `control_token_speed_update_interval_steps` | raw weight 最短更新间隔，当前为 2 |
| `control_token_speed_initial_weight` | 尚无有效 speed 时的初始 raw weight，当前为 3.0 |
| `control_token_speed_min_occurrences` | domain 每步更新 EMA 所需的最少 Control occurrence，当前为 128 |
| `control_token_speed_weight_knots` | speed 到 raw weight 的折线映射 |

## 每步指标

每个 domain 都记录在 `{domain}/control_speed/` 下；`tensorboard_prune_mode: core` 会保留这些指标。

| 指标 | 含义 |
|---|---|
| `control_gap_raw` | 当前 step 的 occurrence-mean absolute log-p gap |
| `control_gap_ema` | 当前 controller 使用的 EMA gap |
| `optimization_speed` | 当前五步平均优化速度 |
| `speed_reference_step` | 速度计算采用的起点 step；无有效速度时为 -1 |
| `speed_computed_this_step` | 当前 step 是否成功得到新 speed |
| `control_weight_mapped_from_speed` | 当前 speed 通过折线直接映射的候选 raw weight |
| `weight_update_triggered` | 当前 step 是否达到更新频率并提交候选 weight |
| `control_weight_applied_raw` | 当前 step backward 真正使用的 raw weight |
| `control_weight_next` | state 中保存、下一 step 将使用的 raw weight |
| `control_weight_applied_normalized` | 当前 Control occurrences 实际使用的 mean-one normalized multiplier 均值 |
| `control_occurrence_count` | 当前 step 的 Control token occurrence 数量 |
| `minimum_occurrences_met` | occurrence 是否达到更新 EMA/speed 的门槛 |
| `observation_available` | 当前 domain 是否观测到 Control token |
| `state_observation_count` | checkpoint 中累计接受的有效 observation 数 |

Controller state 随 optimizer state 写入 checkpoint，因此正常 resume 后会继续原来的 EMA、历史、speed 和 weight。
