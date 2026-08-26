# Domain 与 token loss weighting

这组功能通过 production actor loss 的 backward-only gradient mask 生效。
默认值全部保持原训练行为；只有显式开启对应开关时才改变优化。

## 1. 动态 domain 权重的信息源

```yaml
audit:
  enabled: true
  dynamic_domain_loss_weighting_enabled: true
  dynamic_domain_loss_weighting_signal_source: gradient_norm
```

`dynamic_domain_loss_weighting_signal_source` 支持：

- `gradient_norm`：使用每个 domain 的 `||g_d||₂`。
- `domain_gradient_projection_share`：先计算 signed share
  `<g_d, g_total> / ||g_total||₂²`，保留 signed 值用于日志，再将绝对值
  作为 non-negative controller signal。

两种信号都经过 signal EMA、inverse weighting、上下界裁剪和 weight EMA。
切换信息源时，checkpoint 中旧的 controller EMA 会自动重置，避免量纲混用。

## 2. 调整目标 token 的优化比重

```yaml
audit:
  enabled: true
  control_token_loss_weighting_enabled: true
  control_token_loss_weight: 2.0
  control_token_ids: []
  domain_control_token_ids:
    math: [123, 456]
    code: [456, 789]
    science: [123, 789]
```

当前 Qwen3-4B Connective+Structure follow-up 配置使用 domain-specific
`domain_control_token_ids`；`control_token_ids` 保留为空。两个字段名是为了兼容
既有 runtime schema 而保留的 legacy 名称；在这批 profile 中，候选集合实际表示
`Connective+Structure Rising Top-200`。其中 Connective 是 PDTB exact
single-token connective 与历史 Control-44 ID 的并集，但仅保留相应 domain 的
Rising Top-200 中实际出现的 ID；Structure 是排除扩展 Connective 后的
deterministic Format/boundary token，以及仅在 Code domain 激活的 Python
keyword/builtin/language/module/method/API/programming-lexicon extension。
上述配置把各 domain 匹配 token 对 actor gradient 的贡献乘以 `2.0`。token ID
与 tokenizer 强绑定，不能直接复用其他模型的 ID。

`control_token_loss_weight` 必须是有限非负数：大于 `1.0` 表示放大，
`0.0` 到 `1.0` 表示降权，`0.0` 会屏蔽匹配 token 的 backward gradient。
这个 gradient gate 不改变 forward loss 数值或原有 loss metric。

当前 Qwen3-4B follow-up profile 从独立 4B-OPD baseline 的 Rising Top-200
冻结候选集合，Math / Code / Science 分别包含 48 / 58 / 40 个 unique token IDs；
其中 Connective 为 35 / 31 / 31 个，Structure 为 13 / 27 / 9 个。Control-44
相对旧 selector 新增 9 / 8 / 7 个 ID，不是把全部 44 个 ID 无条件加入每个 domain。
Stable-only IDs 不进入该集合；不同 domain 的 membership 独立维护。完整定义以
`mopd_qwen4b_30b_a3b_instruct_2507_*gpu_math_code_science_topk32_`
`structural_codelex_rising_top200_control_{fixed_w4_b528,speed_pwl_u2}.yaml`
中的 `domain_control_token_ids` 为准。

### 2.1 在线候选审计与动态激活

固定 Control weighting 也可以切换为 online selection submode：

```yaml
audit:
  control_token_loss_weighting_enabled: true
  control_token_loss_weight: 4.0
  control_token_ids: []
  domain_control_token_ids: {}
  control_token_candidate_ids: []
  domain_control_token_candidate_ids:
    math: [123, 456]
    code: [456, 789]
    science: [123, 789]
  control_token_normalize_per_domain: true
  control_token_online_selection_enabled: true
  control_token_online_audit_interval_steps: 3
  control_token_online_window_steps: 3
  control_token_online_min_mean_occurrences_per_step: 20.0
  control_token_online_top_k: 30
```

`domain_control_token_candidate_ids` 是按 domain 冻结的 whitelist，本身不会被
直接加权；其 keys 必须与 `audit.domains` 完全一致。旧的
`control_token_candidate_ids` 仍可作为 global fallback：它会展开到所有 domain，
但两个 candidate 字段不能同时配置。每个
完成的 optimizer step 使用 production forward 已返回的 raw configured
token loss，按 domain 和候选 ID 累加 absolute-loss sum 与 occurrence count。
当 rolling window 已填满且 step 命中 audit interval 时，token 先通过
`window occurrence count / window_steps >= threshold` 的频次门槛，再按
`window absolute-loss sum / window occurrence count` 排名，各 domain 独立激活
Top-K。step `t` 的审计结果从 step `t+1` 生效，避免 same-batch feedback。

rolling window、当前 active IDs 和 audit step 随 optimizer checkpoint 保存；
resume 时配置签名必须一致。step gap 会清空窗口和 stale active set。该模式要求
`ppo_epochs: 1` 且整个 actor update 只有一个 optimizer mini-batch，并与 fixed
IDs、speed controller、phase gate 和 successor-span weighting 互斥。完整记录写入
`<audit.output_dir>/step_XXXXXX/jsonls/online_control_selection.jsonl`。

## 3. 提高所有 domain 共同高 loss token 的优化比重

```yaml
audit:
  enabled: true
  domains: [math, code, science]
  all_domain_shared_token_loss_weighting_enabled: true
  all_domain_shared_token_loss_weight: 1.5
  all_domain_shared_token_selection_mode: per_step_mean_abs_loss
  all_domain_shared_token_top_k: 100
```

`all_domain_shared_token_selection_mode` 支持：

- `per_step_mean_abs_loss`（默认）：每个 optimizer mini-batch 在正式
  backward 前执行：

1. 对每个 domain，按 token ID 聚合当前 `|configured token loss|`。
2. 用 per-occurrence mean absolute loss 给唯一 token ID 排名。
3. 每个 domain 保留 Top-K token ID。
4. 取所有配置 domain 的 token ID 交集。
5. 将交集 token 对 actor gradient 的贡献乘以配置权重。

- `cumulative_abs_loss`：将每个 global step 的 absolute configured-loss
  mass 累加到按 domain/token ID 保存的状态中，用累计 mass 排名，再取各
  domain Top-K 的交集。当前 step 会在 production backward 前纳入累计值；
  状态随 optimizer checkpoint 保存，且同一个 global step 最多累计一次。

两者都是在线 high-loss proxy，不是需要观察下一个 checkpoint 才能计算的
ex-post loss reduction。`all_domain_shared_token_top_k: null` 表示使用所选
时间范围内每个 domain 出现的全部有效 token ID：per-step 模式对应当前
step，cumulative 模式对应 run 开始至当前 step。

为了保证“每 step”确实覆盖完整 actor batch，开启该功能时要求
`ppo_mini_batch_size` 等于 actor batch size，并且 `ppo_epochs: 1`；不满足
时训练会直接报错，而不会静默改成局部 mini-batch intersection。

每次选择的 domain Top-K 和最终交集会写入
`<audit.output_dir>/shared_token_weighting.jsonl`，便于复现实验和检查实际被
加权的 token ID。

## 4. 组合规则与日志

三类权重按乘法组合：

```text
effective_weight =
    domain_weight
    × control_token_weight
    × all_domain_shared_token_weight
```

例如某 token 同时是 Connective+Structure target 和三域共享 token，两个 token factor
都会生效。forward loss 与现有 loss metric 保持不变，改变的是 backward
时各 token 的 gradient contribution。

专项组合 smoke config 将 Control 权重设为 `2.0`、cumulative shared 权重
设为 `3.0`，因此 ordinary / Control-only / Shared-only / overlap token 的
multiplier 分别为 `1x / 2x / 3x / 6x`。dynamic domain weighting 在该配置
中关闭，避免 domain factor 干扰这四类 token multiplier 的核验。

主要新增 metric：

- `{domain}/dynamic_weight/source_signal`
- `{domain}/dynamic_weight/ema_source_signal`
- `{domain}/dynamic_weight/raw_signed_projection_share`
- `global/token_weight/control_token_id_count`
- `global/token_weight/all_domain_shared_token_type_count`
- `{domain}/token_weight/high_loss_token_type_count`
- `global/token_weight/raw_configured_loss_abs_mass`
- `global/token_weight/token_weighted_configured_loss_abs_mass`
- `global/token_weight/effective_configured_loss_abs_mass`
- `global/token_weight/token_weighted_to_raw_abs_loss_mass_ratio`
- `global/token_weight/effective_to_raw_abs_loss_mass_ratio`
- `global/token_weight/mean_token_gradient_multiplier`
- `global/token_weight/amplified_token_occurrence_fraction`
- `global/token_weight/gradient_multiplier_mean_abs_error`

这些 amplification metric 使用第一次 no-grad forward 的原始 configured
token loss 和即将在第二次 production forward 中应用的 multiplier 计算。
gradient gate 保持 forward loss 不变，因此它们可以同时显示 raw loss mass
与实际 backward weighting 的有效 mass。
`gradient_multiplier_mean_abs_error` 比较 production gradient mask 与根据
domain/control/shared 配置独立计算的期望 multiplier，正常应接近 `0`。

原 audit-based domain weighting GPU matrix 已退役；当前研究路径统一使用
`test_grad_configs/mopd_dynamic_budget_qwen0p6b_8b_aw2_fsdp2_b16_4step_3gpu_smoke.yaml`。
历史 matrix 中的 `gradnorm`、`projection`、`projection_control_perstep` 和旧版
global Control-token weighting variants 不再作为 GPU experiment config 维护；
相应算子行为继续由 unit/contract tests 独立覆盖。在上述 Qwen3-4B
Connective+Structure follow-up profiles 中，legacy `control_token_*` 字段承载
broader domain-specific universe；其他 profile 仍以各自配置内记录的 token
universe 为准。
