# 每个 step 实际增强 token 的来源

完成 optimizer step 后、online selector 更新下一步选择之前，记录当前生效权重的
Control / Structure / Other composition。固定 control、online selector（含 Q、
LossRatio）、shared-token weighting 自动启用，不需要修改 YAML。Adaptive
Neighborhood 尚未接入这组指标。

TensorBoard core 模式保留全部新增指标，沿用现有训练日志输出链路。
前缀为 `global/token_weight/amplified_source_`，也提供 `math`、`code`、`science`
等已配置 domain 的同名指标。

| 后缀 | 含义 |
| --- | --- |
| `occurrence_count` | 本 step 实际增强的有效 token 位置总数 |
| `{control,structure,other}_occurrence_count` | 各类增强位置数，重复出现重复计数 |
| `{control,structure,other}_occurrence_fraction` | 各类位置数 / 全部增强位置数 |
| `unique_token_count` | 增强位置对应的去重 token ID 数 |
| `{control,structure,other}_unique_token_count` | 各类去重 ID 数 |
| `{control,structure,other}_unique_token_fraction` | 各类去重 ID 数 / 全部增强 ID 数 |

例如增强位置为 `[control_A, control_A, structure_B]`：occurrence 数量为 2/1、
比例为 2/3 和 1/3；unique token 数量为 1/1、比例各 1/2。比例值范围为 0–1。
分母为零时数量和比例都记 0，读取比例时同时查看对应总数。

## 口径

- 分类来自冻结的 four-baseline global taxonomy：Control=175、Structure=634，
  其余为 Other；不是 candidate provenance，也不是名称为 control 的加权通道。
  runtime 冻结副本在 `mopd_verl/domain_gradient/frozen_taxonomy.py`，测试与
  `analysis-output/four-baseline-global-token-taxonomy/tables/global-taxonomy.csv`
  逐 ID 核对。该 taxonomy 针对本项目 Qwen tokenizer；更换 tokenizer 需重建映射。
- 只计有效 response loss positions，排除 padding 和 loss mask 屏蔽位置。
- 使用实际 production gradient multiplier，剔除动态 domain weighting 的影响：
  `production_multiplier > domain_multiplier + 1e-6` 才计为增强。
  因此已经选中但权重为 1、被 gate 关闭或归一化后未增强的 token 不计入。
- 汇总全部 micro-batches 和 ranks 的 token-ID histogram 后计算比例；global
  unique ID 跨 domain/rank 去重，occurrence 修正 sequence-parallel 副本。
- 不额外进行 model forward/backward。当前在完成 optimizer update 时输出；标准
  单 optimizer update/step 配置对应训练 global step。多 PPO epochs/mini-batches
  配置沿用现有 logger 的 update 聚合方式，不能视为整个 global step 的数量求和。
- 历史日志不会自动补出此指标；运行中的进程需要重新加载新代码才能输出。

验证：CPU fixtures 覆盖 taxonomy、一类重复出现、padding、空 domain、domain
weight、跨 rank 比例/去重、SP 重复计数、固定权重 runtime 与 TensorBoard core。
