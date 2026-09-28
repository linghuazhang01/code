# Code C+S 位置开关修改计划

状态：已在本地实施并通过独立代码审查；尚未部署。现有四卡 V4 训练保持原配置。

## 目标与配置语义

新增 `audit.code_cs_position_gate_enabled: bool | null`，只适用于 Code domain 的
`versioned_cs_selection_mode_by_domain.code` 为 `top_loss` 或
`top_teacher_confidence` 的 V4/V5 Next-Step 配置。

- 未设置（`null`）或显式 `false`：动态 Code 的 C、S 均以所有 valid response 位置参与
  上一步排名，并在下一步的所有 valid 位置加权；不使用类别位置 mask。
- `true`：动态 Code C 仅在现有 Control mask 允许的位置（所有 fenced code blocks 外）
  参与排名与加权；动态 Code S 仅在现有 Structure position mask 允许的位置参与排名与加权。
- Code `position_fixed` 维持旧行为：C 仍按代码块外的 Next-Step TopLoss 选，S 按现有位置
  固定加权且不进入 TopP。若对此模式显式设置新字段（包括 `false`），配置加载时报错。
- Math 的统计和加权不受该字段影响。若配置中没有动态 Code 模式（例如 Code 为
  `position_fixed`，或使用 legacy taxonomy / Current-Step），显式设置则报错。

这是对**尚未部署的本地动态草稿**的有意修正：草稿把 Code C 限在代码块外，却让
动态 S 处处生效。新规则的关闭态让 C/S 都不受位置限制，开启态则分别使用各自 mask；
没有已运行的动态 C+S checkpoint 需要保留草稿的混合行为。

这个开关只改变 **eligible occurrences**，不改变冻结的 C/S token IDs、排名信号、
Next-Step 时序、raw weight、候选池或预算。动态模式始终将 C∪S 共同排名，并只使用
Code 的一份 `control_token_online_top_p_by_domain` 预算。Math 逻辑不变。

## 算法与预算

1. 把每个 domain、类别的 eligible occurrence mask 集中在一个 helper 中。TopLoss、
   Teacher Confidence 的来源 step 统计，以及下一 step 的实际加权，都调用相同规则；
   只在使用位置规则时要求并校验相应 mask 的形状与 valid 范围。
2. TopLoss 沿用正式训练 forward 产生的 raw RKL per-ID 分数；Teacher Confidence 沿用
   batch 中的 teacher chosen-token log-probability，均只聚合 eligible occurrences，
   不增加 student 评分 forward。每个 ID 的累计量是其 eligible occurrence 数，不是
   全部出现数；分数与频次门槛也仅基于 eligible occurrences。各 rank 的分子、计数和
   valid 分母先完成全局汇总，再统一排名。分数并列时沿用 token ID 升序作为 tie-break。
3. TopP 目标继续为 `ceil(p × 该 domain 来源 step 全部 valid response token 数)`。
   按现有规则逐个加入排名后的完整 token ID，达到目标时包含边界 ID，因此可超过目标。
   若通过频次门槛的全部候选出现总数不足目标，则选完这些候选，shortfall 为目标减去
   实际入选的 eligible 出现数；不偷偷改分母或另给 S 预算。
   首步动态 C/S 选择仍为空，下一步的实际覆盖率可能与来源 step 不同。
4. 冻结 taxonomy 继续保证同域 C∩S=∅。C/S 命中位置使用各自配置 raw weight，
   按 microbatch/domain 只做一次 mean-one normalization。

## 兼容、审计与验证

1. 配置从 YAML 经 Hydra、audit metadata 传至运行时。缺省字段不改变旧
   `position_fixed` profile 或正在运行的训练。现有未启动的 `shared_toploss` 与
   `math_teacherconf_code_toploss` 示例均显式设 `false`；另从 `shared_toploss` 派生
   一个仅开关设 `true`、使用独立 run/audit/eval/checkpoint namespace 的 Code 对照示例。
   不覆盖现有 namespace。
2. 将解析后的有效策略（`position_fixed`、`dynamic_ungated` 或 `dynamic_gated`）
   纳入 selector checkpoint 配置签名，而非直接记录原始 `null`/`false`；两者在动态
   Code 中等价。恢复时策略不同则拒绝。旧 checkpoint 缺字段时，仅当旧配置本身为
   `position_fixed` 才兼容：以旧 selector state 中冻结的 C-only candidate map 等
   已保存信息核验，不能只凭当前 YAML 推断；动态 C∪S 的无签名旧状态一律拒绝。
3. 在审计日志中记录策略、来源 step 的 valid 数、C/S 各自 eligible occurrence 数、
   TopP 目标、实际入选出现数和 shortfall，并分开记录下一步 C/S 的实际加权位置数。
   沿用当前 fenced-code 与 Structure mask 的定义，不在本次扩展缩进/行内代码解析规则。
4. 测试 Code 开/关与 TopLoss/Teacher Confidence 的交叉组合：同一 batch 内代码块内外
   的 C、符合/不符合位置规则的 S；确认排名与加权资格一致、C/S 共用一份预算。
   另测首步与下一步、空 eligible 集合、预算不足或边界超额、跨 rank 汇总、checkpoint
   恢复拒绝策略漂移、旧 `position_fixed` 快照不变，以及无额外 student forward。
   更新动态草稿的测试和 `Token.md` 说明，避免保留“Code C 固定限位、S 不限位”的描述；
   原有运行的 manifest/publish report 作为历史记录保持不变。

实施范围仅限本地源码、配置示例、测试和说明文档；不重启或改动现有四卡训练。

## 本地实施与验证（2026-09-28）

- 新增共享位置资格 helper；TopLoss 来源统计、Teacher Confidence 来源统计与实际
  Next-Step 加权使用同一 C/S 位置规则。`null`/`false` 对动态 Code 等价，`true`
  分别使用现有 Control/Structure mask；固定模式和 Math 维持旧规则。
- 新字段经 YAML、launcher、audit metadata 传递，解析后的策略写入 selector
  checkpoint 签名。恢复时拒绝策略漂移；旧的 C-only 固定模式快照有受限兼容路径。
- 来源 step 的 eligible C/S 次数、valid 分母、TopP 目标/入选/缺口和本 step 实际
  加权 C/S 位置数写入审计指标及 `versioned_token_selection.jsonl`。
- 两个未启动示例显式设 `false`，另有独立 namespace 的 `true` 对照示例。
  对照配置的展开差异仅为该开关与运行/输出目录标识。
- 相关 275 项 CPU 回归通过，涵盖 TopLoss/Teacher Confidence × Code 开/关、
  跨 rank 汇总及单 rank 错误同步、首步/次步、空 eligible 集、checkpoint 兼容和旧模式。扩大套件另有
  7 项历史 Qwen4B profile 测试因仓库缺少其引用的两个 YAML 文件而失败，未进入
  本次选择逻辑；这两个文件不在 Git 跟踪列表中。
- 独立审查发现并修复两项兼容问题：非 train metadata 不再传显式位置开关；旧
  `position_fixed` checkpoint 兼容恢复后立即写入有效策略签名。修复后复审无其他
  阻塞问题，`git diff --check` 通过。
