# Task Plan: Global Token Taxonomy and Recall Contract

## Goal

把 Control/Structure 全局 taxonomy、Rising/Stable endpoint Top-200、三套 candidate
pool 和 sliding-window recall protocol 冻结成一个无歧义、可执行、可验证的项目级合同，
最终写入项目根目录 `AGENTS.md`。

## Scope

- 四条 baseline：1.7B-OPD、1.7B-EOPD、4B-OPD、4B-EOPD。
- 三个 domain：Math、Code、Science。
- 两个主 token type：Control、Structure；二者互斥，Other 是全集补集。
- 三套 candidate pool：ExpandedPruned-V2、Robust190、Control-44。
- Taxonomy 与 candidate-pool audit 已开始执行；Rising/Stable 与 recall 尚未重跑。

## Phase 1: Freeze the global taxonomy

- [x] 将 tokenizer 的完整 token-ID 空间定义为 taxonomy universe；记录 tokenizer、
  vocabulary size、revision/hash 和 decode/normalization 规则。
- [x] 审计现有 PDTB、semantic-control surface rules、Control-44、format/boundary、
  code-lexical definitions，区分“语义 taxonomy 来源”和“历史 outcome-conditioned pool”。
- [x] 冻结 domain-invariant `Control(t)`：按明确的 inclusion families、surface
  normalization、tokenization observability 和 exclusion rules 标注完整词表。
- [x] 冻结 domain-invariant `Structure(t)`：以 format/boundary/syntax 为核心；
  domain relevance 作为独立属性，不让同一 token 的主类型随 domain 改变。
- [x] 冻结互斥优先级：`Control` 优先，`Structure = RawStructure \ Control`，
  `Other = Vocabulary \ (Control ∪ Structure)`。
- [x] 输出 machine-readable inventory，至少包含 token ID、raw/decoded surface、
  type、subtype、rule、provenance；验证三类互斥且穷尽完整词表。

## Phase 2: Freeze candidate pools as taxonomy subsets

- [x] 从可追溯 source 文件恢复 ExpandedPruned-V2、Robust190 和 Control-44 的精确
  token-ID membership，不根据本轮结果重新扩充。
- [x] 对每套池计算 `RawPool`、`Pool ∩ Control`、`Pool ∩ Structure` 和
  `Pool \ (Control ∪ Structure)`；任何被排除 token 必须逐项列出，不能静默删除。
- [x] 明确 candidate pool 是 taxonomy 的子集，而不是 Control/Structure 的定义来源。
- [x] 保存 membership、token surface、type、domain applicability 和 provenance/hash。

## Phase 3: Rebuild Rising/Stable endpoint Top-200

- [x] 冻结每条 baseline 的 Rising/Stable start/end optimizer step；不得从结果中重新
  挑选 phase boundary。
- [x] 对每个 `baseline × domain × phase × token` 计算 endpoint optimization speed：
  `speed = (gap_start - gap_end) / (end_step - start_step)`，明确 gap 是 occurrence-weighted
  absolute teacher-student log-probability gap。
- [x] eligibility 默认要求 start 和 end occurrence 都严格 `>20`；不足 200 时记录
  实际 eligible 数，不伪造 Top-200。
- [x] 按 speed 降序、token ID 升序打破 tie，独立生成 Rising Top-200 和 Stable Top-200。
- [x] 用同一份全局 taxonomy 对 Top-200 分类；统计四 baseline、三 domain 下的
  Control/Structure/Other actual count、share、Rising→Stable 变化、overlap/Jaccard。
- [x] 绘制 baseline 分面和 domain 分面图片；图题必须写明“Top-200 内的 taxonomy
  composition”，不得表述成完整词表所有 Control/Structure token 的动态。

## Phase 4: Rebuild the recall module

- [x] 每个 pool、domain、token type 独立构造 candidate set；另做 Control+Structure
  unified-selection ablation，避免把“分开选”和“统一选”混为同一口径。
- [x] 对每个 sliding window `w`，selection cadence 固定为每 `w` 个 optimizer steps
  选择一次；score 只使用当时及以前的数据，禁止 future leakage。
- [x] source-window eligibility 默认要求 token 在参与 score 的每个有效 step 上
  occurrence 都严格 `>20`；若改用 window mean，必须作为单独 ablation 命名。
- [x] 复用并核对代码中的 A/C/E/F 指标公式、aggregation order、direction 和 tie-break；
  不允许报告名与实际公式不一致。
- [x] 同时评估两个 target：
  1. next-step target：`t→t+1` endpoint speed 的 eligible global Top-200；
  2. next-window target：`t→t+w` endpoint speed 的 eligible global Top-200。
- [x] target eligibility 默认要求 target start/end occurrence `>20`；对 next-window
  同时要求 window 内每个参与 step occurrence `>20`，并报告该 gate 的 sensitivity。
- [x] 每个 selection event 从 candidate pool 选 Top-K；若候选不足 K，必须记录
  `actual_selected` 和 `full-K rate`，主最优点要求 full-K rate=100%。
- [x] 主指标遵循用户确认的池内口径：
  `Actual = |CandidatePool ∩ FutureTop200|`，`Hits = |Selected ∩ FutureTop200|`，
  `Precision = Hits/K`，`PoolRecall = Hits/Actual`，
  `PoolF1 = harmonic_mean(Precision, PoolRecall)`。
- [x] 同时报告候选池设计能力：`PoolCapacity`、完整 taxonomy `TypeRecall/TypeF1`；
  不能只凭小池的高 PoolF1 判断其 end-to-end coverage 更好。
- [x] 搜索 `w=1..20`、`K=1..30`（并受 full-K 约束），分别比较 A/C/E/F、三套 pool、
  split/unified selection、next-step/next-window target。

## Phase 5: Aggregation, figures, and decision table

- [x] 保留最细粒度 `token type × domain × baseline` 结果；四条 baseline 等权，
  不把 domain 当作 seed。
- [x] 每个分面展示 Actual、Hits、Precision、Pool Recall、Pool F1、Pool Capacity、
  Type Recall、full-K rate 和最优 `metric/w/K`。
- [x] 输出参数曲线/heatmap、三 pool 主对比图、Control/Structure split-vs-unified 图、
  Rising/Stable composition 图和 exact CSV/Markdown tables。
- [x] 将同数据选参的结果标为 exploratory；如无 held-out trajectory，不做泛化或
  显著性结论。

## Phase 6: Write the project-global contract

- [x] 在 `/Users/linghuazhang/Desktop/Project/OPD/code/AGENTS.md` 增加
  `Token Taxonomy and Recall Evaluation Contract`，写入不可违反的规范、公式、
  primary/secondary metric 和 exact source-of-truth 路径。
- [x] `AGENTS.md` 保持规范性、简洁；完整 token inventory 和 subtype 解释放在
  `Token.md` 及 machine-readable tables，并由 `AGENTS.md` 链接。
- [x] 标记旧报告中把 Top-200 cohort 外推为“全局 Control token”的文字为 legacy，
  防止新旧口径混用。

## Verification Gate

- [x] Taxonomy 三类互斥、穷尽 vocabulary，且重跑计数稳定。
- [x] Control-44 raw membership 恰为 44；Robust190 raw membership 恰为 190；
  ExpandedPruned-V2 raw/effective 数量都有明确记录。
- [x] 所有 phase Top-200 均能从原始 endpoint gap/occurrence 重建。
- [x] Recall 的 Actual/Hits/denominator 可由 event-level token IDs 独立复算。
- [x] selection 无 future leakage，所有主最优点 full-K rate=100%。
- [x] `AGENTS.md`、`Token.md`、分析代码、report labels 四处口径一致。

## Decisions Requiring Confirmation

1. **已确认**：Global Control/Structure 均 domain-invariant；domain subset 只能从
   175/634 的 global taxonomy 按四-baseline domain occurrence `>20` 派生。
2. **已执行**：window 内每个参与 step 都严格 `>20`，没有用 mean gate 替代。
3. **已比较**：split 与 unified 另做等总预算比较；当前结果支持 unified 作为默认。

## Status

**Completed** — taxonomy、phase Top-200、三 pool Recall 网格、等预算 ablation、图片、
token-level evidence、文档与验证 receipt 均已生成。
