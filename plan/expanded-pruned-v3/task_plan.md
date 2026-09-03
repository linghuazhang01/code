# ExpandedPruned-V3 Implementation Plan

## Goal

新增独立的 `ExpandedPruned-V3` candidate pool：继续使用 V2 taxonomy；每条 baseline
内合并 Rising/Stable Top-200，在 1.7B 与 4B 内分别合并 OPD/EOPD，再取两个
model-size 集合的交集作为 Small seed。随后沿用 V2 的 deterministic tokenizer sibling
closure、archived occurrence `>=20` pruning 与 seed-floor contract。

## Completed work

- [x] 将 pool version 与 taxonomy version 解耦；V3 显式映射到 V2 taxonomy。
- [x] 为 V3 合并 Rising/Stable seed，并记录每个 seed 的 phase/baseline provenance。
- [x] 实现 1.7B/4B cross-size support gate，精确验证 170 个 domain-aware seeds。
- [x] 生成 Small、Expanded、ExpandedPruned membership、hash 和 config-ready whitelist。
- [x] 保持 V1/V2 membership hash 不变；V3 允许替换而非仅 add-only 扩充。
- [x] 通过当前 Four-Baseline-Supported domain taxonomy 审计 V3 effective membership。
- [x] 更新 `Token.md`、`AGENTS.md`、Obsidian 实验定义与验证规则。
- [x] 独立 replay A/C/E/F、split/unified、K=1...100 与 window/pre-window=1...20。
- [x] 生成 next-step K25 与 next-window K41 的 Math-only 4--8 GPU standalone configs。

## Frozen contract

- Small seed：每条 baseline 内 `Rising ∪ Stable`；同 size 内 `OPD ∪ EOPD`；最后
  `(1.7B union) ∩ (4B union)`，按 domain。
- Sibling closure：Control lexical、Format signature、Code-only CodeLex，与 V2 相同。
- Pruning：seed 无条件保留；新增 sibling 要求 domain archived maximum occurrence `>=20`。
- 冻结大小：Small Math/Code/Science=`65/54/51`；Raw Expanded=`317/260/270`；
  ExpandedPruned=`146/140/128`。
- V3 selector replay 已完成：主 taxonomy Type F1 下，Math-only next-step 为 unified
  A/i1/w1（pre-update source）/K25，next-window event-supported 为 unified
  A/i7/w7/K41。两者仍是 in-sample exploratory evidence；训练尚未提交，待 held-out
  validation。历史 V2 指标不得迁移为 V3 结论。
