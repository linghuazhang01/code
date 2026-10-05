# SAPD 双语 paper template 与 Opus 讨论

日期：2026-10-02。类别：writing。当前方法配方与训练状态不变。

## 本轮产物

- [中文计划](/Users/linghuazhang/Desktop/Project/OPD/paper_proposal/paper_template_zh.md)
- [English plan](/Users/linghuazhang/Desktop/Project/OPD/paper_proposal/paper_template_en.md)
- [Opus 两轮讨论](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-paper-template-20261002/claude-review/DISCUSSION.md)

基于用户六段 Introduction 指导和 tech-paper-template，定位 Technique Paper。核心 idea：冻结词面/结构候选决定资格，当前 ID-mean RKL32 决定优先级，整 ID occurrence 预算与相对正权重增权保留稠密监督。候选使用作者 2026-10-02 配方 Math394/Code704；不把它混同 V9 或历史 taxonomy。

## 讨论形成的决定

两轮实际返回模型均为 claude-opus-5-5。核心比较新增 E1R：先通过相同 OPD-1.7B 重复频次筛选的全词表 TopLoss。E4 也从同一支持词表抽取规模/频次匹配的随机池；去 Control-44 来源后重建候选的对照进入主文。由此区分先验内容、重复支持、池大小和实际干预强度。

加入 Rock Tokens 竞争假设，使用条件持续率、固定独立 prefix panel 和冻结开发集排除清单的独立诊断；不新增主算法模块。明确超额有数学上界，但小预算下相对超额仍可能大。C2 保持为 C1 的 type-to-occurrence 技术设计，不把常规评估包装成独立创新；mean vs sum/归一化不声称已证明必要性。

## 待补证据

计划可指导起草，但当前配方的匹配效果、多 training seeds、成本/覆盖，以及完整真实 running example 仍待验证。旧 V9 富集和 legacy 单 seed 不支撑新配方的收益。结果句、C3 数值和最终结论保持待定。未运行训练/评测，未改实验代码、配置或旧章节正文。

现有 opd.md 超过用户单文件行数上限，本轮写作增量保存在本条目的独立短记录，未继续扩展该大文件。Obsidian 对应更新为 Writing/SAPD-Introduction-Draft 与 Daily/2026-10-02。
