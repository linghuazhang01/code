# SAPD Methodology：2026-10-02 修订

- 用户确定两阶段候选：Control=PDTB∪Control-44；Structure=(format/boundary∪CodeLex)\Control，且CodeLex加入所有domain。
- 仅使用OPD-1.7B step1–52，按domain保留至少两个不同step occurrence严格>20的ID。三域共享初始5321 IDs（374C+4947S），不使用四baseline交集或V9用法审计。
- 原始156条记录复算：Math129C+265S=394；Code157C+547S=704；Science125C+240S=365。NPZ/CSV/JSON/summary独立逐项核验通过。
- 中英文methodology已重写，17组公式相同，包含离线/在线频次区分、RKL32、ID mean loss、整ID occurrence预算、局部mean-one和Current-Step流程。
- 未修改训练配置或历史实验分数，新pool未部署验证。Control-44历史outcome-conditioned来源仍明确披露。
- 已完成两轮Claude Opus5.5讨论并修订两文：补充ID均值动机、词面先验边界、全局token-mean reduction、固定轨迹求导与复现工件。Claude撤回“position选择脱离候选先验”的原判断；第二轮未见数学硬伤，效果和成本仍待实验。

- 最新作者要求：方法正文仅涵盖 Math 与 Code，已移除 Science 的正文说明及表格行；两域仍均使用 CodeLex，候选总数为 394/704。原始三域复算工件保留用于溯源。

## 产物

- [中文方法](/Users/linghuazhang/Desktop/Project/OPD/paper_proposal/method_zh.md)
- [英文方法](/Users/linghuazhang/Desktop/Project/OPD/paper_proposal/method_en.md)
- [重建报告](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-methodology-20261002/REPORT.md)
- [候选ID](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-methodology-20261002/candidate-ids.json)
- [Claude讨论记录目录](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-methodology-20261002/claude-review)

- [完整讨论与处理决定](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-methodology-20261002/claude-review/DISCUSSION.md)
