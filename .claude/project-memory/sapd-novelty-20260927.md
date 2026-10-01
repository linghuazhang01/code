# SAPD 查新记录 — 2026-09-27

主分析与独立 gpt-6-astra/xhigh reviewer：5/10，PROCEED WITH CAUTION。一般 entropy blind spot 已由 TIP/DEAR 覆盖；可继续检验 Control/Structure 语义先验在相同 loss/entropy/频率/位置/优化预算之外是否提供额外效用，以及阶段性干预差异。Rock Tokens 强调部分持续高 loss 的结构/话语残差效用低，需要区分可纠正与低效用结构位置。

active taxonomy 下，OPD三domain宏平均：1.7B Control14→6.67%、Structure16.67→7.83%；4B Control16.67→11.50%、Structure10.33→8.17%。4B Math Structure9.5→10.0%为例外；不能写所有domain均下降，也没有本轮训练级显著性证明。覆盖replay是旧taxonomy/token-ID聚合，不是逐position entropy selector复现。

优先复用/审核已有taxonomy vs full-vocabulary TopLoss结果，再做等预算语义身份消融；C+S优于单组不自动代表synergy。建议都是后续研究设计，未启动训练。

Claude Code指定Opus5.5及用户授权opus fallback均HTTP403，未得到Claude意见，不存在跨家族共识。完整材料可待认证/权限恢复后复用。

完整报告：[NOVELTY_REPORT.md](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-novelty-20260927/NOVELTY_REPORT.md)。原始评审：[001-codex.response.md](/Users/linghuazhang/Desktop/Project/OPD/.aris/traces/novelty-check/2026-09-27_run01/001-codex.response.md)。

