# 2026-10-01：方法图按最新要求简化为OPD/SAPD

- 作者要求越简洁越易读，并明确不保留Loss-only。v6仅设计未生成；当前[预览v7](/Users/linghuazhang/Desktop/Project/OPD/figures/ai_generated/sapd-method-comparison-20261001-v7.png)从v5原生imagegen编辑，两栏四个Math token IDs。
- OPD为项目uniform teacher-support RKL32基础监督；SAPD frozen C/S资格+候选内TopLoss，Therefore/换行extra+base，Other2/低loss Then仍base。
- 用filled/hollow circles表达额外强调存在与否，消除定量Native bar尺度问题。移除共同流程、第三方法、EOS、batch/count/normalization细节，保留toy与t→t+1。示例loss不是真实评测结果。
- [Caption](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-method-overview-20261001/caption-v7.md)明确OPD无selector、SAPD预算/4-1 local mean-one与whole IDs。当前是raster设计预览，不虚称vector终稿或已验收printing。
- 未改论文、实验、既有动机图；旧v5保留。设计方向待作者反馈，知识库Daily与Writing已同步。

