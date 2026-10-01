---
project_id: opd
date: 2026-09-30
updated: 2026-10-01
kind: writing
status: awaiting-user-feedback
---

# SAPD 三图设计

- 用户要求先根据 `paper_proposal/` 设计动机图、方法图、结果图，同意后再绘制。
- [文字设计稿](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-three-figures-20260930/design.md)包含布局、标签、caption、工具与质量审计。
- Figure1第二轮：主图聚焦1.7B，Math/Code各Rising/Stable C/S/O柱；Math C20.5→8.5%、Code S21.5→9.5%，各−12pp。右侧可配明确标注的entropy–mismatch示意，词面缩入legend，4B/Science放补充。
- Figure2：分层OPD主干 + 功能prior/TopLoss支路，标明Next-Step。具体示例采用已评测Legacy C01：Math C+S390 / Code S551，TopP5%/1%，Fixed4。
- Figure3：九行、Macro8/Math4/Code4三列dot plots；C01相对EOPD的差值为+0.194466/+0.520833/−0.131902 pp。
- 功能prior独立收益与最终主recipe未冻结；不混入V4/V5、Current-Step、模拟值或未绑定消融，不虚构training CI。
- Factual与Consistency reviewers已复核初版；第二轮分域数字与taxonomy另行核验，不能追认为新的Claude意见。
- Phase分类确实对应active Code S551；ExpandedPruned-V3为另一个candidate pool，其有效Code S64。未绑定同run、同taxonomy的entropy–RKL32 joint数据，不定量绘制类别分布。
- 9/30真实Claude讨论未成功（bridge路径失效、CLI403、网页登录）；[历史状态](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-three-figures-20260930/claude-review-status.json)保留。10/1使用既有proxy取得claude-opus-5-5首轮带图与补充文字答复，详见本轮[讨论记录](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-motivation-preview-20261001/claude-content-discussion.md)。
- 2026-10-01用户授权首张动机图预览，已用原生imagegen生成并做一次chart编辑：[v2](/Users/linghuazhang/Desktop/Project/OPD/figures/ai_generated/sapd-motivation-1p7b-preview-20261001-v2.png)。接口未返回model，不能确认GPT-image-2.5。图2/3与终稿未绘制；未改训练/评测代码、proposal或论文，没有启动GPU作业。
- 预览增加固定Rising1–35/Stable36–52与Top200统计说明。最新Token.md 0.1与phase source重新核对；文字数字正确，生成图几何近似，正式统计部分需CSV确定性矢量重绘。[说明与审阅](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-motivation-preview-20261001/brief.md)。
- 用户随后要求图内只说明动机；已用原生imagegen生成[v3](/Users/linghuazhang/Desktop/Project/OPD/figures/ai_generated/sapd-motivation-1p7b-preview-20261001-v3.png)，移除底部1/2/3并收紧画布，phase/token统计移至图外caption草案。v3为最新生成预览，v1/v2保留；内容复审后不建议直接作为论文Figure1。
- 10/1[内容评估](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-motivation-preview-20261001/content-review.md)建议单一实证构成＋少量词面，Claude补充轮同意精简；Math Control/Code Structure分别−12pp，不能泛指合计C+S。该轮只评估，后续作者授权重绘并追加百分比/去空柱要求。
- [动机图v5](/Users/linghuazhang/Desktop/Project/OPD/figures/sapd-motivation-1p7b-v5.png)：去掉Other灰块、以相对百分比标注变化，原图和[记录](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-motivation-preview-20261001/v5-brief.md)保留。先生成原生AI v4试版，再独立CSV绘图；Fresh Codex复核PASS。
- 当前最新[动机图v6](/Users/linghuazhang/Desktop/Project/OPD/figures/sapd-motivation-1p7b-v6.png)按作者进一步要求优化百分比表达：改C/S并列柱、零起点0–30%轴；八个phase份额留柱内，同色括号连接指定家族两phase柱顶，显示Math Control−58.5%、Code Structure−55.8%与relative decrease。全Top200分母不变，Other不绘制。[说明与审阅](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-motivation-preview-20261001/v6-brief.md)。Codex follow-up复核PASS，非Claude新看图；PNG/PDF/SVG、脚本和规格保存，待作者反馈，未改论文/实验或图2/3。
- Obsidian canonical note：[[Writing/SAPD-Figure-Design]]，复用已有Research/opd绑定，未新建顶层registry。
- 10/1作者询问v6是否适合作为动机图、是否解释Rising/Stable。[语义评估](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-motivation-preview-20261001/v6-phase-context-review.md)认为适合真实观察型动机，建议在phase tick下补reward growth / reward plateau，caption保留固定1/35、36/52及各domain×phase独立Top200 gap缩减排序。构成变化仅提出功能prior值得检验，不能推成因果收益或后期token不重要。本轮未重绘、未改论文与实验。
- 作者随后授权继续优化图片描述，最新[动机图v7](/Users/linghuazhang/Desktop/Project/OPD/figures/sapd-motivation-1p7b-v7.png)已绘制：phase tick补reward growth / reward plateau，上方明确Top-200 by teacher–student gap reduction rate；画布176×74mm，八个份额与数据坐标/相对变化均与v6一致。[Caption](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-motivation-preview-20261001/caption-v7.md)说明两domain共享固定历史endpoints与独立Top200，末句提出functional prior待验证问题。PNG/PDF/SVG、规格、审计与改前备份保存；[八项说明](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-motivation-preview-20261001/v7-brief.md)和Codex follow-up复核PASS记录保存（含最终行距检查），非Claude新意见。当前v7待作者查看，仅绘图源和include snippet更新，论文正文、图2/3与实验未改。
- 作者请求对应图片生成prompt，已保存[英文prompt](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-motivation-preview-20261001/prompt-v7.md)：明确当前v7的8个份额、2个relative decreases、Rising/Stable解释、Top200筛选note、配色/括号/图例和200-ID分母；详细统计信息标为不绘制的背景。只写prompt，未新生成图片或修改既有图/实验。
- 作者要求再加入论文动机，已扩充同一prompt的背景段：dense base supervision下额外强调的分配问题；Control/Structure为可复现词法候选prior；functional prior决定eligibility，actual raw distillation loss决定ranking，所有valid response tokens保留基础监督。历史phase构成变化只提出待检验问题，chosen-token gap诊断与训练loss分开，不写成prior因果有效性或token后期不重要。原prompt备份保存，图内数据与标签、图片、论文正文和实验未改。
