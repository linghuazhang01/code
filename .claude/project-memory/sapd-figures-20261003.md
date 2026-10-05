# SAPD figures — 2026-10-03

最新可编辑交付：已将接受的SAPD方法图转换为paper_proposal/fig/sapd-method-overview.drawio，原生文字/面板/token条/柱/连接线可编辑，8个公式保留LaTeX，两个网络图标内嵌为SVG并独立保存于sapd-method-assets。重建代码为paper_proposal/code/method_drawio.py。官方Draw.io31.7 CLI实际预览、完整97元素清单、52文字源、SVG拓扑、字体携带及独立视觉/代码审查通过；PNG为paper_proposal/fig/sapd-method-overview.png。MathJax与原STIX字形略有差别，内容不变；原出版PDF和旧PNG/SVG未改。

最新方法图修订：图面移除Control-44，使用Format Token + Codexlex、Control Token和Structure Token；选中ID的排名柱与raw 4×柱统一红色。第三栏按z_i→r_i→a_i→gradient-equivalent surrogate组织，明确rank-local microbatch/domain均值归一化。正式代码paper_proposal/code/method.py，PDF为paper_proposal/fig/sapd-method-overview.pdf（176×94mm），预览同步figures/sapd-20261003。推导、符号表与英文Method段落见paper_proposal/sapd_weighting_derivation.md。原method_en/zh与训练/候选配方未改；公式恒等式、实现语义、字体/版面、矢量导出及独立审查通过，共享样式改动后效果图的渲染像素和PDF文字保持一致。

最新交付位置：去掉Joint，副标题为Math-only distillation与Math–Code distillation。论文版绘图代码位于paper_proposal/code/effects.py（附drawing.py、author_references.json、requirements.txt和README），仅导出paper_proposal/fig/sapd-effects.pdf，派生数据在code/data。原图稿目录同步预览并保留方法图。新入口在不同cwd下经明确Python3.10.9/Matplotlib3.7.5执行通过，数值、来源hash、PDF正体/无Joint及与旧renderer像素一致均通过。

最新小文字简化：图面删除Math4/Macro8，副标题为Math-only distillation与Joint Math–Code distillation，9pt正体；其余分数/布局不变，评测集合仍在caption定义。PDF/SVG/PNG已更新并核验。

最新措辞：左副标题Math-only distillation · Math4、右副标题Joint Math–Code distillation · Macro8，9pt衬线斜体；两轴统一Macro-average accuracy (%)，评测集合留在副标题。数值、OPD/TIP/ExOPD/SAPD顺序及轴范围不变，导出与字号/画布检查通过。

最新顺序调整：两个面板均为OPD → TIP → ExOPD → SAPD；颜色和纹理随方法移动，分数及20–28%/30–32.5%轴范围不变。已重新导出并核验标签、柱高、配色映射和画布边界。

最新轴范围：按用户要求，Single Domain纵轴改为20–28%（间隔2），Multi Domain改为30–32.5%（间隔0.5），两边保留断轴；图注同步注明两种尺度。全部数据、四方法、配色保持，标签偏移随轴跨度调整；矢量导出、数值/版面检查与独立审查通过。旧版备份backups-before-20-30-axis。

最新图形选择：用户改回蓝色柱状图，指定TIP、ExOPD、OPD、SAPD四方法；双面板176×76mm，左Single Domain的Math4、右Multi Domain的Macro8，深蓝SAPD与三种浅蓝/中蓝纹理baseline，15–35断轴。数值依次22.92/25.4/22.29/26.04与30.35/30.95/30.73/31.66；数据未更改。评测集合不同，不把跨面板高度差解释为跨域增益。旧单面板版已备份，PNG/PDF/SVG与图注、验收记录已同步，独立审查通过；方法图未变。

最新展示要求：ExOPD保留25.4%，图面不加†，英文图注不再包含参考值说明。已更新PNG/PDF/SVG并核验所有分数不变；来源、核验状态和独立archive26.25仍在数据文件中保留，图注只对SAPD–OPD比较说明step60/Avg@8，不扩展成全部方法同协议声明。

最新数值指示：用户明确指定ExOPD参考行37.3/31.5/16.2/16.5与Avg25.4。图中改为ExOPD†25.4，独立记录author_references.json及protocol未独立核验的caption；原archive26.25保存在单独字段和原始记录，其他数值不变。PDF/SVG/PNG及验证记录已更新。

最新范围：用户要求效果图只保留Math-only，已改为88×66mm单面板并同步图注；六方法、+3.75pp、15–35%断轴和配色保持。双面板版已备份，方法图未改。数值与版面复核通过。

- 两张图保存在 `../figures/sapd-20261003/`，每张PDF/SVG/PNG，宽176mm；旧图保留。
- 图1按Math-only训练的Math4与Math+Code训练的Macro8分面，只展示指定两种设置的代表性legacy配置及全部五种真实baseline。直接从原始Math summaries与已核验Macro8 CSV取数；相对OPD +3.75/+0.930242pp，统一15–35%轴且明确断轴。
- 纠正参考值陷阱：归档Math-only OPD22.291667、ExOPD26.25；不用未核验23.1/25.4或synthetic ExOPD。Ours26.041667低于ExOPD0.208333pp；Macro8 ours31.657633高于EOPD0.194466pp。
- 图2按2026-10-02已确认的论文方法配方绘制：recurrence候选Math394/Code704、Current-Step RKL32 ID-mean ranking、整ID occurrence预算、局部mean-one和dense supervision。它不改写Token.md或历史训练配置；图1不证明此新配方的效果。
- PDF无嵌入位图，SVG保留文字；主标签≥8pt，自动布局检查无越界/文字重叠，toy预算和权重数学检查通过。独立数据/代码/视觉审查通过。
- 图注、全部数据、来源hash、完整figure-designer审计见 [README](/Users/linghuazhang/Desktop/Project/OPD/figures/sapd-20261003/README.md)。未更改训练、评测、论文正文或既有图稿。

## 后续视觉修订

- 按用户反馈覆盖同名最新图，旧版备份在`../plan/sapd-figures-20261003/backups-before-visual-revision/`。
- 效果图改为低饱和冷色baseline+砖红SAPD，删底部说明及星号，176×66mm。方法图采用模型图标、token条、loss排名条、1×/4×raw-weight柱，176×80mm，PDF可提取文字量减少63.4%。
- 统计/recipe/局部归一化/toy性质保留于caption。数据逐项与旧版一致，字体及布局检查、独立审查通过。
