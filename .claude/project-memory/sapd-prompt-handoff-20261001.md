# 2026-10-01：停止绘图，交付详细方法prompt

- 作者明确改为只要生图prompt，本轮没有调用imagegen，没有改论文/实验。已有预览保留，README不再推荐失败布局。
- [完整prompt](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-method-overview-20261001/SAPD-image-generation-prompt.txt)：只OPD/SAPD，完整背景与可见简洁标签分开；不锁定旧circle matrix或配置box soup。
- 科学范围沿用Legacy Next-Step/token-ID Math C+S390 / Code S551；覆盖teacher-support reverse KL/同support归一化、frozen membership、全batch/domain/生成ID未加权mean、strict>20、whole-ID occurrence budget、raw4/1 local mean-one与detached gradient gate。
- 指定SAPD必须保留t→t+1提示；Other与低loss候选仍保留监督，raw4非绝对梯度×4，toy例非实测、不承诺优越/FLOPs。
- 两名现有reviewer分别核验科学逻辑/用户约束，一致性均PASS；全文traces保存，未虚称新外部模型审查。
