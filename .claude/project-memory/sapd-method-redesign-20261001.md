# 2026-10-01：方法图按作者参考重画

作者明确否定v1参数流程图，给出ToDi式token监督对比参考。该反馈覆盖继续重画授权；本轮应用figure-designer与imagegen原生工具，未沿用ToDi的FKL/RKLmix。

- 当前[对比预览v5](/Users/linghuazhang/Desktop/Project/OPD/figures/ai_generated/sapd-method-comparison-20261001-v5.png)：同一组Math token IDs，共享raw RKL32，Uniform / loss-only TopLoss / SAPD三栏对应监督分配。
- Source t和raw multipliers at t+1放在主体；SAPD显示frozen C+S eligibility与candidate内TopLoss，Other2保留raw1，Then/EOS候选内低loss保留raw1，选Therefore/换行。类别/词面已独立核验。
- 原生生成及三次编辑收紧比较主体，删除底部重复四框链、展开C/S/O、明确whole-ID budget和局部mean-one。AI条形几何仍近似，不能作精确plot/vector终稿。
- [外部Caption](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-method-overview-20261001/caption-v2.md)声明toy losses、source每ID25次、N1000、p5%=50occurrences、其余ID更低；loss-only为conceptual comparator。不是实测或prior收益证明。
- 这是Legacy Math C+S390示例，不是V4/V5 Math S9或Current-Step。旧v1仅存历史，不应再推荐。
- 完整prompts、review tracing已保存，等待作者对设计方向反馈；未插入论文、未修改实验或其他图。

