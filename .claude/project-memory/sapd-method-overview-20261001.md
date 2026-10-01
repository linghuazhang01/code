# 2026-10-01：SAPD 方法总览图与结果呈现

- 用户授权先绘制方法总览图，询问实验主结果用 table 还是 plot。
- 已生成 [方法图 PNG](/Users/linghuazhang/Desktop/Project/OPD/figures/sapd-method-overview-20261001-v1.png)、PDF/SVG、FigureSpec 与可复现包装脚本；[交付记录](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-method-overview-20261001/README.md)。
- 图为 Legacy C01 Next-Step：Math C+S390 / Code S551、m05/c01、raw Fixed4，单 shared frozen teacher；按全 optimizer batch/domain/token ID 的未加权 raw RKL32 排序，整 ID occurrence budget，下一步生效；rank-local microbatch/domain mean-one 与全 valid-token 监督明确。
- 代码/实际图像独立复核最终 Approve / PASS，三项图评分均 9/10。矢量 PDF 17.6 cm 宽、最小字 8.10 pt、无裁切；导出默认拒绝碰撞。完整 reviewer tracing 已保存。
- 推荐主结果 table（八 benchmark 分组 + Math4/Code4/Macro8；版面紧张时逐 benchmark 放 appendix），plot 优先 matched prior/selector 消融或真实训练动态；只有已归档九行时 dot plot 可辅助呈现 domain trade-off。
- C01−EOPD 为 Math +0.521 / Code −0.132 / Macro +0.194 pp，单 training seed、batch516/528 等条件 unmatched，不作显著性/因果结论；不拼入 Math-only、模拟、未绑定 V4/V5。
- 未插入论文，未改已有动机图、实验配置或启动训练/评测。最终 paper recipe 与作者视觉反馈仍待后续确认。

