# 2026-09-27：OPSA 论文阅读

- 阅读 Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement，arXiv 2608.31046v2（2026-09-24）。
- Canonical 笔记：`/Users/linghuazhang/Desktop/Project/Notes/Obsidian/Research/opd/Papers/Does-On-Policy-Distillation-Really-Distill.md`。
- 论文通过 lowest-logp 20% mask、固定负 advantage、entropy 加权，展示 teacher-free 收益；不能推广成所有 OPD 无知识迁移。
- v2 Appendix C.2 MD-OPSA 等量混合数学/代码 prompt，值得作为 MOPD 的 teacher-free 对照；尚未实施或启动训练。
- 官方代码 commit `f2e6a15ac42acd59c5d4458a8adf395311004872`：selector 与 entropy min–max 按 DP-local packed batch；论文 §4.1 按 response 描述。复现必须声明 scope。
- 本回合只新增阅读文档与 Daily 记录；现有训练、评测与源码均未修改。

## 后续：对当前项目的借鉴建议

- 优先同前缀 occurrence-level 诊断，区分 taxonomy 选择与 low-logp/entropy/disagreement 的重叠；缺失原始字段需重评分，不能由 token-ID 聚合数据推定。
- 当前为 Top-32-support KL 的全 token 监督+局部增权；固定负 K1 advantage 不是直接等价替换。TopLoss 仍依赖 teacher，替换信号后不能叫 teacher-free。
- matched intervention 保留全部基础监督，仅交换额外增权位置，报告实际 coverage/normalization/梯度规模；优先检验结构 prior 和 online adaptation。
- 沿用不做 multi-seed、不重开 Math 网格的已有 sprint 约束；OPSA/MD-OPSA 仅建议为条件性补充 baseline，没有启动或排队。
- 完整分析已追加到 canonical 论文笔记“2026-09-27 后续讨论”部分。

## 用户新提议：低loss降权/置零

- 已建立 `/Users/linghuazhang/Desktop/Project/OPD/plan/tail-loss-20260927/EXPERIMENT_PLAN.md` 与同目录tracker；tail候选范围待用户定义，未改代码/配置、未启动。
- 代码精确时序为t步forward统计、成功optimizer后选IDs、t+1应用；fresh首步无active IDs。
- normalizer是rank-local micro-batch内按domain mean-one；当前microbatch1近似按response，不能误称全step全局domain归一化。
- 三段raw weight拟为4/1/alpha，alpha=.5或0；必须处理zero-total-weight及少量非tail被极度放大的边界，并继续记录未乘权loss用于重新选择。

## 后续：用户质疑为何不用同一步loss选择

- 同一步loss→detach选择/权重→本步backward在算法上可行；一步滞后并非必要条件，也没有本次核查支持其性能更优。window1的滞后本身不提供跨步平滑。
- 当前actor逐micro-batch forward后立即backward，见`third_party/verl/verl/workers/actor/dp_actor.py:994`。保留全step、跨rank的token-ID均值排名时，完成统计前较早micro-batch已经反传；同一步需先无梯度评分再训练forward，或保留全部计算图。若仅按本micro-batch/response选择，可一遍forward完成，但选择范围改变。
- 仓库已有`domain_gradient/occurrence.py:138`的same-step prepass，由`audit.py:2249`调用；预评分后恢复RNG、buffer、module mode，复用当前batch。该分支按出现位置排序且按全局domain归一化，不能直接作为只改变时序的对照；regular配置仍默认token_id。
- 对“当前低loss位置置零”的目标，same-step occurrence更直接，但优劣待实验。若先隔离时序影响，应固定candidate、token-ID聚合、预算与原micro-batch归一化，只改变t统计用于t还是t+1；之后再比较tail multiplier。旧tail草案中的沿用滞后是最小改动假设，尚非最终选择。
- 本回合仅核查源码并记录讨论，未修改训练源码/配置或运行队列；没有新增实验结果。

## 用户授权后的实现：current-step + head/tail + teacher confidence

- 已新增显式`control_token_online_selection_timing=current_step`，默认next_step行为保留；复用prepass、首步生效、当前步重选、不加载lagged active-ID state。
- 新`current_step_selection.py`/`current_step_weights.py`支持token-ID与occurrence，保留configured C+S候选、strict count gate；head优先，tail排除head，独立top-p预算、raw tail weight允许0。显式currentstep采用microbatch/domain mean-one；D0返回零并记录，原始loss持续重新评分。
- teacher confidence=teacher chosen-token logp，type取mean logp；head/tail可独立选loss、confidence或L+C+LC composite，后者低分不等于low-loss与low-confidence双阈值。
- 新增5个Math+Code 4GPU/b528独立profile：same-step head-only、lowloss tail0.5/0、纯TC双端、composite双端；tail示例预算M5%/C1%，未调优。使用说明`docs/current-step-token-weighting.md`；计划/tracker已版本化更新，不再沿用旧滞后草案。
- 11文件合并回归226 passed；CPU实际RKL梯度oracle、真实2-rankGloo、配置链路、legacy、zero/reentry验证。独立review修复权重审计误报，无剩余P1/P2；另外修复既有testhelper的重复import mock隔离。
- 仅本地修改，未GPU FSDP smoke/测吞吐、未同步远端、未训练、未提交Git；原评测队列不变。canonical实验笔记为Obsidian `Experiments/Current-Step-Head-Tail-Weighting.md`。
