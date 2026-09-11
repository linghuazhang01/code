---
project_id: opd
repo_root: /Users/linghuazhang/Desktop/Project/OPD/code
vault_root: /Users/linghuazhang/Desktop/Project/Notes/Obsidian/Research/opd
hub_note: Research/opd/00-Hub.md
language: zh-CN
last_sync_at: 2026-09-06T14:59:00+08:00
last_synced_head: a925de611df850ef2a23b26ca063009b47d6cef2
status: active
auto_sync: true
---
# 项目记忆: opd

## 2026-09-07 14:17：Q/Fixed4 Step60 Math4结果已归档

- job219 COMPLETED/0:0，完整结果、日志下载并checksum验收通过；120题×8rollouts、64shards完整。
- Overall Avg@8=23.23%（223/960），Pass@8=43.33%（52/120）；已更新`../experiments_records/eval/Summary.md` Math-only区及对应RUN_MANIFEST，不进入Standard10排名。
- job218仍RUNNING、step15/70。评测归档heartbeat已暂停。详情`plan/job218-then-q-step60-eval-20260907.md`。

## 2026-09-07 12:17：Q/Fixed4 Step60 Math4单卡评测已启动

- 用户确认Math四集K8并要求立即使用空闲单卡，取消等待训练218完成。已提交job219，RUNNING/1GPU/100G，CUDA witness和64 shards计划通过。
- 精确checkpoint：`/home/shuang_qiu/mopd_code/checkpoints/MOPD/q1p7b-math-top32kl-taxonomy-topp05-qsel-w4-5g4a1t-b256/global_step_60`；run tag `q1p7b_qsel_fixed4_step60_math4_dp1_k8_seed42_20260907`。
- heartbeat每30分钟汇报评测219及训练218，完成后归档Math-only结果；详见`plan/job218-then-q-step60-eval-20260907.md`，不得重复提交。

## 2026-09-07 12:09：V3 5% LossRatio新训练已提交并启动

- Q/Fixed4的Step60完整门禁已通过，五卡释放后启动Slurm job218；scontrol=RUNNING，5GPU/500G/64CPU，12:09:47开始。
- 新run=`q1p7b-v3-topp05-lossratio-i1w1-5g4a1t-b256-r20260907`，用原V3 5%配置的`_r20260907.yaml` overlay仅隔离六个输出标识；fresh training，无checkpoint resume。
- 6项配置测试及resolved equality通过，runtime远端checksum一致，只上传新overlay。详情`plan/v3-lossratio-after-step60-20260907.md`，不得重复提交job218。

## 2026-09-07 07:54：Q训练Step60已完整，V3后续训练等待五卡释放

- heartbeat `step60-v3-5`确认Q/Fixed4 restart当前step61；Step60的12个rank文件、HF JSON及safetensors长度结构、保存日志和上传receipt通过核验。
- 用户已授权下一项V3 TopP5% top_loss/loss_ratio训练（5GPU/500G），当前五卡仍被Slurm外训练占用；未终止旧训练、未提交新任务，待资源释放后去重提交。
- 具体checkpoint、证据和后续操作状态见`plan/v3-lossratio-after-step60-20260907.md`；每30分钟检查并每次直接汇报。

## 2026-09-07 06:17：V3 K25 Step60 Math评测Job217已提交

- SSH恢复；Job217于06:16:46提交/启动，scontrol确认1H200/100G/64CPU/24h，Math4/K8/seed42/DP1TP1。
- 当前guard等待实际分配UUID `GPU-ffbfad45-bf88-6cef-21b3-2cbce3edf575` 空闲，尚未推理。未干预现有训练。
- Jobs215/216因新增guard假设GPU-前缀启动失败；实测PyTorch返回裸UUID，规范化修复后217保持RUNNING。不能将前两次归因于调度分配失败。
- `v3-k25-step60-math` heartbeat已更新为只跟进217；不重提。完成4集K8后下载并更新eval/Summary.md。
- 归档manifest: `../experiments_records/eval/q1p7b_v3_k25_step60_math4_dp1_k8_seed42_20260907/RUN_MANIFEST.md`。无评测分数。

## 2026-09-07：V3 K25 Step60 Math-only 评测准备

- 用户确认四个 Math dataset、K8、1GPU/100G，完成后归档并更新 eval/Summary.md。
- HF revision `089ea552fa85b3ac46090c65897d1ecceba0368f` 的目标 Step60 已下载到远端独立 models/hf_opd_v3_k25_step60_089ea552 目录，权重 SHA256 与 HF LFS 一致。
- launcher dry-run 通过；尚未提交。Slurm 显示空闲但 GPU0–4 有 Ray 训练进程，只有 GPU5 空闲；ConstrainDevices=yes。未修改调度配置、未中断训练。
- 用户随后明确要求直接单卡提交，授权已确认；SSH三次登录前关闭，尚无job。已创建每10分钟跟进的heartbeat `v3-k25-step60-math`，恢复后查重提交、核验1GPU/100G、完成后下载并更新eval/Summary.md；不打断现有训练。详见项目根目录 plan/v3-k25-step60-eval-20260907.md。

## 当前问题
- TODO

## 2026-09-06 14:59：R2远端成功启动

- 用户授权执行/重试后SSH已恢复；Q代码按rsync dry-run→备份→上传→checksum验证同步。
- `start.sh --local --foreground`通过nohup后台启动（远端无screen，不安装）；
  STOP_STALE_RAY=0，不中断任务。GPU0–4，GPU5空闲；4a1t/b256/70steps不变。
- 14:57:30启动，主PID41213；14:59日志进入Training Progress0/70，W&B已同步。
  尚未确认step1完成，不创建monitor、不追加step60取消策略或评测。
- 真实5GPU CUDA/NCCL Q witness通过；Python环境复用，无包安装；既有env凭据未输出。
- Run=`q1p7b-math-top32kl-taxonomy-topp05-qsel-w4-5g4a1t-b256`；详细路径/hash/备份
  见`plan/taxonomy-q-next-experiments-20260906/R2_LAUNCH.md`。Q源码仍未Git提交，
  远端工作树已同步；不能以远端HEAD1539c1b识别实际运行源码。

## 2026-09-06 14:20：exact Q selector + Fixed4，5GPU本地完成

- 用户确认A/B分母为同source step全部valid Math occurrences跨response/rank均值。
  新top_q_loss_entropy以sumL/sumH/sumLH/N还原精确Q；联合5%、count>20、t→t+1、
  Fixed4/other1保持，不用alpha1.75。4a1t/b256/seed42/70steps，展开只改selector+6标识。
- config: `configs/token_selection/math/q_next_step_full_taxonomy_unified_topp0p05_i1_w1_fixed4_5gpu_4a1t_b256.yaml`。
- 457tests通过、1skip，含真实双CPU rank；launcher dry-run通过，无GPU smoke。
  Q尚未commit/push；远端eb85612仅含之前lossratio/alpha。无上传/启动/monitor修改。
- 记录：`plan/taxonomy-q-next-experiments-20260906/R2_IMPLEMENTATION.md`。
  R1负反馈仍未核验，不混淆Q-selector和Q动态weighting。

## 2026-09-06 03:01：用户改 alpha=1.75，补齐 6/7/8 GPU config

- 最新配置使用`lossratio_alpha1p75`，原alpha1p510088五卡文件已改名；原校准JSON保留历史。
- 5GPU=4a1t/b256；新增6GPU=4a2t/b256、7GPU=5a2t/b255、8GPU=6a2t/b258，
  遵循既有taxonomy资源配置，后两者batch为满足actor整除；6–8GPU显式teacher FSDP=2。
- 保留selected/other configured-loss、base边界和alpha后不cap4；总70steps不变。
  历史raw均值在alpha1.75下为4.635492，不再匹配4；不保证新run权重或收益。
- 四个launcher dry-run通过，新增精确配置差异测试；本轮不改训练逻辑、不提交远端任务。
  详见`plan/loss-ratio-alpha-20260906/IMPLEMENTATION.md`最新修订。

## 2026-09-06 02:47：指定 lossratio run 的 alpha 与 sibling config 已实现

- 用户批准实现并生成配置；指定参考run为
  `q1p7b-math-top32kl-topp05-lossratio-5g4a1t-b256`。核实该run是configured-loss
  selector + selected/other-valid occurrence loss ratio，**不是Q/all-valid**。
- 保留原base=clip(ratio,1,4)，selected raw=alpha×base，other=1；alpha后不再cap4。
  空组保持neutral1，alpha默认1；不改变原selector、t→t+1或mean-one语义。
- Source1–59→applied2–60平均raw=2.648852726612563，因此alpha=1.510087729609387；
  历史缩放mean4、P05–P95 3.7012–5.0212、range3.6270–5.0926，不保证新run均值或收益。
- 新config为原lossratio文件名增加`_alpha1p510088`，展开仅alpha与6个run/output标识不同；
  保留5GPU(4a1t)、batch256和70总steps。记录见`plan/loss-ratio-alpha-20260906/IMPLEMENTATION.md`。
- 参数贯通launcher/actor/checkpoint/日志；schema10必须alpha、旧版默认1、resume不匹配拒绝。
  已加float32范围保护和float64 mean-one reduction；406项tests与launcher dry-run通过。
- 代码审查与独立历史权重复算完成。未提交/上传/运行/取消任务，未改monitor或eval Summary。
  此纯lossratio实现不验证用户此前Q/all负结果，也不代表Q生产模式已实现。

## 2026-09-06 02:16：动态权重强度参数讨论（待确认）

- 用户询问动态权重能否增加超参数增强学习强度；建议selected raw=alpha×Q/all，other=1，
  alpha>0固定，alpha=1恢复原式，不截断。只在ratio外乘selected，避免Q缩放或全组缩放抵消。
- 当前mean-one使其控制相对强调而非全局learning rate；effective selected系数为
  alpha*r/(1-p+p*alpha*r)，p须取实际归一化单元的coverage，不把raw增幅当梯度增幅。
- 可先核验R1日志并冻结r_ref，再用alpha=4/r_ref对比Fixed4；如测更强一档，再配对
  alpha=6/r_ref与Fixed6。历史r_ref只近似匹配新run尺度，须记录实际raw/applied分布。
- 这是解释R1负结果的可选weighting分支，与R2的selection问题独立。未确认alpha或预算，
  不改变R2既有待确认优先级，不改代码/远端。详见同目录plan的STRENGTH_PARAMETER.md。

## 2026-09-06 01:39：用户反馈 R1 下降，调整优先级

- 用户明确反馈 configured-loss selector + Q selected/all-valid 已试过且效果下降。
  记为 reported-negative-unverified：run ID、具体Q定义、对照、seed和完整降幅尚未取得。
  不与先前纯loss_ratio实验混淆，也不因旧本地实现状态否认用户的实验。
- 最新顺序：只读核对R0/R1 → Q定义/shadow → R2=Q-selector+Fixed4。取消默认重跑R1；
  R3=Q/Q降为条件性后续，Fixed6仍为独立幅度探针。最小新增1个R2，anchor不可比时另议重跑。
- 单独改权重的负结果不排除Q selector价值；旧proxy的3.36/3.50不是此次R1实测权重，
  不把“强调不足”当作已证实原因。尚未获得足以写canonical Summary的新数值。
- 已版本化更新同目录EXPERIMENT_PLAN/TRACKER；本轮只改计划/记忆，无代码或远端修改。

## 2026-09-06：Taxonomy Q 初版计划（执行顺序已被上方修订）

- 本轮只规划，不改训练代码、不提交/中断任务、不恢复监控；当前队列未核验。
- 建议 FullMath 联合 TopP5%、1.7B，做 selector（configured L / Q）× selected raw
  weight（Fixed4 / Q-all）2×2：先验收旧 R0 provenance，再依次 R1=L/Q、R2=Q/4、R3=Q/Q。
  需新增3个训练；R0无法严格复用则4个。Fixed6与匹配固定权重均为条件性扩展。
- 建议生产 Q 使用 exact abs(configured Top32 loss)+student entropy，all-valid均值归一化，
  occurrence先乘后聚合，selected/all-valid不截断。该定义不同于已接受的gap aggregate
  proxy，必须在实施前确认；先同batch shadow分离loss来源与交叉矩两项差异。
- Step60 Math4完整Avg@8为筛选主指标，+0.5pp为建议实用门槛而非显著性；候选/对照做
  Standard10并补training seeds43/44。旧sampling seed复评不计入训练复现。
- 资源假设5GPU/500G训练+1GPU/100G Math4筛选；正式Standard10默认DP4/400G。
  完整计划：`plan/taxonomy-q-next-experiments-20260906/EXPERIMENT_PLAN.md`；未改eval Summary。

## 2026-09-05：Taxonomy loss + entropy 权重设计与离线proxy

- 用户已接受gap+entropy token-ID aggregate proxy；Q用于FullMath taxonomy联合选择，
  覆盖5%全部valid occurrences，count严格>20；primary ratio对比all-valid（含selected），
  另附other-valid。最新要求不cap4、不floor1、不+1，正ratio分母直接相除。
- Data恢复后真实回放已完成：1.7B 51有效steps（负entropy异常step3整步排除），4B46。
  Q/all主权重中位数3.3624/3.5029，P05–P95为3.2083–3.7739/3.3302–3.7733；
  前10→后10有效steps均值3.6732→3.2584、3.7401→3.4304。Q/other中位数3.8637/4.0591。
- 相较各自gap score，Q权重均值大25.58%/31.45%，但固定Q后换selector的均值仅大
  1.39%/2.53%，主要数值增幅来自score公式，不是训练收益证明。33项tests与独立
  6-step复算通过（ratio误差≤8.88e-16），三张图和CSV/manifest已保存。
- exact configured-loss方案仍缺数据：1.7B无Top32逐token记录，4B历史loss为chosen-token
  PG，两者缺按ID的loss×entropy交叉矩。proxy不能冒充exact，未改生产训练或远端状态。
  报告见`analysis-output/opd-baseline-taxonomy-score-ratio-20260905/proxy-replay/analysis-report.md`；
  Obsidian `Results/Taxonomy-Q-Weight-Proxy-Replay.md`。未改远端任务或恢复监控。

- 用户澄清联合 score 为 `A+B+A*B`。建议两项分别采用全体 valid occurrence
  公共均值尺度，再算 selected/other 的同 score occurrence-mean ratio；A 用
  abs(configured loss)，B 用 student entropy，先乘后平均。归一化方案仍是建议。
- 现有 paired score 代数形式相同，但 raw Top-K loss、逐 response max-normalization
  和每 ID 的 1+mean(score) 不等于该组级 ratio；现有 loss_ratio 仅支持 top_loss。
- 旧 paired 组件 3 项 tests 通过；未实现新模式、未提交实验，暂停的监控未恢复。
- 本轮设计：`plan/taxonomy-loss-entropy-score-ratio-20260905.md`；Obsidian
  `Experiments/Taxonomy-Loss-Entropy-Score-Ratio.md`。不将下方历史队列视为实时状态。

## 研究假设
- TODO

## 当前任务
- `q1p7b-math-top32kl-ns-taxonomy-topp0p01-i1w1-5gpu-3a2t-b258-resume15`
  已在 Slurm job 201 从 `global_step_15` 完整恢复；02:20--03:53 四次约 30 分钟
  健康检查均通过，训练从 step 17 推进到 step 22，日志持续增长且无 Traceback、
  CUDA OOM、RayTaskError 或 NCCL fatal。step 20 已产出四项真实 Avg@4，并保存完整
  3-rank checkpoint；bounded batched reward 与 validation Avg@4 修复已由远端 9 项
  targeted tests 和真实运行共同验证。commit
  `e231273882951ff2f16984efba847b08b3ce19dd` 已推送至 `origin/main`；当前继续运行
  至 step 70，step 20 指标只作为中间 protocol smoke，不作最终实验结论。
- ExpandedPruned-V3 candidate pool 已冻结：使用 V2 taxonomy；每条 baseline 内合并
  Rising/Stable Top-200，同 size 内合并 OPD/EOPD，再取 1.7B/4B intersection。
  Small Math/Code/Science 为 65/54/51，raw ExpandedPruned 为 146/140/128，
  共 414 个 domain-aware entries、268 个 unique IDs；active-taxonomy effective 为
  115/110/99（共 324/214 domain-aware/unique）。V1/V2 hash 未变化。V3 selector
  replay 已覆盖 A/C/E/F、split/unified、K1...100、window/pre-window1...20：主
  taxonomy Type F1 下 next-step 为 unified A/i1/w1（pre-update source）/K25（0.2050），
  next-window event-supported 为 unified A/i7/w7/K41（0.3048）。两个 optimum 的
  Math-only 4--8 GPU standalone full config matrix 已生成，31 targeted tests 与
  10/10 launcher dry-run
  通过；尚未启动训练，仍需 held-out 验证。详见 Obsidian
  `Experiments/ExpandedPruned-V3-Candidate-Pool.md`。
- 监控 Math-only ExpandedPruned-V2 5-GPU training job 189。heartbeat `5gpu-taxonomy` 每 30 分钟复核两个门禁：`global_step_60` 的 4-rank model/optimizer/extra-state 与 HF model/tokenizer 完整后，去重提交四项 Math、K=8、seed42、DP1 的 partial evaluation；job 189 `COMPLETED / 0:0` 且 `global_step_70` 完整后，再去重提交 `configs/token_selection/math/top32kl_next_step_full_taxonomy_split_topp0p05_i1_w1_5gpu_b256.yaml`。
- Online Control token selector 已新增 opt-in occurrence-coverage `top_p` budget；默认继续使用 `top_k`。token type 按 score 排序后，累计其 occurrence count，直到 `selected_occurrences / valid_token_count >= top_p`；rolling window 同时累计分子和分母，grouped taxonomy 在 Top-P 下按 domain candidate union 统一选择。配置、launcher、actor meta、checkpoint schema v8 和 JSONL audit logging 已贯通。
- Math-only EOPD / OPD `global_step_60` 的四项 Math `K=8` 评测 jobs 170/171 已完成、下载并通过本地/远端 aggregate SHA、JSON/JSONL、120 题/960 rollouts 与独立指标重算验收。结果作为独立 `Mass Only（Math-only）` 区块写入 `experiments_records/eval/Summary.md`，不进入 Standard10 active 排名；详见 Obsidian `Results/Math-Only-Step60-EOPD-vs-OPD.md`。
- Math-only FiRE-OPD 4-GPU training job 175 与 Step60 partial eval job 181 均已完成；job 181 的 32/32 shards、120 prompts、960/960 rollouts、JSON/JSONL、SUCCESS、remote/local 内容和独立指标验收通过，已写入 `experiments_records/eval/Summary.md` 与三方法 analysis bundle。
- TIP-TopK32 `global_step_60` Math-only partial eval job 176 已完成、下载并验收：四项 Math、K=8、DP=2、32/32 shards、120 prompts、960/960 rollouts，远端/本地 checksum 一致；结果已更新到 canonical `Summary.md` 的 Math-only 区块与 Obsidian `Results/Math-Only-Step60-ExOPD-vs-TIP.md`。
- Burgundy A100 部署资产已完成 staging，并沉淀为项目 skill `skills/cityu-a100/`：统一记录 node01–12 资源盘点、两层资源申请、OPD 环境/数据/模型安装与验收流程。环境、数据、4B/30B 模型均已落在 `/home/lzhan37/scratch/opd/`；最终 seeded CUDA witness 仍受空闲卡 `reset required` 故障阻塞，待 CSC 修复健康 A100 后仅重跑 `PHASES=verify`。
- 运行 V1_KL_Student_Entropy_Control、V1_Speed_Control、V0_Speed 三个 `global_step_60` 的 Standard10×K8 DP2 评测：V1_KL job 160 与 V1_Speed job 161 已完成下载、checksum/结构验收、official EvalPlus jobs 165/166 post-scoring 与 `Summary.md` 同步；V0_Speed job 162 因 AIME25 shard 0020 为 0 records 在 merge 阶段 fail-closed。其失败 suite/logs 已下载并确认 1,700/1,702 prompts、13,600/13,616 raw rollouts 可恢复；未自动重跑。
- ExOPD / OPD / TIP generation 与 pinned G-OPD official EvalPlus base+plus post-scoring 均已完成、下载和验收；active Code/Overall 已采用 official Plus。Student Base job 147 与 Teacher job 148 的 10×K8 outputs 也已下载验收并完成 official EvalPlus post-scoring，已作为 reference anchors 写入 canonical summary。详见 Obsidian `Experiments/Step60-Standard10-Evaluation.md` 与 `Results/Step60-Standard10-Comparison.md`。
- 检查已导入的仓库结构。
- 补充当前实验和结果。
- 开始关联论文与可沉淀的项目知识。
- 当前仓库变更未检测到需要跟进的任务。
- 决定是否从 checkpoint 恢复 TIP-TopK32 1.7B 与 4B baseline。
- 运行 Top-KL+Student Entropy 的 6/7/8-GPU 拓扑配置（5+1、6+1、6+2），比较单 teacher 与双 teacher 的吞吐和显存。
- 在真实训练环境执行一次 baseline smoke，确认 `<domain>/<category>/...` 与 `domain_step_metrics.jsonl` 按预期落盘。
- EOPD `global_step_65` 的 8-benchmark 评测与下载验证已完成：job 128/133 均成功，结果独立保存于 `experiments_records/eval/eopd_step65_m16_c4_s4_mmlu500_20260825_123812/`。
- 查找或重新提供 EOPD `global_step_55` 的 actor model shards；当前指定目录仅余 `data.pt`，无法评测。
- 监控 Top-KL + Student Entropy Domain Candidate v2 `global_step_60` 的 4-GPU 8-benchmark Slurm job 141；成功后将完整 records、summary 与日志下载到 `experiments_records/eval/` 并验证 6,880 个 rollouts。
- 旧 8-GPU Domain Candidate v2 `global_step_70`/`global_step_75` 的 8-benchmark
  评测与单目录扁平归档已完成；后续若采用 reasoning-aware 新 Code evaluator，
  必须作为新协议结果单独记录。
- 对 Math-only OPD baseline matrix 先运行 5-GPU batch-256 smoke；方法覆盖 OPD、FiRE-OPD、TIP-TopK32 与 EOPD，随后再决定是否展开 4–8 GPU sweep。
- Math-only OPD 4-GPU baseline Slurm job 142 已在 step 60 的 Hugging Face checkpoint upload 因 private repository storage 403 失败；其 student/teacher model load 与前 59 个 training steps 均正常，不再作为进行中任务。
- 已实现按 global step 数组选择性上传 verl checkpoint 到 Hugging Face Hub；命中 step 会强制保存，失败后 resume 会依据本地 receipt 补传，token 推荐在 `start.sh` 外部通过 `export HF_TOKEN` 注入。

## 进行中的实验
- `q1p7b-math-a-ns-expanded-v2-i1w1k29-5gpu-b256`：Slurm job 189 使用 5×H200（4 actor/rollout + 1 ref/teacher）、batch 256、200G；截至 2026-09-02 00:48 +08:00 已运行 04:52，最新完整训练指标 step 29/70、checkpoint `global_step_25`。后续 FullTaxonomy split Top-32 reverse-KL `top_p=0.05` 任务尚未提交。
- `Step60-Standard10-Evaluation`：ExOPD job 143、OPD job 146、TIP job 145、V1 KL Entropy control job 160、V1 Speed control job 161、Student Base job 147 与 Teacher job 148 均已完成、下载、验收和 official EvalPlus post-scoring。V0 Speed job 162 为 FAILED/1:0，失败 archive 已下载；逐 shard 审计确认仅 AIME25 task 0020 缺 16 records。heartbeat 暂停等待用户处理。远端 output root 为 `data/eval_data/results/standard10_20260828/`。
- `qwen1p7b-30b-tip-topk32-rho50-4gpu-b525`：远端任务已中断，W&B 已同步至 step 61/200。
- `qwen4b-30b-tip-topk32-rho50-4gpu-b525`：远端任务在首个 training step 前取消，尚无 history metrics。

- `qwen1p7b-...-eopd/global_step_65` 8-benchmark 评测已完成并验证；`global_step_55` 仍因 actor shards 缺失而阻塞。详见 Obsidian `Experiments/EOPD-Checkpoint-Evaluation.md` 与 `Results/Reports/2026-08-25--eopd-checkpoint-evaluation--r1--step65-8benchmark.md`。
- `qwen1p7b-4gpu-control-online-topklentropy-domaincand-v2-i3w3f20k30w4-b528`：训练 job 132 已于 2026-08-26 22:18 按用户指令取消（`CANCELLED by 1002`，elapsed `1-04:13:17`）。`global_step_60` 的 3/3 actor、optimizer、extra-state shards 与 tokenizer metadata 完整；4-GPU、400G 的统一 8-benchmark 评测 job 141 已启动，Math/Code/GPQA/MMLU-Pro-500 四分支均进入 vLLM generation。
- `qwen1p7b_30b_a3b_instruct_2507_8gpu_..._domaincand_v2_.../global_step_{70,75}`：job 134/135/136/137 均成功完成，结果已验收并扁平归档。详见 Obsidian `Experiments/Domain-Candidate-v2-Step70-Step75-Evaluation.md`。
- `qwen1p7b-30b-math-fire-opd-4gpu-b255_20260830_231139`：training job 175 与四项 Math、K=8、DP=2 partial eval job 181 均已 `COMPLETED / 0:0`；FiRE-OPD Math Avg@8 / Pass@8 为 22.19% / 40.83%，完整归档与比较见 Obsidian `Results/Math-Only-Step60-ExOPD-vs-TIP.md`。

## 近期结果
- 2026-08-31：FiRE-OPD Step60 Math-only job 181 完成、下载并验收；Math Avg@8 / Pass@8 为 22.19% / 40.83%。同协议 ExOPD / TIP / FiRE-OPD 为 26.25/46.67、22.92/39.17、22.19/40.83；FiRE − ExOPD 为 -4.06/-5.83 pp，FiRE − TIP 为 -0.73/+1.67 pp。FiRE length-cap rate 25.10%，高于 ExOPD/TIP 的 16.98%/17.40%。
- 2026-08-31：TIP-TopK32 Step60 Math-only job 176 以 `COMPLETED / 0:0` 完成。32/32 shards、120 prompts、960/960 rollouts、JSON/JSONL 与 remote/local aggregate SHA 验收通过；Math Avg@8 / Pass@8 为 22.92% / 39.17%，比同协议 ExOPD 低 3.33 / 7.50 pp。完整归档与 `Summary.md`、Obsidian result note 已同步。
- 2026-08-30：TIP-TopK32 job 173 完成 70/70 steps（`COMPLETED / 0:0`），final Hugging Face checkpoint 完整；ExOPD step60 Math-only partial eval job 174 完成 32/32 shards、960/960 rollouts，并已下载归档与更新 `Summary.md`。
- 2026-08-29：Math-only Step60 EOPD / OPD jobs 170/171 完成。四项 Math Avg@8 / Pass@8 分别为 EOPD 24.90% / 42.50%、OPD 22.29% / 40.00%，观察差异 +2.60 / +2.50 pp；AIME24/25 驱动主要 Avg@8 增益，HMMT25Nov Pass@8 则由 OPD 高 6.67 pp。该比较为单 checkpoint、单 generation schedule 的独立 Math-only snapshot，不替代 Standard10 或未完成的 ExOPD/FiRE-OPD/TIP matrix。
- 2026-08-28：V1 Speed job 161 已归档 10 datasets × K=8、1,702 prompts、13,616 rollouts；official EvalPlus job 166 完成。Math/Code/Science/Overall Avg@8 为 21.67/42.58/50.95/44.54%，Pass@8 为 37.50/57.35/79.37/64.98%；HE+/MBPP+ official Plus 为 68.22/58.80 Avg@8 与 88.41/75.40 Pass@8。
- 2026-08-28：V0 Speed job 162 为 FAILED/ExitCode 1:0。AIME25 shard 0020 的 records/summary 为 0 bytes 但遗留 SUCCESS marker，merge 期望 16 records；缺最终双 SUCCESS。失败 suite/logs 已归档并审计为 1,700/1,702 prompts、13,600/13,616 raw rollouts；未重跑、未进入 active comparison。
- 2026-08-28：V1 KL Entropy job 160 已归档 10 datasets × K=8、1,702 prompts、13,616 rollouts；official EvalPlus job 165 完成。Math/Code/Science/Overall Avg@8 为 22.60/42.73/49.87/44.24%，Pass@8 为 40.00/58.03/77.22/64.63%；HE+/MBPP+ 为 68.75/58.07 Avg@8 与 87.80/76.98 Pass@8。
- 2026-08-28：Teacher reference job 148 已归档 10 datasets × K=8、1,702 prompts、13,616 rollouts；official EvalPlus job 164 完成。Teacher Math/Code/Science/Overall Avg@8 为 58.33/58.54/73.51/64.67%，Pass@8 为 77.50/71.83/87.11/78.50%；HE+/MBPP+ 为 83.23/77.18 Avg@8 与 95.12/84.13 Pass@8。
- 2026-08-28：candidate official EvalPlus rescore jobs 152/153/154 全部完成。修正后的 Overall Avg@8：ExOPD 44.90%、OPD 43.53%、TIP 44.92%；Overall Pass@8：65.16%、64.16%、65.33%。六项 source/result SHA、K=8、task/sample count 与 SUCCESS 通过验收。
- 2026-08-28：HE+/MBPP+ forensic audit 发现 current scorer 只执行 1,181 / 1,174 条 base asserts，遗漏 source 中 122,683 / 39,841 个 Plus inputs；截图数值算术正确但不能作为 official EvalPlus。Docker scorer 另有 `os._exit(0)` fail-open，当前 outputs 未触发。
- 2026-08-28：Step60 standard10×K8 generation 完成；最初的 48.00% / 42.87% / 46.02% Overall Avg 与 68.04% / 66.69% / 68.68% Pass 来自 compact Code proxy，现已被 official EvalPlus rescore supersede。
- 2026-08-22：从远端取回上述两条 `.wandb` 日志，并使用本机 LZ101 凭据同步到 `lz101-rice-university/MOPD`。
- 1.7B run 的 W&B summary 为 `training/global_step=61`；4B run 仅含配置与 console log，不能据此做效果比较。
- 2026-08-23：统一 baseline observability contract；35 个物理 YAML 展开为 44 个运行实例，全部启用 per-domain training metrics，并为每个实例配置唯一 audit 输出目录和匹配实际 objective 的 `loss_variance_signal`。
- 2026-08-25：EOPD step65 的 8 项结果完成。Avg@K/Accuracy：AIME24 33.125%、AIME25 26.875%、HMMT25Feb 16.250%、HMMT25Nov 13.125%、HumanEvalPlus 59.756%、MBPPPlus 58.598%、GPQA-Diamond 34.848%、MMLU-Pro-500 57.850%。
- 2026-08-26：Domain Candidate v2 step70/step75 评测完成。Step75 相对 step70：
  Math +1.46 pp、Code -0.92 pp、Science -0.93 pp、overall -0.26 pp
  Avg@K；当前多域整体以 step70 略优。

## 最近同步状态
- 2026-09-03T03:58:25+08:00：取消停滞且不会热加载修复的 job 199；job 200 因
  W&B `resume=never` 冲突失败后，将目标配置修正为 `resume=must`，job 201 从 step 15
  完整恢复。四次约 30 分钟健康检查从 step 17 推进到 step 22，均为 `RUNNING` 且无
  fatal error。step 20 的 Avg@4 为 AIME2024 0.3083、AIME2025 0.2750、HMMT25Feb
  0.2000、HMMT25Nov 0.1417，checkpoint 完整。远端受影响测试 9 passed，修复 commit
  `e231273882951ff2f16984efba847b08b3ce19dd` 已推送并与 `origin/main` 对齐。
- 2026-09-03T01:33:08+08:00：定位 FullTaxonomy Top-P=0.01 resume job 199 的
  停滞点为 resume 后第一批 DeepMath reward 的串行 Math-Verify，而非 checkpoint、
  distributed worker 或 validation。新增 pickle-safe bounded batch scorer，目标配置改为
  `reward_manager=batch`，并将 validation 设为 stochastic Avg@4（native metric
  `mean@4`）。本地 targeted tests 为 7 passed / 1 skipped / 1 deselected；远端
  dynamic custom-module + spawn scoring smoke 与完整 Hydra compose 通过，6 个同步文件
  SHA256 一致。job 199 仍保留运行，未自动取消或重提。
- 2026-09-02T18:38:18+08:00：为 V3 next-step A/unified/i1/w1/K25 与
  next-window A/unified/i7/w7/K41 生成 4--8 GPU Math-only standalone full configs，
  使用 effective 115-token unified whitelist。topology/batch 覆盖 3a1t/255、
  4a1t/256、4a2t/256、5a2t/255、6a2t/258；30 tests 与 10/10 launcher dry-run
  通过，训练尚未提交。
- 2026-09-02T18:08:27+08:00：完成 ExpandedPruned-V3 四 baseline × 三 domain
  selector replay。搜索 A/C/E/F、split/unified、K1...100、window/pre-window1...20；
  occurrence 严格 >20，验证 20/20 通过。主 Type F1 下 next-step 为 unified
  A/i1/w1（pre-update source）/K25（0.2050），next-window event-supported 为 unified A/i7/w7/K41
  （0.3048）；原始稀疏长窗口极值不作为推荐。
- 2026-09-02T17:17:42+08:00：按用户确认将 ExpandedPruned-V3 seed source 从单条
  4B-OPD phase union 改为 four-baseline cross-size consensus。Small 共 170 个
  domain-aware entries；sibling closure 后 847，occurrence pruning 后 414；V2 hash
  保持不变，V3 phase/baseline provenance 与 active-taxonomy audit 全部通过。
- 2026-09-02T06:43:01+08:00：冻结 ExpandedPruned-V3 pool construction、phase
  provenance、hash 与 active-taxonomy audit；该 4B-OPD-only seed 定义已在 17:17
  被 four-baseline cross-size consensus 取代。完整 replay 尚未执行。
- 2026-09-02T06:19:19+08:00：根据用户进一步澄清，将 Online Control `top_p` 最终固定为现有 `online_control_occurrence_fraction` 的目标：按 score 排序 token type，累计 occurrence count，取使 `selected_occurrences / valid_token_count >= p` 的最短前缀。selector history 新增 per-domain valid-token denominator，窗口内分子分母同步累计；grouped taxonomy 按 domain union 选择，JSONL 使用 `top_p_basis=selected_occurrences_over_valid_tokens`，并显式记录 target count、是否达到及 shortfall。schema 升至 v8，旧 v7 Top-P history 因缺分母会安全清空，156 项针对性测试通过。
- 2026-09-02T01:46:00+08:00：按用户新增要求，将 heartbeat `5gpu-taxonomy` 改为每 30 分钟执行两阶段流水线。阶段 A 在当前 ExpandedPruned-V2 job 189 的 `global_step_60` 完整保存后，提交 DP1、Math4、K=8、seed42、16 shards/dataset（64 shards / 960 rollouts）、400G/24h 的 partial evaluation，且用 manifest model path 去重；阶段 B 保持 job 189 正常完成后启动 FullTaxonomy `top_p=0.05` 5-GPU 训练。只有两个阶段均已提交/确认存在后才暂停 heartbeat。
- 2026-09-02T00:48:39+08:00：远端 job 189 为 `RUNNING`，最新完整训练指标 step 29/70，最新完整 checkpoint 为 `global_step_25`；日志仍在增长，未发现 Traceback、CUDA OOM、NCCL fatal 或非零退出。创建每小时 heartbeat `5gpu-taxonomy`，完成门禁通过后将用 `MOPD_SLURM_MEMORY=400G ./slurm.sh` 提交 FullTaxonomy split Top-32 reverse-KL `top_p=0.05` 5-GPU config，并在成功启动后暂停 heartbeat。
- 2026-09-01T19:43:23+08:00：Online Control token selection 接通 `control_token_online_budget_mode: top_p` 与 `control_token_online_top_p` 的配置/state/logging 路径，旧 checkpoint 自动迁移为 `top_k`；其最终选择语义已于 2026-09-02 按用户澄清固定为 selected-token occurrence coverage of all valid tokens。
- 2026-09-01T11:03:38+08:00：完成 2025-09-01—2026-09-01 Agentic OPD primary-source 调研。技术主线收敛为 trajectory/occupancy control、selective intervention、exact state validity 与 outcome/cost/support audit；下一 research gate 是在 exact same state 上做 student / teacher-bridge paired continuation，检验 ASCR、KL、entropy、future teacher preference 对真实 outcome delta 的 calibration。详细笔记见 Obsidian `Papers/2026-09-01--Agentic-OPD-Literature-Roadmap.md`，repo 报告见 `../plan/2026-09-01--agentic-opd-last-year-literature-roadmap.md`。
- 2026-08-31T23:24:02+08:00：FiRE-OPD eval job 181 于 22:19:14 `COMPLETED / 0:0`；223-file suite 与 Slurm log 下载到本地，32/32 shards、120 prompts、960/960 rollouts、38 SUCCESS、75 JSON、43 JSONL、remote/local content-check 与独立指标复算通过。新增 `RUN_MANIFEST.md`、三方法 paired analysis/figures，更新 canonical `Summary.md`、Obsidian experiment/result/daily/plan/hub。
- 2026-08-31T21:32:41+08:00：确认 CityU 节点为 6×H200；训练 job 179 占用 4 张后仍有 2 张 scheduler-free GPU。`fire-opd/global_step_60` 的 3/3 actor shards、HF tokenizer/model metadata 与 4.06 GB model 均通过 preflight；远端 parallel-eval 13 tests、standard-suite 9 tests、seeded CUDA witness 通过。按 ExOPD/TIP 同协议提交 FiRE-OPD Math-only partial eval job 181：四项 Math、K=8、DP=2、8 shards/dataset、seed 42、200G、24h。job 已 `RUNNING`，两个 H200 worker 完成 CUDA witness，32-shard manifest 已建立，输出为 `data/eval_data/results/partial_math_20260831/fire_opd_step60_math4_k8_dp2_20260831/`。
- 2026-08-31T02:45:00+08:00：确认 TIP-TopK32 Math-only partial eval job 176 为 `COMPLETED / 0:0`；下载完整 suite 与 Slurm log 到 `experiments_records/eval/tip_step60_math4_k8_dp2_20260830/`。远端/本地 suite aggregate SHA 一致；75 JSON、43 JSONL、32/32 shards、120 prompts、960/960 rollouts 与独立指标复算全部通过。`Summary.md`、实验笔记、结果笔记、Daily 与计划已同步。
- 2026-08-30T23:32:00+08:00：确认节点剩余 2 张 H200 后，以 TIP-TopK32 `global_step_60` 提交 Math-only partial eval job 176。协议固定为四项 Math、K=8、DP=2、8 shards/dataset、seed 42；Slurm 实际分配 2×H200、64 CPU、200G，两个 worker 已开始生成，输出根目录为 `data/eval_data/results/partial_math_20260830/tip_step60_math4_k8_dp2_20260830/`。
- 2026-08-30T23:13:00+08:00：同步缺失的 FiRE-OPD config/method fragment 后完成远端 dry-run，提交 Math-only 4-GPU/batch-255 FiRE-OPD job 175。Slurm 实际分配 4×H200、64 CPU、400G；W&B `o5bon8c8` 已开始同步，训练初始化无 fatal error。
- 2026-08-29T19:58:00+08:00：jobs 170/171 均 `COMPLETED`/ExitCode 0:0；两组完整 Math-only outputs 与 Slurm logs 已下载到 `experiments_records/eval/math_only_{eopd,opd}_step60_k8_dp1_20260829/`。本地/远端文件数、bytes、aggregate SHA 与 log SHA 一致；120 题 × K=8 独立重算通过。`Summary.md` 新增独立 Mass Only 区块与分析 bundle，Obsidian experiment/result/daily 同步。
- 2026-08-29T17:02:23+08:00：纠正方法为 ExOPD。取消误提交的 EOPD job 169（最后完整 step 12），新增并验证 ExOPD lambda 1.25 的 Math-only 4-GPU/batch-255 config，提交 job 172（W&B `tfvf42rx`）。创建每小时 heartbeat，按完整 step 65 checkpoint 将 ExOPD → FiRE-OPD → TIP-TopK32 串行切换，并明确排除评测 jobs 170/171。
- 2026-08-29T15:26:05+08:00：按用户指令重新提交 Math-only EOPD 4-GPU baseline。73 项配置测试通过；同步当前 70-step `_common.yaml`，以 4×H200、64 CPU、400G、72h 提交 Slurm job 169。W&B run 为 `rohnsloa`；EOPD config/data/reference tokenizer/NCCL 已通过，尚无完整 training step。
- 2026-08-29T01:11:34+08:00：按用户指令提交 Math-only EOPD 4-GPU baseline。首次 job 167 因旧 `/home/shuang_qiu/mopd` 部署树被删除而在 model path 解析阶段 fail-fast，无 training step；已用仓库脚本重新下载并校验 Qwen3-1.7B、Qwen3-30B teacher 与 data，重建 compatibility symlinks。job 168 以 4×H200、400G 启动，W&B `u84zyvpb`；step 1 于约 296 秒内完成，actor peak memory 98.65 GB，无 traceback/OOM。
- 2026-08-28T23:54:02+08:00：按用户要求下载 V0 Speed 失败 suite 的 866 个文件与 jobs 159/162 outer logs；checksum dry-run 仅根目录 mtime 不同。逐 shard 审计确认唯一 canonical 缺口为 AIME25 task 0020 的 2 prompts / 16 rollouts，其余 159 tasks 完整。新增 `FAILURE_REPORT.md`，并在 `Summary.md` 状态表登记 failed/incomplete；active 指标与 heartbeat 状态不变。
- 2026-08-28T23:08:59+08:00：V1 Speed job 161 与 official EvalPlus job 166 已完成、下载并通过 generation/rescore checksum、JSON/JSONL、13,616 rollouts、model/source/result SHA、task/sample/K 验收；核心 Overall 为 44.54/64.98 Avg@8/Pass@8，已写入 `Summary.md` 与 Obsidian comparison。V0 Speed job 162 为 FAILED/1:0；AIME25 shard 0020 空 records 但有 SUCCESS marker，merge fail-closed，未自动重跑。`opd-control` heartbeat 暂停等待用户决定。
- 2026-08-28T22:10:41+08:00：V1_Speed job 161 为 156/160（97.50%），V0_Speed job 162 为 154/160（96.25%）；两项均处于最后的 MMLU-Pro-500 wave、failed=0，四张 H200 持续有生成负载，日志未见 Traceback/OOM/CUDA/非零退出。两项尚未生成 `STANDARD_SUCCESS` / `parallel/SUCCESS`，未下载中间 artifacts。
- 2026-08-28T21:34:19+08:00：V1_KL job 160 完整 suite 与 Slurm logs 已下载；1,167,194,887-byte remote payload checksum dry-run 为 0 differences，160/160 shards、10/10 waves、JSON/JSONL、K/model/source/result hash 验收通过。official EvalPlus job 165 exit 0；结果已同步 `experiments_records/eval/Summary.md`、forensic audit 与 Obsidian experiment/result/daily/hub。jobs 161/162 最新为 144/160、142/160，failed=0。
- 2026-08-28T21:26:57+08:00：按用户要求将本地项目 skill 更名为 `Cityu-A100 Skill`（id `cityu-a100`），并更新 invocation 与全部文档路径。Burgundy 仅有 VS Code 内的 password-authenticated 交互会话，新的 BatchMode SSH 被拒绝，因此远端目录更名待下次 authenticated sync；本轮未改动远端目录。
- 2026-08-28T20:44:59+08:00：新增并验证现名为 `Cityu-A100 Skill` 的项目 skill，包含实时资源脚本、A100/Slurm 申请说明与 OPD bootstrap runbook；官方 skill validator、Shell syntax、引用完整性与 Burgundy 实机 inventory 均通过。更名前版本曾同步至 Burgundy。当前 Slurm capacity snapshot 显示 node05 有 2、node07/11/12 各有 1 张 scheduler-free A100；其中 05/07/11 已观测到 reset-required，12 的健康状态尚未验证，不能把 scheduler-free 直接视为可运行。
- 2026-08-28T20:43:42+08:00：V1_KL job 160 于 20:30:39 `COMPLETED`，ExitCode 0:0、160/160 shards、10/10 waves、failed=0、`STANDARD_SUCCESS` 与 `parallel/SUCCESS` 均通过，等待 `opd-control` heartbeat 自动下载、official EvalPlus 和 Summary 归档。jobs 161/162 为 124/160、123/160，均在 LCB v6、failed=0。
- 2026-08-28T20:02:45+08:00：control jobs 160/161/162 均 RUNNING，完成 150/160（最后的 MMLU-Pro-500）、111/160（LCB v5）、111/160（LCB v5）shards，三项 failed=0，尚无 `STANDARD_SUCCESS`。六张 H200 均保持约 122.8 GB resident memory，worker logs 持续更新。
- 2026-08-28T18:56:10+08:00：control jobs 160/161/162 均持续 RUNNING；完成 shard 为 125/160（LCB v6）、87/160（MBPPPlus）、86/160（MBPPPlus），三项 failed=0。六张 H200 显存均约 122.8 GB，GPU utilization 48–100%，worker logs 持续更新。
- 2026-08-28T18:26:10+08:00：Teacher job 148 archive 已下载 1,567 files / 472,381,323 bytes，checksum dry-run 为 0 differences；official EvalPlus job 164 exit 0，source/result SHA、164/378 tasks、1,312/3,024 samples、K=8 与 SUCCESS 均通过。`Summary.md`、Obsidian experiment/result/daily/hub 已同步。首次 job 163 因旧 pinned source path 缺失而 fail-fast；worker 已支持 source env override，remote pinned source hash 验证后重提成功。
- 2026-08-28T17:47:19+08:00：Teacher job 148 已 `COMPLETED`，160/160 shards、failed=0、`STANDARD_SUCCESS`。节点于 17:40:22 导致 control jobs 157/158/159 `NODE_FAIL`；自动 requeue 因未传 `--resume` 立即失败。已用同一 run tag 和 `--resume` 提交 jobs 160/161/162，三项均 RUNNING、各占 2×H200；恢复点分别为 109/160、21/160、21/160，failed=0。服务器 `/dev/sda2` 为 7.0T，已用 3.3T、可用 3.3T（50%），inode 使用 1%。
- 2026-08-28T15:40:00+08:00：远端 `144.214.166.120:22` 连续三次 SSH/TCP 连接超时，无法复核 jobs 148/157/158/159 的当前 Slurm 与 shard 状态。最新可信快照仍为 12:36；本轮未下载可能不完整的 artifacts，也未生成最终 `summary.md`。
- 2026-08-28T12:36:05+08:00：Teacher job 148 完成 151/160 shards（94.38%），正在最后的 MMLU-Pro wave（7 done / 3 running / 6 pending）；V1_KL job 157 完成 90/160（56.25%），正在 MBPPPlus（10 done / 2 running / 4 pending）。两项均 0 failed。V1_Speed 158 与 V0_Speed 159 继续等待 Resources/Priority。
- 2026-08-28T11:06:58+08:00：三个 control checkpoint 均通过 Step60/FSDP shard/dry-run/重复任务 preflight，并以 Standard10×K8、DP2、200G、24h 提交为 jobs 157/158/159。job 157 已确认 `gpu_count=2`、`step60_candidate`、checkpoint_step=60、CUDA witness=2 并进入 RUNNING；jobs 158/159 等待资源。
- 2026-08-28T10:49:51+08:00：Student Base job 147 保持 complete；Teacher job 148 持续 DP3 RUNNING，累计完成 110/160 shards（68.75%）。前 6 个 waves 完成，LCB v5 为 14 done / 2 running / 0 pending / 0 failed；完成这两个 shard 后将进入 LCB v6。当前队列中没有另一条独立的 Teacher Base job。
- 2026-08-28T10:28:19+08:00：Teacher reference job 148 持续 DP3 RUNNING，前 6/10 个 dataset waves 已全部 16/16 shards 完成；当前 LCB v5 为 3 done / 3 running / 10 pending / 0 failed。GPU 1/4/5 各占用约 122.8 GB，利用率 42–82%；GPU 0/2/3 空闲。
- 2026-08-28T10:04:39+08:00：ExOPD/OPD/TIP raw completions 已经 pinned G-OPD exact sanitizer 与 full base+plus evaluator 重评分；jobs 152/153/154 exit 0，artifacts 下载并完成 K/hash/SUCCESS 验收，active Summary/Obsidian comparison 已更新为 official Plus。
- 2026-08-28T09:19:39+08:00：远端复核确认 Student Base job 147 于 08:56 完成，10 datasets × K=8、1,702 prompts、13,616 rollouts 通过 outer finalize 并生成 `STANDARD_SUCCESS`；Teacher job 148 仍为 DP3 RUNNING，三张 H200 显存约 122.8 GB、利用率 55–62%，GPU 0/2/3 空闲。为避免重复 GPU 消耗，本轮未重复提交 reference jobs。
- 2026-08-28T08:58:16+08:00：HE+/MBPP+ official claim 审计 verdict 为 FAIL；active Code/Overall conclusion 暂停，保留 raw completions 并计划 official offline re-score。审计报告与 Obsidian result/plan 已同步。
- 2026-08-28T08:27:13+08:00：jobs 143/145/146 已完成、下载并通过 checksum 与结构化校验；6 张空闲 H200 已拆成两个 DP3 reference tasks，Student Base job 147 与 Teacher job 148 同时 RUNNING。`Summary.md`、analysis bundle 与 Obsidian canonical result 已同步。
- 2026-08-28T02:34:05+08:00：取消 OPD DP4 job 144 并以 DP2 job 146 替换；launcher/manifest 已参数化并通过本地、远端 7 tests 与独立 review。job 146 已获 2×H200、200G 并进入 RUNNING，outer state 记录 `gpu_count=2`、`K=8`。
- 2026-08-28T02:13:42+08:00：job 143 前 6 个 waves 均 16/16 shards、0 failure，当前 LCB v5 有 4 running / 12 pending；jobs 144/145 因 4-GPU resource envelope 排队。
- 2026-08-28T01:05:12+08:00：统一 DP4 strict-wave evaluator 已通过本地/远端门禁；远端 LCB source、原 G-OPD formatter prompt 与 offline tokenizer witness 已验收，jobs 143/144/145 已提交。
- 2026-08-27T23:02:41+08:00：下载 official LCB v5 `v5/` split（167 题）并生成独立 artifact；v5/v6 question ID overlap=0，exact prompt/data_source/manifest hash 校验通过。official runner 固定 `Qwen3-4B-NonThinking` style，v5/v6 默认各 K=8。本轮未提交 GPU job。
- 2026-08-27T22:31:43+08:00：固定 Code user-prompt contract；HumanEvalPlus/MBPPPlus 使用原 G-OPD 3-newline join，LiveCodeBench 使用 `Qwen3NonThinking` template 与 `enable_thinking=false`。三个本地 Code artifacts 已重建并通过 164/378/175-row 与 manifest/hash preflight。本轮未提交远端 GPU job。
- 2026-08-22T10:04Z：TIP-TopK32 1.7B/4B 日志已重新归档到 LZ101 entity；远端当前无 Slurm job。
- 2026-08-22T10:07:28Z：范围 `auto`，git head `395f5f467fafbf5adf4d031dbce2b39eb3b26e6c`，变更文件数=0（无可追踪变更）。
- 已于 2026-08-22T10:07:28Z 完成 bootstrap。
- 2026-08-22T20:41:23+08:00：新增 `mopd_qwen1p7b_30b_a3b_instruct_2507_7gpu_math_code_science_topk32_control_online_topklentropy_i3_w3_f20_k30_w4_b528.yaml`；保持训练参数不变，仅将资源拓扑从 6+2 调整为 6+1，并同步运行标识。
- 2026-08-23T03:53:59+08:00：baseline domain-metrics 配置与回归测试完成；109 个针对性测试及 ruff 检查通过。
- 2026-08-24：补齐 6-GPU Top-KL+Student Entropy 配置（5 student + 1 teacher），并复核 7/8-GPU 配置分别为 6+1 与 6+2；YAML 结构校验及归一化差异检查通过。
- 2026-08-25T23:46:00+08:00：job 132 训练监控与完成后 8-benchmark 评估/下载 heartbeat 已创建；评估口径固定为 4×Math、HumanEvalPlus、MBPPPlus、GPQA-Diamond 与 MMLU-Pro-500 seed42，共 6,880 rollouts。
- 2026-08-26T22:27:00+08:00：按用户指令停止训练 job 132，并以 `start.sh` 生成的 Slurm resource envelope 提交 step60 评测 job 141（4×H200、400G、48h）；FSDP merge 已验证，四个 benchmark 分支并行运行。
- 2026-08-26T22:51:00+08:00：新增 Math-only OPD/FiRE-OPD/TIP-TopK32/EOPD × 4–8 GPU 配置矩阵；batch 255/256/255/258/259 均可被 actor world size 整除，30 项测试与 20 个 launcher dry-runs 通过。
- 2026-08-26T23:15:00+08:00：step70/step75 的四个评测任务全部成功；
  158 个远端 payload、901,053,599 bytes 已归档并通过 SHA256、样本协议与
  checkpoint 路径验收。
- 2026-08-27T02:19:20+08:00：完成 step-selective Hugging Face checkpoint upload；
  配置、save/resume receipt、单节点 shard 安全约束与 exported `HF_TOKEN` 路径已接通，
  81 项针对性测试通过。
- 2026-08-27T04:36:26+08:00：按用户指令提交 Math-only OPD 4-GPU baseline；Slurm job 142 已进入 `RUNNING`，W&B run 为 `gg3m2ype`，启动阶段未见 blocking error。


## 2026-09-05：V3 与 FullTaxonomy 选中数量口径

- 实际 Math V3 effective pool（85 Control + 30 Structure = 115 IDs）是 FullMath taxonomy（124 + 266 = 390 IDs）的子集；raw V3 146 IDs 必须先与 domain taxonomy 相交。
- 当前 Top-P 按 score 降序累计 token ID 的 occurrence，直到 selected occurrences / all valid tokens >= p；并非选候选 ID 数的 p 比例。较小候选池可以因跳过高分低频 IDs 而用更少 IDs 达到相同覆盖，或因池容量不足而耗尽，需检查 target_reached/shortfall。
- 用户提及 taxonomy 80 多、V3 60 多；尚未指定两条 run/step，不能将这些近似数当作已审计指标或确定归因为频率分布。需对齐 p、score、source window 与当步统计。Top-P 对 taxonomy domain union 统一选择，文件名 split 不代表每类各 p。
- 依据：code/AGENTS.md 的 Candidate-pool intersection、control_selection_budget.py::select_ranked_tokens、control_top_loss.py 的 Top-K-only grouped branch。本轮仅核对定义与实现，未改训练配置或运行任务。


2026-09-07 final archive: Job217 COMPLETED (0:0), elapsed05:11:39; 1H200/100G, Math4 K8. Downloaded all results/logs; verified64 shards,120 prompts ×8=960 scored rollouts. Avg@8=222/960=23.125%; Pass@8=51/120=42.50%. Updated experiments_records/eval/Summary.md. User requested deletion of automation v3-k25-step60-math; deletion confirmed. scancel was sent before clarification, but job had already completed and was not interrupted.


## 2026-09-07：α1.75 / 8×L20Y GPU利用率调查

- Run：q1p7b-math-top32kl-topp05-lossratio-a1p75-8g6a2t-b258。W&B约12:22快照包含step1–64、4344条15秒GPU采样；目标启智节点不是项目现有SSH服务器。
- Step1至64指标时间窗口，八卡时间加权利用率26.46%；GPU0–5约30–32%、6–7为12.95/14.10%。以组内任一卡>5%定义活跃，actor-only56.38%、teacher-only34.76%、both0%、均低活动8.86%；0%阈值有0.17%短暂重叠，不宣称绝对互斥。
- Step1–64 training timer均956.94秒，gen442.89秒、actor update93.74秒，420.31秒残差未精确拆分。Teacher活跃时两卡均38.91%；主问题是分组互等叠加推理阶段低效率，优先查rollout长尾、teacher FSDP/microbatch1、全词表Top32 chunk16。
- 远端metadata commit e60ae1d与本地eb85612不一致且对象不可读；代码机制是本地实现诊断，未核验目标节点hash/NCCL trace。Teacher高显存是稳定平台，不能据此认定leak；alpha乘法并无证据是瓶颈。
- 完成独立统计review，报告/原始快照/复算脚本/PNG/PDF保存在 code/profile_output/lossratio-a1p75-8g6a2t-20260907/。未修改训练源码或配置，未启动/停止/重启任务。
- 报告：/Users/linghuazhang/Desktop/Project/OPD/code/profile_output/lossratio-a1p75-8g6a2t-20260907/REPORT.md。后续需完整stage timer和目标节点trace；本轮不创建monitor。


## 2026-09-07：Teacher/rollout优化方案

- 用户要求给出方案，本轮未实施或启动benchmark。方案：code/plan/teacher-rollout-optimization-20260907.md。
- Teacher先补独立chunk透传（当前actor chunk不影响teacher），测试16/64/128，再实施有效tokens≤18432、最多2条的受限分批，最长输入singleton；现有dynamic batching不是硬cap，且预算须≥padding宽度，不能盲设8k/16k。对齐teacher rank forward数并恢复sample顺序。
- Rollout先验证graph与sleep/wake/权重更新；V0 capture上下文默认8192，考虑18432覆盖；保持默认capture sizes，独立试并发24/32/48，显存占比先0.60。Chunked prefill独立验证；动态跨engine队列仅作为长尾仍严重的后续改造。
- 以固定工作量stage/step秒数、数值一致性和每卡live/reserved峰值验收，不预承诺提速；方案经独立只读核查，保留远端代码版本尚未核实的边界。


## 2026-09-07 13:24：Job218 teacher 实测调优已生效

- Run q1p7b-v3-topp05-lossratio-i1w1-5g4a1t-b256-r20260907，5H200/500G，4actor+1teacher。物理GPU5/PID266206为teacher；物理GPU4是独立评测219，未改动。
- 初期5卡均值19.65%，teacher活跃时39.61%，约67.86%采样为teacher忙/actor等待。
- 用户授权扩大batch；真实16样本长度715–16485、总58346tokens。chunk256 + 最多16条/硬token cap49152，完整同输入compute由21.075秒降到warm5.23秒约4×；Top32 IDs完全相同，chosen logprob误差0，topk logprob最大差3.8e-6。15个CPU测试及GPU端到端对照通过。
- 13:24:32已可逆runtime apply到原teacher，不改actor/loss/alpha/rollout，不重启任务。注意启动YAML/W&B config仍保留旧值；W&B summary runtime_teacher_tuning_20260907记录override，重启后失效。回执及rollback_live.py在profiling目录。
- 首个完整256条teacher：1,114,225tokens、27 microbatches、最大49,010tokens、85.469秒、allocated采样峰72.13/139.79GiB、teacher阶段1秒NVML平均69.74%。旧step9 575.02秒→新step10 397.83秒（-30.81%，response长度不同，非严格等量A/B）；训练正常继续。
- 当前vLLM0.23，rollout concurrency/graph需engine重建，不能套用旧0.8.5 max_seq_len_to_capture。详见ROLLOUT-V023-REVIEW.md；本轮未改rollout。
- 本轮新增profiling/测试/报告，未改生产训练源码或commit。Source of truth：/Users/linghuazhang/Desktop/Project/OPD/code/profile_output/v3-5gpu-tuning-20260907/REPORT.md。


## 2026-09-07 14:16：Job218 teacher MoE 再优化已生效

- 在现有chunk256/mb16/token cap49152上叠加stable-sort MoE dispatch，只替换48个teacher MoE instance forwards。PID266206，token42214e66-f172-4798-81b0-5b46c8a62837；重启失效，保留精确rollback。
- 同58,346 tokens六轮交替A/B：stock4.20s→stable2.82s（-32.93%），Top32/chosen logprob完全一致，allocated峰73.74GiB。新64条421,366tokens完整验证35.642s→27.738s（-22.18%）；13,302,784项Top32 entries exact match。源码hash、实际输出PT及两级数值门槛均复核后上线。
- Job220独立1H200/100G五档rollout固定415,712 decode-token回放全部完成：eager24 167.28s；graph24/32/48/64=142.11/127.19/130.67/126.44s，graph利用率约99.7–100%，preemptions0。32与64差0.6%，尚需external_launcher后端复核；active训练rollout仍eager24。
- 后续正在测teacher最多32条/57344tokens及external eager24/graph32/graph64。第二轮报告：/Users/linghuazhang/Desktop/Project/OPD/code/profile_output/v3-5gpu-tuning-20260907/PHASE2-REPORT.md。未修改生产训练源码或commit。


## 2026-09-07 14:48：Teacher/rollout第二轮调优收尾

- teacher32/57344已于14:48:55实际apply，chunk256与stable_sort保持，readback验证compute及48MoE ownership。新batch token1d54909d-2e2b-4977-9aa8-c9ac2ca33940；三层重启失效，回滚须batch→MoE→原teacher tuning。
- 64条421366tokens四轮A/B：baseline34.666(冷)/27.790(暖)，candidate24.542/24.170秒，11→8个microbatch；最慢candidate对最快baseline仍减少11.69%，allocated峰73.57GiB，Top32 IDs/支持100%，chosen误差0，topk最大3.8e-6。
- MoE先行上线后的真实完整teacher：1911791tokens/122.402秒，teacher阶段利用率79.83%，allocated峰72.08GiB；与早期69.74%不是等输入A/B。新57k batch完整训练收益尚未单独对齐统计。
- external_launcher复核job221全3case完成：eager24=178.241s，Graph32=141.218s，Graph64=130.976s；Graph64降时26.52%，NVML生成期82.96%→99.92%，峰84.29GiB、preemptions0、sleep/wake通过。该单engine后端与进程形态对齐，不代表4-engine FSDP weight-sync集成验证已完成。
- 根据新后端证据，首选Graph64，32为长回答/preemption回退；.6显存比例/32768tokens/chunked_prefill=false保持。原job218 rollout仍eager24，重建engine才生效；两份候选YAML已load_config验证仅改2字段，未部署。
- 220/221均COMPLETED 0:0；额外GPU释放，900秒采样器已结束，218仍RUNNING。JSON/CSV/回执本地归档，大64条PT在远端保留及无损gzip（SSH下载曾超时，不宣称全部本地归档）。报告profile_output/v3-5gpu-tuning-20260907/PHASE2-REPORT.md。无生产源码修改/commit/push。

W&B最新batch summary字段写入后未能独立读回；当前有效设置以final-runtime-readback.json和apply-batch-m32-t57344.json为准，未将界面同步当作生效证据。


## 2026-09-07 15:22：Teacher chunk1024 定向测试

- 用户要求仅测chunk1024。job218/PID266206原teacher完成一次256 warmup后ABBA256/1024/1024/256；固定64条421366有效tokens、mb32/token cap57344、8组原序batch、stable-sort MoE。
- 正式256耗时24.566/24.519s，1024为23.567/23.781s；均值24.542→23.674s（耗时-3.54%，保守-3.01%）。
- NVML活动期均值85.06%→82.19%，未提高、更未打满；不能将小幅时间收益声称GPU利用率提升。10ms allocated采样峰73.57→74.44GiB（+0.87GiB）；NVML设备峰值均约139GiB，含cache/context。
- chosen logP、Top32 IDs完全一致，Top32logP最大差3.8147e-6，同chunk重复exact，数值门槛通过。每组保留实际chunk显存guard与至少12GiB余量。
- 运行回执记录forward恢复256、mb32/57344与48层MoE保留；测试没有部署1024，没有修改正式训练配置。较早统一配置固化请求仍未完成。
- 报告：/Users/linghuazhang/Desktop/Project/OPD/code/profile_output/v3-5gpu-tuning-20260907/teacher-chunk1024-ab/REPORT.md；原始输出PT保留远端，本地返回JSON/source已归档。新增probe CPU4tests通过，独立review通过。


## 2026-09-07 15:48：全部配置性能优化已固化并同步

- 已完成之前“所有config统一优化”的请求。覆盖本地/远端合计291 YAML、308展开入口；163 YAML实际更新，其他继承。302可运行，2abstract和4legacy-invalid EOPD仍保留原有算法配置；后者原错误为rollout_correction.rollout_is必须null。
- 所有训练config请求rollout enforce_eager=false/max_num_seqs64，teacher_performance启用chunk1024、max_micro_batch_size32、max_tokens57344、stable_sort MoE及12GiB margin。student训练chunk/算法/全局batch/采样与GPU分配未改。
- 新生产teacher_performance配置/运行/MoE模块和fsdp_workers初始化hook使后续启动可自动应用。仅单rank resident GPU BF16 remove-padding/unfused/SP1启用teacher优化；多rank/offload等有明确fallback。逐组按显存缩batch，单条超预算保留chunk16。MoE要求48层、HF4.57.6及原forward SHA。
- chunk1024只是约3.54%耗时收益，不声称提高NVML利用率；Graph64旧实测降时26.52%。所有topology并未逐个GPU benchmark。
- 本地相关209回归通过、最终teacher/config35专项通过、扩展配置22回归通过；独立review无blocking。rsync dry-run后同步，远端备份已保存，不使用delete。两份历史远端TopP0.1 overlay原有身份/验证差异保留，只继承父级性能设置。
- 远端208文件hash核对通过，302可运行入口和全部raw性能配置通过；实际Hydra传参、Transformers4.57.6 48个tiny真实MoE原/新CPU输出exact通过。未进行fresh四engine GPU重启/权重同步集成smoke，未停止/重启218（读回仍RUNNING），因此运行中rollout仍沿用旧加载配置，磁盘新配置在后续启动生效。
- 完整报告：/Users/linghuazhang/Desktop/Project/OPD/code/plan/performance-config-rollout-20260907/README.md；部署manifest/本地与远端before备份/remote-verification.json同目录。未commit/push。


## 2026-09-07 16:10：5GPU性能重启尚未执行，SSH握手被远端关闭

- 用户明确授权应用性能配置并重启当前5GPU训练。原job218最后成功读回RUNNING、elapsed3:54:24、完成step24/70、正在step25；完整checkpoint tracker20。为避免回退，准备在25完整保存后切换。
- 全局性能源码/config已于15:48同步；本轮远端prepare再次核对208文件hash通过，sbatch dry-run确认5GPU/500G/64CPU/compute/72h、Graph64、teacher1024/32/57344/stable_sort/12GiB、总70steps。restart.yaml使用W&B must，最终CLI resume_path覆盖继承的disable。
- checkpoint20四rank model/optim/extra ZIP和optimizer内loss_ratio alpha1 state可读，含scheduler/RNG/data；后续本地gate加固真实schema10、完整selector契约/四rank hash、FSDP topology、文件稳定性、pin后二次验证。独立review与py_compile通过；加固版gate尚未同步成功/远端验证。
- 约16:05起SSH在认证前kex阶段被远端主动关闭；多次间隔重试失败。没有发出scancel218，没有提交新job，没有实际创建pin。因此不能声称已重启或新配置已在训练进程生效。
- 计划与可复用恢复脚本：`plan/restart-5gpu-performance-20260907/README.md`。网络恢复后先重新sync/prepare/gate并读取最新step，若已超过25则重新选择完整checkpoint边界；不得盲目执行固定25脚本或重复sbatch。旧job当前状态因SSH不可用尚未核实。
- main仍保留已有未提交修改；无commit/push。


## 2026-09-07 16:38：5GPU性能重启完成，正式job223已完成恢复首步

- SSH恢复；原job218已于16:13:48 FAILED/1:0，无需再取消。失败位于step26 teacher F.linear，申请16.22GiB失败，allocated58.41GiB、reserved-unused78.63GiB、free1.42GiB，提示碎片化风险。checkpoint25完整。
- 用户再次明确授权终止旧任务并重启。模型/optimizer/RNG/scheduler/dataloader及真实schema10 selector恢复校验通过，四rank完整selector hash一致；source25与hardlink pin两次验证通过。
- 新增teacher allocator小补丁，仅ref角色且worker_placement.separate_ref_policy=true（独立pool）、现有优化guard通过后设置expandable_segments=True；不影响共享pool/vLLM。36专项测试通过，独立review补齐shared-pool角色guard，远端manifest208文件校验通过。
- 首次job222只打印dry-run命令、1秒COMPLETED/0:0，没有训练；修正生成sbatch遗留--dry-run后16:28:11正式提交job223。不得重复提交。
- job223 RUNNING，5GPU/500G/64CPU，4actor+1teacher、batch256、总70steps，同W&B identity/must；从pin/global_step_25恢复，四卡CUDA Graph捕获与恢复权重同步成功。step26完整完成，已进入step27 rollout。
- teacher进程342192读回chunk1024、batch32、max_tokens57344、margin12GiB、48MoE、allocator expandable_segments=true。首个完整teacher成功，allocated峰75.417GiB，结束后56.903GiB/缓存reserved77.254GiB；未再OOM。
- step26 rollout149.341s/actor update140.988s/整步427.514s，response1900291tokens；旧step25 rollout285.327s/整步575.944s，描述性降时47.66%/25.77%。相邻batch不同，不能作严格A/B或全程GPU平均利用率结论；采样曾见rollout4卡各100%、teacher96%。
- 详细记录`plan/restart-5gpu-performance-20260907/README.md`，提交回执corrected-submit.log、status223-step26.json、first-resumed-step.json及teacher223-after-compute.json。所有checkpoint保留；main未commit/push。


## 2026-09-07 16:46：Job223 teacher与step时间拆分

- 用户问TTR/每step时间分布，按teacher解释。发现核心日志过滤teacher与old_log_prob单项；Ray dashboard关闭，转直接GCS GetTaskEvents只读查询保留任务时间戳成功（73条）。无生产配置/源码/运行wrapper修改，无重启或新GPU任务。
- Step26：rollout149.34s(34.9%)、teacher worker103.28s(24.2%)、student oldlogP四rank关键路径9.10s(2.1%)、actor更新140.99s(33.0%)、剩余24.80s(5.8%)；合计427.51s。
- Step27：rollout115.28s(33.6%)、teacher86.05s(25.1%)、oldlogP6.92s(2.0%)、actor更新110.82s(32.3%)、剩余23.66s(6.9%)；合计342.72s。两步response tokens1900291/1632256，不把差异当严格A/B。
- teacher为Ray worker task:execute墙钟，包含worker内部搬运/后处理；driver RPC与准备归入残差。四actor耗时按并行关键路径而非相加。验证/checkpoint保存位于timing_s/step之外，周期步实际墙钟更长。
- 报告与只读脚本变更表：`profile_output/job223-step-timing-20260907/REPORT.md`；原始GCS JSON和派生step-breakdown.json同目录。job223保持RUNNING。


## 2026-09-07 17:08：生产代码审查、commit 与 push 完成

- 用户明确授权检查当前代码并详细说明 commit 后 push。main 已提交 `29ad9e676a0d33733b15d1746463d40964cbd354`，origin/main 通过 ls-remote 核验 SHA 完全一致，ahead/behind 为 0/0。
- 167 文件，3918 insertions / 274 deletions：exact Q selector/恢复契约/分布式统计；CUDA Graph + rollout 64；teacher chunk1024/32rows/57344tokens/12GiB、受保护 stable-sort MoE 与独立 teacher expandable segments；正式配置覆盖、测试与 GPU 性能说明。
- 修正 .gitignore 对 canonical/fire_token_topk32.yaml 的误排除，精确例外不放开其他 token 凭据。teacher、Q 与 ignore 例外独立 review 无阻塞问题。
- 干净提交副本 CPU 测试 918 passed / 2 deselected，2 个 protobuf deprecation warnings。未执行的两个测试分别缺 LiveCodeBench-v5/test.parquet 和 Eurus/code_train.parquet；未改测试断言掩盖数据缺失。含两 rank Gloo 测试。测试依赖隔离安装，未改全局 Python。
- git diff --check 通过，167 文件及 index 与测试副本 SHA256 一致，凭据模式扫描无命中。未提交日志、checkpoint、连接文件、本地 project memory、30 份历史远端配置快照。
- 本轮未操作或重启 GPU 作业；job223 状态未重新查询，不能把此前 RUNNING 当本轮实时状态。详细 commit 已记录固定输入 benchmark、显存代价与非严格 A/B 限制。
- 验证记录、选定文件清单、commit 正文与最终 pytest log 保留于 plan/git-publish-20260907/；仅清理本轮创建的 temp/git-publish-checkout 与 temp/git-publish-deps。


## 2026-09-07 17:28：Lottery TopP7.5% Fixed4配置与HF补传prompt

- 用户要求参考alpha1.75八卡Math run新增lottery config，并撰写HF token/漏传checkpoint排查补传prompt。澄清未返回，按已告知默认解释：均匀随机排列candidate token类型、TopP occurrence覆盖0.075、Fixed4、纯Math；“六”未定，不假定step6/60。
- 新config：`configs/token_selection/math/lottery_next_step_full_taxonomy_unified_topp0p075_i1_w1_fixed4_8gpu_6a2t_b258.yaml`；新run `q1p7b-math-top32kl-lottery-topp075-w4-8g6a2t-b258`。390候选、strict>20、i1w1、6a2t/b258、Graph64及teacher性能设置继承；alpha显式reset1.0，raw4/1后保留domain归一化。
- 新源码支持selection_mode=lottery与显式seed42，SHA256(version,seed,step,domain,token_id)排序；lottery schema11绑定seed/version，旧非lottery schema10字段集合保留。启动→audit→runtime→checkpoint透传完整。独立review发现mode规范化可绕过version检查，已修复并增7回归，复核无阻塞问题。
- 184核心tests + 31正式Math配置tests通过，配置解析/启动命令/diff检查通过。本地历史server-only YAML的旧test_freq=-1导致首次宽扫描失败，正式集检查排除了既有归档；未改断言或旧文件。未做GPU smoke。
- 可复制prompt：`plan/lottery-config-hf-recovery-20260907/hf-checkpoint-recovery-prompt.md`，目标故障run/step留占位符；包括token来源优先级、权限、异步staging、receipt与Hub验证、仅目标step补传、delete_patterns范围核查。未读取token或执行上传，未启动/重启任务；本轮未commit/push。


## 2026-09-07 17:48：用户纠正 lottery 为V3候选

- 先前FullTaxonomy390候选不符合用户意图，已替换为V3 115候选（85Control+30Structure），fixed raw4、TopP0.075；lottery seed42与6a2t/b258保留。
- 当前config：`configs/token_selection/math/lottery_next_step_expanded_pruned_v3_unified_topp0p075_i1_w1_fixed4_8gpu_6a2t_b258.yaml`；run=`q1p7b-math-top32kl-v3-lottery-topp075-w4-8g6a2t-b258`。已比对现有V3完整ID列表及SHA256并验证启动透传，参考reward/data/objective等保持一致。核心源码未改，未提交/启动任务。
- 本机磁盘曾不足影响写入，清理了可重建的项目Python/pytest缓存约14.8MB后完成；共享uv缓存繁忙，取消了清理，不影响其他进程。


## 2026-09-07 18:18：最终纠正为loss-ratio candidate排序＋V3＋Fixed4

- 用户确认不是lottery，也不是loss-ratio动态weight，而是V3115候选、TopP7.5%、weight固定4，以loss ratio排序。最终模式`top_loss_ratio`，公式mean_abs_loss(type)/max(mean_abs_loss(all other valid domain occurrences),1e-12)，含out-of-pool分母并先跨rank汇总。raw ratio不clip4、不设置weight；选中/其他raw4/1后保持已有domain归一化。
- config=`configs/token_selection/math/lossratio_selector_next_step_expanded_pruned_v3_unified_topp0p075_i1_w1_fixed4_8gpu_6a2t_b258.yaml`；run=`q1p7b-math-top32kl-v3-lrsel-topp075-w4-8g6a2t-b258`。原先lottery源码/helper/tests/config已撤除，HEAD已有Q与teacher优化保留。
- 187专项与回归通过；独立review发现候选count全覆盖但loss总量不一致的phantom denominator缺口，已修复并加6测试，复核无阻塞。V3 IDs/hash、启动命令、AST、diff检查均通过。i1w1、strict>20、6a2t/b258及性能设置保留。
- HF排查补传prompt保留在`plan/lossratio-selector-v3-fixed4-20260907/hf-checkpoint-recovery-prompt.md`，本轮未检查token/上传/启动GPU/commit/push。未做GPU smoke。
- 磁盘不足时用uv自身清理pyarrow/tensordict/orjson/transformers可重下载缓存，工具报告654.2MiB逻辑缓存；未改已安装环境或实验数据。


## 2026-09-07 18:41：按用户要求撤除ratio排序，回到原Top-Loss

- 用户明确只按平均loss排序。撤除新增top_loss_ratio helper/校验/排序接线和2专项tests，5个生产文件经diff核对还原到HEAD29ad9e6；当前生产源码无未提交diff。lottery也已撤除。
- 最终config：`configs/token_selection/math/top32kl_next_step_expanded_pruned_v3_unified_topp0p075_i1_w1_fixed4_8gpu_6a2t_b258.yaml`；run=`q1p7b-math-top32kl-v3-topp075-w4-8g6a2t-b258`。V3115候选、top_loss、TopP0.075、Fixed4（raw4/1后原domain归一化）、i1w1/>20、6a2t/b258及性能参数保留。
- 精确V3 ID/hash与启动参数核验通过，50项原selector/V3回归通过，git diff检查通过。文档已替换错误方案；旧plan标记superseded；HF补传prompt继续保留，未上传/启动GPU/commit/push。


## 2026-09-07 18:59：V3 Top-Loss Fixed4 config提交并push

- 按用户授权commit/push；提交`ca9a38e87a41abaa75c8bc8d21816c3b1a190047`（feat(config): add V3 Top-P 7.5 percent Fixed4 math profile），2文件72insertions：最终V3 TopLoss TopP0.075 Fixed4 8gpu config及Math README启动说明。生产源码无改动；本地工作日志与旧归档未纳入提交。
- origin/main通过ls-remote核验与本地一致；50既有回归通过。run_mopd.sh --slurm --mem=800G --dry-run已验证8GPU/64CPU/800G，sbatch未调用。用户应在训练目录同步到该提交后运行去掉dry-run的命令。
- 交付补传agent prompt：`plan/v3-toploss-fixed4-20260907/agent-hf-recovery-prompt.md`。目标参考旧alpha1.75八卡run，step需填或由agent只读核实唯一漏传步；验证token不泄露，补传后核对Hub固定revision完整性。未创建上传执行任务，未实际上传或启动训练。


## 2026-09-07 20:49：5GPU状态核查与10% Fixed4配置

- 实时Slurm job223 RUNNING（elapsed 4:20:18），最新完整日志step64/70，五卡仍分配；job218为历史FAILED，不能误作当前任务。global_step_60目录存在，本轮未验收完整性。日志尾有Timeout during comparison警告，不能据此判定训练失败。
- 用户明确改用Fixed4。新增`configs/token_selection/math/top32kl_next_step_expanded_pruned_v3_unified_topp0p1_i1_w1_fixed4_5gpu_4a1t_b256.yaml`：V3 Math 115 unified IDs、top_loss、TopP0.1（all-valid occurrence coverage，非10%词表ID）、fixed raw4/1、原domain normalization、i1/w1/>20、4actor+1teacher、batch256、70steps。
- 独立run `q1p7b-math-top32kl-v3-topp1-w4-5g4a1t-b256`，resume disable，保留性能优化与HF steps55/60/65/70。配置解析和launcher透传验证通过；尚未上传或提交GPU任务。
- 本轮只先准备config；后续确认旧任务结束及五卡释放，rsync dry-run/上传/Slurm dry-run后以5GPU/500G启动。训练后用step60执行canonical 10 datasets × K8、默认DP4标准评测，结果归档experiments_records/eval。
- 10% Fixed4相较当前5% loss-ratio同时改变budget和weight模式，不能作为只改变selection比例的单变量消融。


## 2026-09-07 20:58：中断旧5GPU并启动10% Fixed4与旧Step60 Math评测

- 用户明确授权中断旧任务。验收旧V3 5% lossratio Step60后，scancel223；Slurm确认CANCELLED，六卡均14MiB后启动新任务。
- 新训练job224，run=q1p7b-math-top32kl-v3-topp1-w4-5g4a1t-b256，20:56:27启动；5GPU/500G/64CPU/72h，4actor1teacher、batch256、70steps、从头训练。已进0/70 rollout，W&B注册完成。
- 新config为top32kl_next_step_expanded_pruned_v3_unified_topp0p1_i1_w1_fixed4_5gpu_4a1t_b256.yaml，SHA256=5264085e40207dc1d1a04edd0476702606afbee383166a4e0c71115b61f3b2b4。源码checksum一致，只上传新增config；dry-run后用不带dry-run的launcher正式提交。
- Math-only eval225于20:57:17启动，旧run q1p7b-v3-topp05-lossratio-i1w1-5g4a1t-b256-r20260907/global_step_60/actor/huggingface；1GPU/100G/64CPU/24h、Math4/K8/DP1/TP1/seed42/16shards，CUDA_WITNESS通过，64-shard manifest已生成。
- 尚无训练完整step或评测最终成绩；后续完成评测后下载、验收960rollouts、重算Avg@8/Pass@8并写Math-only Summary，不进入Standard10。
- 本地回执与启动日志：plan/selection10-launch-20260907/。评测manifest：../experiments_records/eval/q1p7b_v3_topp05_lossratio_step60_math4_dp1_k8_seed42_20260907/RUN_MANIFEST.md。


## 2026-09-07 22:33：Math评测225进度核查

- job225 RUNNING，约1小时35分；最新56/64 shards成功：AIME24、AIME25、HMMT25Feb各16/16，HMMT25Nov8/16。无失败shard，suite SUCCESS尚不存在。
- 新10% Fixed4训练job224同时RUNNING，本轮未检查训练step。
- 尚未满足完整结果下载验收条件，未修改Summary.md正式结果、未宣称完整分数。后续完成后下载原始结果/日志、验收Math4 K8并更新Summary.md。


## 2026-09-07：eval225完整归档

- job225 COMPLETED/0:0，22:46:27结束，耗时01:49:10；完整suite及Slurm日志已下载。64/64shards、120题每题8次、960条唯一记录，raw/merged correctness及汇总一致；远端checksum无文件内容差异。
- 旧V3 TopP5% TopLoss+LossRatio Step60：AIME24 38.33/63.33，AIME25 24.58/43.33，Feb17.08/26.67，Nov13.33/33.33；Overall Avg@8=23.33%，Pass@8=41.67%（224/960、50/120）。
- 相对Q-selector Fixed4 +0.10/−1.67pp，OPD +1.04/+1.67pp，ExOPD seed42 −2.92/−5.00pp；单checkpoint描述性比较。
- experiments_records/eval/Summary.md已更新Math-only统一表及独立章节；archive=/Users/linghuazhang/Desktop/Project/OPD/experiments_records/eval/q1p7b_v3_topp05_lossratio_step60_math4_dp1_k8_seed42_20260907。训练224本轮未查询。


## 2026-09-07：Summary实际训练config与机制核对

- 按用户要求直接通过W&B GraphQL读取12个training run.config，冻结resolved快照与run URL到experiments_records/eval/_comparisons/config_mechanisms_20260907/。未改动远端runs。
- Summary新增Math-only机制表（13条自研评测行、10个训练配置；baseline留白）及标准10-dataset自研V1机制表；详细索引给出完整YAML、W&B、恢复入口、预算/选择/weight公式。
- 纠正ExpandedPruned-v2 Top29旧Adaptive Neighborhood误名：W&B确认为邻域false、top_logp_diff、Fixed4；成绩不变。TopK没有固定occurrence百分比，Taxonomy TopP1/2/5/7.5/10%均TopLoss+Fixed4；V3 TopP5%是TopLoss+LossRatio；Q是5%+Fixed4。
- 特别记录job223实际restart.yaml+step25恢复、7.5%resume55、1%resume15与10%历史config/当前avg4身份差异。12个live快照独立复核通过，config链接有效；本地experiments_records是跨卷symlink，YAML链接改用绝对路径。


## 2026-09-07：每 step 增强 token 的 taxonomy composition

- 新增 global/domain `token_weight/amplified_source_*`：Control/Structure/Other occurrence 与 unique ID 数量及各自占比；分母是实际增强 token。使用固定175/634 global taxonomy，排除 loss mask 位置，按本 step 实际 multiplier 判定，online selector 更新前统计。跨 rank 汇总后求比例，global unique 跨 domain 去重，修正 SP 副本。
- 固定/online/shared weighting 自动输出，TensorBoard core 保留；Adaptive Neighborhood 尚未接入。无需额外 forward/backward。历史/运行中旧进程不会自动新增指标。
- 本地162项相关测试通过，含 teacher-prefix mask-only 回归；未提交、未上传或重启远端。说明：`docs/token-source-metrics.md`。

## 2026-09-08：Summary训练run的60步Control/Structure选中量

- 完整恢复9/12条selector训练run（8条Math-only+V1 KL Entropy三domain），660个step×domain记录全部通过applied coverage与词表counts交叉校验，逐步selector链连续。包括warmup，均值分母60；global taxonomy Control175/Structure634，旧V1保留Other。
- Taxonomy TopP5%：每step Control/Structure occurrences=54,710.38/30,625.13（1.79:1）；V3 TopP5% LossRatio=76,437.10/6,870.57（11.13:1）。Split21两类ID均值均20.65，但occurrences=2,826.12/4,862.55（0.58:1）。选中量不同于normalized weight>1的精确分类型增强量。
- C05 resume55采用56–58最后记录；W&B step57旧尝试123963与最终audit128564不同，最终selector链与词表counts均支持后者。C09 parent1–15+resume16–60。
- TopP2%实际训练run是-r2，从1重跑，原配置索引旧run只有1–8步；r2完整W&B可确认92.22 IDs、38,177.85 occurrences每step，但不能分类。V3 K25可确认24.58 IDs、5,051.42 occurrences；V1 Speed各domain平均28.50 IDs，无occurrence字段。
- 缺少2%/V3 K25/V1 Speed的原始applied IDs与词表counts；metadata定位启智root=/mnt/tidal-alsh01/usr/zhongmeizhi/liyihang/git-repos/code，两台已配置SSH主机和HF checkpoint repo均无这些audit，本机无qzcli访问配置。不能用预算猜测分类量。
- Summary.md已追加统计，原评测分数未变。完整报告：[REPORT.md](/Users/linghuazhang/Desktop/Project/OPD/experiments_records/eval/_comparisons/selected_token_counts_60steps_20260907/REPORT.md)；逐步per_step.csv、汇总per_run_domain.csv、raw与SHA256均在同目录。


## 2026-09-08：Math-only主结果表合并机制及C/S计数

- 按用户明确示例，将config、selection预算、token选择、权重及Control IDs/step、Structure IDs/step、Control occurrences/step、Structure occurrences/step、C:S occurrences合并到Summary主表，21条实验记录（含表头22行）、16列；删除原重复机制表，保留口径和底部统计来源。
- 统计来自selected_token_counts_60steps_20260907；核对8条Math训练每条Step1–60完整，与per_step逐条重算均值和ratio-of-sums一致。含warmup零步，applied IDs口径；baseline新增列空白，2%和V3K25缺分类audit不填估计。
- 原7列分数/状态逐cell一致，降序保持。备份code/plan/summary-config-mechanisms-20260907/Summary.before-main-table-merge-20260908.md。


## 2026-09-08：TopLoss5%与Q-selector5%分差解释

- 对照相同FullTaxonomy390/TopP5%/Fixed4的TopLoss eval196与Q eval219。原始merged汇总核验：Avg@8 26.04→23.23（250→223正确rollouts/960，−2.81pp）；Pass@8 46.67→43.33（56→52题/120，−3.33pp）。四集Avg均下降，AIME25 Pass反升50→56.67。
- 已冻结W&B config对比：核心selector top_loss→top_q_loss_entropy；此外Q启用entropy_from_logits_with_chunking，test_freq −1→20；其余输出身份与新增未生效默认字段有差异。因此不是严格只改一个开关的实验证据。
- Applied C/S统计：TopLoss54710.38/30625.13 occurrences每步、C:S1.79；Q54974.17/30830.28、C:S1.78，数量和粗分类几乎相同。不能把差异归因为少选token或C/S比例大变。
- 机制假设：Q的L/meanL+H/meanH+LH/(meanL meanH)可提高高entropy位置排名，未必等价于更有价值的teacher纠偏信号；在线选择后影响on-policy轨迹，差异可累积。尚未证明这是本次分差的因果来源。
- 建议验证（未执行）：同一批rollout同时计算两selector，比较ID overlap/所选loss mass/entropy/实际occurrences；然后统一其余配置多seed训练。不可将960rollouts视作960道独立题，只有120题。


## 2026-09-08：Q/TopLoss 原始 audit 下载核验

- 已下载两组全部已有词表向量、selection、domain metrics、loss variance、training cost，959个JSONL与远端逐文件SHA256一致；120个Step1–60 selection与旧抽取哈希一致。不是约20GB完整目录镜像，保留了初期下载的少量逐位置log-prob/gap文件。
- Step1 logp_abs_vocab原文件SHA相同，首次next集合76/80 IDs，交集71、Jaccard83.53%；实际覆盖5.7273%/5.2850%。聚合文件相同不等于已核对完整rollout序列。
- Step2–60 applied跨run平均Jaccard72.75%；各run全部59次next→applied连接正确。Q轨迹相邻集合更稳定（74.76% vs69.16%），不能推为掉分原因。
- 两组冻结config均关闭逐词表entropy、Top32 loss与response-level保存；Q所选token mean_abs_loss为空。下载不能补回未记录的数据，logp_abs/gap不能冒充configured KL loss。因果判断仍需固定batch补算两selector及loss/entropy分量。
- 报告与下载清单：`/Volumes/Data/OPD/experiments_records/eval/_comparisons/selector_audit_20260908/REPORT.md`、`remote_sha256.json`。未启动训练或评测，Summary分数未改。


## 2026-09-08：相同step选择相似度

- 已生成 selector_audit_20260908/SAME_STEP_SIMILARITY.md、applied_same_step_similarity.csv 和曲线图；Step1空集合记N/A，Step2–60按实际applied对齐。
- 平均Jaccard72.749%，Dice84.175%；平均交集96.49 IDs，占TopLoss所选IDs83.30%、Q85.20%；共同ID在各自轨迹所选occurrences中的占比均值85.16%/85.42%。
- 分段Jaccard：2–10 74.54%、11–20 74.27%、21–30 72.44%、31–40 72.81%、41–50 71.10%、51–60 71.52%。最低step43 63.38%，最高step2 83.53%，step60 72.26%。这是跨训练轨迹同进度比较，不是同batch因果消融。


## 2026-09-08：Q Taxonomy TopP5% Fixed4切换准备

- 新独立config *_r20260908.yaml已解析、35项Q回归通过、独立审计无阻断问题；TopP按all-valid occurrences累计，Fixed4为raw4:1后domain normalize。
- 远端job224（V3 TopP10% Fixed4）01:08核查RUNNING，41/70，5GPU/500G。本轮未中断、未提交GPU任务。
- 配置和checkpoint/eval helpers已按dry-run上传；训练预演5GPU/500G通过；eval预演因Step60尚不存在拒绝。远端无pytest，未声称远端测试通过。
- 已授权每小时等job224 Step60完整后停止它，从base启动新Q训练5GPU/500G，并1GPU/100G对旧job224 Step60做Math4 K8。详细runbook：/Users/linghuazhang/Desktop/Project/OPD/code/plan/q-taxonomy-step60-switch-20260908/STATUS.md。


## 2026-09-08 02:14 +08：小时检查 SSH 超时

- 本轮SSH在55秒内未返回远端输出，无法确认job224当前step、Slurm状态或Step60保存情况。最近成功观测仍为01:08的41/70，不能当作实时状态。
- 没有发送scancel、sbatch或评测提交；新train/eval job ID仍无。未改变远端任务。
- heartbeat保持ACTIVE，下一个小时重试；必须重新验收Step60、去重及预演后才能切换。


## 2026-09-08 02:26 +08：手动继续，连接恢复

- SSH恢复；精确job224 RUNNING，elapsed05:29:58，日志最新training/global_step=54、54/70。global_step_60目录尚不存在。
- 最近step约354秒，按近期速度距离60约36分钟，仅为估计；必须等checkpoint实际完整。
- 尚未满足停止门槛，没有取消或新提交；每小时heartbeat继续执行既定切换与旧Step60 Math4单卡评测。


## 2026-09-08 03:19 +08：训练226与评测227已启动

- job224 Step60的12rank文件、HF JSON、4,063,515,640-byte safetensors及保存日志完整，03:17:13安全CANCELLED后六卡释放。
- 新Q job226 RUNNING，5GPU/500G，W&B已注册，进入0/70；run=q1p7b-taxonomy-q-topp05-fixed4-5g4a1t-b256-r20260908。
- 旧V3 TopP10% Fixed4 Step60 Math评测job227 RUNNING，1GPU/100G，CUDA_WITNESS H200 NVL，64shards manifest身份与K8/DP1核验通过。尚无最终结果。
- 禁止重复提交。heartbeat已改为每1h监控226/227，eval完成后验收归档并暂停，不停止新Q训练。startup manifest与RUN_MANIFEST已归档experiments_records/eval/q1p7b_v3_topp10_fixed4_step60_math4_dp1_k8_seed42_20260908。


## 2026-09-08 04:17 +08：小时检查

- job226 RUNNING，最新training/global_step=15/70，最近100KB训练日志Traceback/CUDA OOM/RayTaskError计数0。
- job227 RUNNING，35/64 shards成功；AIME24与AIME25 wave已完成，HMMT25Feb进行中。FAILED文件0，suite SUCCESS未生成。
- 尚未满足最终归档条件，未提前汇报指标；未取消、重启或重复提交。heartbeat继续每小时跟进。


## 2026-09-08 05:21 +08：评测227完成归档，监控目标完成

- job227 05:04:29 COMPLETED/0:0，耗时01:46:14；suite原始文件117397646 bytes及Slurm日志已下载，checksum无文件内容差异（仅根目录mtime不同）。
- 验收64/64shards、120题每题K8、960唯一records、raw/merged correctness和汇总一致。AIME24 78/240、18/30；AIME25 71/240、13/30；Feb38/240、8/30；Nov41/240、11/30。Overall Avg@8=23.75%，Pass@8=41.67%。56/960达到token上限。
- 归档experiments_records/eval/q1p7b_v3_topp10_fixed4_step60_math4_dp1_k8_seed42_20260908；RUN_MANIFEST、LOCAL_SHA256SUMS、独立计数及Summary.md Math-only主表/详细段落已更新，不进Standard10。主表保留原有行，新增16列行按Avg降序；Summary备份在本runbook目录。
- 新Q训练job226仍RUNNING、step22/70。训练切换和旧checkpoint单卡Math评测归档目标均完成，heartbeat已PAUSED，未停止新Q训练。


## 2026-09-08 13:23：Q226结束并启动单卡Math评测228

- 训练226已70/70、10:24:20 COMPLETED/0:0；W&B teardown连接错误保留。新Q run r20260908 Step60完整验收。
- 用户授权单卡评测，eval228于13:22:42 RUNNING，1GPU/100G/DP1/TP1，Math4 K8 seed42，CUDA_WITNESS H200 NVL，64shards manifest已生成。禁止重复提交。
- 详情code/plan/q226-step60-math-eval-20260908/STATUS.md；尚无最终结果，未恢复已暂停的旧定时任务。


## 2026-09-08 15:33：FullTaxonomy TopLoss TopP5% Fixed6已启动

- 用户明确授权5GPU实验。新增configs/token_selection/math/top32kl_next_step_full_taxonomy_unified_topp0p05_i1_w1_fixed6_5gpu_4a1t_b256_r20260908.yaml；对原TopLoss5%Fixed4仅改raw weight4→6及输出身份，batch256、4actor1teacher、seed42、70steps、resume disable保留。
- 独立审查通过：fixed6不受loss_ratio max4影响；主进程动态config diff/launcher透传通过。rsync dry-run→upload、继承链SHA一致、Slurm预演5GPU500G通过。
- job229于15:33:08 RUNNING，实际5GPU/500G/64CPU，日志已进入训练数据加载/过滤；尚未观察到完整optimizer step。run=q1p7b-taxonomy-toploss-topp05-fixed6-5g4a1t-b256-r20260908。
- 回执及状态code/plan/toploss-topp05-fixed6-20260908/；禁止重复提交。未提交或推送Git，保留既有未提交改动。


## 2026-09-08 16:18 +08：job229 step14，磁盘容量警告

- Slurm RUNNING，elapsed00:45:24，最新step12→13→14/70；近期step耗时320/368/386秒。actor GPU0–3利用率100%、约90GB，teacher GPU5约80GB，此瞬间利用率0%；GPU4空闲。
- checkpoint目录step5/10存在；step10占45G，本轮未进行完整checkpoint验收。Step60未生成。
- 新实质警告：Ray每10秒报告/tmp所在文件系统>95%满，若需要object spilling可能失败。df核实/tmp和checkpoint共用/dev/sda2，7.0T、96%、剩余285G。训练当前仍在推进，不能称为失败。
- 未删除文件、未停止/重启训练；提示用户容量风险，后续继续观察磁盘与训练/checkpoint进度。


## 2026-09-08 17:53 +08：连续两轮SSH超时

- SSH再次55秒无远端输出，已连续两轮无法查询。最后可验证状态仍为16:50 RUNNING/step18，不能推断当前状态或磁盘空间。
- 无取消、重启、删除、配置变更或新提交。将监控连接异常通知用户，heartbeat保持ACTIVE，下轮继续重试；不把连接失败归为训练失败。

## 2026-09-09：Fixed6四卡从头重启job230

- 用户授权：job229从Step0重启，资源改为4GPU。
- 原job229于2026-09-09 09:59:42 FAILED/1:0，重排队后W&B resume=never与已存在run冲突。原checkpoint20保留。
- 新run：q1p7b-taxonomy-toploss-topp05-fixed6-4g3a1t-b255-r20260909。
- 新job230于2026-09-09 10:11:41启动，Slurm RUNNING，Restarts=0，4GPU/400G/64CPU。
- 3 actor/rollout + 1 teacher，batch/mini-batch255（原256改为3可整除），Top32-KL/FullTaxonomy390/TopLoss/TopP0.05/Fixed6/domain normalization/i1w1/seed42/70steps不变。
- 新runID与全部输出路径隔离；继承extra_overrides trainer.resume_mode=disable，从base开始。
- 本地load_config与launch dry-run通过，独立审查无blocking；rsync preview→upload，仅新增overlay，四层继承链SHA256一致；Slurm dry-run请求4GPU/400G；W&B API新ID无已有run；磁盘checkpoint/log/tmp同挂载点2908GiB，超过500GiB门槛。
- 最新启动日志确认resume_mode=disable，处于模型初始化。尚未确认完成optimizer step。
- W&B：https://wandb.ai/lz101-rice-university/MOPD/runs/q1p7b-taxonomy-toploss-topp05-fixed6-4g3a1t-b255-r20260909
- 配置：configs/token_selection/math/top32kl_next_step_full_taxonomy_unified_topp0p05_i1_w1_fixed6_4gpu_3a1t_b255_r20260909.yaml
- 远端checkpoint：/home/shuang_qiu/mopd_code/checkpoints/MOPD/q1p7b-taxonomy-toploss-topp05-fixed6-4g3a1t-b255-r20260909
- 远端audit：/home/shuang_qiu/mopd_code/audit/q1p7b-taxonomy-toploss-topp05-fixed6-4g3a1t-b255-r20260909
- 回执：launch-receipt.txt；启动验证startup.txt；不重复提交。未commit/push；保留既有工作区改动。

- 2026-09-10 W&B复查：新run状态 `crashed`，连续记录 `_step=1..69`，最后 `training/global_step=69`；总目标70步，Step70未完成。

- 10:13启动验收：job230 RUNNING；W&B新run成功初始化并sync，Training Progress 0/70，首个batch255已进入teacher输入准备；确认从Step0开始，尚无完整optimizer step。

## 2026-09-10：26.04% checkpoint 源码溯源审查

- job193/Step60的Slurm和W&B均记录HEAD1539c1bf1a3a5c595b1750dbbc8fb2e2c8a50154，但该commit缺少实际使用的TopP selector；因此不能以其直接复现，实际含HEAD之外的源码。
- 已归档原始Hydra配置。本地3c60721仅作时间邻近参照；相对此版本未发现Fixed4数学路径回归，但当前同名配置rollout/teacher性能/验证频率已变化。
- Fixed4 26.04 vs Fixed6 23.75不是纯weight消融（batch256/255、4actor/3actor不同）；题目配对bootstrap差值区间包含0，尚不能证明weight4最优。
- 详细证据：[审查报告](/Users/linghuazhang/Desktop/Project/OPD/plan/checkpoint-2604-audit-20260910/REPORT.md)。未启动训练，未修改训练代码。

## 2026-09-11：确认历史训练采用rsync源码部署

远端HEAD及唯一保留reflog仍为8月6日clone的1539c1b；现存72份含commit训练日志和68份W&B元数据均记同一HEAD。历史部署脚本/清单/回执确认rsync只同步指定源码和配置，不包含.git。旧HEAD不能代表训练源码版本。远端只读核验；详细证据见[报告](/Users/linghuazhang/Desktop/Project/OPD/plan/checkpoint-2604-audit-20260910/REPORT.md)。

## 2026-09-11：MOPD 三域 batch 对齐

- `experiments_records` 中已归档的三域训练记录显示 per-domain 实际采样为 `175–176`，对应 global batch `525–528`；同为 8GPU、6 actor/rollout + 2 ref/teacher 的记录使用 Math/Code/Science 各 `176`、global `528`。
- 新提交的 TopLoss + TopP5% Fixed2/4/6 配置已统一为 `data.train_batch_size=528`、`actor.ppo_mini_batch_size=528`，等权三域由 sampler 分配为 `176/176/176`；文件名及运行、审计、评测、checkpoint 标识同步为 `b528`。
- 本地 config load、domain allocation、launcher dry-run 和 `tests/test_mopd_profiles.py`（9 passed）通过，未提交或启动远端任务。
