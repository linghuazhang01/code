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

## 2026-10-01：当前 V6 Math5%/Code1% Current-Step 只读核对

- 最新Rising补全后的pool为Math C118/S156=274；Code V6 C23/S196=219（S lexical144/format52）、V7 S196、V8 S341。下方早先审计记录为补全前冻结版本。
- 四卡config为`mopd_math_code_current_step_token_v6_toploss_m05_c01_fixed4_4gpu_colocated.yaml`；实际current_step/token_id/joint top_loss，Math5%/Code1% occurrence budget，strict>20，raw4/1后rank-local microbatch/domain归一。Code C/S均在全部valid位置参与，无fence gate；Structure不是全池固定增权。
- Student1.7B、teacher30B-A3B-Instruct-2507、teacher-support Top32 RKL、batch528/MB1、60steps、seed42、colocated4GPU。专项40tests通过。Code C23精确清单与[参数核对](/Users/linghuazhang/Desktop/Project/OPD/code/plan/v6-next-step-definition-audit-20261001/current-v6-inspection.md)已保存。
- 用户最新要求先查看；未新增Next-Step、未修改训练config/源码/Token.md/生成器，未启动任务。发现runtime文档旧数量与Token.md code_structure旧M263/C21残留，以及generator恒真subset/type推断，尚未修正。

## 2026-10-01：Current-Step G0124 GPU 利用率诊断

- 后续按用户授权完成独立teacher chunk通用默认：`topk_logprob_chunk_enabled=true`、1024，`enabled=false`或四卡colocated不再屏蔽chunk；batch/MoE/统计guard、ref/actor MB1及collective顺序保留。AGENTS固化不得随profile/resume静默回退，显存fallback有effective chunk/原因回执。250 passed / 1 skipped，428profile覆盖、mypy/核心lint通过；CPU FP32/BF16固定输入16/256/1024的Top32/logP/loss一致。[实现与验收](/Users/linghuazhang/Desktop/Project/OPD/code/plan/current-step-g0124-performance-20261001/teacher-chunk-implementation.md)。仅本地修改，未同步/重启当前worker；新代码在后续worker初始化生效，当前拓扑GPU收益与峰值未测。

- Chunk回退补查：9月7日`29ad9e6`已持久化1024，job223读回确认生效；9月22日`c3ea0eb`创建四卡base时显式`teacher_performance.enabled=false`，本run继承后chunk1024字段仍在但wrapper未装，forward实际默认16。应表述为已有优化在四卡profile失效，不能当成从未完成的新调优；chunk须与多rank动态batching保护分开验证接线。[Git证据](/Users/linghuazhang/Desktop/Project/OPD/code/profile_output/current-step-g0124-util-20261001/chunk-provenance.json)。本次追踪未改训练代码或运行状态。
- 02:57远端只读快照：原launcher52670与四worker55523–55526继续运行，日志完整Step29/60，正执行下一步ref；W&B读取时Step28。九个关键源码SHA与本地一致。本轮未重启、改参、同步或启动任务。
- 最新Step21–29平均57.886分钟：teacher26.385（45.58%）、actor12.172（21.03%）、CPU reward8.384（14.48%）、gen7.408（12.80%）。Current-Step prepass50.111秒仅1.44%，属于update内部；不能将残差全叫audit。
- 物理GPU0/1/2/4；全恢复13.07小时设备均值48.99/48.57/48.31/62.42%，GPU4另有1768MiB外部进程，不能解释为本run独占利用率。012互为NV6，4到012为SYS；teacher FULL_SHARD4的通信开销需要trace定量。
- 优先现有async reward重叠（理论完整隐藏时wall time最多减14.48%，未测）；teacher默认chunk16需独立接通，enabled直接打开会因colocated多rankguard回退。保留actor MB1以保持microbatch/domain mean-one，复制teacher或3+1布局属于后续验收/独立实验。
- 完整[诊断与变更清单](/Users/linghuazhang/Desktop/Project/OPD/code/profile_output/current-step-g0124-util-20261001/REPORT.md)、[后续计划](/Users/linghuazhang/Desktop/Project/OPD/code/plan/current-step-g0124-performance-20261001/PLAN.md)；修改前备份在同profile目录，未改eval Summary或Hub。

## 2026-10-01：Token V6/V7/V8 候选池配置审计与定义登记

- [Token.md §0.4](/Users/linghuazhang/Desktop/Project/OPD/code/Token.md:250) 登记 V6/V7/V8 为 active global taxonomy 的 candidate-pool versions，保持 legacy runtime；共同 Math C116/S147=263，Code 分别为 C21/S164=185、C0/S164、C0/S316。
- 48 配置加载/launcher 与逐 ID provenance 全部通过；36 run 实为24 namespace、18种配置组合。V6 Code Structure-only与V7参数等价，V7/V8的该变体为同namespace精确alias。Code无位置门控，旧Next-Step剪枝与教师context proxy不能证明新Current-Step效果。
- 补入冻结ID fingerprint回归与HF Step60精确断言，114项相关CPU tests通过；48配置生效值未变，未启动训练/评测。完整[审计报告](/Users/linghuazhang/Desktop/Project/OPD/code/plan/token-v6-v8-config-audit-20261001/audit-report.md)及逐文件JSON已保存。
- 按用户后续要求修复Git ignore：新增5行完整命名白名单，48新YAML与专项测试全部可见；12个protected path检查通过。48配置展开后的96个Math/Code候选池按decoded文本复核，包含`**`的ID为0，完整词表仍保留这些IDs。见 Obsidian `Experiments/Token-V6-V8-Candidate-Pools.md` 与 `Daily/2026-10-01.md`。

## 2026-09-30：M05/C01 与 M05/C02 Structure-only token 审计

- 已绑定 31.66% Macro8 的 C01 8GPU 与 30.64% 的 C02 4GPU；两者候选集相同：Math Control124/Structure266，Code Control0/Structure551。C02 60 步在线 audit 的字段内容已取回并验证 59 对 source→next lag，完整[分析报告](/Users/linghuazhang/Desktop/Project/OPD/analysis-output/c01-c02-structure-token-audit-20260930/analysis-report.md)。
- C02 source-step 零次入选：Math Control8、Math Structure131、Code Structure253；Code 高频 `not`、`Output`、` Python`、`---\n\n` 为 60/60。Code 有 298/551 个 Structure ID 曾入选，完整逐 ID 与逐 step 清单在分析目录。
- C01 W&B 仅有 60 步数量，Code 每步新选均值50.88（C02 76.22），缺原始 token ID audit；C01 零次/高频具体 ID 不可核实，不以 C02 名单代替。
- 用户要求按精确 run ID 复查远端：CityU 上 C02 有 60 份 selector audit；C01 只有 Step60 HF 恢复模型的 9 个文件及评测。HF 对象仓库同样只有这 9 个文件，未列 C01 audit；原启智 `/mnt/tidal-alsh01/...` 在 CityU 未挂载，不能断言原训练盘不存在。证据见[远端搜索记录](/Users/linghuazhang/Desktop/Project/OPD/analysis-output/c01-c02-structure-token-audit-20260930/remote-cityu-search.md)及[C01 来源记录](/Users/linghuazhang/Desktop/Project/OPD/analysis-output/c01-c02-structure-token-audit-20260930/c01/SOURCE.md)。C01 精确 run 为 8GPU 6+2，C02 为 4GPU。
- C02 对 C01 的 Code4 Avg@8 +0.2653 pp、Math4 −2.2917 pp、Macro8 −1.0132 pp；Code TopP 1%→2% 与训练拓扑同时变化，各仅一个 checkpoint/seed，不能归因于 token 或单独的 TopP。

## 2026-09-30：V4/V5 token 逐 step 选择审计

- 已按 `Token.md` revision-3 核对 V4 默认、V5 默认、V5 C+S Shared TopLoss 三条 Next-Step run 的 step 1–60，并导出逐步数量、可得的 token ID/词面、零次与高频表；完整材料在 [分析报告](/Users/linghuazhang/Desktop/Project/OPD/analysis-output/token-v4-v5-selection-20260930/analysis-report.md)。
- V4 默认 Control Math/Code 分别有 117/119、137/153 个 ID 在 60 次 source-step 选择中至少入选一次，零次为 2/16；V5 Shared 的 Math Control/Structure 均全覆盖，Code Control/Structure 零次为 4/18。两条 run 的逐 ID audit 已校验；V5 默认 W&B 仅能恢复逐步 Control 数量与固定 Structure 的加权汇总数，不能推断逐 ID 频次。
- 历史 Current-Step V4/V5 仍缺可绑定的 run ID/audit，不能以配置快照或评测分数反推出 60 步选择。三条现有轨迹机制不同，不能将其差异归因为 taxonomy 单因素效果。

## 2026-09-29：V4/V5 Math+Code 八项结果与下一步决策

- 用户提供现在 V4、旧 V4、旧 V5 的 Macro8 Avg@8 / Pass@8，分别为
  `29.76/48.88`、`30.80/46.42`、`29.86/46.21`（%）。现在 V4 − 旧 V4 的
  Math4/Code4/Macro8 Avg@8 为 `−1.88/−0.22/−1.04 pp`，Math Pass@8 `+5.00 pp`。
  旧两组的 run ID、训练配置与逐题原始记录未确认，不能把差异归因于 taxonomy 或时序。
- 用户已取消先评测再训练的旧顺序；V5 r3 C+S Shared TopLoss 四卡 run 已启动，
  最近本地状态见 `plan/v5-shared-toploss-direct-4gpu-20260929/RUN_MANIFEST.md`。
  本轮没有查询实时远端状态、启动或排队新的 GPU 任务。
- 条件式计划见 `plan/v4-v5-post-eval-20260929/PLAN.md`：先做逐题配对与配置
  provenance 审计；完成既有 V5 shared 的同协议评测；首个建议新增训练是匹配四卡
  V5 r3 Next-Step + Structure position-fixed，用来与现在 V4/新 V5 shared 形成最小矩阵。
  Code 门控与 Current-Step 时序对照仅在前述结果支持时再考虑。

## 2026-09-28 23:31 +08：V4/V5 Step60 双模型评测准备

- GPU0/1/2/4 的 V4 r3 Next-Step 四卡训练仍在运行，远端日志已到 Step 50/60；Step60
  模型尚未保存，公开上传程序仍等待完整 checkpoint。
- 最近的 8-GPU Token V5 r3 run 已完成 Step60；公开 HF 固定提交
  `52ef604f8daa7797ca0893876af65d2cb5204c46` 的九文件模型已恢复到远端独立
  `checkpoints/HF_RESTORED/<V5-run>/global_step_60/actor/huggingface`，模型/tokenizer
  LFS SHA-256 与 safetensors 结构均验收通过。
- 用户授权删除七个明确列出的旧非最终 checkpoint，实测释放 312.04 GiB；相关 `/`
  挂载在 V5 恢复后约有 683.42 GiB 可用。保留所有旧 Step60、旧 TopLoss Step30 与
  当前 V4 训练文件；最终评测启动前仍须实时确认 ≥500 GiB 与四张 GPU 空闲。
- 用户确认“MAS”指 Math，本次评测允许 GPU4。V4 Step60 与上传完成后，以两个互不
  重叠的 DP2/TP1 lane 做 Math4+Code4、K8、seed42 八项专项评测，标记
  partial/non-canonical；小时监控 `opd-v4-next-step` 持续跟进。执行依据见
  `plan/v4-v5-step60-math-code-eval-20260928/PLAN.md`。

## 2026-09-28：V4/V5 动态 Code C/S 位置开关已本地实现

- 动态 `top_loss` / `top_teacher_confidence` 的 Code C∪S 继续共同排名、共享一份
  TopP 预算；新开关 `audit.code_cs_position_gate_enabled` 关闭时 C/S 均可在全部有效
  response 位置参与来源 step 排名和下一 step 加权，开启时分别使用现有 Control 与
  Structure mask。旧 `position_fixed` 与 Math 规则保持。
- 同一位置资格用于 TopLoss、Teacher Confidence 与实际加权；有效策略纳入 selector
  checkpoint 签名，审计记录来源 C/S eligible 次数、预算/缺口及当前加权位置数。
  独立审查的非 train metadata 和旧 fixed checkpoint 签名问题均已修复。
- 274 项相关 CPU 回归通过；扩大套件的 7 项历史 Qwen4B profile 测试因仓库缺少
  两个非 Git 跟踪的旧 YAML 而失败。三个对照配置均未启动，现有 GPU0/1/2/4 V4
  训练未重启、未同步此本地改动。计划与验证见
  `plan/versioned-shared-cs-selection-20260928/CODE_POSITION_GATE_PLAN.md`。

当前 Math+Code 评测状态（2026-09-27）见 [六卡恢复记录](math-code-eval-20260927.md)：用户已授权所有6卡，controller2594875运行中；后文9月26日四卡队列和等待恢复记录均为历史状态。

## 2026-09-28：Token V4/V5 Current-Step 实现完成，尚未启动训练

- 冻结 Qwen3 Token V4/V5：V4 Math/Code Control=121/154，V5=59/65；共享
  Structure=6/32，artifact SHA256 为
  `27884fadeea94145df23ede69109c7f8181c866de2f1c9f93d7659786719d817`。
- runtime 已分离 Control 与 Structure：Control-only 做 Current-Step TopLoss，Structure
  按答案格式位置 Fixed4，两者合并后只做一次 per-domain mean-one normalization。
- 已生成 18 个 Math5/Code5,2,1 × Token V4/V5 × 3/4/8 GPU profile；全部 batch528。
  8-GPU 是 8 actor + 8-way FSDP teacher colocated，旧 6+2 保留作为 fallback。
- 166 个 focused tests 通过；无 GPU smoke、无训练提交、无远端同步。下一步先做 1–2 step
  8-GPU colocated memory/throughput smoke。详见 Obsidian
  `Experiments/Token-V4-V5-Current-Step.md`。

## 2026-09-15：Math-only full-vocabulary Top-Loss 两种拓扑配置完成

- 新增两份 `Math-only / full_vocabulary / Top-Loss / Top-P=5% / i1-w1 /
  Fixed4 / 70 steps` 配置：4-GPU student/teacher co-located（batch256）与
  8-GPU 6 actor/student + 2 FSDP teacher（batch258）。
- 两份配置均直接继承无 taxonomy candidate groups 的 Math `_common`，并显式清空
  candidate IDs/groups，避免 Control/Structure pool 经 YAML deep merge 残留。
- 配置解析、launcher contract 和 topology/output identity 测试共 12 passed；尚未启动
  训练、未 commit/push。详情见 Obsidian
  `Experiments/Taxonomy-Loss-Entropy-Score-Ratio.md`。

## 2026-09-15：Step60 后续 heartbeat 已激活

- 当前远端 Fire OPD co-located run `qwen1p7b-30b-fire-opd-native-4gpu-b528-colocated`
  初始实时观测约为 `step49/200`，`global_step_60` 尚不存在；远端磁盘可用约 2.3T。
- 用户授权 Step60 checkpoint 完整后先做 canonical 三 domain/10-dataset K=8 评测，再
  启动 4GPU co-located full-vocabulary Top-Loss Top-P5% Fixed4 训练；评测与新训练串行，
  默认只使用 GPU0–3。
- 已创建 hourly automation `step60-fullvocab`，包含精确 run/checkpoint gate、查重、
  非 delete 全代码同步和 500G 磁盘门禁。详细状态见
  `plan/step60-fullvocab-20260915/STATUS.md`。

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

## 2026-09-14：新增三域四卡 co-located taxonomy 配置

- 新增 `configs/token_selection/math_code_science/taxonomy/mopd_qwen1p7b_30b_a3b_instruct_2507_4gpu_math_code_science_toploss_topp_m05_c02_s05_code_structure_only_fixed4_b528_colocated.yaml`。
- 配置为 4 GPU teacher/student co-located，启用 math/code/science 三域，Top-P 分别为 5%/2%/5%，`interval=1`、`window=1`，Code taxonomy 仅保留 `Structure`，沿用 Fixed4 与 batch 528。
- resolved config 检查和 launcher `--dry-run` 通过；未启动训练、未提交或推送。

## 2026-09-15：Full-vocabulary Top-Loss Top-P 5% 对照已实现

- 现有 taxonomy selector 的 `top_p=0.05` 分母是全部 valid response occurrences，
  但候选仅为配置的 Control/Structure pool；现已新增显式
  `control_token_online_candidate_scope=full_vocabulary`。
- 全集模式按 domain/token-ID 聚合 `abs(configured token loss)`，跨 rank FP64
  all-reduce；严格 `count > 20` 后按 occurrence mean 排序，取覆盖至少 5% valid
  occurrences 的最小完整 token-type 前缀，结果从下一 step 生效。
- 新配置：`configs/token_selection/math_code_science/full_vocabulary/mopd_qwen1p7b_30b_a3b_instruct_2507_4gpu_math_code_science_fullvocab_toploss_topp05_fixed4_b528_colocated.yaml`。
- selector state/checkpoint 升至 schema 12，旧 schema 默认迁移为 `configured`；JSONL
  记录 scope、tokenizer universe 与 observed/effective candidate counts。
- 相关测试 `206 passed`（含真实 two-rank CPU/Gloo all-reduce），compile/diff check
  通过；尚未启动训练、commit 或 push。


# 2026-09-20 论文叙事更新

依据当前Standard10归档，推荐主线为结构先验约束的在线token-ID增权，替代旧ASCR phase controller故事。

- 主候选m05_c01_s01：10-dataset macro Avg@8=34.30%，OPD=31.88%；Math=25.00% vs20.10%。属于探索性单run观察。
- 相对EOPD的macro优势仅0.29pp，主要来自Math；Overall/Pass@8无全面领先。多次benchmark调参需披露。
- 优先补matched多seed、全词表/非语义匹配候选池、固定token集合对照。taxonomy内随机不够证明结构语义价值。
- 实际机制：lagged TopLoss，occurrence coverage5%/1%/1%，raw Fixed4、domain normalization；Science含teacher confidence信号；非选中token仍接受监督。
- 完整大纲：[PAPER_PLAN.md](/Users/linghuazhang/Desktop/Project/OPD/PAPER_PLAN.md)。已完成独立GPT-6-Astra xhigh审阅；未新启动实验、未改训练代码或评测总表。


## 2026-09-20 性能优化实验设计

优先冻结Math5%/w4，做Code/Science额外增权2×2开关（仍三域训练）；按结果决定selector与局部p/weight搜索。第一批3个新run，未启动。最终同recipe OPD/anchor/final多training seeds确认。

已查到full-vocabulary Math-only5%/w4结果23.85%；此前主文仅定位Standard10证据，后续不应说全词表对照完全不存在。历史taxonomy26.04存在recipe/provenance混杂，不能直接证明结构优越。

详细方案：[EXPERIMENT_PLAN.md](/Users/linghuazhang/Desktop/Project/OPD/refine-logs/EXPERIMENT_PLAN.md)。


## 2026-09-20 MOPD是否保留：研究方向建议（尚未采纳为执行变更）

独立reviewer：广泛优于强基线claim=no，Math集中收益partial，confidence medium、same-family/provisional。建议保留全部多领域结果作为适用性/边界证据，暂停广泛Code/Science调参，优先确认Math matched baselines、多training seeds和未调参测试集。不能宣称负迁移或Math-only必然更优。没有删除数据、取消任务、启动实验或改写既有执行计划。


## 2026-09-20 用户约束：20天不做multi-seed

用户明确计算成本过高，本轮所有新增设置只用一个固定training seed，不把multi-seed设为完成门槛。计划改为六种核心设置（ours/OPD/strong baseline/fullvocab/matched随机pool/frozen IDs），合规旧run复用；可选最多两个新run，用于邻域调参或第二规模成对验证。D12冻结结果，D20(10/9)形成可提交稿。单seed限制如实披露；未启动/取消任何远端任务。

当前计划：[PAPER_SPRINT.md](/Users/linghuazhang/Desktop/Project/OPD/plan/paper-sprint-20260920/PAPER_SPRINT.md)。


## 2026-09-20 Math实验复用与Math+Code方向

用户指出Math搜索已较完整，建议只Math+Code做MOPD。本轮计划改为复用Math，不重开网格；双域OPD/ours/仅Math增权/一个强baseline四行，条件性补机制对照。单seed约束持续有效。双域8benchmarks专项不称Standard10；旧Science结果保留，三域抽取两域分数不等于双域训练。仅规划，未启动任务。当前方案：refine-logs/EXPERIMENT_PLAN.md。


## 2026-09-20 2Domain 4/8卡审阅计划

推荐4H200共置，若8张同规格且全部可用则2×4并行；旧8卡日志为L20Y，不能据卡数断言更快。建议global batch528保持不随GPU改变、双域264/264，单seed42、60steps；四行MC1完整增权/MC0同RKL32无增权/MC2仅Math增权/MC3 TIP rho0.5。MC0不能误用native chosen-token PG。8benchmark1004题8032rollouts，专项不称Standard10。用户要求先看，未生成生产配置或启动任务。

[计划](/Users/linghuazhang/Desktop/Project/OPD/plan/two-domain-20260920/EXPERIMENT_PLAN.md)。


## 2026-09-20 用户要求纳入全部已跑baseline

双域计划扩为10训练设置：OPD TopK32、OPD Native、EOPD、ExOPD、TIP TopK32、FiRE Native六baseline，加完整方法/仅Math增权、V1 KL+Entropy/V1 Speed历史对照。Base/Teacher优先复用评测，SFT历史checkpoint/协议待查。仅存在profile但未确认完成的其他方法列查漏项，不伪称已跑。单seed60step不变；8同型H200可5波2×4，4卡串行；未启动。


## 2026-09-20 Math+Code 六 baseline 配置落地

用户最新明确 batch 约512、vLLM显存比例0.75。本轮在 code/configs/baselines/math_code 新建12个入口（六方法×四卡共置/八卡6+2）、_common.yaml和README。统一batch516（最接近512且被4/6整除），Math/Code每步258/258；此设置取代此前规划中的528，仅限本次新双域配置。seed42、60steps，HF仅Step60。方法为OPD TopK32 Uniform、OPD Native、EOPD Native(tau0.8/alpha1/K16)、ExOPD lambda1.25、TIP TopK32 rho0.5、FiRE Native。共置teacher FSDP4且关闭自适应batching；独立teacher FSDP2且开启同步batching。

全部12份通过typed配置加载、canonical actor字段比较、领域/batch/placement检查、launcher dry-run；独立code reviewer未发现阻塞问题。没有GPU测试或启动训练。本地缺模型/Eurus数据，沿用现有remote路径约定。0.75不等于算力利用率。建议四卡能稳定运行时优先两组四卡并行，八卡耗时需小于四卡一半才赢得该六方法排程总耗时与卡时；实际依赖硬件与峰值显存。正式评测另走Step60双域8dataset×K8，不能冒称Standard10。

[配置与布局说明](/Users/linghuazhang/Desktop/Project/OPD/code/configs/baselines/math_code/README.md)


## 2026-09-20 双域配置审计与提交

已审计并本地提交 math_code 配置、README、AUDIT，共15文件，commit df9c503。12个新配置contract与launcher dry-run通过；关联回归100通过、3失败，均为HEAD已有缺失的旧Math配置引用，详见configs/baselines/math_code/AUDIT.md。未包含其他工作区改动，未运行GPU。已通过系统配置的本地代理推送到 origin/main（0ad5844→df9c503），分支与远端同步。


## 2026-09-20 EOPD→ExOPD 远端串行训练

用户要求用 `start.sh` 轮流运行 EOPD 与 ExOPD。远端 EOPD 已由 `start.sh --local --foreground` 启动，run_id=`eopd_native_4gpu_colocate_b516_s42_20260920`，GPU0–3，目标60 steps；配置与本地 sha256 一致。启动前检查显示 GPU4 为独立 vLLM 服务、GPU5 空闲，根文件系统可用约1.57TB；全代码 rsync dry-run/实际同步完成，0个文件变化。

已创建当前线程 heartbeat `eopd-exopd`，每30分钟监控 EOPD 的 step、日志、GPU、磁盘和错误；仅当 EOPD 成功退出且 `global_step_60/actor/huggingface` 存在、GPU0–3空闲时，再用同一 `start.sh --local` 启动 ExOPD run_id=`exopd_native_lambda1p25_4gpu_colocate_b516_s42_20260920`。不得占用 GPU4/5，不重复启动，不自动重启失败任务。EOPD 当前处于初始化/首 step，日志显示 Total training steps=60、Math+Code batch516。


## 2026-09-26 八模型 Step60 Math+Code 评测启动

用户授权现有4模型先评测，并下载原清单缺失4项；已从 public icemoon28/opd-checkpoints 固定 revision 221e385514f877f0817e7b03c21e7a8bee586281 恢复 OPD Uniform、TIP、MOPD C05、MOPD C01 cstruct，SHA256/Hub LFS/safetensors/tokenizer均验证。baseline上传路径使用config stem，已核对experiment_name精确对应。

远端控制器744860，配置 plan/mc-step60-eval-queue-20260926/all8_ready.json。四波 EOPD+C01、ExOPD+FiRE、OPD Uniform+C05、TIP+C01 cstruct；各GPU0,1/2,3、DP2/TP1，不用4/5。首波两模型18:58 +08已开始AIME24，首分片完成，无D状态/CUDA错误。八datasets×K8、seed42、temp1、top_p1、16384tokens，Code Docker async2/pending2、官方EvalPlus sanitize+base+plus后才判完成。Math+Code partial/non-canonical，不写作Standard10。

远端 logs/status 在 experiments_records/eval/mc_step60_queue_20260926/all8_ready；每suite位于 experiments_records/eval/<run_tag>。每项 QUEUE_EVAL_SUCCESS/status complete才完整完成，失败停止后波不重试。heartbeat math-code-c01已改为监控自动队列，全部归档后删除。M05/C02训练已完成但不纳入本次八模型。

修复 sparse G-OPD 缺LCB runner：从本地固定37371a4c源码同步并验证v5=167/v6=175数据hash。首次控制器仅在preflight误判常驻Xorg后退出，无eval已启动；改为无compute、0util、<1GiB的Xorg可用后正常启动。未手动nvidia-smi、未杀进程、未覆盖旧结果。详情 plan/mc-step60-eval-queue-20260926/README.md。


## 2026-09-26 22:54 +08：首波失败，后续六模型未启动

SSH 恢复后的实时核验：controller 744860 已退出，队列 FAILED.json 记录 22:40:37 +08 Wave 1 failed; all subsequent waves halted。含官方评分的完整完成数为 0/8。

- EOPD：86/128 分片有 SUCCESS（四项 Math 各 16/16、HumanEval+ 16/16、MBPP+ 6/16、LCB v5/v6 0/16）。20:20 +08 worker 日志显示 Docker 评分 15 秒超时，随后 docker rm 清理又 5 秒超时，异常导致退出。
- MOPD C01 regular：128/128 分片有 SUCCESS，merge.log 确认 merged shards=128；official EvalPlus finalizer 因 Missing MERGE_SUCCESS 在入口退出，官方评分尚未完成。suite_manifest 的 complete/普通 SUCCESS 不等于完整评测完成。
- 其余六模型没有启动。未重启、未重试、未杀进程；保留全部输出和缓存。heartbeat 已更新为失败后的只读状态，不重复通知相同错误，等待用户恢复指令。

远端证据：experiments_records/eval/mc_step60_queue_20260926/all8_ready/FAILED.json、两项 status.json 与 .log；EOPD logs/gpu_worker_1.log；MOPD C01 logs/merge.log。


## 2026-09-26 23:14 +08：恢复代码验证完成，启动前检查退出

已修复 start.sh local launcher 的 custom8 official EvalPlus finalization、worker --resume 传递、defer merge旧完成标记撤销；Docker cleanup仅对专属容器有界重试并确认清理，评分timeout/隔离参数不变。相关本地59项测试、远端22项unittest、实际Docker pass/resource-limit测试与两suite resume signature校验通过。远端无pytest，未安装额外包。

恢复配置 all8_recovery_20260926.json 计划第一波保留EOPD86/128、MOPD128/128分片并resume，余六首次启动。源码同步前dry-run通过；旧远端源码保存于all8_recovery_20260926/sources_before。cleanup hash仅显式from/to迁移、原shard provenance保留。

nohup controller2591024于15:12:32 UTC在preflight发现系统colord-sane PID2591028为D状态，立即退出；没有GPU查询、没有新suite或worker、没有修改原suite manifest。只读复查该PID已消失。按既定“不自动重试”要求未再次提交，当前仍0/8完整完成。新FAILED.json与recovery_cpu_preflight.json在远端experiments_records/eval/mc_step60_queue_20260926/all8_recovery_20260926。heartbeat已记最新状态，等待明确恢复指令。


## 2026-09-27：SAPD 查新与定位

主分析和独立Codex评审5/10、PROCEED WITH CAUTION；TIP/DEAR核心重叠，Rock Tokens机制边界，待验证语义身份的条件效用。Claude Code Opus5.5及授权opus fallback均403，无跨家族意见。详见 [查新记录](sapd-novelty-20260927.md) 与Obsidian `Papers/SAPD-Novelty-and-Positioning.md`。未启动训练或改变现有队列。


## 2026-09-27：SAPD motivation 跨模型辩论完成

按用户指定，实际使用 gpt-6-astra / medium 与 Claude Code claude-opus-5-5 / xhigh。两轮Claude调用成功：沿用已有bridge代理，通过临时npx Claude Code2.1.283解决默认2.1.274版本不支持问题，未修改默认安装/认证。此前403为历史状态。

双方收敛为：保留具体阶段观察与类别覆盖差异，由此提出功能先验能否改善误差驱动选择；phase share不是因果效用，entropy类别低覆盖不自动证明同一批high-gap遗漏，TopLoss不是已测得学习价值。V3有phase构造provenance；缺少本轮联合核验不等于作者未做；runtime时序按实际版本说明。

完整往返与最终草稿：[DEBATE_SUMMARY.md](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-story-debate-20260927/DEBATE_SUMMARY.md)、[MOTIVATION_DRAFT.md](/Users/linghuazhang/Desktop/Project/OPD/plan/sapd-story-debate-20260927/MOTIVATION_DRAFT.md)。未改代码、未启动训练，无新增实验结论。


## 2026-09-27 SAPD Introduction

完整英文六段稿：`../plan/sapd-introduction-20260927/INTRODUCTION_DRAFT.md`。Codex生成，Astra6 medium分析，Claude Code Opus5.5 xhigh两轮对抗已完成；12篇引用独立核验通过。删除未支持的beyond-mismatch承诺，正面对SAPD适用Rock Tokens风险，区分category density与neighborhood halo。未补造SAPD vs OPD数值，作者主结果待补。详细修订：同目录REVISION_LOG.md；Obsidian Writing/SAPD-Introduction-Draft.md。

## 2026-09-28：Token V4/V5 四 baseline 第二版

按用户要求主代理规划、gpt-6-sol/max 执行新版定义。V4 Control Math/Code=121/154；V5=64/74；共享Structure=6/37。V5新增sibling后不再是V4子集，交集59/65。647行membership的2588个baseline计数与NPZ一致，严格>20，artifact SHA为c45b5f7b6bd824140922706b2cf75630dce7130d580c346c638a3358833fd97b。

Code Control代码块外同一mask参与评分与加权；Structure仅最终闭合块I/O、签名、入口语句，排除注释和字符串。Math支持小写final标题。原18个w4配置迁移r2，另18个C/S同时w8变体；Math5/Code5,2,1、batch528、3/4/8GPU拓扑不变。r1 artifact及配置有快照。报告：`plan/token-v4-v5-fourbaseline-20260928/implementation-report.md`。未启动GPU或同步远端。

## 2026-09-28：Token V4/V5 第三版

用户提供r3后，由主代理规划、GPT-6 Sol/max执行最小迁移。C类final/answer/conclusion整体移到Structure：V4 Math119/9 Code153/45，V5 Math61/9 Code72/45。666行membership与四baseline共2664个NPZ计数一致，source与runtime JSON逐字节一致。第三版SHA：2a6e62db4c0986d4bd4825bc0891cf223fb2b1a955d1bc5404eed2ff42dd22d8。

保留36个原配置路径与超参数，更新r3 pins/namespace并备份r2。Code答案标记限代码块外标题/加粗；Math另允许答案词与字面\\boxed同一行。报告：`plan/token-v4-v5-revision3-20260928/implementation-report.md`。未训练、未同步远端。

## 2026-09-28 05:15 +08：远端 GPU 拓扑只读检查

CityU GlobalProtect 连接成功后实查：6张H200 NVL；0/1/2/4/5各14MiB且无compute进程，3被vLLM占用。0–3两两NV6、NUMA0；4/5在NUMA1，相互NODE，与0–3为SYS，无NVLink。驱动P2P read/write全部非对角OK，跨组PCIe P2P为OK，因此不能将缺少NVLink等同于完全无法通信。未做CUDA传输/NCCL实测，不保证五卡联训稳定或加速。V4/V5 batch528不可均分5个actor；3学生+2教师可作为待验证设计。GPU4/5归属限制保留，未占用或启动任务。证据：`profile_output/gpu-topology-20260928/REPORT.md`。

## 2026-09-28：用户指定 GPU 0/1/2/4，先核对 V4 配置

用户已明确授权本次通信测试和V4四卡训练使用物理GPU0/1/2/4，构成GPU4限制的本次例外；不包括GPU5。用户要求先仔细说明config，本轮展开并校验 `mopd_math_code_current_step_token_v4_toploss_m05_c01_fixed4_4gpu_colocated.yaml`：r3、Math5%/Code1%、C/S raw4、batch528、60steps、teacher FSDP4、colocated、BF16；taxonomy SHA匹配，原配置没有NCCL环境覆盖。05:21 +08只读复核四张目标卡均空闲、根盘/tmp同挂载可用525.36GiB，超过500GiB门槛；正式测试/启动前需重新检查。完整参数及launcher覆盖保存于 `plan/v4-4gpu-0124-20260928/`。截至本轮参数说明，未同步代码、未跑NCCL测试、未启动训练。


## 2026-09-28：本次 V4 四卡切换 Next-Step

用户要求复用上一 step 的计算结果进行 token 选择。实际待跑配置改为 `mopd_math_code_next_step_token_v4_toploss_m05_c01_fixed4_4gpu_colocated.yaml`，新 namespace `q1p7b4g-mc-v4-r3-cs-next-m05c01-f4-b528-s60`。补齐原先仅支持Current-Step的V4/V5位置加权路径，正式forward的raw RKL在成功更新后聚合供下步；无额外评分forward。首步Control raw1、Structure raw4，次步开始应用历史Control IDs；Code围栏限制、strict>20、全response top-p分母、C/S单次归一化不变。预算基于上一step，下步实际比例可能变化。Math5%/Code1%、Fixed4、batch528、60steps、模型和GPU0/1/2/4计划不变。

283个不同CPU测试通过（273项回归+补充/重跑25项，含15项重叠；4项最初被沙箱共享内存限制的双进程测试已在允许环境重跑通过）；独立code-review两项alias/unit校验问题已修复并复审通过，launcher dry-run通过。原配置及参数快照已备份。完整记录与更新后的resolved config在 `plan/v4-4gpu-0124-20260928/next-step-report.md`。本轮未同步远端、未进行GPU通信测试、未启动训练；原05:21资源快照需在后续启动前重新检查。


## 2026-09-28 05:50 +08：V4 Next-Step GPU0124训练启动

先检查磁盘525.35GiB和空闲GPU0124，四rank NCCL all-reduce/all-gather/reduce-scatter均通过（default transport，无P2P禁用）。完成1798源码文件的dry-run/备份式全量同步及关键hash/完整config校验后，通过start.sh --local启动run `v4_next_0124_20260928`，PID3283417。05:52模型加载完成并进入60step循环、W&B在线记录正常；尚无完成step指标。V4 r3 / Next-Step TopLoss / Math5% Code1% / Fixed4 / batch528不变，tail=0且不使用teacher confidence选token。

自动审批因未明确授权公开目的地，拒绝原配置Step60上传到icemoon28/opd-checkpoints。使用显式trainer.huggingface_checkpoint.enabled=false覆盖后训练已启动，本地checkpoint正常保存；公开上传授权问题已发给用户，待答复。两子agent仅在启动后开始本地latest Current-Step矩阵迁移，不同步到正在跑的远端。详情：`plan/v4-4gpu-0124-20260928/RUN_MANIFEST.md`。

## 2026-09-28 06:03 +08：首步完成与上传授权

该训练step 1/60已完成；首步Control加权0，step 1统计选出供step 2使用的Math 69、Code 19个Control IDs。用户明确授权Step 60模型checkpoint公开上传到`icemoon28/opd-checkpoints`。因为训练进程内的HF功能启动时关闭，独立CPU上传进程PID3299246等待完整Step60本地checkpoint后复用既有model-only上传函数；当前状态waiting，未声称已发布。最新范围只迁移37个无tail/TC的active公开Current-Step配置和8 helper；4个tail/TC示例暂缓，不能静默变算法。详情见`plan/v4-4gpu-0124-20260928/RUN_MANIFEST.md`。

06:05独立审查后修正上传程序的陈旧收据与仓库公开性核查，原等待进程3299246在Step60目录尚不存在时替换为PID3303128；训练不重启。新的上传状态仍为waiting。


## 2026-09-28：Current-Step 配置独立审计与无 tail/TC 迁移验收

用户要求最新提交的 current-step 配置改为依据上一 step 的 raw loss 选择 token，暂不加入 tail 或 teacher confidence。用户明确授权本审计 chat 向实施 chat 同步发现，由实施 chat 统一修改；审计 chat 保持独立验收。初始 41 个公开配置均仍为 current_step；36 个 V4/V5 本来没有 tail/TC，另 4 个实验例子通过非零 domain override 启用了 tail。最终 37 个公开 canonical profiles（36 V4/V5 + 1 legacy 纯 TopLoss）及旧名 alias 全部解析为 next_step/token_id/top_loss，tail 关闭、interval/window=1；8 个 helper 同步迁移，4 个 tail/TC 示例退出 active 配置并保留历史。全 resolved config 对比仅 timing 与六类 namespace 改变，正在运行的 V4 四卡配置与保护快照完全一致。

本审计完成 130 项 focused tests；随后其中 24 项受 fixture 修改影响的测试在不含 plan/ 和实验输出的独立源码副本中再次通过，临时副本已清理。独立 runtime review 未发现阻塞；本 chat 未执行 GPU 训练、commit/push 或远端同步。原来的预算、taxonomy/位置规则、C/S raw4/raw8、3/4/8-GPU topology、batch528 与 60steps 保持。

完整审计：[REVIEW.md](/Users/linghuazhang/Desktop/Project/OPD/code/plan/config-audit-no-tail-tc-20260928/REVIEW.md)；机器核验：同目录 config-inventory.json、post-migration-audit.json、isolated-validation.json。后续优先使用 next_step canonical 文件名，不能以旧 alias 的文件名推断 current-step 机制。

## 2026-09-28 06:21 +08：训练step 2与迁移推送

V4 GPU0124训练step 2/60完成：实际Math/Code active Control IDs=69/19，与step1的next-active完全一致；step2新选出74/25供step3使用，说明上一step选择在远端实跑生效。独立Step60公开模型上传进程PID3303128仍为waiting，尚未发布。本轮配置迁移150项相关测试与独立审查通过，commit`6dd711d833c40e64385aa1e45d197b021163940f`已推送`origin/main`；运行中的远端训练没有重新同步或重启。启动时1798文件源码快照已逐一校验并归档至`plan/v4-4gpu-0124-20260928/launch-source.tar.gz`，临时stage已清理。

## 2026-09-29：评测后追加 V5 Shared TopLoss 训练

用户要求：当前 V4 四卡训练及 V4/V5 两项 Math+Code Step60 评测均成功结束后，追加运行 Token V5 r3 的 C+S shared Next-Step TopLoss，关闭 Code C/S 位置门控。计划见 `plan/v5-shared-toploss-after-eval-20260928/PLAN.md`，已有小时 heartbeat 已延长到新训练终态。现默认使用已有三卡配置与物理 GPU0/1/2；GPU4 现有许可仅适用于前述评测，若用户另行明确授权本次训练用 GPU4，才改用对应四卡配置。新 profile 保持 batch528、Step60、Math5%/Code1%、Fixed4，不自动上传 HF。此时只排队，没有同步远端或启动新训练；转阶段前必须重查两项评测的 suite/official EvalPlus、磁盘500GiB、GPU空闲、源码同步与独立 run namespace。

## 2026-09-30 13:57 CST：旧 checkpoint 清理与 Current-Step Step15 恢复

用户授权将旧三域 checkpoint 清理、已完成Step60仅保留模型及加载资产。93个精确操作（58个旧三域目录、35个model-only）实际释放1736.641GiB，盘可用2204.594GiB；40个完整HF导出共415文件前后SHA256一致，logs/audit/评测原始数据保留。执行清单和receipt：`plan/remote-checkpoint-cleanup-20260930/`。

原GPU0124 Math Control124 / Code Structure551 Current-Step run日志完成16，最后完整checkpoint15；旧PID退出，host boot=2026-09-30 10:11:31 Taipei，未见traceback/OOM。Step10/15完整保留。CPU验证四rankmodel/optimizer/extra结构、scheduler15、全部RNG、data.pt及HF config/tokenizer通过。恢复overlay显式resume_path=Step15、wandb_resume=must；batch528、seed42、total60、Current-Step TopLoss/Fixed4、Math5%/Code1%保持不变。1868源码文件先dry-run再完整sync并SHA核验，源代码snapshot归档。

新run `mc_ctrl_struct_current4_resume15_20260930`，launcher PID52670，GPU0/1/2/4，start.sh --local --foreground + nohup/setsid；正在初始化，实际restore还在验收。详见 `plan/current-step-control-structure-4gpu-20260930/recovery-step15/RUN_MANIFEST.md`。

恢复验收补充：日志确认global_step15、原W&B resume与model/optimizer/RNG/scheduler实际加载；GPU0124四worker已进入第16步rollout、利用率100%，无traceback/OOM。PID52670；本地source stage已清理，完整source tar/hash、crash log、清理receipt保留。未commit/push。

## 2026-09-30 17:53 +08：疑似再次中断的只读核查

同一run `mc_ctrl_struct_current4_resume15_20260930` 实际持续运行，PID52670与四个原worker不变；17:49保存Step20，17:53四GPU0124利用率100%、actor_rollout_generate_sequences正在执行Step21。W&B API为running（summary19滞后于本地20）。Step16–20耗时42–49分钟/step（平均46.59），teacher/ref约21分钟是最大实测组成，Current-Step prepass44–46秒。磁盘2161.10GiB、boot仍10:11:31；未见应用错误栈，dmesg权限拒绝不能排除所有kernel历史事件。

Step20已核验12份rank PT archive结构、scheduler.last_epoch20、可CPU加载data.pt、311-tensor HF文件header/payload与latest marker20；实际从20恢复未测试。独立bug-analyzer与code-reviewer复核通过。没有重启/终止/sync/删除远端文件、没有commit/push；完整诊断见 `plan/current-step-control-structure-4gpu-20260930/second-crash-investigation/README.md`。判断无指标更新时需结合长step耗时、worker阶段与GPU活动，当前不需要重新断点跑。

## 2026-09-30 18:05 +08：原任务关机原因得到直接日志证据

用户澄清要问恢复前原run为何挂了。只读读取previous-boot journal，09:03:02明确记录sudo账号shuang_qiu、TTY pts/2、USERroot、COMMAND=/usr/sbin/poweroff，随后09:03:09 session shutdown、wtmp09:04:33 system down、10:11:43 reboot。原GPU monitor09:02:46四目标卡仍100%，旧log完成16后在17/ref准备处停止，无硬异常栈。证据支持账号触发主机关机打断原训练；具体人的身份/意图不能由日志确认，不归因于桌面firmware-notifier异常。最近完整保存15，从15恢复正确。归档 `plan/current-step-control-structure-4gpu-20260930/recovery-step15/ORIGINAL_STOP_CAUSE.md` 与 `original-stop-system-evidence.json`；当前训练未操作。


## 2026-09-30：SAPD四份规范研究文档归档

- 按research-workflow-controller整理 [Idea](/Users/linghuazhang/Desktop/Project/OPD/paper_proposal/idea.md)、[Proposal](/Users/linghuazhang/Desktop/Project/OPD/paper_proposal/proposal.md)、[Method](/Users/linghuazhang/Desktop/Project/OPD/paper_proposal/method.md)、[Experiment](/Users/linghuazhang/Desktop/Project/OPD/paper_proposal/experiment.md)，paper_proposal仅允许这四个文件。
- 当前主线为SAPD功能候选prior+TopLoss相对增权；legacy、V4/V5 default固定Structure、Shared与9/30 Current-Step分别记录。Shared的Math Structure不沿用default位置限制。
- 保留作者已有density/C+S/selector观察；实测C01 Structure-only Macro8=31.657633%，EOPD=31.463167%，差+0.194466pp；batch528/516及topology/objective等差异限制因果解释。展示稿模拟值不进入实测。最终recipe和matched prior独立收益尚待确认。
- 真实历史Claude为2轮motivation+2轮Introduction回应；本轮只核对归档，无新exchange，不追认完整科学gate通过。六篇最近邻仅作身份/版本/摘要局部核对，Zotero仅metadata fallback；不是新全文/全面novelty审计。
- 当前训练/评测状态按本地有日期记录保留；没有启动或操作训练/评测。本轮来源、备份、tracker和核对报告见 [RUN state](/Users/linghuazhang/Desktop/Project/OPD/temp/research-workflow/20260930-canonical-summary-8213a4c7/state.md)。


## 2026-09-30：SAPD Introduction 与论文初稿

- 使用 intro-drafter 与 paper-writer，基于四份 canonical 输入完成 [Introduction 初稿](/Users/linghuazhang/Desktop/Project/OPD/paper_proposal/intro_draft.md) 和 [Paper 框架初稿](/Users/linghuazhang/Desktop/Project/OPD/paper_proposal/paper_draft.md)；两份英文初稿由用户明确授权放入 paper_proposal，该目录现允许原四文件及这两个精确新增文件。四份输入 SHA256 未变。
- Introduction 为六段正文、17 项引文；Paper 含 Abstract、九个主章节、方法公式、四种 implementation profile、九行 exploratory 结果、Discussion 与附录。[分章节 LaTeX 入口](/Users/linghuazhang/Desktop/Project/OPD/latex/main.tex)；[PDF 预览](/Users/linghuazhang/Desktop/Project/OPD/latex/main.pdf)。
- 旧 LaTeX 已完整备份到本轮 RUN/backups/latex；当前入口使用 SAPD，不沿用旧 ASCR phase controller。编译从正式 latex 目录执行 make pdf 成功，PDF13页，无未定义引用或 overfull 警告；公式、表格、全部初次页面及调整后附录/参考文献已目视检查。
- 22 项引文通过 fresh-context reviewer 的74条真实查询核对，最终无 unresolved citation issue。修正 RFT 正式题名与 Reasoning-highlighted Fine-Tuning 名称，以及 DistiLLM adaptive off-policy 归因；核对范围为身份/元数据/摘要级主张，不是完整全文或方法复现。
- 主结果来自 legacy，其中 C01 的 Structure-only 仅指 Code S551，Math仍为C+S390。功能 prior 的独立收益、唯一主 recipe、matched ablation 与训练跨 seed 证据仍待补齐。本轮没有运行训练/评测或查询 live run。
- 工作记录与引文/科学核对见 [本轮 writing RUN](/Users/linghuazhang/Desktop/Project/OPD/temp/research-workflow/20260930-sapd-writing-fb230dc0/state.md)。


## 2026-09-30：SAPD 两份初稿中文译文

- 按用户“翻译给我看看”的要求，依据 paper-polish 的语义保真规则完成 [Introduction 完整中文译文](/Users/linghuazhang/Desktop/Project/OPD/latex/translations/intro_draft_zh.md) 与 [Paper 框架完整中文译文](/Users/linghuazhang/Desktop/Project/OPD/latex/translations/paper_draft_zh.md)。Introduction 保留六段；Paper 覆盖 Abstract、九个主章节、方法公式、结果表与全部附录。
- 两份英文源文件 SHA256 未变；正文引文序列、17/22 项参考文献条目、47 处数学内容及结果表数字均与英文一致。保留探索性、作者报告、计划、单 training seed 等证据边界；本轮没有新增文献或科研主张。
- fresh-context 语义核对逐章通过，未发现实质误译或遗漏；采用 token type 与 Pass@8 两处精度微调。核对结论仅针对翻译保真，不代表新的科学或文献全面审查。
- 中文文件放在 latex/translations，paper_proposal 继续保留原四份规范文档与两份已获授权的英文初稿；目录核对 PASS。工作记录：[翻译 RUN](/Users/linghuazhang/Desktop/Project/OPD/temp/research-workflow/20260930-sapd-translation-be807bce/state.md)。


## 2026-10-01：SAPD Introduction 直接相关文献扩充

- 按用户要求在线检索并加入9篇直接近邻：ToDi、TIDE、USD、Multi-Granularity Semantic Revision、TRACE、OmniOPD、reasoning-prefix OPD、ReNIO与OPSA。Introduction从17增至26项，完整版从22增至31项；其中新增3篇正式会议文献、6篇arXiv preprint。
- [英文Introduction](/Users/linghuazhang/Desktop/Project/OPD/paper_proposal/intro_draft.md)、[中文Introduction](/Users/linghuazhang/Desktop/Project/OPD/latex/translations/intro_draft_zh.md)、两份完整版、分章节LaTeX与BibTeX已同步。[PDF](/Users/linghuazhang/Desktop/Project/OPD/latex/main.pdf)正式目录编译14页，无未定义引用或overfull；最终Introduction及参考文献已目视核对。
- 26项Introduction引文由fresh-context两组核验，执行78个基础查询及3个额外venue查询；全部VERIFIED。三处既有文献描述与OPSA方法归属按核验意见澄清；中英/LaTeX一致性复核通过。该审查为same-family/provisional的出处和引用内容核对，不是全面novelty review或全文复现。
- 六段结构保留，方法/贡献两段、公式和结果表不变；四份canonical文档SHA-256不变。没有新增实验效果或Claude exchange。论文差异与来源见[新增文献说明](/Users/linghuazhang/Desktop/Project/OPD/temp/research-workflow/20261001-sapd-intro-citations-cc5105f3/citation-selection-report.md)。


## 2026-10-01：code/ 未提交修改精简审查

- 按用户要求审查23个tracked修改及1301个untracked；最小精简raw未评分summary、MANIFEST重复事件和重复四卡配置测试，5个独有训练约束合并保留；修正Token active旧数并统一runtime说明来源。
- 8条定向ignore排除902个生成产物，磁盘原件保留；57个配置SHA均不变，teacher独立chunk1024/guards、async评分、LCBformatter和Dockercleanup保留。本轮未覆盖其他chat的teacher runtime。
- V6–V8测试改为独立baseline Rising小fixture（来源SHA与154个IDs逐组核验），去本地分析目录/pandas依赖。209项CPU回归、干净导出40项、最终18/10项复跑和shell检查通过；独立review Approve。
- 待修P2：custom仅单项HumanEvalPlus/MBPPPlus + score_code绕过official收口而发布complete；canonical10/双项custom正常。完整[审查与备份](/Users/linghuazhang/Desktop/Project/OPD/plan/repo-change-review-20261001/REVIEW.md)。main未commit/push，未连接远端或运行GPU任务。

- 收尾并发变化：另一处工作将plan/profile_output/refine-logs整目录ignore，53个文件退出Git索引且原件均在；本轮未执行该索引操作，去掉被整树规则覆盖的7条子目录ignore。最终main状态与并发证据见上述审查目录。


## 2026-10-01：V6 Math5% Code1% 四卡配置参数展开

- 使用 load_config 与 build_overrides 核对 mopd_math_code_current_step_token_v6_toploss_m05_c01_fixed4_4gpu_colocated.yaml：Qwen3-1.7B/30B-A3B、batch528（Math264/Code264）、60steps、LR5e-6、teacher Top32 reverse KL、teacher独立chunk1024与12GiB guard，actor chunk16。
- C/S联合池 Math118+156=274、Code23+196=219；current_step/token_id/top_loss、strict count>20，5%/1%为domain全部valid response occurrences预算，整ID选择可overshoot。
- Fixed4反向gradient mask按rank-local microbatch/domain mean-one归一化：selected=4/(1+3r)，其他=1/(1+3r)；前向KL值保持不变。另乘detached token IS（上限5），全局valid-token mean；样本1:1不保证gradient贡献1:1。
- 本轮为本地只读配置/实现核对，未修改训练YAML、未启动GPU任务。


## 2026-10-01：V6 四卡训练已启动

- 按用户授权停止旧 Current-Step run（已完成31、latest checkpoint30），旧日志与checkpoint保留；四rank12份训练archive及HF权重结构核验通过。
- 新run `mc_v6_cs_current4_g0124_20261001`，launcher PID494574、TaskRunner497085，direct-local，物理GPU0/1/2/4；GPU4保留他人PID369498约1.7GiB，用户在获知占用后再次明确要求该四卡组合。
- Canonical V6 Math5% / Code1% C+S Current-Step，pool274/219、raw4/mean-one、batch528、60steps；teacher独立chunk1024/margin12GiB，actor chunk16。HF Step60 public、resume disabled。
- 87项本地检查和独立review通过，1941文件完整rsync带备份且SHA精确一致，STOP_STALE_RAY=0。05:50 Asia/Taipei仍在数据预处理，尚未声称完成训练step或测得有效chunk/提速。
- [运行记录](/Users/linghuazhang/Desktop/Project/OPD/plan/v6-current-step-4gpu-0124-20261001/RUN_MANIFEST.md)；[W&B](https://wandb.ai/lz101-rice-university/MOPD/runs/q1p7b4g-mc-v6-cs-current-m05c01-f4-b528-s60)。


## 2026-10-01 06:06 +08：V6 四卡启动核验完成

- 05:57远端核验：物理GPU0/1/2/4分别对应新worker498206/498212/498215/498219，均存活；新W&B已建立，日志进入首个训练step（0/60，未声称完成Step1），未匹配到Traceback/Ray/NCCL/OOM错误。
- Teacher真实worker配置确认独立TopK chunk已开启，requested/configured chunk1024、margin12GiB、FSDP/world_size4；尚未看到首次forward的effective_chunk日志，不能声称已测得实际chunk或提速。
- 后续SSH连接超时；06:06另经W&B API只读核验state=running、heartbeatAt=2026-09-30T22:06:17Z（本地06:06:17，查询时约2秒前），确认run仍有新心跳。未重启或再次停止训练；完整启动日志下载未完成，已有四worker与日志摘录证据保留。
- [运行记录](/Users/linghuazhang/Desktop/Project/OPD/plan/v6-current-step-4gpu-0124-20261001/RUN_MANIFEST.md)；启动核验证据startup-progress-5.json、wandb-status.json。


## 2026-10-01 06:07 +08：SSH恢复与teacher实际chunk核验

- 06:06:50重试SSH成功；launcher494574与四worker均存活，物理GPU0/1/2/4正在ref_compute_ref_log_prob，首step仍为0/60，没有完整Step1或提速结果。
- 06:00:11真实首次teacher forward日志：configured_chunk=1024、effective_chunk=1024、tokens=190、capacity=94418、memory guard passed；不能外推后续所有shape。GPU3仍由vLLM占约129GiB，GPU4他人已知进程保留。
- 完整启动日志、outer log、PID、launch脚本和GPU CSV已下载，zip与逐文件SHA核验通过；临时base64重复payload已删除。W&B API心跳查询时实际约2秒前；SSH超时已恢复，未为此重启训练。
- 启动切换已完成；证据见运行目录startup-final.json、final-ssh-probe.txt、runtime-evidence-receipt.json与runtime-evidence.zip。


## 2026-10-01：实际启动config参数复核

- 用户追问启动config与具体参数；对照启动日志CONFIG、冻结resolved-config及launcher overrides确认使用V6 m05/c01 fixed4 4gpu colocated。GPU0/1/2/4及Python由启动环境指定，初始化Qwen3-1.7B、resume disable。
- Math274/Code219为C+S联合TopLoss候选；5%/1%是全domain valid-response occurrence预算，strict count>20，本step选/用，局部mean-one raw4/1。teacher batching=false但独立TopK chunk=true1024（首次forward已实证），actor chunk16；继承的max_micro_batch32/max_tokens57344不表示本run启用teacher批处理。
- 参数来源与完整展开配置：/Users/linghuazhang/Desktop/Project/OPD/plan/v6-current-step-4gpu-0124-20261001/resolved-config.json。未改训练配置或操作远端进程。


## 2026-10-01：V6训练速度诊断与三卡布局可行性

- 12:37远端日志已完成step1–10、checkpoint5/10。均值40.00min：teacher25.12min(62.8%)、reward7.30min(18.25%)、actor update4.50min；Current-Step prepass归约33.81s包含在update，不能重复加总。step10为44.00min，response mean由1008增至3988。W&B查询到step9且新心跳，日志step9计时一致；sampledHistory本次0行。
- GPU0/1/2为NV6，GPU4为跨NUMA SYS；GPU4其他用户PID369498虽仅1768MiB，12:41 pmon持续24%SM。通信/共享算力是teacher慢的重点怀疑项，尚无NCCLtrace或净提速测量。Reward实际naive同步串行逐条评分528行，Math Verify/prime_code；已有batched scorer未被使用，不归因于Eval Docker。
- 更正先前参数说明：ref.param_offload=True是配置请求，当前GPU teacher实际CPUOffload=None；actor才有手动参数/optimizer offload。独立chunk1024只优化Top32后处理，不减少teacher forward/通信。
- 用户问teacher只0/1/2：现成V6 m05/c01 3gpu_colocated可用且load_config/build_overrides核验通过，student/teacher都3卡、batch528、MB1、候选274/219、5%/1%、Fixed4、teacher chunk1024/margin12GiB保持。actor4/ref3重叠部分colocation当前不支持；separate两个池需要7GPU，单改fsdp_size3不成立。
- 本轮未停止、重启或切换训练，未修改训练源码/配置，也未新建自动监控或发通知。具体提速和3卡显存需独立验证，当前三卡模板resume disable不代表已验证4rank续训。
- [完整诊断](/Users/linghuazhang/Desktop/Project/OPD/profile_output/v6-current-step-slow-20261001-1234/REPORT.md)；[三卡配置](/Users/linghuazhang/Desktop/Project/OPD/code/configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v6_toploss_m05_c01_fixed4_3gpu_colocated.yaml)。


## 2026-10-01：V6 student4 / teacher3，共享GPU新config完成（未启动）

- 用户明确保留student物理GPU0/1/2/4，teacher仅GPU0/1/2。现已新增opt-in共享ref pool支持与[新config](/Users/linghuazhang/Desktop/Project/OPD/code/configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v6_toploss_m05_c01_fixed4_4student_3teacher_shared.yaml)，独立actor world4/ref world3，ref FSDP3，实际总占4卡。通过实际actor CUDA mask反查teacher bundle，拒绝缺失/重复/多GPU可见性，不依赖rank顺序；两个launcher按物理4卡计数。
- 完整继承原V6 Math5%/Code1%、C+S pools274/219、Current-Step、strict occurrence>20、rawFixed4局部mean-one、batch/mini528、MB1、Top32、LR5e-6/seed42/steps60。Teacher batching仍false，独立chunk1024/margin12GiB保留；vLLM仍sync TP1/KV0.75。新namespace q1p7b4s3t-mc-v6-cs-current-m05c01-f4-b528-s60，fresh resume disable，public HF60。
- 共享PG每bundle GPU1/CPU2，actor/ref各0.5逻辑GPU；只预留4-bundle。Teacher先init并清独立进程缓存，每次ref RPC完成后再synchronize/empty_cache交还显存。首版限single-node CUDA/FSDP/sync TP1/SP1/单teacher；拒绝LoRA和base/secondary模型路径。
- 131项回归通过；最终36项shared测试通过（含LoRA空adapter拒绝，与前项重叠）；另1项真实Ray2.43/Torch2.5.1 Gloo CPU witness通过，验证7进程、独立world4/3/all-reduce10/6、打乱CUDA mask仍选0/1/2、528 rows经132/176分片后row/uid/mask/Top32顺序一致。新文件lint/AST/shell语法通过，独立review无剩余当前config阻塞。CPU fake GPU slots不等于CUDA/NCCL/FSDP/vLLM模型验收。
- Teacher每卡常驻权重约增加33%，实际启动显存及提速尚待远端验证；此配置使用start.sh --local和物理GPU_IDS=0,1,2,4，不是Slurm-relative mask。本轮没有SSH、远端同步、停止/启动、commit/push或通知；当前远端run未切换。
- [实现与验证记录](/Users/linghuazhang/Desktop/Project/OPD/plan/v6-4student-3teacher-shared-20261001/REPORT.md)，resolved-config/overrides、verification.json、代码备份和本轮patch保留。


## 2026-10-01 13:44 +08：V6 student4 / teacher3 远端切换完成

- 按用户明确要求，以 cityu-hk-vpn-login skill 和已有 Keychain 凭据连接 VPN，定向停止旧 run `mc_v6_cs_current4_g0124_20261001`（已完成11，latest checkpoint10）；旧四rank训练archive/data.pt/HF权重结构核验通过，179个已确认session成员退出，checkpoint10及日志保留，没有全局Ray stop或他人进程信号。
- 完整1950文件先rsync dry-run、带backup同步且逐文件SHA精确一致（聚合3b5d6d86ea12fcb756d1b5578ce4172d86ea4d4a892dc513d649f7654bafbdb1），有效checkpoint/log/audit/tmp/Ray/W&B文件系统约2014.92GiB可用，超过500GiB门槛。GPU4已授权PID369498保留，GPU3/5未动。
- 新run `mc_v6_cs_current4s3t_g0124_20261001`于13:28:19通过start.sh --local启动，launcher649093/TaskRunner651645。使用[4-student/3-teacher shared config](/Users/linghuazhang/Desktop/Project/OPD/code/configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v6_toploss_m05_c01_fixed4_4student_3teacher_shared.yaml)；fresh resume disable，从Step0开始，未加载旧checkpoint。
- 实际布局已验证：student652905/652906/652907/652908分别物理GPU0/1/2/4、world4/port38575；teacher653318/653323/653327分别GPU0/1/2、world3/port40115，ref FSDP3。总占4张物理卡；GPU映射以NVIDIA UUID/实际进程及Ray CUDA检查为准，/proc继承环境不一定反映Python后续CUDA mask修改。
- 原V6 Math5%/Code1% C+S Current-Step、pool274/219、rawFixed4局部mean-one、batch/mini528、MB1、LR5e-6/seed42/steps60保持；teacher batching=false，独立TopK chunk1024/margin12GiB，actor chunk16；vLLM sync TP1/KV0.75，public HF Step60计划保持。
- 模型、四个vLLM和首rollout完成初始化；共享卡rollout snapshot约129.2–129.8GiB，vLLM sleep后回到常驻模型水平。13:42:59真实teacher forward确认effective_chunk=1024，memory guard passed（tokens190/1420，capacity88840/88851）；13:44三ref仍在首step计算，无已观察到的OOM/NCCL/Traceback。完整optimizer step、cache-release-after-full-ref、速度提升和最终训练效果尚未验收。
- [W&B新run](https://wandb.ai/lz101-rice-university/MOPD/runs/q1p7b4s3t-mc-v6-cs-current-m05c01-f4-b528-s60)已初始化并API确认running/新心跳。[运行记录与全日志证据](/Users/linghuazhang/Desktop/Project/OPD/plan/v6-4student-3teacher-restart-20261001/RUN_MANIFEST.md)保留PID、config/overrides、源码快照/backup、GPU映射和下载hash收据；main未commit/push。仅补本地精确.gitignore例外让该非secret训练YAML可见，启动冻结源码保持原样。


## 2026-10-01 14:35 +08：V6 四 Student / 三 Teacher 首次半小时进度

- 新run `mc_v6_cs_current4s3t_g0124_20261001` 完成1/60，Step2正在teacher/ref；所有7个GPU worker和launcher仍在。首step1900.032s（31m40s），Teacher1279.294s（21m19s，67.3%），rollout55.920s，reward276.052s，Student update238.005s，内部Current Step prepass33.187s，不能重复相加。
- 检查时间14:34:57 +08；Step2约26.4min、Teacher阶段约17.8min均按audit/log mtime估算。启动初始化约8.5min，与step计时分开。W&B running、heartbeat14:34:44，但training history/summary暂为空；耗时由远端完整log与audit cost交叉确认。无OOM/NCCL/进程退出；14:12:32有comparison timeout warning，训练继续。
- 仅1步样本外推约10月2日21:17完成，剩余约30.7h；可靠性低、未计后续checkpoint/HF上传额外开销。半小时heartbeat仍ACTIVE；本轮只读，未重启或改参数。证据与连续状态见[监控记录](/Users/linghuazhang/Desktop/Project/OPD/code/plan/v6-4student-3teacher-monitor-20261001/report-20261001T062738.md)。


## 2026-10-01 14:59 +08：V6 半小时监控 Step2

- 完成2/60，Step3 teacher/ref约6.2min（log mtime推算）；Step2总2023.576s（33m44s），rollout69.823s、Teacher1254.490s（20m54s）、Student update232.354s（3m52s，内含prepass32.618s）、reward414.065s（6m54s）。相比Step1，reward多138.013s是step增长123.544s的主要来源，Teacher少24.804s。
- 2步mean/median1961.804s（32m42s）；仅2样本低可靠外推10月2日22:19完成，剩余约31.4h，未含后续checkpoint/HF上传额外耗时。W&B running、Step1已同步并与log一致，Step2由log+audit确认。全部7workers存活，无新增OOM/NCCL/退出或comparison timeout；只读、未重启/改参数，heartbeat继续。证据见[本轮监控](/Users/linghuazhang/Desktop/Project/OPD/code/plan/v6-4student-3teacher-monitor-20261001/report-20261001T065738.md)。


## 2026-10-01 15:30 +08：V6 半小时监控 Step3

- 完成3/60；Step4 teacher/ref约1.75min（log mtime估算），全step约12min。Step3总2156.028s（35m56s），rollout66.416s，Teacher1264.861s（21m05s），Student update230.901s（3m51s，内含prepass32.092s），reward539.365s（8m59s）。Step3较Step2慢132.452s，reward多125.300s是主要增长阶段。
- 最近3步mean2026.545s（33m47s）、median2023.576s（33m44s）；粗估10月2日23:23 +08完成，剩余約31.9h，趋势上升使预测不稳定，未计未来checkpoint/HF上传额外时间。W&B已返回Steps1–2，与log一致；Step3由log+audit确认。
- Step4新增15:22:57/15:27:01 comparison timeout warnings，15:30复查已进入Teacher；全部7workers、launcher存活，无OOM/NCCL/硬异常。只读、未改参数/重启，heartbeat继续。证据见[本轮监控](/Users/linghuazhang/Desktop/Project/OPD/code/plan/v6-4student-3teacher-monitor-20261001/report-20261001T072741.md)。


## 2026-10-01 16:01 +08：V6 半小时监控 Step4

- 完成4/60，Step4总2131.820s（35m32s），rollout136.609s（2m17s），Teacher1271.680s（21m12s），Student update221.710s（3m42s，内含prepass32.395s），reward442.739s（7m23s）。较Step3快24.207s：reward回落96.626s、rollout增加70.194s。
- 最近4步mean2052.864s（34m13s），median2077.698s（34m38s）；粗估10月2日23:50 +08完成，剩余约31.8h，未计checkpoint/HF上传额外开销。W&B running，已返回Steps1–3并与log一致，Step4由log+audit确认。
- 15:58 GPU0/1/2/4空闲；16:01复查7workers等待、TaskRunner CPU样本54.9%，距Step4结束约8min；具体Step5阶段/起点暂无日志证据，不声称teacher或reward已运行某时长，也不判为卡死。无新增OOM/NCCL/退出或comparison warning，只读未重启/改参数，heartbeat继续。证据见[本轮监控](/Users/linghuazhang/Desktop/Project/OPD/code/plan/v6-4student-3teacher-monitor-20261001/report-20261001T075741.md)。


## 2026-10-01 16:44 +08：V6 半小时监控 Step5 与 Teacher 耗时诊断

- 当前新run `mc_v6_cs_current4s3t_g0124_20261001` 完成5/60，Step5于16:30:38写入audit。主step2182.056s（36m22s），rollout116.927s、Teacher1260.399s（21m00s）、Student update218.701s（3m39s，内含prepass32.071s），reward523.109s；checkpoint5另31.763s。16:44检查Step6三个ref RPC执行中，阶段约1m44s按prepared-ref日志mtime估算，未写未完成step的虚构耗时。
- 最近5步均值2078.702s（34m39s）、中位2131.820s（35m32s），粗估剩余31.52h，10月3日00:16 +08完成；未含未来checkpoint/HF上传，样本/response/reward波动限制预测。W&B running，heartbeat16:44:29，Steps1–4 timing与日志一致；Step5由log+audit确认，W&B延迟一step。
- 全部7worker/launcher存活，无OOM/NCCL/进程退出或新增comparison warnings；CPU/Teacher阶段GPU低util不判卡死。Teacher5步均值21m06s，实际MB1/DP3=>176次完整forward/rank，经48层FULL_SHARD通信；性能bundlefalse使batching、stable_sort MoE和fused statistics均未安装。chunk1024仅TopK后处理；entropy audit每步true，chosen logP/entropy/TopK分别全vocab处理含prompt。Teacher窗口GPU0/1/2粗采样平均util14–23%，支持通信/dispatch/同步开销嫌疑，尚无kernel trace或占比测量。源码与运行snapshot一致并经独立只读review。
- 仅监控/诊断，未改训练代码/config或停止/重启。建议固定rollout先单独测试MoE dispatch、再Teacher分组2/4并核验3rank同步/显存/数值，最后fused统计；不能把待用max_micro_batch32当成当前已启用批量。短暂本机ENOSPC后空间恢复，未清理文件，远端训练持续正常。
- [本轮监控](/Users/linghuazhang/Desktop/Project/OPD/code/plan/v6-4student-3teacher-monitor-20261001/report-20261001T082742.md)；[Teacher完整诊断](/Users/linghuazhang/Desktop/Project/OPD/profile_output/v6-4student-3teacher-20261001/REPORT.md)；[W&B](https://wandb.ai/lz101-rice-university/MOPD/runs/q1p7b4s3t-mc-v6-cs-current-m05c01-f4-b528-s60)。Heartbeat保持ACTIVE。


## 2026-10-01 17:00 +08：V6 半小时监控，无新增完成step

- 检查17:00:05：仍5/60，上次已汇报Step5，本轮无新完成step。Step6三Teacher RPC活跃，Teacher阶段约17分07秒（最终prepared-ref日志mtime16:42:58估算）；整体step距上一audit约29分27秒，未结束不写完整耗时。
- 最近5步mean34分39秒、median35分32秒，粗估剩余31.27h，10月3日00:16 +08完成，未含未来checkpoint/HF上传；初始化约8分31秒单记。当前Teacher阶段仍低于近期约21min完整耗时，无明显停滞证据。
- 全部7GPU workers与launcher/TaskRunner存活，GPU0/1/2 util21/9/18%，GPU4 actor等待。无OOM/NCCL/退出或新增comparison警告。W&B running、heartbeat16:59:59，history Steps1–4与log一致，Step5仍由log+audit确认且待提交；网络可达。
- 仅只读监控，未改配置/重启，heartbeat ACTIVE。[本轮记录](/Users/linghuazhang/Desktop/Project/OPD/code/plan/v6-4student-3teacher-monitor-20261001/report-20261001T085909.md)；[W&B](https://wandb.ai/lz101-rice-university/MOPD/runs/q1p7b4s3t-mc-v6-cs-current-m05c01-f4-b528-s60)。


## 2026-10-01 17:34 +08：V6 半小时监控 Step6，VPN恢复

- 完成6/60，本轮新增Step6：总2271.158s（37分51秒）、rollout102.831s（1分43秒）、Teacher1271.460s（21分11秒）、Student update223.605s（3分44秒，包含prepass32.307s）；reward605.147s（10分05秒），checkpoint本步未记录。较Step5总增89.102s，reward增82.038s为主要增长阶段。
- 17:34:33检查Step7正在三Teacher ref RPC，阶段约13分32秒（prepared-ref日志mtime17:21:01估算），整步约26分03秒；未结束不填完整耗时。最近5步Steps2–6 mean35分53秒、median35分56秒，粗估剩余31.86h、10月3日01:26 +08完成；reward/response增长、未来checkpoint/HF开销未计，使预测不稳定；初始化8分31秒单记。
- W&B running/heartbeat17:34:14，Steps1–5 timing与log一致，Step6由log+audit确认。9个target进程均存活，GPU0/1/2 util19/16/14%，GPU4等待，无OOM/NCCL/退出或新增comparison警告。
- 首次SSH超时后按既有授权使用cityu-hk-vpn-login及既有Keychain恢复，GlobalProtect界面确认连接且SSH复读成功，实际凭据未输出；仅网络故障不判训练失败。未改训练参数/停止/重启，heartbeat ACTIVE。[本轮记录](/Users/linghuazhang/Desktop/Project/OPD/code/plan/v6-4student-3teacher-monitor-20261001/report-20261001T092909.md)；[W&B](https://wandb.ai/lz101-rice-university/MOPD/runs/q1p7b4s3t-mc-v6-cs-current-m05c01-f4-b528-s60)。
