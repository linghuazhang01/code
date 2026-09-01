# MOPD 远端调试规则

- 本地仓库是源码唯一 source of truth；所有修改必须先在本地完成。
- 固定流程：本地修改与检查 → `rsync --dry-run` → 上传到 `/home/shuang_qiu/mopd_code` → 远端运行 → 下载日志 → 回本地修复。
- 禁止直接修改远端源码；远端只允许运行、测试、训练和生成运行产物。
- `../ssh.sh` 第 1 行是 SSH 连接命令，第 2 行是密码。不得修改、打印、提交或整体执行该文件；只执行第 1 行，并在密码提示时输入第 2 行。
- 默认禁止 `rsync --delete`，不得覆盖远端独有的 dataset、model、logs 或 checkpoints。
- 后续所有远端 Slurm 任务（训练、评测和 smoke）的内存请求 hard cap 为 `400G`，`--mem` 不得超过 `400G`。若 launcher 默认值高于 `400G`，提交前必须显式设置 `MOPD_SLURM_MEMORY=400G`（或更低）；无法满足时不得提交。

## MOPD 标准评测与归档规则

- 后续标准评测固定使用 Math、Code、Science 共 10 个数据集，并将这 10 个数据集作为一个完整 evaluation batch；少于 10 个数据集的运行只能标记为 partial/smoke，不得作为标准评测汇报。canonical 清单如下：
  - Math（4）：`AIME2024`、`AIME2025`、`HMMT25Feb`、`HMMT25Nov`。
  - Code（4）：`HumanEvalPlus`、`MBPPPlus`、官方增量版 `LiveCodeBench v5`（pinned dataset 的 `v5/` split，167 题）、官方增量版 `LiveCodeBench v6`（`test6.jsonl` / `v6` split，175 题）。pinned source 中不存在官方 `test5.jsonl`；本仓库仅为兼容 G-OPD loader，从两个 v5 parquet shard 确定性生成并记录 hash 的 `test5.jsonl`。不得把累计且大范围重叠的 `release_v5` / `release_v6` 当作这两个独立 dataset。
  - Science（2）：`GPQA-Diamond`、固定的 `MMLU-Pro-500 seed42` subset。
- 10 个数据集中的每个 dataset 均固定执行 8 次 rollout，即统一使用 `K=8`；不得继续沿用旧结果中 Math `K=16`、Code/Science `K=4` 的口径作为新标准。
- 标准评测默认使用 `DP=4`；当用户为集群调度明确指定 `--gpus 2` 或 `--gpus 3` 时允许 `DP=2/3`，并必须在 `RUN_MANIFEST.md` 记录实际 topology。每个 replica 都是常驻的单卡 `TP=1` vLLM worker，所有 worker 同时处理同一个 dataset；10 个 dataset 必须按严格 dataset wave 顺序执行，当前 wave 完成前任何 worker 不得进入下一项。每个 dataset 默认拆成最多 16 个 micro-shards，由当前 worker pool 动态领取以减少长短题造成的尾部空闲；每个 prompt 必须在一次 vLLM sampling request 中使用 `n=8` 生成八个 rollout。不得采用“一卡一个 dataset”的跨数据集并发，也不得对可单卡容纳的 checkpoint 使用多卡 `TP` 代替 data parallelism。Student Base/Teacher 可用显式 `--reference-anchor` 纳入同协议 anchor 评测；manifest 必须将其标为 `reference_anchor`，不得伪称 `global_step_60`。
- Code user prompt 必须逐字对齐 G-OPD 原仓库的 Eval 实现并使用 `enable_thinking=false`：`HumanEvalPlus`/`MBPPPlus` 的 stripped 原题与 G-OPD Python code-fence / "think first" suffix 之间固定为 3 个换行；`LiveCodeBench v5/v6` 使用同一个 G-OPD `Qwen3NonThinking` prompt，并保留 `You will NOT return anything except for the program.` preamble。运行前必须用 artifact preflight 拒绝旧 prompt parquet；修改 prompt builder 后必须重建对应 parquet、记录 prompt/source hash，并重新评测，不得把旧 prompt 的 Code rollout 混入新标准结果。LiveCodeBench public+private scoring 必须走 G-OPD official runner；在 input/output scorer 完成隔离前，不得用通用 Docker Code scorer 冒充有效 LCB 分数。G-OPD 的 LCB formatter 固定使用 `Qwen/Qwen3-4B` tokenizer/chat template 和 `enable_thinking=false`，而不是 target checkpoint 的 chat template；run provenance 必须记录这个 formatter identity、完整 checkpoint path，并在声称 token-level prompt 完全相同时另外固定及验证 tokenizer revision/chat-template hash。
- `HumanEvalPlus`/`MBPPPlus` 的 active 分数必须对 raw completions 运行 pinned G-OPD official EvalPlus `sanitize + base + plus` post-scoring；Plus correctness 要求同一 completion 同时通过 base 与 plus tests。只执行 parquet 中 compact `assert_case` 的通用 Docker Code scorer 只能标记为 `compact base-assert proxy`，不得进入 active Code/Overall。official post-scoring 可复用已生成的 K=8 completions，无需重新占用 GPU inference；必须记录 G-OPD/EvalPlus commit、官方 source hash、`min_time_limit`、`gt_time_limit_factor`、sanitized samples/result hash、task/sample count 与每题 K=8，并使用全新 fail-closed output directory，禁止静默复用旧 `eval_results.json`。
- 标准评测固定采用 checkpoint/global step 60（`global_step_60`）。使用其他 step 的结果必须明确标记为非标准或补充实验，不得与标准结果混报。
- `/Users/linghuazhang/Desktop/Project/OPD/experiments_records/eval/` 是 active 结果区，保留 `global_step_60` 的候选模型评测，以及 canonical Student Base/Teacher sampled reference anchors；其他非 Step-60 训练 checkpoint、greedy 历史结果和旧 comparison 统一移动到同级 `/Users/linghuazhang/Desktop/Project/OPD/experiments_records/eval_candidates/`。仅满足 Step-60、或仅作为 Base/Teacher reference anchor、但尚未完成 10-dataset × K=8 的旧结果必须标记为 `pending standard rerun` 或 `reference anchor`，不得进入正式标准横向比较。
- 所有标准评测原始结果、日志、manifest 和派生统计统一保存到 `/Users/linghuazhang/Desktop/Project/OPD/experiments_records/eval/` 下，并为每次运行使用独立、可追溯的子目录。
- 指标统计统一维护在 `/Users/linghuazhang/Desktop/Project/OPD/experiments_records/eval/Summary.md` 中；新结果按照该文件当前的章节、表格和 `Avg@K / Pass@K` 格式追加或扩展，同时记录 model/checkpoint、10 个 dataset、`K=8`、运行配置和数据来源。
- 每次完成评测总结时，必须在 `Summary.md` 对应模型的“模型或 checkpoint”或“数据来源”位置记录该次评测所用 checkpoint 的完整远端路径，并链接到该次评测子目录中的 `RUN_MANIFEST.md`；不得只写 `step 60` 或模型简称而缺少可定位地址。
- 远端 checkpoint provenance 的详细 source of truth 保存为 `<eval-run-dir>/RUN_MANIFEST.md`，至少记录 remote host/cluster 标识、完整 remote checkpoint path、checkpoint/global step、模型与训练配置标识、Slurm job ID、评测时间以及本地归档目录。`Summary.md` 保留简要路径和 manifest 链接，`RUN_MANIFEST.md` 保留完整细节；两处信息必须一致，且不得写入密码、token 或其他 secrets。
- 历史评测及其原始口径继续保留；只有同时满足 10-dataset、每 dataset 8 rollouts、step 60 的结果，才可进入后续标准横向比较。每次运行前应核验 dataset 数量为 10，运行后应确认所有数据集完整且成功，再更新 `Summary.md`。

## Four-Baseline Global Control/Structure Taxonomy

以下定义是项目后续 token composition、Rising/Stable Top-200 和 recall 分析的全局
source of truth。旧的 Control-44、PDTB、VR Rising Top-200、candidate seed label 或
某次 selector 的 selected tokens 均不能单独定义 Control/Structure 类型。

### Global taxonomy

- 固定四条 baseline：`1.7B-OPD`、`1.7B-EOPD`、`4B-OPD`、`4B-EOPD`；固定 domain：
  `math`、`code`、`science`。
- Base Control 使用现有 V2 `connective_structure ∪ frozen Control-44`；Base Structure
  使用全局 `format_structure ∪ code_lexical_structure` 并排除 Base Control。base 类型
  在三个 domain 中不变。
- 对 token `t` 和 baseline `b` 定义
  `M_b(t) = max_{domain,step} occurrence(b,domain,step,t)`。threshold 必须严格使用
  `>20`，不能写成 `>=20`。
- `GlobalControl = {t ∈ BaseControl | 对每条 baseline b，M_b(t)>20}`。
- `GlobalStructure = {t ∈ BaseStructure | 对每条 baseline b，M_b(t)>20}`。
- 冻结计数为 Control `175`、Structure `634`，合计 `809` 个 global unique token IDs；
  `Other = tokenizer vocabulary \ (GlobalControl ∪ GlobalStructure)`。三类互斥且穷尽
  vocabulary。

### Domain subsets

- Domain 集合只能从上述 809-token global taxonomy 获取，禁止回到 Base V2 taxonomy
  重新取并集或按 domain 改类型。
- 对 domain `d` 定义 `M_{b,d}(t)=max_step occurrence(b,d,step,t)`；
  `DomainType(d,T) = {t ∈ GlobalType(T) | 对每条 baseline b，M_{b,d}(t)>20}`。
- 冻结数量：Math Control/Structure=`124/266`（390），Code=`157/551`（708），
  Science=`128/244`（372）。同一 global token 可以属于多个 domain subset，但类型不可变。

### Candidate-pool intersection

- Candidate pool 是 taxonomy 子集，不是 taxonomy 来源。对任何 raw pool `P_d`，selector
  的有效集合必须为 `P_d ∩ (DomainControl_d ∪ DomainStructure_d)`，类型由 global
  taxonomy 决定；历史 `candidate_type`/seed provenance 只能用于审计。
- 冻结 raw pool：ExpandedPruned-V2 Math/Code/Science=`116/146/105`；Robust190=
  `69/74/47`（190 个 domain-token entries）；Control-44 每个 domain 使用同一 44 IDs。
- 与 domain taxonomy 相交后的有效 Control/Structure 数量分别为：
  ExpandedPruned-V2 Math=`66/23`、Code=`68/47`、Science=`59/19`；Robust190
  Math=`46/20`、Code=`39/30`、Science=`30/14`；Control-44 Math=`40/0`、
  Code=`40/0`、Science=`38/0`。被排除项必须保留在 audit 表中，不能静默重分类。

机器可读 source of truth：
`analysis-output/four-baseline-global-token-taxonomy/tables/global-taxonomy.csv`、
`domain-subsets.csv`、`candidate-pool-membership.csv`、`definition.json`。重建入口为
`analysis-output/four-baseline-global-token-taxonomy/build_taxonomy.py`。

### Rising/Stable endpoint Top-200 contract

- phase boundary 固定为：1.7B-OPD Rising `1→35`、Stable `36→52`；1.7B-EOPD
  Rising `1→37`、Stable `37→65`；4B-OPD Rising `6→20`、Stable `21→45`；
  4B-EOPD Rising `1→20`、Stable `21→55`。不得根据本次结果重新挑 boundary。
- 对 token `j` 计算
  `speed_j=(gap_start(j)-gap_end(j))/(end_step-start_step)`；这里只使用 phase start/end
  的净优化量。start、end occurrence 必须都严格 `>20`，gap 必须有限。
- 在完整 eligible tokenizer vocabulary 中按 speed 降序排名，tie 按 token ID 升序；
  Rising/Stable 独立选 Top-200。eligible 少于 200 时必须报告实际 Top-K。
- Top-200 membership 再按当前 domain 的 `DomainControl/DomainStructure` 分类，其余为
  Other。所有图表只能称为“Top-200 内的 taxonomy composition”，不得外推为完整
  Control/Structure taxonomy 自身的数量变化。

### Candidate recall replay contract

- 固定搜索 A/C/E/F、`window=1..20`、`K=1..30`。selection boundary 从该 window
  最早的 supported step 锚定，之后每隔恰好 `window` 个 optimizer steps 选择一次；
  score 只能使用 boundary 及历史数据。
- source candidate 必须属于有效 candidate-pool/domain-taxonomy 交集，并在 score 所需
  每个 source snapshot occurrence 严格 `>20`。A/C 是 source window 内 occurrence-
  weighted mean gap/entropy；E 是现有 aggregate vectors 能恢复的 normalized
  `A+C+A*C` proxy；F=`(gap[t-window]-gap[t])/window`。
- `next_step` target 使用 `t→t+1` endpoint speed；`next_window` 使用 `t→t+window`
  endpoint speed。二者都从完整 eligible vocabulary 取 global Top-200；next-window
  additionally 要求 token 在 `t..t+window` 的每个 snapshot occurrence 严格 `>20`。
- 令 `DynamicPool` 为当次通过 source gate 的 candidate set。主分母为
  `Actual=|DynamicPool∩FutureTop200|`，并定义 `Precision=Hits/K`、
  `PoolRecall=Hits/Actual`、`PoolF1=2PR/(P+R)`；`Actual=0` 时 PoolRecall/PoolF1
  unavailable。另报 `PoolCapacity=Actual/TypeActual`、`TypeRecall=Hits/TypeActual`
  和 TypeF1，防止小池凭 pool-internal recall 被误判为完整 taxonomy coverage 更强。
- split 模式分别从 Control、Structure 各选 K；unified 模式从两类并集共选 K。
  比较模式优劣时必须另做等总预算：split `K/类型` 对 unified `2K/并集`。Control-44
  没有 Structure candidate，split complete evaluation 必须标为 unavailable。
- 保留 `baseline × domain × token type` cell；cell 内用 ratio-of-sums，跨 cell 等权
  macro。主最优点必须覆盖所有 expected cells 且 full-K rate=`100%`。window 导致的
  event 数差异必须报告，稀疏长-window grid maximum 不能直接当稳健部署结论。

当前重放 source of truth：
`analysis-output/four-baseline-global-token-taxonomy/phase-top200-recall/`。
