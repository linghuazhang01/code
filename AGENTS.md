# MOPD 远端调试规则

## 三 domain 训练 batch size 约束

- Math/Code/Science 三 domain 训练的 global `data.train_batch_size` 应保持约 526–528，默认使用 **528**；`actor.ppo_mini_batch_size` 默认同步设为 **528**。既有 525 配置可为满足 GPU 整除约束保留，但不得因扩大 GPU 数量自动增大 global batch。
- 必须根据实际 actor data-parallel 数核对整除条件；6 actor + 2 teacher 的 8-GPU 布局使用 528，三 domain 等权时每步各 176 条。526 只是用户要求的近似规模，不能在不满足整除要求时机械使用。
- 禁止新建、继承或恢复 **batch size 576 / b576** 训练模板。启动前必须检查继承展开后的 `data.train_batch_size`、`actor.ppo_mini_batch_size` 和最终启动参数，不得只看文件名。
- 其他 batch size 必须取得用户针对该任务的明确指示；本约束不改变 math-only 配置，也不允许改写历史实验记录或删除已有 checkpoint。

## Token.md 定义优先级

- `Token.md` 是本项目中 `V3`、`Taxonomy`、`Control`、`Structure`、`Other`、domain subset、candidate pool 及其相关版本、集合关系、数量和计算口径的唯一规范 source of truth。凡涉及上述术语的代码、配置、实验、评测、分析或文档，执行前必须先阅读同目录下最新的 `Token.md`。
- 若 `AGENTS.md`、代码、配置、实验记录或其他文档与 `Token.md` 不一致，以 `Token.md` 为准；本文件只补充工作流约束，不得自行重新定义或覆盖 `Token.md`。
- 特别注意：`V3` 是 candidate-pool version，不是新的 Control/Structure taxonomy version。V3 的 pool 构造、seed 和 provenance 遵循 `Token.md` 的 V3 定义；但最终的 Control/Structure/Other 分类、taxonomy 交集、Top-200 composition、Type Recall denominator 和审计必须使用 `Token.md` 当前 active 的完整 tokenizer-vocabulary taxonomy，不能改用 legacy V2 taxonomy，也不能把 pool 的 `candidate_type`/seed provenance 当作 taxonomy type。`GlobalControl ∪ GlobalStructure` 的 809 个 token 只是该完整 taxonomy 的非-Other 部分，`GlobalOther` 是完整词表补集。

- 本地仓库是源码唯一 source of truth；所有修改必须先在本地完成。
- **远端训练代码同步要求**：凡启动远端训练，必须先将本地全部代码同步（sync）到远端 `/home/shuang_qiu/mopd_code`，再执行远端命令；不得只上传单个配置或局部代码文件。同步前仍需完成本地修改与检查，并先运行 `rsync --dry-run`；禁止使用 `rsync --delete`。
- 固定流程：本地修改与检查 → `rsync --dry-run` → 上传到 `/home/shuang_qiu/mopd_code` → 远端运行 → 下载日志 → 回本地修复。
- 禁止直接修改远端源码；远端只允许运行、测试、训练和生成运行产物。
- `../ssh.sh` 第 1 行是 SSH 连接命令，第 2 行是密码。不得修改、打印、提交或整体执行该文件；只执行第 1 行，并在密码提示时输入第 2 行。
- 默认禁止 `rsync --delete`，不得覆盖远端独有的 dataset、model、logs 或 checkpoints。
- **远端任务运行方式（当前默认）**：训练、评测、smoke 及其他 GPU 任务统一通过 SSH 在远端 GPU 节点直接以 local process 启动，不提交 Slurm，不依赖 `sbatch`、`srun`、`squeue` 或 `scontrol`。长任务必须使用 `nohup`/`setsid`/`tee` 等方式保留后台进程、PID 和日志；启动命令必须显式设置并记录 `CUDA_VISIBLE_DEVICES`、Python executable、run directory 和完整日志路径。只有用户针对某个具体任务明确要求 Slurm 时，才允许走 Slurm；此时才适用下述 Slurm 资源申请规则。
- **训练启动入口**：远端训练必须从同步后的仓库根目录使用 `start.sh --local` 启动，不得直接调用 `python -m mopd_verl.launch` 或绕过 `start.sh` 的训练入口。启动时必须显式传入 config、`GPU_IDS`、实际 Python executable，并设置正确的 conda 环境根目录/`PATH`，确保 `ninja`、FlashInfer 等运行时依赖可见；长任务可在 `start.sh --local --foreground` 外层使用 `nohup`/`setsid`，同时保留 start.sh 生成的 run ID、日志和 GPU 监控文件。
- **Local 评测启动入口**：远端四 GPU 评测使用同步后的仓库根目录 `start.sh --eval --local --model_path PATH` 启动；可通过 `--run_tag`、`--output_root`、`--gpu_ids`、`--datasets`、`--gopd_dir`、`--score_code` 和 `PYTHON` 指定运行标识、输出目录、物理 GPU IDs、数据集、G-OPD checkout、Code scorer 与 Python executable。默认是 Math-only 四数据集、`K=8`；需要标准 3-domain 评测时必须使用 `--standard_protocol`，固定为 canonical 十数据集、DP=4、四个单卡 `TP=1` worker、`K=8`，并写入 `scheduler: direct-local` 的 `RUN_MANIFEST.md`。`start.sh --eval` 不带 `--local` 仍保留历史 Slurm standard-evaluation 分支，但默认远端任务不得使用该分支。
- **直接运行的评测 provenance**：即使复用历史 `slurm_*` worker 脚本，也只能作为普通 local process 执行，禁止调用 Slurm submit/queue 命令；`RUN_MANIFEST.md` 必须明确记录 `scheduler: direct-local`、实际 GPU IDs、完整 checkpoint path 和日志路径，不得把 synthetic run ID 伪称为 Slurm allocation。
- **训练步数与 Hugging Face checkpoint 约束**：所有训练配置的 `trainer.total_training_steps` 最大只能为 `65`，禁止运行超过 65 个 training steps；Hugging Face checkpoint steps 必须落在实际训练范围内，默认使用 `[55, 60, 65]`，禁止新增或恢复 `global_step_70` 上传。Hugging Face 必须使用 public repository 传输方式，即明确设置 `huggingface_checkpoint.private: false`；不得改回 `true`。启动前必须核对 resolved command 同时满足 `trainer.total_training_steps<=65`、`trainer.huggingface_checkpoint.private=False`，且所有上传 step 不超过总训练步数。
- 若用户明确要求 Slurm，远端 Slurm 任务才必须按 GPU 数量线性分配 host memory：`1 GPU = 100G`，即 `--mem = GPU_COUNT × 100G`（例如 1/2/5 GPU 分别请求 100G/200G/500G）。提交前必须核对 launcher 输出与 `scontrol` 中的 GPU 数和 `ReqMem`；不得擅自多申请或少申请。只有用户对某个具体任务明确指定例外时才允许偏离该公式，并必须在运行记录中注明。当前 taxonomy job 193 使用 `200G` 是用户明确指定的一次性例外，不得据此改变后续任务的默认公式。
- GPU 使用约束：远端 GPU 4、5 属于其他用户，后续训练、评测、smoke 及其他 GPU 任务默认不得使用 GPU 4/5；只有用户针对当前具体任务明确授权时才允许例外。启动任务前必须核对并记录实际 GPU 分配，优先选择项目明确可用的其他 GPU。

- **远端任务启动门槛（磁盘可用空间）**：每次启动远端任务（包括训练、评测、smoke 和重启）前，必须检查任务实际使用的远端文件系统，包括 checkpoint、日志/评测输出及临时目录（如 `/tmp` 或实际 Ray temp 目录）所在挂载点；同一挂载点只检查一次。以 `df` 的 Available 或等效文件系统查询为准，按 `500 GiB` 核验。任一相关文件系统可用空间小于 `500G` 时，不得启动任务，必须提示用户对应路径/挂载点、当前可用空间、500G 门槛及本次未启动的任务；无法读取或确认磁盘可用空间时同样不得启动并说明原因。达到门槛仅允许继续其他资源检查；不得为满足门槛擅自删除文件。此门槛指磁盘可用空间，不是主机 RAM 或 GPU 显存。

## MOPD 标准评测与归档规则

- **`Summary.md` 格式保护**：`/Users/linghuazhang/Desktop/Project/OPD/experiments_records/eval/Summary.md` 必须保持现有的章节结构、表格格式、列数、命名和链接风格。
- 对 `Summary.md` 的更新只能做与已完成评测直接相关的最小范围增量；禁止重写、重排、覆盖全文或引入新的总结模板，不得进行大范围修改。
- 修改前必须先检查并保留可恢复副本，修改后必须核对文本 diff、Markdown 表格结构和文件完整性；临时状态、未完成任务和推测性结论不得写入 `Summary.md`。

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

以下是与 `Token.md` 当前 active definition 对齐的执行摘要，完整定义和版本语义以
`Token.md` 为准。旧的 Control-44、PDTB、VR Rising Top-200、candidate seed label 或
某次 selector 的 selected tokens 均不能单独定义 Control/Structure 类型。

### Global taxonomy

- `Taxonomy` 指完整 tokenizer vocabulary 上的三分类：`GlobalControl`、`GlobalStructure` 和 `GlobalOther`；其中 `GlobalControl ∪ GlobalStructure` 的 809 个 token 是 global C/S inventory，不能误称为完整 taxonomy，`GlobalOther` 补足其余全部 vocabulary。
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

- Domain Control/Structure 集合只能从上述 active full-vocabulary taxonomy 的
  `GlobalControl`/`GlobalStructure` 获取；`809` 是 C/S global inventory，`GlobalOther`
  仍是完整词表补集。禁止回到 legacy Base V2 taxonomy 重新取并集或按 domain 改类型。
- 对 domain `d` 定义 `M_{b,d}(t)=max_step occurrence(b,d,step,t)`；
  `DomainType(d,T) = {t ∈ GlobalType(T) | 对每条 baseline b，M_{b,d}(t)>20}`。
- 冻结数量：Math Control/Structure=`124/266`（390），Code=`157/551`（708），
  Science=`128/244`（372）。同一 global token 可以属于多个 domain subset，但类型不可变。

### Candidate-pool intersection

- Candidate pool 是 taxonomy 子集，不是 taxonomy 来源。对任何 raw pool `P_d`，selector
  的有效集合必须为 `P_d ∩ (DomainControl_d ∪ DomainStructure_d)`，类型由 global
  taxonomy 决定；历史 `candidate_type`/seed provenance 只能用于审计。
- 冻结 raw pool：ExpandedPruned-V2 Math/Code/Science=`116/146/105`；
  ExpandedPruned-V3=`146/140/128`；Robust190=`69/74/47`（190 个 domain-token
  entries）；Control-44 每个 domain 使用同一 44 IDs。V3 是 candidate-pool version；
  Small seed 先对每条 baseline 合并 Rising/Stable Top-200，再分别合并 1.7B/4B 的
  OPD/EOPD，最后取两个 model-size 集合的交集。V3 的最终 taxonomy 类型和有效交集
  必须回到 `Token.md` 当前 active 的完整 vocabulary taxonomy，不能沿用 legacy V2
  taxonomy 或 pool label。
- 与 domain taxonomy 相交后的有效 Control/Structure 数量分别为：
  ExpandedPruned-V2 Math=`66/23`、Code=`68/47`、Science=`59/19`；Robust190
  Math=`46/20`、Code=`39/30`、Science=`30/14`；ExpandedPruned-V3 Math=`85/30`、
  Code=`46/64`、Science=`69/30`；Control-44 Math=`40/0`、Code=`40/0`、
  Science=`38/0`。被排除项必须保留在 audit 表中，不能静默重分类。
- ExpandedPruned-V3 已按下述 protocol 独立 replay；主 taxonomy Type F1 下，Math-only
  部署参数为 next-step unified A/i1/w1（pre-update source）/K25，next-window event-supported unified
  A/i7/w7/K41。该结论来自当前四条 trajectories，仍属于 in-sample exploratory
  evidence；禁止把它表述为 held-out 泛化优势。

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

- 固定搜索 A/C/E/F、`window=1..20`、`K=1..100`。selection boundary 从该 window
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
