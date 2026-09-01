---
project_id: opd
repo_root: /Users/linghuazhang/Desktop/Project/OPD/code
vault_root: /Users/linghuazhang/Desktop/Project/Notes/Obsidian/Research/opd
hub_note: Research/opd/00-Hub.md
language: zh-CN
last_sync_at: 2026-09-01T19:43:23+08:00
last_synced_head: 395f5f467fafbf5adf4d031dbce2b39eb3b26e6c
status: active
auto_sync: true
---
# 项目记忆: opd

## 当前问题
- TODO

## 研究假设
- TODO

## 当前任务
- Online Control token selector 已新增 opt-in cumulative score-mass `top_p` budget；默认继续使用 `top_k`。配置、launcher、actor meta、checkpoint schema、统一/分组候选选择和 JSONL audit logging 已贯通，相关回归测试通过。
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
- 2026-09-01T19:43:23+08:00：Online Control token selection 支持 `control_token_online_budget_mode: top_p` 与 `control_token_online_top_p`。selector 按非负 ranking-score mass 选择达到阈值的最短前缀；grouped candidates 按组独立累计。旧 checkpoint 自动迁移为 `top_k`，新配置/state/logging 路径与 172 项相关测试通过。
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
