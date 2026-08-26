---
project_id: opd
repo_root: /Users/linghuazhang/Desktop/Project/OPD/code
vault_root: /Users/linghuazhang/Desktop/Project/Notes/Obsidian/Research/opd
hub_note: Research/opd/00-Hub.md
language: zh-CN
last_sync_at: 2026-08-26T23:15:00+08:00
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
- 监控 Math-only OPD 4-GPU baseline Slurm job 142；该任务使用 3 actor/rollout + 1 ref/teacher、global batch 255、200 steps，并在 W&B run `gg3m2ype` 记录训练。
- 已实现按 global step 数组选择性上传 verl checkpoint 到 Hugging Face Hub；命中 step 会强制保存，失败后 resume 会依据本地 receipt 补传，token 推荐在 `start.sh` 外部通过 `export HF_TOKEN` 注入。

## 进行中的实验
- `qwen1p7b-30b-tip-topk32-rho50-4gpu-b525`：远端任务已中断，W&B 已同步至 step 61/200。
- `qwen4b-30b-tip-topk32-rho50-4gpu-b525`：远端任务在首个 training step 前取消，尚无 history metrics。

- `qwen1p7b-...-eopd/global_step_65` 8-benchmark 评测已完成并验证；`global_step_55` 仍因 actor shards 缺失而阻塞。详见 Obsidian `Experiments/EOPD-Checkpoint-Evaluation.md` 与 `Results/Reports/2026-08-25--eopd-checkpoint-evaluation--r1--step65-8benchmark.md`。
- `qwen1p7b-4gpu-control-online-topklentropy-domaincand-v2-i3w3f20k30w4-b528`：训练 job 132 已于 2026-08-26 22:18 按用户指令取消（`CANCELLED by 1002`，elapsed `1-04:13:17`）。`global_step_60` 的 3/3 actor、optimizer、extra-state shards 与 tokenizer metadata 完整；4-GPU、400G 的统一 8-benchmark 评测 job 141 已启动，Math/Code/GPQA/MMLU-Pro-500 四分支均进入 vLLM generation。
- `qwen1p7b_30b_a3b_instruct_2507_8gpu_..._domaincand_v2_.../global_step_{70,75}`：job 134/135/136/137 均成功完成，结果已验收并扁平归档。详见 Obsidian `Experiments/Domain-Candidate-v2-Step70-Step75-Evaluation.md`。
- `qwen1p7b-30b-math-opd-4gpu-b255_20260827_043512`：Slurm job 142 于 2026-08-27 04:35 启动并处于 `RUNNING`；4 张 H200 已分配，config 校验、dataset load、Ray/NCCL/vLLM 初始化与 W&B 登录均通过，当前等待首个 training step 完成。详见 Obsidian `Experiments/Math-Only-OPD-Baseline-Matrix.md`。

## 近期结果
- 2026-08-22：从远端取回上述两条 `.wandb` 日志，并使用本机 LZ101 凭据同步到 `lz101-rice-university/MOPD`。
- 1.7B run 的 W&B summary 为 `training/global_step=61`；4B run 仅含配置与 console log，不能据此做效果比较。
- 2026-08-23：统一 baseline observability contract；35 个物理 YAML 展开为 44 个运行实例，全部启用 per-domain training metrics，并为每个实例配置唯一 audit 输出目录和匹配实际 objective 的 `loss_variance_signal`。
- 2026-08-25：EOPD step65 的 8 项结果完成。Avg@K/Accuracy：AIME24 33.125%、AIME25 26.875%、HMMT25Feb 16.250%、HMMT25Nov 13.125%、HumanEvalPlus 59.756%、MBPPPlus 58.598%、GPQA-Diamond 34.848%、MMLU-Pro-500 57.850%。
- 2026-08-26：Domain Candidate v2 step70/step75 评测完成。Step75 相对 step70：
  Math +1.46 pp、Code -0.92 pp、Science -0.93 pp、overall -0.26 pp
  Avg@K；当前多域整体以 step70 略优。

## 最近同步状态
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
