# Multi-Teacher OPD Math + Code Training

本目录是当前 OPD/MOPD 训练入口。训练代码从本仓库启动，`verl` runtime 已放在 `third_party/verl`，不再要求远端额外存在一个独立的 `G-OPD` checkout。

## 当前路径约定

仓库本身可以迁移；训练配置尽量使用相对路径：

- 代码目录：`OPD-code/`
- 数据目录：`OPD-code/data/G-OPD-Training-Data/`
- vendored verl：`OPD-code/third_party/verl/`
- 模型目录：`OPD-code/../models/`
- 日志目录：`OPD-code/logs/`
- checkpoint 目录：`OPD-code/checkpoints/`
- audit 目录：`OPD-code/audit/`

默认远端例子仍使用当前机器的路径 `/root/autodl-tmp/opd_mopd/OPD-code`，但这是同步脚本的默认目标，不是训练代码硬编码依赖。迁移到其他服务器时覆盖 `REMOTE_HOST`、`REMOTE_PORT`、`REMOTE_CODE_DIR` 即可。

## 代码功能

- `configs/mopd_formal_single_a800.yaml`：当前 single-A800 正式训练配置，0.6B student、两个 4B teacher、本地 parquet 数据、TensorBoard logger、full-gradient 与 sample-gradient audit。
- `configs/mopd_formal_dual_a800.yaml`：当前 dual-A800 诊断训练配置，16K response、TP=2、full-gradient audit 与 sample gradient norm。
- `configs/mopd_formal_dual_a800_pg_loss.yaml` / `configs/mopd_formal_dual_a800_teacher_topk.yaml` / `configs/mopd_formal_dual_a800_student_topk.yaml`：dual-A800 蒸馏目标对照配置，分别对应 chosen-token PG/OPD、teacher top-k LSM、student top-k LSM。
- `configs/mopd_formal_4gpu_a800.yaml` / `configs/mopd_formal_8gpu_a800.yaml`：从 dual-A800 profile 扩展到 4/8 卡，保持约 128 prompts/GPU。
- audit_all / audit_light 不再维护整份复制配置；使用 base config 加命令行 Hydra overrides 控制开关。
- `configs/mopd_math_code.yaml`：paper-style 两教师配置。
- `configs/mopd_general_reasoner.yaml`：General-Reasoner-Qwen3-14B 作为 reasoning teacher、Qwen3-4B 作为 student 的 WebInstruct MOPD 配置。
- `configs/mopd_audit_smoke.yaml`：one-step smoke test 配置。
- `scripts/run_mopd.sh`：通用本地 launcher，可启动任意 YAML config。
- `mopd_verl/launch.py`：把 YAML 转成 `verl.trainer.main_ppo` 的 Hydra overrides。
- `mopd_verl/settings.py`：typed config dataclasses。

配置文件、蒸馏目标切换、audit 开关和常用 override 的集中说明见
[`CONFIG_GUIDE.zh.md`](CONFIG_GUIDE.zh.md)。
- `mopd_verl/domain_sampling.py`：按 `domain_train_files` 和 domain 权重构造 batch 内固定配额采样器。
- `mopd_verl/verl_audit.py`：训练、validation、full-gradient audit 的 JSONL 与 TensorBoard scalar 逻辑。
- `scripts/sync_and_start_remote_mopd.sh`：本地执行，rsync 本仓库到远端；可选择同步后直接启动训练。
- `scripts/start_remote_mopd_training.sh`：远端执行，只负责检查环境/数据/模型并启动 screen 训练。
- `scripts/start_general_reasoner_mopd_training.sh`：远端执行，专门选择 `configs/mopd_general_reasoner.yaml` 并启动 General-Reasoner/GReasoner 14B teacher MOPD。
- `scripts/setup_remote_training_env.sh`：远端执行，创建 conda 环境并安装 vendored `third_party/verl` 所需依赖。
- `scripts/download_mopd_data.sh`：下载训练 parquet 到 `data/G-OPD-Training-Data`，并把 validation parquet staging 到 `eval/domains/`。
- `scripts/download_mopd_models.sh`：下载或检查 formal config 需要的模型目录。
- `scripts/run_remote_one_step_smoke.sh`：远端 one-step smoke test。
- `eval/scripts/run_paper_eval_suite.sh` / `eval/scripts/prepare_paper_eval_data.sh`：legacy external paper eval。它们仍依赖完整 G-OPD eval 目录，默认正式训练不启用。

## 从零启动训练

### 1. 本地同步代码与数据到远端

如果是从 GitHub clone 这个仓库，先安装 Git LFS 并拉取 LFS 管理的数据和 wheel 文件：

```bash
git clone git@github.com:linghuazhang01/code.git OPD-code
cd OPD-code
git lfs install
git lfs pull
```

如果没有执行 `git lfs pull`，`data/G-OPD-Training-Data/` 下的 parquet 和 `third_party/verl/` 下的大 wheel 只会是很小的 pointer 文件，后续数据检查和训练启动都会失败。

先在本地代码目录确认 `data/G-OPD-Training-Data` 存在；如果没有，先下载：

```bash
cd /Users/linghuazhang/Desktop/Project/OPD/code
bash scripts/download_mopd_data.sh
```

同步到当前远端，不启动训练：

```bash
cd /Users/linghuazhang/Desktop/Project/OPD/code
ASSUME_YES=1 bash scripts/sync_and_start_remote_mopd.sh --sync-only
```

这个脚本会：

- 从本地 `ssh.sh` 读取 SSH 密码，但不会打印密码；
- 使用 `rsync -az --delete --partial` 同步整个代码目录；
- 显示传输进度和统计；
- 同步 `data/G-OPD-Training-Data` 和 `third_party/verl`；
- 排除 `.git`、`ssh.sh`、`.env`、`logs`、`hf_home`、`smoke_data`、`checkpoints`、`audit`、`eval_outputs` 等运行产物。

迁移到其他服务器时示例：

```bash
REMOTE_HOST=root@new.server \
REMOTE_PORT=22 \
REMOTE_CODE_DIR=/workspace/OPD-code \
ASSUME_YES=1 \
bash scripts/sync_and_start_remote_mopd.sh --sync-only
```

### 2. 远端安装训练环境

登录远端后，在同步后的代码目录执行：

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
bash scripts/setup_remote_training_env.sh
```

可覆盖的关键变量：

```bash
CONDA_ROOT=$HOME/miniconda3
ENV_NAME=mopd-verl
INSTALL_VERL_DEPS=1
FORCE_REINSTALL=0
INSTALL_SGLANG=0
USE_MEGATRON=0
```

脚本会创建或复用 `mopd-verl` 环境，安装 `third_party/verl/scripts/install_vllm_sglang_mcore.sh` 需要的依赖，并生成 smoke 数据。环境信息写到 `logs/env.sh`。

在 Jupyter/Notebook 中，使用独立的非交互安装脚本：

```python
!bash scripts/setup_notebook_training_env.sh
```

该脚本会自动查找或安装 Miniconda，通过
`conda-forge --override-channels` 创建 `mopd-verl`，避免 Anaconda ToS
交互确认，然后调用常规训练环境安装流程并注册
`MOPD (mopd-verl)` Jupyter kernel。安装完成后切换到该 kernel，或重启当前
kernel。

### 3. 下载或检查模型

当前 formal config 使用 0.6B student 和两个 teacher checkpoint。下载脚本也会准备 4B base model，方便后续把 student 切到 `Qwen3-4B`：

```text
../models/Qwen3-0.6B
../models/Qwen3-4B
../models/Qwen3-4B-Non-Thinking-RL-Math-Step500
../models/Qwen3-4B-Non-Thinking-RL-Code-Step300
```

下载/检查：

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
bash scripts/download_mopd_models.sh
```

脚本默认会下载 `Qwen/Qwen3-0.6B` student、`Qwen/Qwen3-4B` base model，以及两个 Keven16 teacher checkpoint：

- `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`
- `Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300`

只有不需要 `../models/Qwen3-4B` base model 时，才设置 `DOWNLOAD_BASE_4B=0`。
设置 `DOWNLOAD_TEACHERS=0` 会跳过 math/code teacher 下载；如果两个 teacher 目录已经存在、只想校验不想下载，使用 `DOWNLOAD_TEACHERS=0 REQUIRE_MATH_CODE_TEACHERS=1`。如果要换成别的 hub 源，仍然可以覆盖 `MATH_TEACHER_MODEL_ID` 和 `CODE_TEACHER_MODEL_ID`。

如果要准备 General-Reasoner 14B teacher 的本地 checkpoint：

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
DOWNLOAD_STUDENT=0 \
REQUIRE_STUDENT=0 \
DOWNLOAD_TEACHERS=0 \
REQUIRE_MATH_CODE_TEACHERS=0 \
DOWNLOAD_REASONING_BASE_14B=0 \
DOWNLOAD_REASONING_TEACHER=1 \
bash scripts/download_mopd_models.sh
```

默认会准备：

- `../models/Qwen3-4B`，hub id `Qwen/Qwen3-4B`
- `../models/General-Reasoner-Qwen3-14B`，hub id `TIGER-Lab/General-Reasoner-Qwen3-14B`

默认不会额外下载 `../models/Qwen3-14B`。只有 teacher 是 adapter、确实需要单独的 14B base checkpoint 时，才显式设置 `DOWNLOAD_REASONING_BASE_14B=1`。

使用本地 checkpoint 时，可在 `configs/mopd_general_reasoner.yaml` 中把 `model.reasoning_teacher_path` 改成 `../models/General-Reasoner-Qwen3-14B`。`reasoning` teacher 不需要 `secondary_teacher_path`；该 slot 主要给 `code` teacher 使用。

如果使用 ModelScope：

```bash
MODEL_BACKEND=modelscope bash scripts/download_mopd_models.sh
```

### 4. 远端启动正式训练

在远端直接启动：

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
bash scripts/start_remote_mopd_training.sh configs/mopd_formal_single_a800.yaml --run-id mopd_manual_$(date +%Y%m%d_%H%M%S)
```

脚本会在启动前检查：

- `third_party/verl/verl/trainer/main_ppo.py` 是否存在；
- conda 环境能否 import `yaml`、`verl`、`verl.trainer.main_ppo`；
- config 中的 train/validation parquet 是否存在；
- config 中的本地模型路径是否存在；
- `screen` 是否可用；
- GPU 是否空闲；
- stale Ray 是否已停止。

日志路径会打印出来，也会写到：

```text
logs/opd_target_run_id
logs/opd_target_log
logs/opd_target_config
logs/opd_target_gpu_csv
```

查看日志：

```bash
tail -f "$(cat logs/opd_target_log)"
```

只想从本地同步并启动，也可以：

```bash
cd /Users/linghuazhang/Desktop/Project/OPD/code
ASSUME_YES=1 bash scripts/sync_and_start_remote_mopd.sh configs/mopd_formal_single_a800.yaml --run-id mopd_manual_$(date +%Y%m%d_%H%M%S)
```

## One-Step Smoke Test

环境装完后可以先跑一个最小 smoke：

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
bash scripts/run_remote_one_step_smoke.sh
```

它使用 `smoke_data/train.parquet` 与 `smoke_data/val.parquet`，并把 student/ref/teacher 都临时覆盖为 `Qwen/Qwen3-0.6B`。这个 smoke 只验证训练链路能完成一次 optimizer step，不代表模型质量。

## General-Reasoner 14B Teacher 训练

先准备 WebInstruct-verified 的 train/test parquet：

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
bash scripts/prepare_general_reasoner_data.sh
```

先 dry-run 检查生成的 verl 命令：

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
bash scripts/run_general_reasoner_mopd.sh --dry-run -- \
  trainer.total_training_steps=1
```

启动 General-Reasoner/GReasoner 14B teacher MOPD：

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/start_general_reasoner_mopd_training.sh
```

等价的显式命令是：

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/start_remote_mopd_training.sh configs/mopd_general_reasoner.yaml \
  --run-id greasoner_14b_mopd_$(date +%Y%m%d_%H%M%S)
```

`configs/mopd_general_reasoner.yaml` 默认 `trainer.n_gpus_per_node=8`，launcher 会在启动 Ray/verl 前检查 `GPU_IDS` 是否暴露了 8 张 GPU。单卡/少卡运行需要同步调整 `trainer.n_gpus_per_node`、tensor parallel 和 batch size；只把命令改成 `GPU_IDS=0` 不够。

该配置使用：

- student: `../models/Qwen3-4B`
- reasoning teacher: `../models/General-Reasoner-Qwen3-14B`
- train parquet: `data/GeneralReasoner/WebInstructVerified/train.parquet`
- validation parquet: `eval/domains/greasoner/data/WebInstructVerified/test.parquet`
- teacher base slot: `null`
- domain label: `extra_info.opd_teacher=reasoning`
- thinking mode: `data.enable_thinking=true`

训练时 `reasoning` 会路由到 primary ref teacher slot；`code` 仍然路由到 secondary/base-ref slot。因此 math/search/tool/reasoning 这类非-code teacher 可以复用 primary teacher 逻辑，code teacher 保持原来的 secondary teacher 逻辑。

## Formal Single-A800 配置

`configs/mopd_formal_single_a800.yaml` 当前关键设置：

- `data.domain_train_files.math`: `data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet`
- `data.domain_train_files.code`: `data/G-OPD-Training-Data/Eurus/code_train.parquet`
- validation: `eval/domains/math/data/AIME24`、`eval/domains/math/data/AIME25`、`eval/domains/math/data/HMMT25Feb`、`eval/domains/math/data/HMMT25Nov`、`eval/domains/code/data/HumanEvalPlus`、`eval/domains/code/data/MBPPPlus`、`eval/domains/code/data/LiveCodeBench`
- domain sampling: `math: 0.5`、`code: 0.5`
- `data.train_batch_size`: `512`
- `data.val_batch_size`: `1024`
- `data.max_prompt_length`: `2048`
- `data.max_response_length`: `16384`
- `actor.ppo_mini_batch_size`: `512`
- `actor.ppo_micro_batch_size_per_gpu`: `1`
- `actor.use_dynamic_bsz`: base profile 默认 `false`。重型 top-k/audit run 建议打开，
  让 actor update 按 token 数动态分 micro-batch，而不是固定按样本数分。
- `rollout.gpu_memory_utilization`: 多卡 base profile 为 `0.8`，single-A800 base 为
  `0.9`。重型 top-k/audit run 可调低，给 actor backward 留显存余量。
- `trainer.logger`: `["console","tensorboard"]`
- `audit.log_sample_level_freq_steps`: `1`
- `audit.log_validation_metrics_freq_steps`: `1`
- `audit.full_gradient_enabled`: `true`
- `audit.full_gradient_freq_steps`: `1`
- `audit.full_gradient_train_max_samples_per_domain`: `null`
- `audit.full_gradient_micro_batch_size_per_gpu`: `1`
- `audit.sample_gradient_enabled`: `true`
- `audit.sample_gradient_freq_steps`: `1`
- `audit.sample_gradient_norm_enabled`: `true`
- `audit.sample_gradient_cos_enabled`: `false`
- `audit.sample_gradient_log_sample_level_freq_steps`: `1`
- `audit.token_gap_enabled`: `true`，`audit.token_gap_freq_steps: 1`，按 domain 记录 `teacher_logp - student_logp` 与 `abs(teacher_logp - student_logp)` 的分布统计，并把 raw domain vector 写到 `token_gap_vectors.jsonl`。
- `audit.entropy_enabled`: `true`，`audit.entropy_freq_steps: 1`，按 domain 记录 teacher entropy、student entropy、teacher-student cross entropy 的 sum 和分布统计，并把 raw vector 写到 `entropy_distribution_vectors.jsonl`。开启 top-k distill 时，teacher-student cross entropy 使用 `actor.topk_distill_support_source` 选出的同一个 local support 和重归一化口径。
- `audit.token_conflict_enabled`: `true`，`audit.token_conflict_freq_steps: 1`，记录 token-level teacher/student disagreement 摘要，并把 top token 明细写到 `token_conflict_attribution.jsonl`。
- `audit.token_gradient_enabled`: 默认 `false`。开启后先按 domain 收集本 step 全局所有 valid response token 的 `gap_abs = abs(teacher_logp - student_logp)` 分布，再在这个全量分布上选择 `top100_gap_abs` 和 top-p mass token 集合做额外 gradient recompute。若 `autograd.grad()` 断图，会回退到安全的 backward diagnostic。开启后 domain target chunks 会临时使用 FP32 存储，以降低 restore `.grad` 的量化误差。小 batch debug 时可以额外打开 `audit.token_gradient_strict_grad_restore=true`，fallback 前直接备份原始 `.grad`，fallback 后恢复这份原始快照。
- `audit.token_gradient_freq_steps`: base profile 默认 `10`。
- `audit.token_gradient_top_p`: `0.10`，token-gradient audit 会同时统计 `top100_gap_abs`，以及覆盖该比例 domain `gap_abs` mass 的最小 token 集合。
- `actor.topk_distill_enabled`: 默认 `false`。开启后使用 local-support matching，在 `actor.topk_distill_k` 个 support token 内重归一化做 KL。`actor.topk_distill_support_source=teacher` 表示 support 来自 teacher top-k；`student` 表示 support 来自 old actor/student top-k，然后 teacher/current-student 都在同一个 support 上 gather logprob。默认使用 reverse-KL；若要做 forward-KL ablation，设置 `actor.topk_distill_kl_direction=forward`；若要使用旧的 tail-bucket 目标，可显式设置 `actor.distill_mode=topk_forward_kl_with_tail` 或 `topk_reverse_kl_with_tail`。
- `rollout.teacher_prefix_sampling_enabled`: 默认 `false`。开启后 trainer
  读取每条样本里的 `rollout.teacher_prefix_dataset_key` 字段，tokenize 后截断到
  `rollout.teacher_prefix_length`，再让 student 从 `prompt + teacher_prefix`
  继续采样 suffix。launcher 会自动开启 `actor.teacher_prefix_enabled`，
  以便正确路由 teacher-prefix mask。默认 `actor.teacher_prefix_loss_region=suffix_only`，
  因此 teacher-prefix token 只作为上下文，不产生 loss；student suffix token
  继续使用配置里的 OPD/top-k objective。若希望 prefix 也用 forward-KL 训练，
  可设置 `actor.teacher_prefix_loss_region=prefix_and_suffix`。
- 对 `max_response_length=16384` 的 top-k 蒸馏，`actor.use_dynamic_bsz=true`
  可以降低长短样本混在一起时的 actor backward 峰值，但不能把单条超长
  response 切成多次 backward。若 debug run 采样到接近 16K 的 response，
  建议先通过命令行覆盖 `data.max_response_length=8192` 或 `4096`，再开启所有
  gradient audit。
- 正式单卡配置使用 `audit.full_gradient_storage_dtype: bfloat16`。顺序 backward tracker 只在 CPU 保存两个 domain target；cosine 的 dot/norm 仍使用 FP32 累加。
- `paper_eval.enabled`: `false`

120 GB CPU RAM 的单 A800 节点不要直接同时运行 batch 512、optimizer
offload 和每步 full-gradient/sample-gradient audit。先用下面的低内存启动
确认正式数据训练链路：

```bash
bash scripts/start_remote_mopd_training.sh configs/mopd_formal_single_a800.yaml \
  --run-id mopd_a800_lowmem_$(date +%Y%m%d_%H%M%S) \
  -- \
  data.train_batch_size=128 \
  data.val_batch_size=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  trainer.val_before_train=false
```

该命令保留 full-gradient 和 sample-gradient audit，只缩小单次 actor update
需要缓存和重复 backward 的样本总数。actor mini-batch 中每个样本都会计算
sample-to-domain cosine 和 projection share。不要通过提高
`RAY_memory_usage_threshold` 掩盖该问题；节点已接近物理内存上限时，这只会
把 Ray worker kill 推迟成系统 OOM。

## Formal Dual-A800 配置

两张 NVIDIA A800 80GB 当前诊断实验使用
`configs/mopd_formal_dual_a800.yaml`。该配置使用
`train_batch_size=ppo_mini_batch_size=256`、`max_response_length=16384`、
replicated actor audit 坐标系（`actor.fsdp_size=1`）、rollout TP=2、
`gpu_memory_utilization=0.8`，并把 `total_training_steps` 设为 10。

普通的 per-domain data、OPD loss、teacher confidence/gap、reward 和 cost
指标保持开启。full-parameter audit 与 sample gradient norm 保持开启；
sample-to-domain cosine 默认关闭，等待 two-pass FSDP 实现完成后再启用。

额外提供三份 dual-A800 蒸馏目标对照配置，保持相同 model/data/rollout
profile，只切换训练 objective：

| 配置 | Objective | Top-k support |
| --- | --- | --- |
| `configs/mopd_formal_dual_a800_pg_loss.yaml` | chosen-token OPD / PG-style loss | 关闭 |
| `configs/mopd_formal_dual_a800_teacher_topk.yaml` | top-k local support matching，reverse-KL，`k=5` | teacher top-k |
| `configs/mopd_formal_dual_a800_student_topk.yaml` | top-k local support matching，reverse-KL，`k=5` | old actor/student top-k |

```bash
cd /root/OPD-code
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_dual_a800.yaml \
  --run-id mopd_dual_a800_$(date +%Y%m%d_%H%M%S)
```

`paper_eval.enabled=false` 是刻意的：portable 从头训练只依赖本仓库、vendored `verl`、parquet 数据和模型目录。外部 paper eval 仍可单独启用，但需要完整 G-OPD eval 目录。

当前指标定义以 [`metrics_zh.md`](metrics_zh.md) 为准。正式配置会记录这些 audit 指标族：

- per-domain data、OPD loss、teacher confidence/gap、calibration、reward、advantage sign、response length；
- token-level conflict attribution summary 和 top-token JSONL 明细；
- full-parameter train gradient：domain grad norm、math-vs-code cosine/conflict、domain-vs-total cosine、signed projection share；
- sample-gradient：当前 actor update mini-batch 内全部样本的 sample grad norm；dual A800 默认不计算 sample-to-domain cosine 和 projection share；
- cost：step seconds、tokens/sec、peak memory、full-gradient backward time。

本轮中文分析报告见
[`reports/2026-06-13--mopd-full-sample-gradient-report.zh.md`](reports/2026-06-13--mopd-full-sample-gradient-report.zh.md)。

## Formal Scaled-A800 配置

多卡 A800 profile 从当前双卡设置线性扩展，保留 16K response、逐 step
full-gradient audit 和 sample gradient norm 统计。sample-to-domain cosine
继续关闭，等待 two-pass FSDP 路径完成后再启用。

| 配置 | GPU 数 | Train/PPO batch | Response | Rollout TP | vLLM util | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `configs/mopd_formal_4gpu_a800.yaml` | 4 | 512 | 16384 | 4 | 0.8 | 一个 TP=4 rollout group，约 128 prompts/GPU。 |
| `configs/mopd_formal_8gpu_a800.yaml` | 8 | 1024 | 16384 | 4 | 0.8 | 两个 TP=4 rollout group，约 128 prompts/GPU。 |

启动示例：

```bash
GPU_IDS=0,1,2,3 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_4gpu_a800.yaml \
  --run-id mopd_4gpu_a800_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_8gpu_a800.yaml \
  --run-id mopd_8gpu_a800_$(date +%Y%m%d_%H%M%S)
```

## Formal A800 Audit Overrides

base 版 1/2/4/8 卡 A800 配置保留当前标准训练/audit 设置：gap、entropy、
token-conflict、full-gradient、sample-gradient norm 开启；sample-gradient
cosine、token-gradient audit 和 top-k distill 关闭。之前的
`*_audit_all.yaml` / `*_audit_light.yaml` 复制配置已删除；需要固定开关组合时，
使用 base config 并在 `--` 后追加 Hydra overrides。

`audit_all` 会开启 `log_sample_level`、`log_validation_metrics`、
`full_gradient_enabled`、sample gradient norm/cos、逐 step token-gradient
audit，并启用 teacher-top-5 蒸馏。若要切到 student-top-k 版本，把
`topk_distill_support_source` 改为 `student`：

```bash
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_dual_a800.yaml \
  --run-id mopd_dual_audit_all \
  -- \
  mopd_audit.log_sample_level=true \
  mopd_audit.log_validation_metrics=true \
  mopd_audit.full_gradient_enabled=true \
  mopd_audit.sample_gradient_enabled=true \
  mopd_audit.sample_gradient_norm_enabled=true \
  mopd_audit.sample_gradient_cos_enabled=true \
  mopd_audit.token_gap_enabled=true \
  mopd_audit.entropy_enabled=true \
  mopd_audit.token_conflict_enabled=true \
  mopd_audit.token_gradient_enabled=true \
  mopd_audit.full_gradient_freq_steps=1 \
  mopd_audit.sample_gradient_freq_steps=1 \
  mopd_audit.sample_gradient_cos_freq_steps=1 \
  mopd_audit.sample_gradient_log_sample_level_freq_steps=1 \
  mopd_audit.token_gap_freq_steps=1 \
  mopd_audit.entropy_freq_steps=1 \
  mopd_audit.token_conflict_freq_steps=1 \
  mopd_audit.token_gradient_freq_steps=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.actor.policy_loss.distill_mode=chosen_token_reverse_kl \
  actor_rollout_ref.actor.policy_loss.topk_distill_enabled=true \
  actor_rollout_ref.actor.policy_loss.topk_distill_kl_direction=reverse \
  actor_rollout_ref.actor.policy_loss.topk_distill_k=5 \
  actor_rollout_ref.actor.policy_loss.topk_distill_support_source=teacher \
  actor_rollout_ref.actor.policy_loss.topk_distill_tail_bucket=false
```

所有重型 audit 都可以用各自的 `*_freq_steps` 独立节流。触发规则是
`step % freq_steps == 0`；把频率调大表示开关仍然开启，但更少 step 执行和写
JSONL。Top-k distill 是训练 objective，不属于 audit 统计，因此由
`actor.topk_distill_enabled` 控制，不单独加 audit freq。

`token_gradient_max_samples_per_domain`、`token_gradient_top_k_per_sample` 和 `token_gradient_min_teacher_diff` 仍会保留在配置里兼容旧启动文件，但 token-gradient audit 当前不再用它们截断候选池；候选池语义是 `global_candidate_scope=all_valid_response_tokens`。

`audit_light` 会关闭你这次指定的额外 audit 输出：

```bash
scripts/run_mopd.sh configs/mopd_formal_single_a800.yaml -- \
  mopd_audit.log_sample_level=false \
  mopd_audit.log_validation_metrics=false \
  mopd_audit.full_gradient_enabled=false \
  mopd_audit.sample_gradient_enabled=false \
  mopd_audit.sample_gradient_norm_enabled=false \
  mopd_audit.sample_gradient_cos_enabled=false \
  mopd_audit.token_gap_enabled=false \
  mopd_audit.entropy_enabled=false \
  mopd_audit.token_conflict_enabled=false \
  mopd_audit.token_gradient_enabled=false \
  actor_rollout_ref.actor.policy_loss.topk_distill_enabled=false
```

在 `audit_light` 下，audit 仍保留最核心的 per-domain row 和 cost row：
data/token count、OPD loss mean/std/variance、advantage sign、response length、
reward/accuracy（如果 batch 里有 `token_level_scores`）、calibration、
duplicate rate、历史核心的 `teacher_student_gap_mean` 和
`teacher_confidence_mean`、global loss/data summary，以及 `training_cost.jsonl`。
如果要完全无 audit，额外设置 `audit.enabled=false`。

## Formal Single-H200 配置

单张 NVIDIA H200 141GB 使用 `configs/mopd_formal_single_h200.yaml`。该配置
保持 math/code 采样、模型路径、序列长度以及
`train_batch_size=ppo_mini_batch_size=1024` 不变，只利用额外 HBM 和带宽调整
执行层：

- optimizer 常驻 GPU：`optimizer_offload=false`；
- 启用 vLLM CUDA graph：`enforce_eager=false`；
- `log_prob_micro_batch_size_per_gpu=2`；
- `max_num_batched_tokens=65536`；
- `max_num_seqs=16`。

首次 full-length 训练仍保留 `ppo_micro_batch_size_per_gpu=1`。由于 response
最长可达 16K tokens，在实测 peak memory 前直接增大 backward micro-batch
并不稳妥。

启动：

```bash
cd /root/OPD-code
GPU_IDS=0 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_single_h200.yaml \
  --run-id mopd_h200_$(date +%Y%m%d_%H%M%S)
```

如果前 5-10 step 的显存峰值稳定低于约 120 GiB，下一步可以尝试
`actor.ppo_micro_batch_size_per_gpu=2`。不要因为 GPU 更大就直接增大全局
train batch，因为那会改变 rollout freshness 和实验语义。

## 数据采样逻辑

训练采样支持多个 domain train files，并按权重给每个训练 batch 分配固定 domain 配额。当前配置：

```yaml
data:
  domain_train_files:
    math:
      - data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet
    code:
      - data/G-OPD-Training-Data/Eurus/code_train.parquet
  domain_sampling_weights:
    math: 0.5
    code: 0.5
  domain_sampling_replacement: true
```

启动时 `mopd_verl/launch.py` 会把 `domain_train_files` 展开成 `data.train_files`，并额外传给 Hydra。patched `RLHFDataset` 会根据文件来源把样本标成 `math` 或 `code`，patched `ray_trainer` 会优先使用 `DomainBatchSampler`。

`DomainBatchSampler` 使用 largest-remainder 分配整数配额。例如 `train_batch_size=1024` 且 `math: 0.5, code: 0.5` 时，每个训练 batch 固定为 `512` 条 math + `512` 条 code；如果是 `0.7/0.3`，则是 `717/307`。`domain_sampling_replacement: true` 表示每个 domain 内可重复抽样，避免某个 domain 数据量较少时提前耗尽。

## TensorBoard 与日志

正式训练的 logger 是 `["console","tensorboard"]`。TensorBoard event 由 verl trainer 写出；日志文件由 `start_remote_mopd_training.sh` 写到 `logs/<run_id>.log`。

full-gradient、sample-gradient、domain loss、advantage、response length、validation probe 等 audit scalar 会通过 `mopd_verl/verl_audit.py` 写到 TensorBoard。典型 tag 包括：

```text
math/length/response_mean
math/advantage/positive_frac
math/sample_grad/norm_mean
math/sample_grad_cos/domain_cos_mean
math/sample_grad_contribution/projection_share_mean
math/token_conflict/teacher_teacher_diff_mean
math/token_conflict/combined_diff_mass
math/token_gap/gap_abs_p95
math/entropy/teacher_entropy_p50
math/token_grad_conflict/other_cos_negative_frac
math/token_grad/top100_gap_abs_gap_abs_mass_frac
math/token_grad/topp10_gap_abs_mass_gap_abs_mass_frac
global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k
global/full_grad_contribution/math_to_total/signed_projection_share
```

JSONL audit 文件写到 config 中的 `audit.output_dir`，formal 默认是：

```text
audit/formal_single_a800/
```

重点文件包括 `domain_step_metrics.jsonl`、`loss_variance_sample.jsonl`、`token_gap_vectors.jsonl`、`entropy_distribution_vectors.jsonl`、`token_conflict_attribution.jsonl`、`token_grad_metrics.jsonl`、`sample_grad_metrics.jsonl`、`validation_probe.jsonl`、`validation_gain_variance.jsonl`、`training_cost.jsonl` 和 `audit_errors.jsonl`。

## Legacy Paper Eval

`eval/scripts/run_paper_eval_suite.sh` 和 `eval/scripts/prepare_paper_eval_data.sh` 仍保留，但它们依赖 G-OPD 的 `math_eval/`、`code_eval/`、LiveCodeBench 等外部 eval 代码。它们不是当前 portable training 的必需步骤。

如果要启用，需要先准备完整 G-OPD eval checkout，并在启动时传入：

```bash
bash scripts/start_remote_mopd_training.sh configs/mopd_formal_single_a800.yaml -- \
  +paper_eval.enabled=true \
  +paper_eval.script_path=eval/scripts/run_paper_eval_suite.sh
```
