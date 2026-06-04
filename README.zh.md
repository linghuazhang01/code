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
- `configs/mopd_math_code.yaml`：paper-style 两教师配置。
- `configs/mopd_audit_smoke.yaml`：one-step smoke test 配置。
- `mopd_verl/launch.py`：把 YAML 转成 `verl.trainer.main_ppo` 的 Hydra overrides。
- `mopd_verl/settings.py`：typed config dataclasses。
- `mopd_verl/domain_sampling.py`：按 `domain_train_files` 和 domain 权重构造 batch 内固定配额采样器。
- `mopd_verl/verl_audit.py`：训练、validation、full-gradient audit 的 JSONL 与 TensorBoard scalar 逻辑。
- `scripts/sync_and_start_remote_mopd.sh`：本地执行，rsync 本仓库到远端；可选择同步后直接启动训练。
- `scripts/start_remote_mopd_training.sh`：远端执行，只负责检查环境/数据/模型并启动 screen 训练。
- `scripts/setup_remote_training_env.sh`：远端执行，创建 conda 环境并安装 vendored `third_party/verl` 所需依赖。
- `scripts/download_mopd_data.sh`：下载训练与 validation parquet 到 `data/G-OPD-Training-Data`。
- `scripts/download_mopd_models.sh`：下载或检查 formal config 需要的模型目录。
- `scripts/run_remote_one_step_smoke.sh`：远端 one-step smoke test。
- `scripts/run_paper_eval_suite.sh` / `scripts/prepare_paper_eval_data.sh`：legacy external paper eval。它们仍依赖完整 G-OPD eval 目录，默认正式训练不启用。

## 从零启动训练

### 1. 本地同步代码与数据到远端

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
只有当两个 teacher 目录已经存在、只想校验不想下载时，才设置 `DOWNLOAD_TEACHERS=0`。如果要换成别的 hub 源，仍然可以覆盖 `MATH_TEACHER_MODEL_ID` 和 `CODE_TEACHER_MODEL_ID`。

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
- config 中的 train/validation/full-gradient validation parquet 是否存在；
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

## Formal Single-A800 配置

`configs/mopd_formal_single_a800.yaml` 当前关键设置：

- `data.domain_train_files.math`: `data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet`
- `data.domain_train_files.code`: `data/G-OPD-Training-Data/Eurus/code_train.parquet`
- validation: `PaperEval/AIME24`、`AIME25`、`HMMT25Feb`、`HMMT25Nov`、`HumanEvalPlus`、`MBPPPlus`、`LiveCodeBench`
- domain sampling: `math: 0.5`、`code: 0.5`
- `data.train_batch_size`: `128`
- `data.val_batch_size`: `1024`
- `data.max_prompt_length`: `2048`
- `data.max_response_length`: `16384`
- `actor.ppo_mini_batch_size`: `128`
- `actor.ppo_micro_batch_size_per_gpu`: `1`
- `rollout.gpu_memory_utilization`: `0.8`
- `trainer.logger`: `["console","tensorboard"]`
- `audit.full_gradient_enabled`: `true`
- `audit.full_gradient_freq_steps`: `1`
- `audit.full_gradient_train_max_samples_per_domain`: `null`
- `audit.full_gradient_micro_batch_size_per_gpu`: `1`
- `audit.sample_gradient_enabled`: `true`
- `audit.sample_gradient_norm_enabled`: `true`
- `audit.sample_gradient_cos_enabled`: `true`
- `audit.sample_gradient_cos_max_samples_per_domain`: `8`
- `paper_eval.enabled`: `false`

`paper_eval.enabled=false` 是刻意的：portable 从头训练只依赖本仓库、vendored `verl`、parquet 数据和模型目录。外部 paper eval 仍可单独启用，但需要完整 G-OPD eval 目录。

当前指标定义以 [`metrics_zh.md`](metrics_zh.md) 为准。正式配置会记录这些 audit 指标族：

- per-domain data、OPD loss、teacher confidence/gap、calibration、reward、advantage sign、response length；
- full-parameter train gradient：domain grad norm、math-vs-code cosine/conflict、domain-vs-total cosine、signed projection share；
- sample-gradient：当前 actor update mini-batch 内每个样本的 sample grad norm 分布，以及每个 domain 最多 8 个被选中样本的 sample-to-domain cosine 和 projection share；
- cost：step seconds、tokens/sec、peak memory、full-gradient backward time。

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

`DomainBatchSampler` 使用 largest-remainder 分配整数配额。例如 `train_batch_size=128` 且 `math: 0.5, code: 0.5` 时，每个训练 batch 固定为 `64` 条 math + `64` 条 code；如果是 `0.7/0.3`，则是 `90/38`。`domain_sampling_replacement: true` 表示每个 domain 内可重复抽样，避免某个 domain 数据量较少时提前耗尽。

## TensorBoard 与日志

正式训练的 logger 是 `["console","tensorboard"]`。TensorBoard event 由 verl trainer 写出；日志文件由 `start_remote_mopd_training.sh` 写到 `logs/<run_id>.log`。

full-gradient、sample-gradient、domain loss、advantage、response length、validation probe 等 audit scalar 会通过 `mopd_verl/verl_audit.py` 写到 TensorBoard。典型 tag 包括：

```text
math/length/response_mean
math/advantage/positive_frac
math/sample_grad/norm_mean
math/sample_grad_cos/domain_cos_mean
math/sample_grad_contribution/projection_share_mean
global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k
global/full_grad_contribution/math_to_total/signed_projection_share
```

JSONL audit 文件写到 config 中的 `audit.output_dir`，formal 默认是：

```text
audit/formal_single_a800/
```

重点文件包括 `domain_step_metrics.jsonl`、`loss_variance_sample.jsonl`、`sample_grad_metrics.jsonl`、`validation_probe.jsonl`、`validation_gain_variance.jsonl`、`training_cost.jsonl` 和 `audit_errors.jsonl`。

## Legacy Paper Eval

`scripts/run_paper_eval_suite.sh` 和 `scripts/prepare_paper_eval_data.sh` 仍保留，但它们依赖 G-OPD 的 `math_eval/`、`code_eval/`、LiveCodeBench 等外部 eval 代码。它们不是当前 portable training 的必需步骤。

如果要启用，需要先准备完整 G-OPD eval checkout，并在启动时传入：

```bash
bash scripts/start_remote_mopd_training.sh configs/mopd_formal_single_a800.yaml -- \
  +paper_eval.enabled=true \
  +paper_eval.script_path=scripts/run_paper_eval_suite.sh
```
