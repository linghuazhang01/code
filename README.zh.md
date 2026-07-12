# Multi-Teacher OPD Math + Code Training

本目录是当前 OPD/MOPD 训练入口。训练 runtime 从本仓库的 `third_party/verl` 导入，不再依赖远端额外的独立 `G-OPD` checkout。

## 路径约定

- 代码目录：`OPD-code/`
- 数据目录：`OPD-code/data/G-OPD-Training-Data/`
- vendored verl：`OPD-code/third_party/verl/`
- 模型目录：`OPD-code/../models/`
- 日志目录：`OPD-code/logs/`
- checkpoint 目录：`OPD-code/checkpoints/`
- audit 目录：`OPD-code/audit/`

## 新机器安装

本地新机器或新 checkout 使用下面这些脚本：

| 用途 | 脚本 |
| --- | --- |
| 根据 `environment.yml` 创建或刷新 conda/Python 训练环境 | `scripts/setup_training_env.sh` |
| 只下载或校验训练数据 | `scripts/download_mopd_data.sh` |
| 只下载或校验模型 | `scripts/download_mopd_models.sh`、`scripts/download_qwen30b_teacher.sh` |
| 下载并校验当前训练所需的数据 + 模型 | `scripts/download_training_assets.sh` |
| 启动本地训练 | `scripts/run_local_mopd_training.sh` |
| 只渲染/检查 config，不启动训练 | `scripts/run_mopd.sh --dry-run` |

在本地 checkout 中先安装训练环境：

```bash
cd /path/to/OPD-code
bash scripts/setup_training_env.sh
source logs/activate_training_env.sh
```

然后下载并校验 Qwen30B 四领域训练所需数据与模型：

```bash
MODEL_ROOT=$(pwd)/../models \
MIN_FREE_GB=300 \
  scripts/download_training_assets.sh
```

对当前 Qwen30B 四领域 config，推荐入口是
`scripts/download_training_assets.sh`。它会准备 `math`、`code`、`if`、
`science` 四个训练 parquet，下载 Qwen3-4B student，并下载四个 domain
共用的 Qwen3-30B-A3B teacher。

如果只想下载数据，使用同一个入口关闭模型下载：

```bash
DOWNLOAD_MODELS=0 \
REQUIRE_MODELS=0 \
  scripts/download_training_assets.sh
```

如果只想下载模型，使用同一个入口关闭数据下载：

```bash
DOWNLOAD_DATA=0 \
MODEL_ROOT=$(pwd)/../models \
MIN_FREE_GB=300 \
  scripts/download_training_assets.sh
```

也可以直接调用底层单项脚本：

```bash
# 四领域训练 parquet 数据。
scripts/download_mopd_data.sh

# 当前 asset bundle 使用的 Qwen3-4B student/base helper。
MODEL_ROOT=$(pwd)/../models \
DOWNLOAD_STUDENT=0 \
DOWNLOAD_BASE_4B=1 \
  scripts/download_mopd_models.sh

# Qwen3-30B-A3B teacher。
MODEL_ROOT=$(pwd)/../models \
MIN_FREE_GB=300 \
  scripts/download_qwen30b_teacher.sh
```

IF/science validation parquet 只有在 config 启用对应验证路径时才必须存在；
如需一并准备并强制校验：

```bash
IF_VAL_SOURCE=/path/to/raw_if_val.parquet \
SCIENCE_VAL_SOURCE=/path/to/raw_science_val.parquet \
REQUIRE_M2RL_EVAL_DATA=1 \
  scripts/download_training_assets.sh
```

也可以用一条命令完成环境和 assets，并让下载脚本在新 conda env 中运行：
对 `scripts/setup_training_env.sh` 设置 `DOWNLOAD_ASSETS=1`，模型/数据相关
环境变量会继续透传。

## 配置文件

现在保留三个正式 MOPD 版本，每个版本提供 2/4/6/8 卡配置；另外保留指标 smoke profile：

| 配置 | 用途 |
| --- | --- |
| `configs/mopd_formal_audit_all_2gpu.yaml` | 2 卡正式 4B math/code 训练，启用全部 MOPD audit。 |
| `configs/mopd_formal_audit_all_4gpu.yaml` | 4 卡 all-audit 训练，同 objective，放大全局 batch。 |
| `configs/mopd_formal_audit_all_6gpu.yaml` | 6 卡 all-audit 训练，TP=2，三个 rollout data-parallel group。 |
| `configs/mopd_formal_audit_all_8gpu.yaml` | 8 卡 all-audit 训练，TP=4，并保留两个 rollout data-parallel group。 |
| `configs/mopd_formal_audit_loss_only_2gpu.yaml` | 2 卡 all-audit 训练，但 token-gradient selection 只用 loss magnitude。 |
| `configs/mopd_formal_audit_loss_only_4gpu.yaml` | 4 卡 loss-only token-gradient audit 训练。 |
| `configs/mopd_formal_audit_loss_only_6gpu.yaml` | 6 卡 loss-only token-gradient audit 训练，使用 fsdp=2 sequence replay。 |
| `configs/mopd_formal_audit_loss_only_8gpu.yaml` | 8 卡 loss-only token-gradient audit 训练。 |
| `configs/mopd_formal_audit_off_2gpu.yaml` | 2 卡同模型/数据/objective，关闭全部 audit。 |
| `configs/mopd_formal_audit_off_4gpu.yaml` | 4 卡 audit-off 训练。 |
| `configs/mopd_formal_audit_off_6gpu.yaml` | 6 卡 audit-off 训练，使用 TP=2 的显存安全配置。 |
| `configs/mopd_formal_audit_off_8gpu.yaml` | 8 卡 audit-off 训练。 |
| `configs/mopd_formal_audit_all_smoke.yaml` | 2 卡 one-step 指标 smoke，打开全部 audit 输出和 full-vocab vectors。 |
| `configs/mopd_formal_audit_loss_only_smoke.yaml` | 2 卡 one-step smoke，token-gradient selection 只用 loss magnitude。 |
| `configs/mopd_formal_audit_grad_consistency_2gpu_smoke.yaml` | 2 卡 Qwen3-0.6B smoke，用于验证 full/sample/token gradient 闭合。 |
| `configs/mopd_formal_audit_grad_consistency_2gpu_b16_1step_smoke.yaml` | 2 卡 batch size 16、1 step 的 gradient consistency smoke。 |
| `configs/mopd_formal_audit_grad_consistency_2gpu_b32_2step_smoke.yaml` | 2 卡 batch size 32、2 step 的 gradient consistency smoke。 |
| `configs/mopd_formal_audit_grad_consistency_2gpu_b64_3step_smoke.yaml` | 2 卡 batch size 64、3 step 的 gradient consistency smoke。 |

正式 2/4/6/8 卡配置共同使用：

- student: `../models/Qwen3-4B`
- math teacher: `../models/Qwen3-4B-Non-Thinking-RL-Math-Step500`
- code teacher: `../models/Qwen3-4B-Non-Thinking-RL-Code-Step300`
- train files: `DeepMath-103K/train_filtered_level6.parquet` 与 `Eurus/code_train.parquet`
- teacher top-k local-support distillation，`topk_distill_k=32`
- teacher model 默认通过 `model.teacher_model_device: cpu` 放在 CPU；只有在 GPU 显存足够时才建议在 YAML 里改成 `gpu`。

卡数 scaling：

| GPU 数 | 配置后缀 | train/mini batch | rollout TP | Ray CPUs |
| --- | --- | --- | --- | --- |
| 2 | `_2gpu` | 256 | 2 | 8 |
| 4 | `_4gpu` | 512 | 4 | 16 |
| 6 | `_6gpu` | 768 | 2 | 24 |
| 8 | `_8gpu` | 1024 | 4 | 32 |

`mopd_formal_audit_all_*gpu.yaml` 额外打开：

- sample-level 与 validation audit rows
- full-gradient audit
- sample-gradient norm 与 sample-to-domain cosine
- token gap occurrence vector 与 full-vocab vector
- teacher/student entropy occurrence vector 与 full-vocab vector
- token conflict attribution
- token-gradient audit，支持 domain-level signed-gap、gap-abs 与 loss top-k/top-p selection

`mopd_formal_audit_loss_only_*gpu.yaml` 使用相同的 loss-only token-gradient selection：`token_gradient_gap_selection_enabled=false`、`token_gradient_gap_abs_selection_enabled=false`、`token_gradient_loss_abs_selection_enabled=true`。2/4/8 卡 profile 保持完整 all-audit surface，包括 sample-gradient 指标。6 卡 loss-only profile 是显存安全的 fsdp=2 profile：它通过 sequence replay 保留 full-gradient 和 token-gradient audit，将 `data.max_response_length` 限制为 `10240`，显式设置 `rollout.max_model_len=12288`，使用 `token_gradient_top_p=0.15`，但关闭 sample-gradient 指标，因为每个 worker 只拥有 sharded parameter view。

对于 fsdp=2 token-gradient run，必须保持 `sequence_masked_target_enabled=true` 与 `sequence_masked_target_use_as_primary=true`。`token_gradient_top_p=1.0` 可作为 full-token closure check：`topp100_*` token-gradient selection 应覆盖全部候选 token，并且与对应 domain gradient 的 cosine/projection/norm-ratio 接近 1。

`mopd_formal_audit_off_*gpu.yaml` 设置 `audit.enabled=false`，并显式关闭所有 audit 子开关。

all-audit 和 loss-only smoke profile 作为指标测试 profile 保留并纳入测试。它们使用 `data.train_batch_size=32`、`actor.ppo_mini_batch_size=32`、`trainer.total_training_steps=1`，但保持正式 `data.max_response_length=16384`，并保留 full-vocab token gap 与 entropy vectors。

gradient consistency smoke profile 使用 `../models/Qwen3-0.6B` 降低闭合验证成本。它们启用 sequence-masked domain target、sample/token backward recompute，并通过 token `top_p=1.0` closure 检查把 full-token gradient 与 domain gradient 对齐比较。

### Qwen30B 四领域与 Eval

Qwen30B split-teacher profiles 使用 Qwen3-4B student 和 Qwen3-30B-A3B math/code/IF/science teachers。训练数据路径为：

- math: `data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet`
- code: `data/G-OPD-Training-Data/Eurus/code_train.parquet`
- IF: `data/G-OPD-Training-Data/IF/train.parquet`
- science: `data/G-OPD-Training-Data/Science/train.parquet`

IF/science 使用以下 validation parquet 路径：

| Domain | Validation parquet | Validation reward |
| --- | --- | --- |
| `if` | `data/eval_data/ifbench/IFBench_test.parquet` | `mopd_verl/mixed_reward.py` -> IFBench/verifiable-instructions strict reward |
| `science` | `data/eval_data/science/gpqa.parquet` | `mopd_verl/mixed_reward.py` -> GPQA option-letter reward |

准备这些 eval 文件：

```bash
IF_VAL_SOURCE=/path/to/raw_if_val.parquet \
SCIENCE_VAL_SOURCE=/path/to/raw_science_val.parquet \
  scripts/prepare_m2rl_eval_data.sh
```

或者从 Nemotron RL JSONL 中过滤：

```bash
NEMOTRON_RL_SOURCE=/path/to/instruction_following.jsonl \
M2RL_EVAL_MAX_SAMPLES=512 \
  scripts/prepare_m2rl_eval_data.sh
```

Qwen30B configs 中 IF/science validation 路径默认保持注释，因为 verl 即使在 `trainer.test_freq=-1` 时也会启动时构造 validation dataset；如果文件不存在，训练会直接失败。生成 parquet 后，再取消注释 `eval/domains/...` 两行，并按需要打开 `trainer.test_freq` / `trainer.val_before_train`。

配置默认使用 `logger: '["console","tensorboard","wandb"]'`、`runtime.wandb_entity: lz101-rice-university` 和 `runtime.env_file: .env.local`。在启动训练的机器上把 `WANDB_API_KEY` 放入 `.env.local`；该文件已 gitignore，不能提交。无需 W&B 的本地 dry-run 可以覆盖 `runtime.wandb_mode=disabled`。

运行 `scripts/setup_training_env.sh` 创建或更新本地 Conda 环境。
`environment.yml` 是唯一依赖定义，setup script 不再维护第二套 pip requirements
或依赖安装脚本。IF/science validation parquet 使用
`scripts/prepare_m2rl_eval_data.sh` 单独准备。

## 启动

在本地 checkout 中启动：

```bash
cd /path/to/OPD-code
GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_2gpu.yaml \
  --run-id mopd_audit_all_2gpu_$(date +%Y%m%d_%H%M%S)
```

`scripts/run_local_mopd_training.sh` 默认使用 `CONDA_ROOT=$HOME/miniconda3`
和 `ENV_NAME=mopd-verl`，并在 launch 日志里打印实际 Python 路径。

启动 audit-off 版本：

```bash
GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_off_2gpu.yaml \
  --run-id mopd_audit_off_2gpu_$(date +%Y%m%d_%H%M%S)
```

启动 loss-only token-gradient audit 版本：

```bash
GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_loss_only_2gpu.yaml \
  --run-id mopd_audit_loss_only_2gpu_$(date +%Y%m%d_%H%M%S)
```

4/8 卡使用对应 GPU 列表与 YAML：

```bash
GPU_IDS=0,1,2,3 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_4gpu.yaml \
  --run-id mopd_audit_all_4gpu_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1,2,3,4,5 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_6gpu.yaml \
  --run-id mopd_audit_all_6gpu_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_8gpu.yaml \
  --run-id mopd_audit_all_8gpu_$(date +%Y%m%d_%H%M%S)
```

本地 dry-run：

```bash
scripts/run_mopd.sh configs/mopd_formal_audit_all_2gpu.yaml --dry-run
```

下载完整 Qwen30B 四领域训练 assets：

```bash
MODEL_ROOT=$(pwd)/../models \
MIN_FREE_GB=300 \
  scripts/download_training_assets.sh
```

指标 smoke：

```bash
GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_smoke.yaml \
  --run-id mopd_metrics_smoke_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_loss_only_smoke.yaml \
  --run-id mopd_metrics_loss_only_smoke_$(date +%Y%m%d_%H%M%S)
```

## Audit 文件

使用 `scripts/run_local_mopd_training.sh --run-id RUN_ID` 启动 `mopd_formal_audit_all_*gpu.yaml` 或 `mopd_formal_audit_loss_only_*gpu.yaml` 时，JSONL audit 文件会写入配置目录下的 run 子目录，例如 `audit/formal_audit_all_2gpu/RUN_ID/` 或 `audit/formal_audit_loss_only_2gpu/RUN_ID/`。如果显式传入 `mopd_audit.output_dir` override，则使用手动指定的目录。

重点文件包括：

- `domain_step_metrics.jsonl`
- `loss_variance_domain_step.jsonl`
- `loss_variance_sample.jsonl`
- `token_gap_vectors.jsonl`
- `token_gap_vocab_vectors.jsonl`
- `entropy_distribution_vectors.jsonl`
- `entropy_vocab_vectors.jsonl`
- `token_conflict_attribution.jsonl`
- `token_grad_metrics.jsonl`
- `sample_grad_metrics.jsonl`
- `validation_probe.jsonl`
- `validation_gain_variance.jsonl`
- `training_cost.jsonl`
- `audit_errors.jsonl`

full-vocab vector 文件使用 token-id 坐标：第 `v` 维对应 tokenizer token id `v`。`token_gap_vocab_vectors.jsonl` 保存 signed/absolute log-prob gap 的 sum 和 mean vector；`entropy_vocab_vectors.jsonl` 保存 `student_entropy` 与 `teacher_student_cross_entropy` 的 sum 和 mean vector。

详细 metric 定义见 [metrics_zh.md](metrics_zh.md)。配置字段和常用 override 见 [CONFIG_GUIDE.zh.md](CONFIG_GUIDE.zh.md)。
