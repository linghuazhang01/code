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

### Blackwell / CUDA 12.8 环境

RTX PRO 6000 Blackwell 等 `sm_120` GPU 必须使用独立的 PyTorch 2.8 / CUDA
12.8 环境；CUDA 12.4 PyTorch wheel 不包含 `sm_120` kernel。

```bash
cd /path/to/OPD-code
ENV_NAME=mopd-verl-blackwell \
ENV_FILE=$(pwd)/environment.blackwell.yml \
  bash scripts/setup_training_env.sh
source logs/activate_training_env.sh
```

该环境固定使用 `torch==2.8.0+cu128`、`vllm==0.11.0`、
`transformers==4.55.4`、`tensordict==0.10.0`，以及适配 Python 3.10、
PyTorch 2.8、CUDA 12 和 CXX11 ABI TRUE 的官方
`flash-attn==2.8.3.post1` wheel。该组合可在 `sm_120` 上使用默认
`flash_attention_2` backend。第一次 correctness run 可继续关闭
remove-padding：

```bash
ENV_NAME=mopd-verl-blackwell \
GPU_IDS=0,1,2 \
  bash scripts/run_local_mopd_training.sh \
  test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml \
  --run-id blackwell_fsdpsize2_smoke_$(date +%Y%m%d_%H%M%S) -- \
  actor_rollout_ref.model.use_remove_padding=false \
  trainer.save_freq=-1
```

旧实现的 PyTorch 2.8 / CUDA 12.8 远端验证发现：
`world_size=2, fsdp_size=1` 把逻辑 `(ddp=2, fsdp=1)` mesh 直接传给
FSDP1，请求的
`HYBRID_SHARD` 被降级为只在 singleton shard group 上运行的 `NO_SHARD`。
两个 rank 的 gradient 没有同步，随后的无条件 `reshard(True)` 又报
`AssertionError: Expects sharded strategy`。

当前实现已经为该拓扑增加专门语义：verl 仍保留 `(2,1)` logical mesh 做
dispatch 和 replica accounting，但 FSDP1 wrapper 显式使用
`NO_SHARD + WORLD process group`。因此每个 rank 保存完整 model/optimizer，
每次 backward 由 FSDP 底层 all-reduce gradient；所有手工 `reshard` 入口也
会先检查 effective strategy。本机 PyTorch 2.8 / Gloo 两 rank oracle 已验证
rank gradient/parameter 完全一致，global gradient 逐坐标误差为 `2.61e-8`，
micro-batch accumulation 误差为 `2.98e-8`，optimizer update 误差为
`9.31e-9`。对应的
`fsdp_size=2` `FULL_SHARD` 回归同时通过。旧 4B/8B 失败日志仍作为
negative regression evidence 保留；修复后的 PyTorch 2.8 CUDA 实模复跑结果
应以新的实验日志为准。

这里的 `rollout.temperature` 同时用于 actor/ref 的 log-prob scoring，必须是
finite 且严格大于 `0`。若要做 deterministic greedy rollout，应设置
`rollout.do_sample=false` 与 `rollout.temperature=1.0`；不要用
`rollout.temperature=0`，否则 scoring 路径会执行 `logits / 0` 并产生 NaN。

在上述 Blackwell 环境完成的 4-step paired smoke 中，`fsdp_size=2` 的 audit
on/off 在每一步的 71 个非 timing、非 audit 训练指标上逐项完全一致。启用
BF16 domain-gradient vector 后，step 2/4 的 gradient closure relative L2 分别为
`0.004433`/`0.004472`，training parity relative L2 分别为 `4.02e-8`/`0`，
全部通过阈值。这说明 audit replay 只读取统计 gradient，没有改变 production
optimizer update。

然后下载并校验 Qwen3-30B-A3B-Instruct-2507 math/code/IF/science
训练所需数据与模型：

```bash
MODEL_ROOT=$(pwd)/../models \
MIN_FREE_GB=300 \
  scripts/download_training_assets.sh
```

对当前 Qwen30B teacher config，推荐入口是
`scripts/download_training_assets.sh`。它会准备四个 domain 的训练 parquet，
下载 Qwen3-4B student 和所有 domain 共用的
Qwen3-30B-A3B-Instruct-2507 teacher。

如果只想下载数据，使用同一个入口关闭模型下载：

```bash
DOWNLOAD_MODELS=0 \
REQUIRE_MODELS=0 \
  scripts/download_training_assets.sh
```

如果只想下载模型，使用同一个入口关闭数据下载：

```bash
DOWNLOAD_DATA=0 \
REQUIRE_MATH_CODE_TRAIN_DATA=0 \
REQUIRE_4DOMAIN_TRAIN_DATA=0 \
MODEL_ROOT=$(pwd)/../models \
MIN_FREE_GB=300 \
  scripts/download_training_assets.sh
```

也可以直接调用底层单项脚本：

```bash
# MOPD 训练 parquet；当前 profile 覆盖 math、code、IF 与 science。
scripts/download_mopd_data.sh

# 当前 asset bundle 使用的 Qwen3-4B student/base helper。
MODEL_ROOT=$(pwd)/../models \
DOWNLOAD_STUDENT=0 \
DOWNLOAD_BASE_4B=1 \
  scripts/download_mopd_models.sh

# Qwen3-30B-A3B-Instruct-2507 teacher。
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

> **FSDP1 `fsdp_size=1` 语义：**每个 actor rank 保存完整 model、gradient
> 和 optimizer state，FSDP 通过 WORLD process group 同步 gradient。该模式
> 已支持多 rank，但显存和 checkpoint 空间明显高于 `FULL_SHARD`；只有单卡能
> 放下完整 student 时才应使用。`fsdp_size=-1` 或等于 actor world size 时，
> 仍使用跨全部 actor rank 的 `FULL_SHARD`。

| 配置 | 用途 |
| --- | --- |
| `configs/mopd_formal_audit_all_2gpu.yaml` | 2 卡 4B math/code domain-gradient profile；`fsdp_size=1` 使用同步的完整副本。 |
| `configs/mopd_formal_audit_all_4gpu.yaml` | 4 卡 domain-gradient profile；`fsdp_size=1` 使用同步的完整副本。 |
| `configs/mopd_formal_audit_all_6gpu.yaml` | 6 卡 domain-gradient profile；`fsdp_size=1` 使用同步的完整副本。 |
| `configs/mopd_formal_audit_all_8gpu.yaml` | 8 卡 domain-gradient profile；`fsdp_size=1` 使用同步的完整副本。 |
| `configs/mopd_formal_audit_loss_only_2gpu.yaml` | 2 卡 compatibility profile，包含 domain-gradient 与 loss observations。 |
| `configs/mopd_formal_audit_loss_only_4gpu.yaml` | 4 卡 compatibility profile，包含 domain-gradient audit。 |
| `configs/mopd_formal_audit_loss_only_6gpu.yaml` | 6 卡 domain-gradient profile，使用 `fsdp_size=2`。 |
| `configs/mopd_formal_audit_loss_only_8gpu.yaml` | 8 卡 compatibility profile，包含 domain-gradient audit。 |
| `configs/mopd_formal_audit_off_2gpu.yaml` | 2 卡同模型/数据/objective，关闭全部 audit。 |
| `configs/mopd_formal_audit_off_4gpu.yaml` | 4 卡 audit-off 训练。 |
| `configs/mopd_formal_audit_off_6gpu.yaml` | 6 卡 audit-off 训练，使用 TP=2 的显存安全配置。 |
| `configs/mopd_formal_audit_off_8gpu.yaml` | 8 卡 audit-off 训练。 |
| `configs/mopd_formal_audit_all_smoke.yaml` | 2 卡 one-step 指标 smoke，打开全部 audit 输出和 full-vocab vectors。 |
| `configs/mopd_formal_audit_loss_only_smoke.yaml` | 2 卡 one-step domain-gradient 与 loss-metric smoke。 |
| `configs/mopd_formal_audit_grad_consistency_2gpu_b16_1step_smoke.yaml` | 2 卡 batch size 16、1 step 的 gradient consistency smoke。 |
| `configs/mopd_formal_audit_grad_consistency_2gpu_b32_2step_smoke.yaml` | 2 卡 batch size 32、2 step 的 gradient consistency smoke。 |
| `configs/mopd_formal_audit_grad_consistency_2gpu_b64_3step_smoke.yaml` | 2 卡 batch size 64、3 step 的 gradient consistency smoke。 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math.yaml` | 原始 Math-only：4 张 actor/rollout GPU + 2 张 teacher/ref GPU。 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_code.yaml` | 原始 Code-only，使用相同的 6 卡 topology。 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_if.yaml` | 原始 IF-only，并使用 IFBench validation。 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_science.yaml` | 原始 Science-only，并使用 GPQA validation。 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code.yaml` | 原始 Math+code 等权训练。 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code_science.yaml` | 原始 Math+code+science 等权训练。 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math.yaml` | Math-only：6 张 actor/rollout GPU + 2 张 teacher/ref GPU，batch 504。 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_code.yaml` | Code-only，使用相同 topology、batch 与 audit surface。 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_if.yaml` | IF-only，并使用 IFBench validation。 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_science.yaml` | Science-only，并使用 GPQA validation。 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code.yaml` | Math+code 等权训练，使用相同 topology 与 audit surface。 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science.yaml` | Math+code+science 等权训练，使用相同 topology 与 batch 504。 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_topk32.yaml` | Top-32 Math+code+science distillation，使用相同 8-GPU topology、`fsdp_size=1` 与 batch 504。 |

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
- full/domain-gradient norm、pair cosine、domain-to-total projection、closure 与 training parity
- token gap occurrence vector 与 full-vocab vector
- teacher/student entropy occurrence vector 与 full-vocab vector
- token conflict attribution
- 可选的 teacher/student top-k cross-entropy 与 log-probability vectors

clean rebuild 已移除旧的嵌套 sample/token backward replay。所有 formal profile 都设置 `sample_gradient_enabled=false` 与 `token_gradient_enabled=false`；重新打开任一开关会 fail fast，不再回退到 FSDP private-state 操作。`loss_only` 文件保留历史 selection 字段只是为了配置兼容，不会触发 gradient replay。6 卡 loss-only profile 仍是显存安全的 `fsdp_size=2` 版本，并限制 `data.max_response_length=10240`、`rollout.max_model_len=12288`。

domain decomposition 使用 value-preserving gradient gate，因此不会改变 production loss denominator。所有 backward collective 都由 FSDP 控制；audit 不调用 private finalize hook，也不增加第二次手工同步。audit 将同步后的 local `g_total` 与 `g_domain` shard 保存到 CPU，直接计算 cosine/dot；`D` 个 domain 共需要 `D + 1` 次 backward replay。两类向量都使用 `full_gradient_storage_dtype`，BF16 vector 在指标累计时按 chunk 转成 FP64。

`mopd_formal_audit_off_*gpu.yaml` 设置 `audit.enabled=false`，并显式关闭所有 audit 子开关。

all-audit 和 loss-only smoke profile 作为指标测试 profile 保留并纳入测试。它们使用 `data.train_batch_size=32`、`actor.ppo_mini_batch_size=32`、`trainer.total_training_steps=1`，但保持正式 `data.max_response_length=16384`，并保留 full-vocab token gap 与 entropy vectors。

gradient consistency smoke profile 使用 `../models/Qwen3-0.6B` 降低闭合验证成本。它们比较 domain-gradient sum 与 unmasked audit gradient，并在 optimizer step 前比较 audit gradient 与真实 training gradient。

### Qwen3-30B-A3B-Instruct-2507 Teacher Profiles

原有六套 6-GPU 配置保持不变，共同设置为：

- student：`../models/Qwen3-4B`
- teacher：`../models/Qwen3-30B-A3B-Instruct-2507`
- 6 张可见 GPU：4 actor/rollout + 2 teacher/ref
- actor `fsdp_size=2`，即两个 replica、每个 replica 两个 shard
- rollout TP=2、micro batch 1
- 单域/双域 train/mini batch 512，三域 train/mini batch 504
- 统一 chosen-token policy-gradient objective
- 每两步统计一次 domain gradient，BF16 CPU vector，并检查 training parity；
  开启 token-gap vector，关闭嵌套 sample/token backward

新增六套 8-GPU base 配置沿用相同 domains、数据、objective 与 audit surface，
但统一使用：

- 8 张可见 GPU：6 actor/rollout + 2 teacher/ref
- actor `fsdp_size=1`，使用同步的完整 actor model replicas
- rollout TP=2，所有 domain 组合的 train/mini batch 均为 504
- 独立的 8-GPU experiment、audit、checkpoint 与 paper-eval 输出路径

独立 Top-32 profile 使用相同的 `fsdp_size=1` topology 与 batch 504，
并保留自身的 distillation objective 和 audit cadence。

Math 使用 `DeepMath-103K/train_filtered_level6.parquet`，code 使用
`Eurus/code_train.parquet`，IF 使用 `IF/train.parquet`，science 使用
`Science/train.parquet`，mixed profiles 分别按 math/code 或
math/code/science 等权采样。两套 base 配置均使用
`data.load_parquet_direct=true` 与 `mopd_verl/mixed_reward.py`。

配置默认使用 `logger: '["console","tensorboard","wandb"]'`、
`runtime.wandb_entity: null` 和 `runtime.env_file: .env.local`。需要 W&B 时
通过 override 或环境设置 entity，并在启动训练的机器上把 `WANDB_API_KEY`
放入 `.env.local`；该文件已 gitignore，不能提交。无需 W&B 的本地 dry-run
可以覆盖 `runtime.wandb_mode=disabled`。

运行 `scripts/setup_training_env.sh` 创建或更新本地 Conda 环境。
`environment.yml` 是 CUDA 12.4 compatibility 定义，
`environment.blackwell.yml` 是 PyTorch 2.8 / CUDA 12.8 Blackwell 定义；
setup script 不再维护额外的 pip requirements 或依赖安装脚本。IF/science validation parquet 使用
`scripts/prepare_m2rl_eval_data.sh` 单独准备。

## 启动

在本地 checkout 中启动：

```bash
cd /path/to/OPD-code
GPU_IDS=0,1,2 bash scripts/run_local_mopd_training.sh \
  test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml \
  --run-id mopd_fsdpsize2_smoke_$(date +%Y%m%d_%H%M%S)
```

`scripts/run_local_mopd_training.sh` 默认使用 `CONDA_ROOT=$HOME/miniconda3`
和 `ENV_NAME=mopd-verl`，并在 launch 日志里打印实际 Python 路径。

历史 `mopd_formal_*` 中的 multi-rank `fsdp_size=1` 现在会走同步的
`NO_SHARD` replication；启动前仍需确认每个 actor GPU 能容纳完整 student、
gradient 和 optimizer state。

本地 dry-run：

```bash
scripts/run_mopd.sh \
  test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml \
  --dry-run
```

下载 Qwen3-30B-A3B-Instruct-2507 四领域训练 assets：

```bash
MODEL_ROOT=$(pwd)/../models \
MIN_FREE_GB=300 \
  scripts/download_training_assets.sh
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
- `validation_probe.jsonl`
- `validation_gain_variance.jsonl`
- `training_cost.jsonl`
- `audit_errors.jsonl`

full-vocab vector 文件使用 token-id 坐标：第 `v` 维对应 tokenizer token id `v`。`token_gap_vocab_vectors.jsonl` 保存 signed/absolute log-prob gap 的 sum 和 mean vector；`entropy_vocab_vectors.jsonl` 保存 `student_entropy` 与 `teacher_student_cross_entropy` 的 sum 和 mean vector。

详细 metric 定义见 [metrics_zh.md](metrics_zh.md)。配置字段和常用 override 见 [CONFIG_GUIDE.zh.md](CONFIG_GUIDE.zh.md)。
