# Multi-Teacher OPD Math + Code Training

本目录是对官方 `RUCBM/G-OPD` verl fork 的轻量封装。当前仓库只保留本地研究侧最小代码，真正的 PPO/OPD 训练执行委托给 `verl.trainer.main_ppo`。

## 代码功能

- `configs/mopd_math_code.yaml`：保存 paper-style 的两教师 MOPD 配置。
- `configs/mopd_formal_single_a800.yaml`：保存当前 single-A800 正式训练配置：0.6B student、两个 4B teacher、训练 batch size 1024，以及 `max_response_length=1024`。
- `mopd_verl/settings.py`：把 YAML 配置加载为 typed dataclasses。
- `mopd_verl/launch.py`：把 typed config 转换为 `python -m verl.trainer.main_ppo` 所需的 Hydra overrides。
- `mopd_verl/prepare_data.py`：合并 math/code parquet 文件，验证 `extra_info.opd_teacher`，并把 paper math eval JSONL 转换成 verl validation parquet。
- `mopd_verl/paper_eval.py`：被注入到 G-OPD/verl trainer，在 validation 后调用七个 paper benchmark 的外部评测脚本。
- `mopd_verl/smoke_data.py`：生成远程 one-step smoke test 使用的极小 synthetic parquet 文件。
- `mopd_verl/verl_audit.py`：被注入到 G-OPD/verl trainer 的 MOPD audit logger，负责 JSONL 与 TensorBoard scalar 输出。
- `configs/mopd_audit_smoke.yaml`：开启 audit + TensorBoard 的 one-step smoke 配置。
- `scripts/apply_gopd_audit_patch.py`：幂等 patch G-OPD checkout，使 verl dataset/trainer 传递并记录 audit 字段。
- `scripts/run_math_code_mopd.sh`：通用训练启动脚本。
- `scripts/setup_remote_training_env.sh`：在远程机器上初始化 conda + G-OPD 环境。
- `scripts/prepare_paper_eval_data.sh`：在远端生成 AIME/HMMT validation parquet，并下载/检查 HumanEval+、MBPP+、LiveCodeBench 数据。
- `scripts/run_paper_eval_suite.sh`：训练期 validation 后运行 AIME24、AIME25、HMMT25 Feb.、HMMT25 Nov.、HumanEval+、MBPP+、LCB 的 full paper eval suite。
- `scripts/run_remote_one_step_smoke.sh`：远程环境完成安装后运行 one-step training smoke test，并在日志中写入完成标记。

## 训练栈

默认训练栈遵循官方 G-OPD / ExOPD recipe：

- Codebase：`RUCBM/G-OPD`，其训练代码基于 verl v0.6.1。
- Entrypoint：`verl.trainer.main_ppo`。
- Objective：reverse-KL OPD，使用 ExOPD reward scaling，`lambda_vals=1.25`。
- Multi-teacher 开关：`actor_rollout_ref.actor.policy_loss.multi_teacher_distill=true`。
- Teacher routing 字段：`extra_info.opd_teacher`。Audit 还会读取 `extra_info.domain`、`extra_info.source_domain` 和 `extra_info.sample_id`。

## Formal Single-A800 配置

当前正式单卡配置文件是 `configs/mopd_formal_single_a800.yaml`。它不是 paper-exact 的 8-GPU 配置，而是适配当前远端 A800 80GB 的可运行配置：

- Student / training model：`Qwen3-0.6B`
- Math teacher：`Qwen3-4B-Non-Thinking-RL-Math-Step500`
- Code teacher：`Qwen3-4B-Non-Thinking-RL-Code-Step300`
- Training batch：`data.train_batch_size=1024`
- PPO minibatch：`actor.ppo_mini_batch_size=16`
- PPO microbatch per GPU：`actor.ppo_micro_batch_size_per_gpu=1`
- Prompt length：`data.max_prompt_length=2048`
- Response length：`data.max_response_length=1024`
- vLLM cache budget：`rollout.gpu_memory_utilization=0.5`
- Logger：`["console","tensorboard"]`
- Training-time validation parquet：AIME24、AIME25、HMMT25 Feb.、HMMT25 Nov.、Eurus code validation
- External paper eval：AIME24、AIME25、HMMT25 Feb.、HMMT25 Nov.、HumanEval+、MBPP+、LCB

其中 `max_response_length=1024` 对速度影响很大。当 `response_length/mean` 接近 1024 且 `response_length/clip_ratio` 接近 1 时，step 会主要卡在 generation 和 validation，而不是 actor update 或 audit 本身。

`paper_eval.enabled=true` 会在每次 `_validate()` 之后启动 full paper eval suite。这个配置严格覆盖 paper 中报告的 7 个 dataset，但在 single-A800 上会显著拉长每次 validation 时间；如果只做 smoke test，应通过 Hydra override 临时设为 `+paper_eval.enabled=false`。

## 启动训练

在远端 `OPD-code` checkout 中启动正式 single-A800 训练：

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mopd-verl

MOPD_CONFIG=configs/mopd_formal_single_a800.yaml \
PYTHON_BIN=python \
bash scripts/run_math_code_mopd.sh
```

只打印生成的 Hydra command，不真正启动训练：

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mopd-verl

DRY_RUN=1 \
MOPD_CONFIG=configs/mopd_formal_single_a800.yaml \
PYTHON_BIN=python \
bash scripts/run_math_code_mopd.sh
```

使用 `screen` 后台运行并保存日志：

```bash
mkdir -p /root/autodl-tmp/opd_mopd/logs
screen -dmS mopd_formal bash -lc '
cd /root/autodl-tmp/opd_mopd/OPD-code &&
source /root/miniconda3/etc/profile.d/conda.sh &&
conda activate mopd-verl &&
MOPD_CONFIG=configs/mopd_formal_single_a800.yaml PYTHON_BIN=python \
bash scripts/run_math_code_mopd.sh 2>&1 | tee /root/autodl-tmp/opd_mopd/logs/formal_$(date +%Y%m%d_%H%M%S).log
'
```

临时 Hydra override 可以追加在 `--` 后。例如保留正式配置但关闭外部 paper eval，用于更快 sanity run：

```bash
MOPD_CONFIG=configs/mopd_formal_single_a800.yaml \
PYTHON_BIN=python \
bash scripts/run_math_code_mopd.sh -- +paper_eval.enabled=false
```

如果只做 one-step smoke test，用 `scripts/run_remote_one_step_smoke.sh`，不要用正式配置。

## 训练数据

Production 数据：

- Train：`G-OPD-Training-Data/math_and_code/train.parquet`
- Validation：`G-OPD-Training-Data/PaperEval/AIME24/test.parquet`
- Validation：`G-OPD-Training-Data/PaperEval/AIME25/test.parquet`
- Validation：`G-OPD-Training-Data/PaperEval/HMMT25Feb/test.parquet`
- Validation：`G-OPD-Training-Data/PaperEval/HMMT25Nov/test.parquet`
- Validation：`G-OPD-Training-Data/Eurus/code_validation.parquet`

每条训练样本需要符合 verl RL dataset 的基本结构：`data_source`、`prompt`、`ability`、`reward_model` 和 `extra_info`。对于 MOPD，`extra_info` 必须包含：

```json
{"opd_teacher": "math"}
```

或：

```json
{"opd_teacher": "code"}
```

Smoke-test 数据：

- 通过 `python -m mopd_verl.smoke_data <output_dir>` 生成。
- 包含两条极小样本，一条 routed to math teacher，一条 routed to code teacher。
- 只用于验证远程 verl stack 能否启动并完成一次 optimizer step，不代表模型质量。

Paper eval 外部评测数据：

- Math：`data/aime24/test.jsonl`、`data/aime25/test.jsonl`、`data/hmmt25_feb/test.jsonl`、`data/hmmt25_nov/test.jsonl`
- EvalPlus：`code_eval/data/HumanEvalPlus.jsonl`、`code_eval/data/MbppPlus.jsonl`
- LiveCodeBench：`code_eval/coding/LiveCodeBench/code_generation_lite/test*.jsonl`

远端首次运行前先准备数据：

```bash
bash /root/autodl-tmp/opd_mopd/OPD-code/scripts/prepare_paper_eval_data.sh
```

训练中的 full suite 输出默认写到：

```text
/root/autodl-tmp/opd_mopd/eval_outputs/paper_suite/formal_single_a800/step_XXXXXXXX/
```

## 训练模型

Paper-style 默认设置：

- Student/reference：`Qwen/Qwen3-4B`
- Math teacher：`Qwen3-4B-Non-Thinking-RL-Math`
- Code teacher：`Qwen3-4B-Non-Thinking-RL-Code`

远程 smoke-test 默认设置：

- Student/reference/math-teacher/code-teacher：`Qwen/Qwen3-0.6B`

Smoke test 故意让所有角色都使用同一个小型公开模型。这样可以降低首次远程检查成本，同时仍然覆盖 multi-teacher routing 和 OPD launcher path。真实训练时，需要把模型路径改回 paper-style 的 Qwen3-4B student 和两个 domain teacher。

## 远程环境设置

`code/ssh.sh` 描述的远程服务器当前环境包括 `/root/miniconda3`、Ubuntu 22.04、CUDA 12.8 和一张 A800 80GB GPU。setup 脚本假设在该服务器上运行：

```bash
bash /root/autodl-tmp/opd_mopd/OPD-code/scripts/setup_remote_training_env.sh
```

重要环境变量：

```bash
REMOTE_ROOT=/root/autodl-tmp/opd_mopd
CONDA_ROOT=/root/miniconda3
ENV_NAME=mopd-verl
INSTALL_SGLANG=0
FORCE_REINSTALL=0
```

脚本会创建：

- `${REMOTE_ROOT}/G-OPD`
- `${REMOTE_ROOT}/smoke_data/train.parquet`
- `${REMOTE_ROOT}/smoke_data/val.parquet`
- `${REMOTE_ROOT}/env.sh`
- `${REMOTE_ROOT}/logs/`

当前 setup 脚本还 pin 了 A800 container 上验证过的关键依赖：

- `transformers[hf_xet]==4.51.3`
- `ray[default]==2.46.0`
- `click<8.2`
- `numpy<2.0.0`
- OpenTelemetry packages at `1.26.0`

`HF_ENDPOINT` 默认设置为 `https://hf-mirror.com`，因为测试过的远程 container 无法直接访问 `huggingface.co`。默认跳过 FlashInfer 安装，以避免 GitHub wheel 下载过慢；vLLM 会 fallback 到 PyTorch-native sampler，这对 smoke test 已经足够。

## One-Step Smoke Training

完成环境设置后运行：

```bash
bash /root/autodl-tmp/opd_mopd/OPD-code/scripts/run_remote_one_step_smoke.sh
```

Smoke 脚本会把 production config 覆盖为：

- 使用 1 张 GPU；
- batch size 为 1；
- 关闭 W&B；
- 开启 TensorBoard；
- 开启 step 0/step 1 validation，用来记录 validation gain；
- 关闭 checkpoint saving；
- 在 `trainer.total_training_steps=1` 后停止；
- 日志写到 `${REMOTE_ROOT}/logs/`；
- audit JSONL 写到 `${REMOTE_ROOT}/audit/smoke/`；
- TensorBoard event 写到 `${TENSORBOARD_DIR:-${REMOTE_ROOT}/tensorboard}`；
- 启动前清理 stale Ray state，并在脚本退出时停止 Ray。

已验证的远程结果：

- Server：`autodl-container-857546be50-cbac1eda`
- GPU：`NVIDIA A800 80GB PCIe`
- Env：`/root/miniconda3/envs/mopd-verl`
- Code：`/root/autodl-tmp/opd_mopd/OPD-code`
- G-OPD/verl：`/root/autodl-tmp/opd_mopd/G-OPD`
- Log：`/root/autodl-tmp/opd_mopd/logs/one_step_smoke_20260601_003357.log`
- Result：completed `step:1` with `training/global_step:1`
- Step time：`27.805183600634336` seconds
- Peak allocated GPU memory：`31.825812339782715` GB
- Throughput：`1.5105097165782364` tokens/s
- Post-run status：无活跃 training/Ray 进程，GPU memory 回到 `0 MiB`

TensorBoard 层级：

- Event dir：`/root/autodl-tmp/opd_mopd/tensorboard/audit_smoke`
- Scalar tag count：`134`
- audit scalar 默认使用 `domain/category/metric` 层级，一级就是 domain 名字。例如
  `math/loss/token_opd_loss_variance`、`math/full_grad/grad_norm`、
  `math/full_grad_anchor/AIME2024/full_grad_cosine_i_j`、`math/validation/score`、
  `math/validation_gain/score`。
- 非 domain 专属指标统一放在 `global/category/metric`，例如
  `global/data/domain_mix_entropy`、`global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k`、
  `global/cost/gpu_seconds_step`。

JSONL audit 文件：

- `domain_step_metrics.jsonl`：每个 step、每个 domain 的 domain-level 指标。
- `loss_variance_domain_step.jsonl`：每个 step、每个 domain 的 OPD loss variance 摘要。
- `loss_variance_sample.jsonl`：sample-level loss variance 明细；不直接写成 TensorBoard per-sample scalar，避免 tag 爆炸。
- `validation_probe.jsonl`：validation metric 和相邻 validation gain。

## 本地验证

运行 wrapper tests：

```bash
PYTHONPATH=code python3 -m unittest discover -s code/tests
```

只构建 Hydra command，不启动训练：

```bash
DRY_RUN=1 code/scripts/run_math_code_mopd.sh
```

检查 teacher labels：

```bash
PYTHONPATH=code python3 -m mopd_verl.prepare_data inspect /path/to/train.parquet
```

## Audit 指标计算方式与计划覆盖状态

完整 TensorBoard metrics 说明见 [`metrics_zh.md`](metrics_zh.md)。这份文档是当前权威版本，逐项说明了保留 tag 的含义、计算方式和 JSONL 输出。

当前只保留真实 full-parameter gradient 诊断：

- `grad` / `grad_anchor` / `grad_conflict` 低成本 proxy 已删除，不再计算、不写 JSONL，也不会进入 TensorBoard。
- `full_grad` / `full_grad_anchor` / `full_grad_conflict` 会在 actor worker 内执行真实 backward，并读取完整 actor 参数 `.grad`。full-gradient path 现在调用 verl 的 PPO policy-loss function，并使用 MOPD `advantages=-reverse_kl`。
- validation anchor 不再直接相加各 batch 的 mean gradient，而是按 response token count 维护 token-weighted running mean gradient。
- 正式配置中 `full_gradient_train_max_samples_per_domain: null`、`full_gradient_validation_max_samples_per_domain: null`，表示当前 step 的 training batch / 当前 validation pass 中每个 domain 不做样本截断。
- 这里的“full training gradient”指当前 on-policy training batch 的完整 domain 数据；不是每个 step 重新扫完整 train parquet。后者需要对全训练集重新 rollout、计算 teacher/ref log-probs 并 backward，成本接近每个 step 额外跑一轮完整训练数据。
- `predicted_val_opd_loss_delta_i_j` 是一阶 OPD/PPO surrogate 预测，不是实际执行 Adam optimizer step 后的 validation delta。

当前保留的 JSONL 输出：

- `domain_step_metrics.jsonl`
- `loss_variance_domain_step.jsonl`
- `loss_variance_sample.jsonl`
- `validation_probe.jsonl`
- `validation_gain_variance.jsonl`
- `training_cost.jsonl`
- `audit_errors.jsonl`

已删除的可选指标不会再计算或写盘，包括 rank stability、shadow probe、sample influence、额外 teacher-logprob 诊断、CI、dot-only fields、tail percentiles、trend stability、coverage diversity 等。即使 `tensorboard_prune_mode=none`，这些已删除指标也不会恢复。

validation gain 的 TensorBoard 一级层级只在 metric key 能解析出配置 domain 时使用 domain 名。`val/math/score` 会写成 `math/validation_gain/score`；`val-core/AIME2024/reward/mean@1` 这类 benchmark key 会写成 `global/validation_gain/val-core_AIME2024_reward_mean_1`，dataset 名折叠进 metric 名。

### 代码实现原理

实现目标是低侵入地把 MOPD 诊断接入 G-OPD/verl，不改主训练算法：

1. `mopd_verl/settings.py` 定义 `AuditConfig`，保留 output、domain、TensorBoard layout/pruning、validation anchor、full-gradient probe、calibration 和 sample-level loss variance 等必要配置。
2. `mopd_verl/launch.py` 把 typed config 转成 Hydra overrides，例如 `+mopd_audit.enabled=true`、`+mopd_audit.output_dir=...`、`+mopd_audit.full_gradient_enabled=true`。
3. `scripts/apply_gopd_audit_patch.py` 幂等 patch 远端 G-OPD checkout：dataset 侧透传 `domain`、`sample_id`、`source_domain`；trainer 侧创建 `MOPDAuditLogger`，在训练、validation、timing metrics 出来后分别写 audit metrics。
4. `MOPDAuditLogger.log_training_step()` 从 verl `DataProto` 读取 `batch.batch` 与 `batch.non_tensor_batch`，用 `response_mask` 对 token-level OPD loss 做 masked aggregation，生成当前保留的 per-domain metrics、per-sample loss variance rows 和 TensorBoard scalar。
5. full-parameter gradient metrics 不在 Python logger 里用 proxy 近似，而是通过 patch 后的 actor worker `compute_mopd_full_gradient_metrics()` 运行真实 backward：train mode 计算各 train domain 的参数梯度，validation-anchor mode 在 validation pass 中按 validation domain 和 response token count 累计 token-weighted mean anchor gradient。
6. scalar metrics 在写入 verl 原有 logger 前会经过 `MOPDAuditLogger.filter_tensorboard_metrics()`；正式配置使用 `tensorboard_prune_mode=core`，只保留 validation gain、full-gradient、核心 loss、teacher/calibration、coverage、cost 和训练健康指标。
7. 失败策略是 fail-soft：audit 计算异常会写入 `audit_errors.jsonl`，并返回 `global/audit/error=1.0`，不会直接中断训练。

### 验证

本地轻量环境无 torch 时会跳过 synthetic audit test；远端 `mopd-verl` 环境会实际执行该 test，验证 domain/category TensorBoard metric key 和 JSONL 文件生成：

```bash
PYTHONPATH=code python3 -m unittest discover -s code/tests
```
