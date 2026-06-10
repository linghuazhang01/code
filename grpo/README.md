# GRPO Teacher 训练运行说明

这个目录存放专门用于训练 GRPO teacher model 的代码。当前支持两条训练链路：

- ToolRL：训练 tool-use / function-calling teacher。
- General-Reasoner：训练 general reasoning teacher，使用 verifier model 做 reward。

共享训练入口仍然是 `mopd_verl.launch`，共享 `verl` runtime 位于 `third_party/verl`。`grpo/` 只放 GRPO-specific 的 config、data adapter、reward adapter 和 verifier worker。

## 目录结构

```text
grpo/
  configs/
    toolrl.yaml
    general_reasoner.yaml
  data/
    toolrl.py
  rewards/
    toolrl.py
    general_reasoner.py
  workers/
    general_verifier.py
```

请从 `code/` 目录运行下面所有命令：

```bash
cd /Users/linghuazhang/Desktop/Project/OPD/code
export PYTHONPATH="$PWD:$PWD/third_party/verl:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS=0
```

## 1. 运行 ToolRL

ToolRL 上游代码已经拉到：

```text
/Users/linghuazhang/Desktop/Project/OPD/temp/grpo_sources/ToolRL
```

### 1.1 准备数据

把 ToolRL 原始 parquet 转成当前 shared verl schema：

```bash
python -m mopd_verl.prepare_data prepare-toolrl \
  --input ../temp/grpo_sources/ToolRL/dataset/rlla_4k/train.parquet \
  --output data/ToolRL/rlla_4k/train.parquet \
  --split train

python -m mopd_verl.prepare_data prepare-toolrl \
  --input ../temp/grpo_sources/ToolRL/dataset/rlla_4k/test.parquet \
  --output data/ToolRL/rlla_4k/test.parquet \
  --split test
```

### 1.2 先 dry-run 检查命令

```bash
DRY_RUN=1 scripts/run_toolrl_grpo.sh
```

如果命令里能看到这些字段，说明配置路径正确：

```text
data/ToolRL/rlla_4k/train.parquet
custom_reward_function.path=grpo/rewards/toolrl.py
actor_rollout_ref.rollout.n=4
```

### 1.3 正式启动训练

```bash
scripts/run_toolrl_grpo.sh
```

默认配置文件是：

```text
grpo/configs/toolrl.yaml
```

默认 checkpoint 输出到：

```text
checkpoints/toolrl-qwen2.5-3b-grpo
```

### 1.4 常用 override

换 base model：

```bash
scripts/run_toolrl_grpo.sh -- \
  actor_rollout_ref.model.path=/path/to/Qwen2.5-3B-Instruct \
  actor_rollout_ref.model.base_model_path=/path/to/Qwen2.5-3B-Instruct
```

换 GPU 数：

```bash
scripts/run_toolrl_grpo.sh -- \
  trainer.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
```

启用 ToolRL reward variant：

```bash
WITHLENGTH=1 scripts/run_toolrl_grpo.sh
REFINEDREWARD=1 scripts/run_toolrl_grpo.sh
SCHEDULEREWARD=1 scripts/run_toolrl_grpo.sh
```

## 2. 运行 General-Reasoner

General-Reasoner 上游代码已经拉到：

```text
/Users/linghuazhang/Desktop/Project/OPD/temp/grpo_sources/General-Reasoner
```

### 2.1 准备数据

从 Hugging Face 下载并转换 `TIGER-Lab/WebInstruct-verified`：

```bash
python -m mopd_verl.prepare_data prepare-general-reasoner-hf \
  --output-dir data/GeneralReasoner/WebInstructVerified
```

当前 config 把 validation parquet 放在 `eval/domains/greasoner/...`，所以需要 staging 一份：

```bash
mkdir -p eval/domains/greasoner/data/WebInstructVerified
cp data/GeneralReasoner/WebInstructVerified/test.parquet \
  eval/domains/greasoner/data/WebInstructVerified/test.parquet
```

### 2.2 准备 verifier 和 backbone

General-Reasoner 训练需要 verifier reward model。可以先下载到本地模型目录：

```bash
huggingface-cli download TIGER-Lab/general-verifier \
  --local-dir ../models/general-verifier

huggingface-cli download Qwen/Qwen3-4B \
  --local-dir ../models/Qwen3-4B
```

### 2.3 先 dry-run 检查命令

```bash
DRY_RUN=1 scripts/run_general_reasoner_grpo.sh
```

如果命令里能看到这些字段，说明配置路径正确：

```text
data/GeneralReasoner/WebInstructVerified/train.parquet
custom_reward_function.path=grpo/rewards/general_reasoner.py
+reward_model.worker.path=grpo/workers/general_verifier.py
reward_model.strategy=verifier
```

### 2.4 正式启动训练

建议显式传入本地 verifier 和 backbone 路径：

```bash
scripts/run_general_reasoner_grpo.sh -- \
  reward_model.model.path=../models/general-verifier \
  actor_rollout_ref.model.path=../models/Qwen3-4B \
  actor_rollout_ref.model.base_model_path=../models/Qwen3-4B
```

默认配置文件是：

```text
grpo/configs/general_reasoner.yaml
```

默认 checkpoint 输出到：

```text
checkpoints/general-reasoner-qwen3-4b-grpo
```

### 2.5 常用 override

如果不是 8 卡环境，至少改这些参数：

```bash
scripts/run_general_reasoner_grpo.sh -- \
  trainer.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
```

如果显存不够，优先调小：

```bash
scripts/run_general_reasoner_grpo.sh -- \
  data.train_batch_size=256 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096
```

## 3. 远程机器运行

如果代码已经同步到远程机器，可以直接用已有远程启动脚本：

```bash
scripts/start_remote_mopd_training.sh grpo/configs/toolrl.yaml --run-id toolrl_grpo

scripts/start_remote_mopd_training.sh grpo/configs/general_reasoner.yaml --run-id general_reasoner_grpo -- \
  reward_model.model.path=../models/general-verifier \
  actor_rollout_ref.model.path=../models/Qwen3-4B \
  actor_rollout_ref.model.base_model_path=../models/Qwen3-4B
```

如果需要先从本地同步再启动：

```bash
scripts/sync_and_start_remote_mopd.sh grpo/configs/toolrl.yaml --run-id toolrl_grpo

scripts/sync_and_start_remote_mopd.sh grpo/configs/general_reasoner.yaml --run-id general_reasoner_grpo -- \
  reward_model.model.path=../models/general-verifier \
  actor_rollout_ref.model.path=../models/Qwen3-4B \
  actor_rollout_ref.model.base_model_path=../models/Qwen3-4B
```

## 4. 常见问题

### `No module named yaml`

当前 Python 环境缺少 `PyYAML`。切到训练环境，或安装项目依赖：

```bash
pip install -r requirements.txt
```

### 找不到 parquet 数据

先确认是否已经运行对应的数据准备命令：

```bash
ls data/ToolRL/rlla_4k
ls data/GeneralReasoner/WebInstructVerified
ls eval/domains/greasoner/data/WebInstructVerified
```

### General-Reasoner verifier 启动失败

确认训练环境安装了 GPU 版 `vllm`，并且 `reward_model.model.path` 指向可加载的 verifier 模型目录：

```bash
scripts/run_general_reasoner_grpo.sh -- \
  reward_model.model.path=/absolute/path/to/general-verifier
```

### 只想看最终 verl 命令

两个脚本都支持：

```bash
DRY_RUN=1 scripts/run_toolrl_grpo.sh
DRY_RUN=1 scripts/run_general_reasoner_grpo.sh
```
