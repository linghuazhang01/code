# OPD 评测说明

这个目录保存 OPD 的评测实现、评测数据和运行结果。唯一面向用户的 eval
启动入口是：

```bash
scripts/run_local_eval.sh --model-path /path/to/model [options]
```

请从 `code/` 目录运行。不要直接调用 `python -m eval.runner`，也不要直接运行
`eval/scripts/` 下的 model-eval 脚本；这些文件只作为内部实现与兼容代码保留。
`eval/scripts/` 下的数据准备工具仍可单独运行。

## 目录结构

- `runner.py`: Qwen thinking / non-thinking 模式对比评测器。
- `common.py`: parquet 加载、prompt 归一化、token 统计和结果汇总。
- `report.py`: 为已完成或正在运行的 eval 生成 JSON / Markdown 报告。
- `paper_eval.py`: patched verl validation 调用的运行时入口。
- `data_prep/`: 将 paper-eval JSONL 转换为 verl parquet 的数据准备代码。
- `domains/`: 各 domain 的 metadata、数据准备脚本和评测数据。
- `scripts/`: 内部或 legacy eval helper，不是公开启动入口。
- `../data/eval_data/results/`: 公开本地 eval 入口的输出目录。

## Domain 划分

| Domain | 代码位置 | 评测数据 | 状态 |
|---|---|---|---|
| Math | `domains/math/` | `../data/eval_data/math/{AIME24,AIME25,HMMT25Feb,HMMT25Nov}/test.parquet` | 已就绪 |
| Code | `domains/code/` | `../data/eval_data/code/{HumanEvalPlus,MBPPPlus,LiveCodeBench}/test.parquet` | HumanEvalPlus/MBPPPlus 已就绪；LiveCodeBench 用 `prepare_paper_eval_data.sh` 生成 |
| IF | `mopd_verl/m2rl_reward.py` | `../data/eval_data/if/{IFBench,IFEval}/test.parquet` | 完整数据用 `python -m eval.data_prep.m2rl_eval` 生成；shell helper 只准备 IFBench |
| Science | `domains/science/` | `../data/eval_data/science/{GPQA,HLE,MMLU-Pro,SuperGPQA}/test.parquet` | GPQA/HLE 用 `python -m eval.data_prep.m2rl_eval` 生成；MMLU-Pro/SuperGPQA 提供 official evaluator |
| Training ceiling | 复用现有 domain scorer | `../data/eval_training_data/{math,code,if,science}/test.parquet` | 每个 domain 10,000 条、与训练源重叠，只用于 training-performance 诊断 |
| ToolRL | `domains/toolrl/` | `../data/eval_data/toolrl/{BFCL,API-Bank,Bamboogle}/test.parquet` | 数据与内部 evaluator 已存在；ToolRL datasets 尚未接入 `run_local_eval.sh` |

SearchQA 仍保留在 `domains/search/`，因为 thinking evaluator 可以继续包含
`data/SearchQA/test.parquet`。不过 SearchQA 不是这次整理出的四个核心 eval
domain 之一。

## 数据准备

从 G-OPD checkout 准备 Math / Code paper-eval 数据：

```bash
eval/scripts/prepare_paper_eval_data.sh
```

下载 MMLU-Pro 与 SuperGPQA official 数据：

```bash
python -m eval.domains.science.download_official_data --force
```

重建对应的可复现 paper subset：

```bash
python -m eval.domains.science.prepare_subsets
```

将本地 ToolRL JSONL 暂存为 verl eval parquet：

```bash
python -m eval.domains.toolrl.prepare_data \
  --dataset BFCL \
  --input /path/to/bfcl.jsonl \
  --output data/eval_data/toolrl/BFCL/test.parquet
```

在 canonical dataset path 下准备完整 M2RL paper evaluation bundle
（IFBench、IFEval、GPQA、HLE）：

```bash
python -m eval.data_prep.m2rl_eval
```

如果只需从本地数据源准备 training validation pair（IFBench、GPQA）：

```bash
IF_VAL_SOURCE=/path/to/raw_if_val.parquet \
SCIENCE_VAL_SOURCE=/path/to/raw_science_val.parquet \
  scripts/prepare_m2rl_eval_data.sh
```

也可以从 Nemotron RL JSONL 中准备同一组 IFBench/GPQA validation subset：

```bash
NEMOTRON_RL_SOURCE=/path/to/instruction_following.jsonl \
M2RL_EVAL_MAX_SAMPLES=512 \
  scripts/prepare_m2rl_eval_data.sh
```

## Prompt 构建

Math prompt 与原 G-OPD paper eval 对齐：

```text
{problem}
Please reason step by step, and put your final answer within \boxed{}.
```

Code prompt 由 `domains/code/prompting.py` 统一构建：

- `HumanEvalPlus` / `MBPPPlus`: 使用原 EvalPlus Qwen/chat instruction，在题目后追加
  markdown Python code block 要求和 paper 里的 "think first" 句子。
- `LiveCodeBench`: 使用 G-OPD 对齐的增量 `v6`（仅 `test6.jsonl`，175 题），
  默认使用 paper 代码中的 `Qwen3NonThinking` prompt 内容。它不是累计 1,055 题的
  `release_v6`。生成的 parquet 包含完整 private tests，因此由 Git 忽略；
  `manifest.json` 仅记录固定 revision 与 source checksum。

G-OPD 的 official LiveCodeBench protocol 是每题 4 samples、`temperature=1.0`、
`top_p=1.0`、`max_tokens=16384`，并执行 public + private tests。请使用
`eval/scripts/run_paper_eval_suite.sh` 复现；`run_local_eval.sh` 更适合统一接口下的
smoke/debug evaluation。

## 从训练数据生成 Performance-Ceiling Eval

从 Math、Code、IF、Science 各确定性抽取 10,000 条，并按
whitespace-normalized、casefolded prompt 分组，使重复 prompt 在抽样时保持在一起：

```bash
python scripts/split_domain_eval_training_data.py \
  --eval-size 10000 \
  --seed 42 \
  --overwrite
```

- Eval: `data/eval_training_data/<domain>/test.parquet`
- Audit manifest: `data/eval_training_data/manifest.json`

这些样本故意与原始 training files 重叠，用于估计 training-data performance
ceiling，而不是 leakage-free generalization benchmark。当前 workflow 不修改原始
parquet，也不消费或刷新 `data/training_data_split/` 下的旧 remainder。

注意：`runner.py` 仍然会根据 eval mode 控制 tokenizer 的 `enable_thinking`：

- `thinking`: `enable_thinking=True`
- `non_thinking`: `enable_thinking=False`

也就是说，数据里的用户侧 problem instruction 已经对齐 paper；thinking /
non-thinking 的对比仍由当前 runner 显式控制。

## 运行 Eval

所有本地 eval 都通过唯一公开入口启动：

```bash
scripts/run_local_eval.sh \
  --model-path ../models/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
  --datasets aime24,humaneval_plus,ifeval,gpqa_diamond \
  --modes non_thinking \
  --max-samples 8 \
  --save-completions
```

基础参数：

- `--datasets`：逗号分隔的 dataset key。
- `--modes`：`non_thinking`、`thinking`，或逗号分隔的两种模式。
- `--max-samples`：每个 dataset 最多评测多少条。
- `--max-new-tokens`：所有选中模式的生成长度上限。
- `--num-samples`、`--temperature`、`--top-p`、`--seed`：采样行为。
- `--backend transformers|vllm`：推理后端。
- `--tensor-parallel-size`、`--batch-size`、`--gpu-memory`：vLLM 参数。
- `--score-code`：执行生成代码并评分，只能在隔离环境使用。
- `--save-completions`：保存完整 completion。
- `--dry-run`：只校验参数并打印最终命令，不启动模型。

支持的 dataset key：`aime24`、`aime25`、`hmmt25feb`、`hmmt25nov`、
`humaneval_plus`、`mbpp_plus`、`livecodebench`、`ifeval`、`ifbench`、
`gpqa_diamond`。Training-performance key 包括聚合四个 domain 的
`training_ceiling`，以及单域 `training_math`、`training_code`、
`training_if`、`training_science`。原始完整 training-data key 包括
`training_full`、`training_full_math`、`training_full_code`、
`training_full_if` 和 `training_full_science`。

## 两模型三类评测脚本

以下三个脚本按顺序评测 `Qwen3-1.7B` 与
`Nemotron-Research-GooseReason-4B-Instruct`，固定使用单卡 `TP=1`：

两个模型完全使用同一套 GooseReason training profile 的 validation-inference
参数：`non_thinking`、`max_new_tokens=16384`、greedy `n=1`、
`temperature=0`、`top_p=1`、`seed=42`、`max_model_len=18432`、
`max_num_batched_tokens=32768`、`max_num_seqs=24`、`enforce_eager=true`，
并关闭 chunked prefill。唯一有意调整的是按 141 GiB 单卡改为 `TP=1`；request
batch 默认 24，vLLM memory utilization 为 0.9。

```bash
# Held-out OOD benchmark suite
scripts/run_two_model_ood_eval.sh

# 四域各 10,000 条 deterministic training ceiling
scripts/run_two_model_training_ceiling_eval.sh

# 四域全部原始 training rows；必须显式确认高成本运行
CONFIRM_FULL_TRAINING=1 scripts/run_two_model_full_training_eval.sh
```

三个入口均启用 `--save-completions`。原始
`thinking_eval_samples.jsonl` 和面向分析的
`prompt_response_records.jsonl` 会保存 `prompt`、`response`、dataset、
sample ID、model path、run ID、ground truth、rollout index、generation seed，
以及包含源文件、原始行号、parquet `extra_info`、reward config 和额外 source
columns 的 `sample_metadata`。

OOD 脚本默认使用 9 个 held-out dataset，并统一采用每题 1 个 greedy rollout；它是
跨 domain diagnostic，不是各 dataset 的 G-OPD paper protocol（例如 Math paper eval
采用 sampled K=32）。LiveCodeBench 的 official protocol
要求 temperature 1.0、每题 4 个 sampled rollout 和 16,384 tokens，因此不混入默认
greedy suite；只有在同步调整协议时才通过 `OOD_DATASETS=...` 显式加入。这里的 OOD
表示 held-out benchmark split，并不等价于已完成对所有 training source 的 prompt-hash
leakage audit。

完整 training eval 默认以每 shard 10,000 rows 顺序运行，避免一次把全部
training parquet 与生成结果留在内存。可通过 `SHARD_SIZE` 调整；中断后使用同一
`RUN_TAG` 并设置 `RESUME=1 CONFIRM_FULL_TRAINING=1` 续跑。每个完成 shard
都会写入 `SUCCESS` 标记。
真实运行默认要求输出文件系统至少保留 50 GiB；请根据 completion 长度和 rollout 数
调整 `MIN_FREE_GB`。
输出根目录的 `suite_manifest.json` 会记录每个 source range、model、shard seed、
预期 record 数、输出目录、source SHA-256 与当前 `SUCCESS` 状态。Resume 会先严格
比对 immutable suite signature；配置或数据身份不一致时，在跳过任何 shard 前拒绝运行。
三个脚本的 `DRY_RUN=1` 都只做校验并打印计划，不创建输出目录、log、shard marker
或 manifest。非 resume 运行会拒绝复用非空输出目录，不会删除已有 rollout。

### W&B 上传

三个入口默认把 scalar metrics 和 summary table 上传到 W&B project
`mopd-eval`。每个 model run（full-training 中为每个 10k shard）还会上传一个
`evaluation-results` artifact，包含 report、generation config、compact records、
完整 prompt/response rollout JSONL 和 summary files；同一次 launcher 执行的 runs
会放在同一个 W&B group 下。

评测会复用 training launcher 的 `.env.local` parser，包括 `export KEY=value` 格式。
原有的 legacy key 名称可以直接使用，并只在进程内映射为 W&B 标准变量：

```bash
export Wandb_Key=<your-wandb-api-key>
```

同时支持 `WANDB_API_KEY`；如果 shell 或 `.env.local` 已经提供该标准变量，则它优先。
Key 不会被写入 report、status file 或 artifact。默认读取 `<code-root>/.env.local`，
可用 `EVAL_WANDB_ENV_FILE` 覆盖路径。

Full-training 会先为本地完成的 shard 写 `SUCCESS` 并加入本地队列；所有 generation
和最终 suite manifest 完成后，才逐个上传 W&B artifact，因此网络慢不会延迟下一个
generation shard。每个上传默认 timeout 为 1,800 秒；可通过
`EVAL_WANDB_TIMEOUT_SECONDS` 修改，设为 `0` 表示不限制；该 timeout 同样适用于
OOD 与 training-ceiling。

可选环境变量包括 `EVAL_WANDB_ENTITY`、
`EVAL_WANDB_MODE=online|offline|disabled`、`EVAL_WANDB_UPLOAD_RAW=0` 和
`EVAL_WANDB_ENABLED=0`。project 可通过 `EVAL_WANDB_PROJECT` 覆盖，默认值为
`mopd-eval`。Offline W&B 状态保存在每个结果目录的 `wandb/` 下，具体 local run
directory 会记录到 `wandb_upload_status.json`。

Raw artifact 含 prompt、response、ground truth 与 source metadata。启动前请确认目标
W&B project 的 visibility 与数据策略适合这些内容；如果只允许 metrics 离开机器，设置
`EVAL_WANDB_UPLOAD_RAW=0`。

本地评测文件仍是 source of truth。认证或网络失败不会让已经完成的 generation
失效，而是写入 `wandb_upload_status.json`，状态为 `upload_pending`。修复网络后可
对单个结果目录或 full-training 的全部已完成 shard 重试：

```bash
python -m eval.wandb_upload \
  --output-dir data/eval_data/results/<suite>/<RUN_TAG>/<model> \
  --project mopd-eval \
  --group <suite>_<RUN_TAG> \
  --env-file .env.local

python -m eval.wandb_upload \
  --output-root data/eval_data/results/two_model_full_training/<RUN_TAG> \
  --project mopd-eval \
  --group two_model_full_training_<RUN_TAG> \
  --env-file .env.local
```

三个脚本均支持 `MAX_SAMPLES`（OOD/ceiling）或
`MAX_SAMPLES_PER_DOMAIN`（full training）做 smoke test；当
`NUM_SAMPLES>1` 时必须同时设置非零 `TEMPERATURE`。

例如，对四个 domain 各运行最多 100 条 training ceiling eval：

```bash
scripts/run_local_eval.sh \
  --model-path /path/to/model \
  --datasets training_ceiling \
  --max-samples 100
```

`training_ceiling` 不是 official benchmark，结果必须标记为 training-data
performance。Code 默认只生成 completion；只有在隔离环境中显式传入
`--score-code`，且完整 vendored verl reward 依赖可用时，才执行代码评分。
Training Code 数据不会回退到 Math scorer。

IF scoring 需要 `scripts/setup_training_env.sh` 安装的
`verifiable_instructions`，或通过以下命令准备本地 official evaluator：

```bash
scripts/prepare_ifbench_runtime.sh
```

例如，使用两张 GPU 和 vLLM 对比 thinking / non-thinking：

```bash
CUDA_VISIBLE_DEVICES=0,1 scripts/run_local_eval.sh \
  --model-path /path/to/model \
  --datasets aime24,gpqa_diamond \
  --modes non_thinking,thinking \
  --backend vllm \
  --tensor-parallel-size 2 \
  --batch-size 8
```

输出会写入：

```text
data/eval_data/results/<RUN_ID>/
```

主要输出文件：

- `thinking_eval_samples.jsonl`
- `thinking_eval_summary.json`
- `thinking_eval_summary.csv`
- `records.jsonl`
- `README.md`

## Scoring

- Math 使用 boxed-answer 风格 scoring；MMLU-Pro 与 SuperGPQA 使用 Science
  official evaluator。
- Code 通过 vendored verl reward router 调用 `mopd_verl/code_reward.py`。
- IF / Science validation 与训练共用同一条 verl reward 路径：
  `mopd_verl/mixed_reward.py` 将 `m2rl_ifbench` 路由到 IFBench /
  verifiable-instructions strict scoring，将 `m2rl_gpqa` 路由到 GPQA
  option-letter scoring。
- ToolRL parquet 数据可以被加载，用于 token / cost 报告。
- ToolRL 官方 benchmark wrapper 已支持 API-Bank 本地 scoring、BFCL handler launcher、
  以及 Bamboogle search + judge scoring。

## 训练配置引用

MOPD 配置已经将 validation path 指向当前目录：

- `configs/mopd_qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_*.yaml`

训练数据仍保留在 `data/G-OPD-Training-Data/`，不会和 `eval/` 下的评测数据混在一起。

## 推荐使用方式

1. 先运行 `eval/scripts/prepare_paper_eval_data.sh` 准备或刷新 Math / Code eval parquet。
2. 使用 `scripts/run_local_eval.sh` 启动全部本地 eval。
3. 查看输出目录里的 `README.md`、JSON 和 CSV 结果。

如果刚修改过 prompt builder，需要重新运行数据准备脚本，否则已有 parquet 里仍可能保留旧 prompt。

## 内部 Evaluator

`eval/runner.py`、`eval/official_runner.py` 和 `eval/scripts/` 下的 model-eval
脚本作为开发、兼容实现保留，但不再作为独立的用户启动入口。数据准备工具不是
eval 启动入口，仍可直接使用。如果要公开新的 benchmark 或 eval 行为，应先将其
接入 `scripts/run_local_eval.sh`。
