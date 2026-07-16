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

## 从训练数据生成 Holdout Eval

默认从 Math、Code、IF、Science 各抽取约 1,000 条，并按 whitespace-normalized、
casefolded prompt 分组，避免同题跨入 train/eval：

```bash
python scripts/split_domain_eval_training_data.py --write-remainders
```

- Eval: `data/eval_training_data/<domain>/test.parquet`
- Train remainder: `data/training_data_split/<domain>/train.parquet`
- Audit manifest: `data/eval_training_data/manifest.json`

脚本不修改原始 parquet。正式把这些数据作为 holdout 时，训练 config 必须改用
`data/training_data_split/` 下的 remainder；若仍训练完整原文件，eval 会发生泄漏。

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
`gpqa_diamond`。

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
