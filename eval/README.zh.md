# OPD 评测说明

这个目录是 OPD 项目中评测代码、评测数据和评测运行结果的统一入口。

## 目录结构

- `runner.py`: Qwen thinking / non-thinking 模式对比评测器。
- `common.py`: parquet 加载、prompt 归一化、token 统计和结果汇总。
- `report.py`: 为已完成或正在运行的 eval 生成 JSON / Markdown 报告。
- `paper_eval.py`: patched verl validation 调用的运行时入口。
- `data_prep/`: 将 paper-eval JSONL 转换为 verl parquet 的数据准备代码。
- `domains/`: 各 domain 的 metadata、数据准备脚本和评测数据。
- `scripts/`: 本地或远程评测启动脚本。
- `results/`: 本地评测输出目录。

根目录下的 `scripts/*eval*.sh` 只是兼容 wrapper。新的 eval 相关代码应放在
`eval/` 下面。

## Domain 划分

| Domain | 代码位置 | 评测数据 | 状态 |
|---|---|---|---|
| Math | `domains/math/` | `domains/math/data/{AIME24,AIME25,HMMT25Feb,HMMT25Nov}/test.parquet` | 已就绪 |
| Code | `domains/code/` | `domains/code/data/{HumanEvalPlus,MBPPPlus,LiveCodeBench}/test.parquet` | 已就绪 |
| IF | `domains/ifbench/` | `domains/ifbench/data/IFBench_test.parquet` | 与同级 GRPO workspace 对齐的 verl validation 路径；用 `scripts/prepare_m2rl_eval_data.sh` 生成 |
| Science | `domains/science/` | `domains/science/data/gpqa.parquet` | 与同级 GRPO workspace 对齐的 verl validation 路径；用 `scripts/prepare_m2rl_eval_data.sh` 生成 |
| GReasoner | `domains/greasoner/` | `domains/greasoner/data/official/{MMLU-Pro,GPQA-D,SuperGPQA,TheoremQA,BBEH}/test.parquet` | 已接 General-Reasoner 论文五个 benchmark；WebInstructVerified 仅用于训练/verl validation |
| ToolRL | `domains/toolrl/` | `domains/toolrl/data/{BFCL,API-Bank,Bamboogle}/test.parquet` | 已接 API-Bank / BFCL / Bamboogle wrapper；BFCL 需要外部 harness，Bamboogle optional paid |

SearchQA 仍保留在 `domains/search/`，因为 thinking evaluator 可以继续包含
`data/SearchQA/test.parquet`。不过 SearchQA 不是这次整理出的四个核心 eval
domain 之一。

## 数据准备

从 G-OPD checkout 准备 Math / Code paper-eval 数据：

```bash
eval/scripts/prepare_paper_eval_data.sh
```

下载 General-Reasoner 论文评测数据：

```bash
python -m eval.domains.greasoner.download_official_data --force
```

这会准备：

- `MMLU-Pro`
- `GPQA-D`
- `SuperGPQA`
- `TheoremQA`
- `BBEH`

准备 General-Reasoner / WebInstructVerified 训练或 verl validation subset：

```bash
python -m eval.domains.greasoner.prepare_data \
  --from-hf \
  --output-dir eval/domains/greasoner/data/WebInstructVerified \
  --max-samples 100
```

将本地 ToolRL JSONL 暂存为 verl eval parquet：

```bash
python -m eval.domains.toolrl.prepare_data \
  --dataset BFCL \
  --input /path/to/bfcl.jsonl \
  --output eval/domains/toolrl/data/BFCL/test.parquet
```

准备与同级 GRPO workspace 对齐的 M2RL IF / Science validation parquet：

```bash
IF_VAL_SOURCE=/path/to/raw_if_val.parquet \
SCIENCE_VAL_SOURCE=/path/to/raw_science_val.parquet \
  scripts/prepare_m2rl_eval_data.sh
```

也可以从 Nemotron RL JSONL 中过滤一个 validation subset：

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
- `LiveCodeBench`: 默认使用 paper 代码中的 `Qwen3NonThinking` prompt 内容。

注意：`runner.py` 仍然会根据 eval mode 控制 tokenizer 的 `enable_thinking`：

- `thinking`: `enable_thinking=True`
- `non_thinking`: `enable_thinking=False`

也就是说，数据里的用户侧 problem instruction 已经对齐 paper；thinking /
non-thinking 的对比仍由当前 runner 显式控制。

## Thinking-Mode Validation

运行 Qwen thinking / non-thinking 对比：

```bash
eval/scripts/run_qwen3_thinking_validation.sh
```

常用环境变量：

- `MODEL_PATH=/path/to/model`
- `MAX_SAMPLES_PER_DATASET=8`
- `INCLUDE_GREASONER=0`
- `INCLUDE_TOOLRL=0`
- `INCLUDE_SEARCH=0`
- `BACKEND=hf` 或 `BACKEND=vllm`

输出会写入：

```text
eval/results/<RUN_ID>/
```

主要输出文件：

- `thinking_eval_samples.jsonl`
- `thinking_eval_summary.json`
- `thinking_eval_summary.csv`
- `records.jsonl`
- `README.md`

## Scoring

- Math 和 GReasoner 使用 boxed-answer 风格 scoring；如果项目 reward router
  可用，则走原项目 reward。
- Code 通过 vendored verl reward router 调用 `mopd_verl/code_reward.py`。
- IF / Science validation 与训练共用同一条 verl reward 路径：
  `grpo/rewards/mixed.py` 将 `m2rl_ifbench` 路由到 IFBench /
  verifiable-instructions strict scoring，将 `m2rl_gpqa` 路由到 GPQA
  option-letter scoring。
- ToolRL parquet 数据可以被加载，用于 token / cost 报告。
- ToolRL 官方 benchmark wrapper 已支持 API-Bank 本地 scoring、BFCL handler launcher、
  以及 Bamboogle search + judge scoring。

## 训练配置引用

MOPD 配置已经将 validation path 指向当前目录：

- `configs/mopd_qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_*.yaml`
- `grpo/configs/m2rl_if.yaml`
- `grpo/configs/m2rl_science.yaml`
- `grpo/configs/m2rl_if_science_mix.yaml`

训练数据仍保留在 `data/G-OPD-Training-Data/`，不会和 `eval/` 下的评测数据混在一起。

## 推荐使用方式

1. 先运行 `eval/scripts/prepare_paper_eval_data.sh` 准备或刷新 Math / Code eval parquet。
2. 再运行 `eval/scripts/run_qwen3_thinking_validation.sh` 做 thinking / non-thinking 对比。
3. 用 `eval/report.py` 或输出目录里的 `README.md` 查看聚合指标和样本级记录。

如果刚修改过 prompt builder，需要重新运行数据准备脚本，否则已有 parquet 里仍可能保留旧 prompt。

## 官方 Benchmark Eval

除了 parquet-based thinking evaluator，`eval/` 也提供官方 benchmark wrapper：

```bash
eval/scripts/run_official_eval.sh \
  --domains greasoner toolrl \
  --datasets mmlupro api_bank \
  --model-path /path/to/model \
  --output-dir eval/results/official_smoke
```

可选择的 domain：

- `greasoner`
- `toolrl`

可选择的 dataset：

- GReasoner: `mmlupro`, `gpqa_d`, `supergpqa`, `theoremqa`, `bbeh`
- ToolRL: `api_bank`, `bfcl`, `bamboogle`
- `all`: 运行所选 domain 下所有 dataset

示例：只跑 General-Reasoner 论文五个 benchmark：

```bash
eval/scripts/run_official_eval.sh \
  --domains greasoner \
  --datasets mmlupro gpqa_d supergpqa theoremqa bbeh \
  --model-path /path/to/model \
  --tensor-parallel-size 4 \
  --judge-base-url "$OPENAI_BASE_URL" \
  --judge-api-key "$OPENAI_API_KEY"
```

其中 `theoremqa` 是 open-ended QA，按论文评测逻辑需要 judge/equality API。

专门的 Qwen3-4B non-thinking 官方评测启动脚本：

```bash
scripts/run_qwen3_4b_nonthinking_official_eval.sh
```

脚本会自动 source 根目录 `api.sh`，并兼容：

- `dashscope_ak` -> `OPENAI_API_KEY`
- `dashscope_baseurl` -> `OPENAI_BASE_URL`
- `model` -> `JUDGE_MODEL`

默认运行：

- GReasoner: `mmlupro gpqa_d supergpqa theoremqa bbeh`
- ToolRL: `api_bank`

可用环境变量覆盖 `DOMAINS`、`DATASETS`、`MODEL_PATH`、`MAX_SAMPLES`、
`OUTPUT_DIR`、`API_BANK_LEVELS`。

示例：只跑 ToolRL API-Bank：

```bash
eval/scripts/run_official_eval.sh \
  --domains toolrl \
  --datasets api_bank \
  --model-path /path/to/model \
  --api-bank-dir ../temp/grpo_sources/ToolRL/benchmarks/API-Bank
```

BFCL 当前提供 ToolRL/RLLA handler 和外部 harness launcher。运行时需要提供
BFCL 官方 harness 命令：

```bash
eval/scripts/run_official_eval.sh \
  --domains toolrl \
  --datasets bfcl \
  --model-path /path/to/model \
  --bfcl-command "python -m bfcl_eval ..."
```

该 launcher 会向外部命令注入：

- `BFCL_MODEL_PATH`
- `BFCL_OUTPUT_DIR`
- `BFCL_HANDLER`
- `BFCL_API_BASE_URL`
- `BFCL_API_KEY`

Bamboogle 需要搜索 API 和 judge API，均支持自定义 endpoint：

```bash
eval/scripts/run_official_eval.sh \
  --domains toolrl \
  --datasets bamboogle \
  --model-path /path/to/model \
  --serper-base-url https://google.serper.dev/search \
  --serper-api-key "$SERPER_API_KEY" \
  --judge-base-url "$OPENAI_BASE_URL" \
  --judge-api-key "$OPENAI_API_KEY" \
  --judge-model gpt-4o
```
