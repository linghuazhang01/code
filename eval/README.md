# OPD Evaluation

This directory contains OPD evaluation implementations, data, and run artifacts.
The only user-facing evaluation launch entrypoint is:

```bash
scripts/run_local_eval.sh --model-path /path/to/model [options]
```

Run it from `code/`. Do not invoke `python -m eval.runner` or model-evaluation
scripts under `eval/scripts/` directly; they are internal implementation details.
Data-preparation utilities under `eval/scripts/` may still be run separately.

## Layout

- `runner.py`: thinking/non-thinking model evaluator.
- `common.py`: shared parquet loading, token accounting, and summarization.
- `report.py`: JSON/Markdown report generation for completed or live runs.
- `paper_eval.py`: runtime hook used by patched verl validation.
- `data_prep/`: JSONL-to-parquet conversion code for paper-eval datasets.
- `domains/`: domain-specific metadata, preparation scripts, and eval data.
- `scripts/`: internal and legacy evaluation helpers; not public launch entrypoints.
- `../data/eval_data/results/`: outputs from the public local-eval entrypoint.

## Domains

| Domain | Code | Eval data | Status |
|---|---|---|---|
| Math | `domains/math/` | `../data/eval_data/math/{AIME24,AIME25,HMMT25Feb,HMMT25Nov}/test.parquet` | Ready |
| Code | `domains/code/` | `../data/eval_data/code/{HumanEvalPlus,MBPPPlus,LiveCodeBench}/test.parquet` | HumanEvalPlus/MBPPPlus ready; generate LiveCodeBench with `prepare_paper_eval_data.sh` |
| IF | `domains/ifbench/` | `../data/eval_data/ifbench/IFBench_test.parquet` | verl validation path; generate with `scripts/prepare_m2rl_eval_data.sh` |
| Science | `domains/science/` | `../data/eval_data/science/gpqa.parquet` | verl validation path; generate with `scripts/prepare_m2rl_eval_data.sh` |
| GReasoner | `domains/greasoner/` | `../data/eval_data/greasoner/official/{MMLU-Pro,GPQA-D,SuperGPQA,TheoremQA,BBEH}/test.parquet` | Data/internal evaluators exist; official datasets are not yet exposed by `run_local_eval.sh` |
| ToolRL | `domains/toolrl/` | `../data/eval_data/toolrl/{BFCL,API-Bank,Bamboogle}/test.parquet` | Data/internal evaluators exist; ToolRL datasets are not yet exposed by `run_local_eval.sh` |

SearchQA support remains in `domains/search/` because the thinking evaluator can
still include `data/SearchQA/test.parquet`, but SearchQA is not one of the four
domains requested for this eval layout.

## Preparing Data

Math/code paper-eval data from a G-OPD checkout:

```bash
eval/scripts/prepare_paper_eval_data.sh
```

This pins LiveCodeBench `v6` (`test6.jsonl`, 175 incremental problems), not the
1,055-problem cumulative `release_v6`. For G-OPD's official public+private test
protocol, use `eval/scripts/run_paper_eval_suite.sh`. The generated LiveCodeBench
parquet is intentionally ignored by Git because it contains the full private
test payload; `manifest.json` records its pinned revision and source checksum.

Create deterministic four-domain training holdouts and matching train
remainders without modifying the original parquet files:

```bash
python scripts/split_domain_eval_training_data.py --write-remainders
```

Eval files are written to `data/eval_training_data/<domain>/test.parquet`, and
remainders to `data/training_data_split/<domain>/train.parquet`. Future training
configs must use the remainders for the eval data to be leakage-free.

General-Reasoner paper eval data:

```bash
python -m eval.domains.greasoner.download_official_data --force
```

This prepares `MMLU-Pro`, `GPQA-D`, `SuperGPQA`, `TheoremQA`, and `BBEH`.

General-Reasoner/WebInstruct training or verl validation subset:

```bash
python -m eval.domains.greasoner.prepare_data \
  --from-hf \
  --output-dir data/eval_data/greasoner/WebInstructVerified \
  --max-samples 100
```

ToolRL local JSONL staging:

```bash
python -m eval.domains.toolrl.prepare_data \
  --dataset BFCL \
  --input /path/to/bfcl.jsonl \
  --output data/eval_data/toolrl/BFCL/test.parquet
```

Prepare M2RL IF/science validation data:

```bash
IF_VAL_SOURCE=/path/to/raw_if_val.parquet \
SCIENCE_VAL_SOURCE=/path/to/raw_science_val.parquet \
  scripts/prepare_m2rl_eval_data.sh
```

or from the Nemotron RL JSONL blend:

```bash
NEMOTRON_RL_SOURCE=/path/to/instruction_following.jsonl \
M2RL_EVAL_MAX_SAMPLES=512 \
  scripts/prepare_m2rl_eval_data.sh
```

## Running Evaluation

Run all local evaluations through the single public entrypoint:

```bash
scripts/run_local_eval.sh \
  --model-path ../models/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
  --datasets aime24,humaneval_plus,ifeval,gpqa_diamond \
  --modes non_thinking \
  --max-samples 8 \
  --save-completions
```

Important options:

- `--datasets`: comma-separated dataset keys.
- `--modes`: `non_thinking`, `thinking`, or both as a comma-separated list.
- `--max-samples`: maximum examples per dataset.
- `--max-new-tokens`: generation limit for every selected mode.
- `--num-samples`, `--temperature`, `--top-p`, `--seed`: sampling controls.
- `--backend transformers|vllm`: inference backend.
- `--tensor-parallel-size`, `--batch-size`, `--gpu-memory`: vLLM controls.
- `--score-code`: execute generated code for Code scoring; use only in isolation.
- `--save-completions`: retain full completions.
- `--dry-run`: validate inputs and print the resolved command without running it.

Supported dataset keys are `aime24`, `aime25`, `hmmt25feb`, `hmmt25nov`,
`humaneval_plus`, `mbpp_plus`, `livecodebench`, `ifeval`, `ifbench`, and
`gpqa_diamond`.

For example, compare thinking and non-thinking with vLLM:

```bash
CUDA_VISIBLE_DEVICES=0,1 scripts/run_local_eval.sh \
  --model-path /path/to/model \
  --datasets aime24,gpqa_diamond \
  --modes non_thinking,thinking \
  --backend vllm \
  --tensor-parallel-size 2 \
  --batch-size 8
```

Outputs are written to `data/eval_data/results/<RUN_ID>/`:

- `thinking_eval_samples.jsonl`
- `thinking_eval_summary.json`
- `thinking_eval_summary.csv`
- `records.jsonl`
- `README.md`

## Scoring

- Math and GReasoner use boxed-answer style scoring through the project reward
  router when available.
- Code uses `mopd_verl/code_reward.py` through the vendored verl reward router.
- IF/science validation uses the same verl reward path as training:
  `mopd_verl/mixed_reward.py` routes `m2rl_ifbench` to IFBench/verifiable-instructions
  strict scoring and `m2rl_gpqa` to GPQA option-letter scoring.
- ToolRL parquet data is loadable for cost/token reports.
- ToolRL official benchmark wrappers support API-Bank local scoring, the BFCL
  handler launcher, and Bamboogle search + judge scoring.

## Training Config References

MOPD configs now point validation paths to this directory:

- `configs/mopd_qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_*.yaml`

Training data remains under `data/G-OPD-Training-Data/` and is intentionally not
mixed with eval data.

## Internal Evaluators

`eval/runner.py`, `eval/official_runner.py`, and model-evaluation scripts under
`eval/scripts/` remain implementation modules for development and compatibility.
They are not supported as independent user launch entrypoints. Data-preparation
utilities are not launch entrypoints and remain directly usable. Extend
`scripts/run_local_eval.sh` first when exposing another benchmark or evaluation
behavior.
