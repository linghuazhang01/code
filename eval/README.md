# OPD Evaluation

This directory is the single home for OPD evaluation code, evaluation data, and
evaluation run artifacts.

## Layout

- `runner.py`: thinking/non-thinking model evaluator.
- `common.py`: shared parquet loading, token accounting, and summarization.
- `report.py`: JSON/Markdown report generation for completed or live runs.
- `paper_eval.py`: runtime hook used by patched verl validation.
- `data_prep/`: JSONL-to-parquet conversion code for paper-eval datasets.
- `domains/`: domain-specific metadata, preparation scripts, and eval data.
- `scripts/`: launch wrappers for local/remote evaluation.
- `results/`: local evaluation outputs.

Root-level `scripts/*eval*.sh` are compatibility wrappers. New eval code should
go under `eval/`.

## Domains

| Domain | Code | Eval data | Status |
|---|---|---|---|
| Math | `domains/math/` | `domains/math/data/{AIME24,AIME25,HMMT25Feb,HMMT25Nov}/test.parquet` | Ready |
| Code | `domains/code/` | `domains/code/data/{HumanEvalPlus,MBPPPlus,LiveCodeBench}/test.parquet` | Ready |
| IF | `domains/ifbench/` | `domains/ifbench/data/IFBench_test.parquet` | GRPO-aligned verl validation path; generate with `scripts/prepare_m2rl_eval_data.sh` |
| Science | `domains/science/` | `domains/science/data/gpqa.parquet` | GRPO-aligned verl validation path; generate with `scripts/prepare_m2rl_eval_data.sh` |
| GReasoner | `domains/greasoner/` | `domains/greasoner/data/official/{MMLU-Pro,GPQA-D,SuperGPQA,TheoremQA,BBEH}/test.parquet` | General-Reasoner paper benchmarks ready; WebInstructVerified is only for training/verl validation |
| ToolRL | `domains/toolrl/` | `domains/toolrl/data/{BFCL,API-Bank,Bamboogle}/test.parquet` | API-Bank / BFCL / Bamboogle wrappers ready; BFCL needs the external harness, Bamboogle is optional paid eval |

SearchQA support remains in `domains/search/` because the thinking evaluator can
still include `data/SearchQA/test.parquet`, but SearchQA is not one of the four
domains requested for this eval layout.

## Preparing Data

Math/code paper-eval data from a G-OPD checkout:

```bash
eval/scripts/prepare_paper_eval_data.sh
```

General-Reasoner paper eval data:

```bash
python -m eval.domains.greasoner.download_official_data --force
```

This prepares `MMLU-Pro`, `GPQA-D`, `SuperGPQA`, `TheoremQA`, and `BBEH`.

General-Reasoner/WebInstruct training or verl validation subset:

```bash
python -m eval.domains.greasoner.prepare_data \
  --from-hf \
  --output-dir eval/domains/greasoner/data/WebInstructVerified \
  --max-samples 100
```

ToolRL local JSONL staging:

```bash
python -m eval.domains.toolrl.prepare_data \
  --dataset BFCL \
  --input /path/to/bfcl.jsonl \
  --output eval/domains/toolrl/data/BFCL/test.parquet
```

M2RL IF/science validation data, aligned with the sibling GRPO workspace:

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

## Thinking-Mode Validation

Run the Qwen thinking/non-thinking comparison:

```bash
eval/scripts/run_qwen3_thinking_validation.sh
```

Useful switches:

- `MODEL_PATH=/path/to/model`
- `MAX_SAMPLES_PER_DATASET=8`
- `INCLUDE_GREASONER=0`
- `INCLUDE_TOOLRL=0`
- `INCLUDE_SEARCH=0`
- `BACKEND=hf` or `BACKEND=vllm`

Outputs are written to `eval/results/<RUN_ID>/`:

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
  `grpo/rewards/mixed.py` routes `m2rl_ifbench` to IFBench/verifiable-instructions
  strict scoring and `m2rl_gpqa` to GPQA option-letter scoring.
- ToolRL parquet data is loadable for cost/token reports.
- ToolRL official benchmark wrappers support API-Bank local scoring, the BFCL
  handler launcher, and Bamboogle search + judge scoring.

## Training Config References

MOPD configs now point validation paths to this directory:

- `configs/mopd_qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_*.yaml`
- `grpo/configs/m2rl_if.yaml`
- `grpo/configs/m2rl_science.yaml`
- `grpo/configs/m2rl_if_science_mix.yaml`

Training data remains under `data/G-OPD-Training-Data/` and is intentionally not
mixed with eval data.

## Official Benchmark Wrappers

Standalone official benchmark wrappers are available through:

```bash
eval/scripts/run_official_eval.sh \
  --domains greasoner toolrl \
  --datasets mmlupro api_bank \
  --model-path /path/to/model
```

Supported datasets:

- GReasoner: `mmlupro`, `gpqa_d`, `supergpqa`, `theoremqa`, `bbeh`
- ToolRL: `api_bank`, `bfcl`, `bamboogle`
- `all`: every dataset under the selected domains

External API configuration is passed through CLI flags or environment variables:

- `--judge-base-url`, `--judge-api-key`, `--judge-model` for Bamboogle judge.
- `--serper-base-url`, `--serper-api-key` for Bamboogle search.
- `--api-base-url`, `--api-key` for BFCL external harness integration.

BFCL ships as a configured RLLA handler plus launcher because the upstream file
is a handler for the BFCL harness, not a complete standalone runner.

`theoremqa` is open-ended and requires a judge/equality API through
`--judge-base-url`, `--judge-api-key`, and `--judge-model` for paper-aligned
scoring.
