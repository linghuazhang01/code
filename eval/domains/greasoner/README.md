# GReasoner Eval Domain

This domain tracks General-Reasoner-style reasoning evaluation data.

## Model

- Teacher candidate: `TIGER-Lab/General-Reasoner-Qwen3-4B`
- Base family: `Qwen/Qwen3-4B`
- Thinking mode: enabled. This is not a Qwen3-Instruct non-thinking checkpoint.

## Paper Eval Data

General-Reasoner paper evaluation uses these benchmark datasets:

- `data/official/MMLU-Pro/test.parquet`
- `data/official/GPQA-D/test.parquet`
- `data/official/SuperGPQA/test.parquet`
- `data/official/TheoremQA/test.parquet`
- `data/official/BBEH/test.parquet`

Download them with the `datasets` package:

```bash
python -m eval.domains.greasoner.download_official_data --force
```

These files are local data artifacts and are ignored by git.

## Training / VERL Validation Data

`TIGER-Lab/WebInstruct-verified` is the General-Reasoner RL training dataset.
The upstream preprocessing script also creates a 100-sample `test.parquet` for
training-time validation; it is not the paper benchmark table.

Prepare the WebInstruct parquet with:

```bash
python -m eval.domains.greasoner.prepare_data \
  --from-hf \
  --output-dir data/eval_data/greasoner/WebInstructVerified \
  --max-samples 100
```

For a local JSON/JSONL/parquet test split:

```bash
python -m eval.domains.greasoner.prepare_data \
  --input /path/to/test.jsonl \
  --output data/eval_data/greasoner/WebInstructVerified/test.parquet \
  --split test \
  --max-samples 100
```

## Code

- `__init__.py`: dataset metadata.
- `prepare_data.py`: eval-focused wrapper around the General-Reasoner parquet
  converter.
- `download_official_data.py`: downloads the five paper benchmark datasets.
- `official_eval.py`: internal evaluators for official GReasoner benchmarks
  including MMLU-Pro, GPQA-D, SuperGPQA, TheoremQA, and BBEH.
- `mopd_verl/general_reasoner_data.py`: shared train/eval converter.

## Launch Policy

The only user-facing evaluation entrypoint is `scripts/run_local_eval.sh`.
The General-Reasoner official benchmark modules are internal implementations
and are not launched directly. Add their dataset routing to `run_local_eval.sh`
before exposing them as supported public evaluation modes.

The wrapper preserves the original benchmark prompts and scoring style while
making model path, tensor parallelism, output directory, sample cap, and
`enable_thinking` configurable.

`theoremqa` is an open-ended benchmark and its internal evaluator requires a
judge API for paper-aligned scoring.
