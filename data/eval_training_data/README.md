# Training-domain evaluation holdouts

Generated with:

```bash
python scripts/split_domain_eval_training_data.py \
  --eval-size 10000 \
  --seed 42 \
  --overwrite
```

The four ignored parquet files under `math/`, `code/`, `if/`, and `science/`
contain 10,000 rows each. `manifest.json` records source hashes, row counts,
selected prompt-group hashes, and output hashes.

Prompts are whitespace-normalized and casefolded before grouping. This keeps
the 166 duplicated Code problems spanning taco/codecontests together during
selection. The original source parquet files are not modified.

This dataset intentionally samples the training data to estimate a model's
training-data performance ceiling. Train/eval overlap is expected; do not use
it as a leakage-free generalization benchmark. The files in
`data/training_data_split/` are left unchanged.

Run all four domains through the public local-eval entrypoint with:

```bash
scripts/run_local_eval.sh \
  --model-path /path/to/model \
  --datasets training_ceiling
```

The per-domain keys are `training_math`, `training_code`, `training_if`, and
`training_science`. Code samples are generated without execution-based scoring
unless `--score-code` is explicitly enabled in an isolated environment with
the complete vendored verl reward dependencies. IF scoring requires either the
`verifiable_instructions` package installed by `scripts/setup_training_env.sh`
or an IFBench checkout prepared with `scripts/prepare_ifbench_runtime.sh`.
