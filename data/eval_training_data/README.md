# Training-domain evaluation holdouts

Generated with:

```bash
python scripts/split_domain_eval_training_data.py \
  --eval-size 1000 \
  --seed 42 \
  --write-remainders
```

The four ignored parquet files under `math/`, `code/`, `if/`, and `science/`
contain 1,000 rows each. `manifest.json` records source hashes, row counts,
selected prompt-group hashes, and output hashes.

Prompts are whitespace-normalized and casefolded before grouping. This keeps
the 166 duplicated Code problems spanning taco/codecontests on one side of the
split. The original source parquet files are not modified.

For a leakage-free experiment, train against the corresponding files in
`data/training_data_split/`, not the original full training files.
