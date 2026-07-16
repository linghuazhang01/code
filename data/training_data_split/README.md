# Training remainders

These ignored parquet files are the complements of the 1,000-example domain
holdouts in `data/eval_training_data/`:

| Domain | Train remainder rows |
|---|---:|
| Math | 56,046 |
| Code | 24,276 |
| IF | 15,575 |
| Science | 18,670 |

They preserve each source parquet's Arrow schema. See
`data/eval_training_data/manifest.json` for source/output SHA-256 values and
the exact deterministic split metadata.
