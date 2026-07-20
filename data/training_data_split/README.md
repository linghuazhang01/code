# Training remainders

These ignored parquet files are the original complements of the earlier
1,000-example domain holdouts. They remain unchanged when
`data/eval_training_data/` is refreshed with the 10,000-example
training-data performance-ceiling sample:

| Domain | Train remainder rows |
|---|---:|
| Math | 56,046 |
| Code | 24,276 |
| IF | 15,575 |
| Science | 18,670 |

They preserve each source parquet's Arrow schema. The current
`data/eval_training_data/manifest.json` audits only the 10,000-example eval
sample and does not redefine these training files. The `training_ceiling`
local-eval dataset intentionally evaluates against the original training
distribution and does not consume these remainders.
