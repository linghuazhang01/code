# Math4 re-evaluation at a 32768-token cap

The author chose this on 2026-11-01. It separates two readings of the Math-only 1.7B ignore / only scores: the checkpoint cannot solve the problem, or it did not finish inside the 16384-token cap. At 16k, list-12 only is truncated on 59% of samples and termlook on 46%, against 7% for list-12 ignore.

| arm | checkpoint (step 60) | Math4 at 16k |
|---|---|---:|
| `q1p7b_math_opdnative` | `checkpoints/math/q1p7b-math-opdnative-toklog-3g3c-b255` | 23.05 ± 0.73 |
| `q1p7b_math_l12ign` | `checkpoints/math/q1p7b-math-opdnative-l12ign-toklog-8g8c-b256-s42-20261020` | 19.48 ± 0.42 |
| `q1p7b_math_l12only` | `checkpoints/math/q1p7b-math-opdnative-l12only-toklog-8g8c-b256-s42-20261020` | 22.99 ± 1.56 |
| `q1p7b_math_termlook` | `checkpoints/math/q1p7b-math-opdnative-l12only-termlook-toklog-8g8c-b256-s42` | 19.58 ± 0.65 |

Each arm runs at evaluation seeds 42, 1042 and 2042, K=8, CUDA graphs, 12 suites in all. Seed 42 finishes first.

## Files

- `run_eval_jobs_a100_v12_math32k.sh`: single-A100 queue. It differs from the 16k runner in the cap (`max_new_tokens` 32768, `max_model_len` and `max_num_batched_tokens` 34816), the tag (`math4_32k_k8_s60_<arm>_a100_cg_s<seed>_20261101`) and in accepting math4 jobs only.
- `launch_math32k.sh`: starts the queue inside a RUNNING hold job. Several cards can share the list, because each suite takes a lock.
- `collect_math32k.py`: budget curves and paired contrasts. It uses the standard library only.

## Run

The runner and launcher read `OPD_EVAL_BASE`, default `$HOME/scratch/opd`. That directory needs the `mopd_code_a100mc` checkout, the `mopd-verl-a100` environment and the evaluation data. Copy the runner and the collector into it and the launcher into its `logs/`.

```bash
cd "$HOME/scratch/opd" && nohup bash logs/launch_math32k.sh <job> > logs/math32k_<job>.out 2>&1 &
```

```bash
cd "$HOME/scratch/opd" && python3 collect_math32k.py
```

A sample counts as correct within budget B only if it is correct and finished before B tokens. Math is the mean of the four dataset accuracies.

## Checked

- Plans for all three seeds match the 16k suites on all 64 shards (dataset ranges, K, generation seeds).
- On the 16k suites the collector gives 19.48, 22.99 and 19.58, with truncation 7%, 59% and 46%.

## Not checked

- The runner has not generated on a GPU at this cap.
- The Math-only naive 16k suites are no longer on the evaluation machine. Its 16k figure comes from samples finished within 16384 tokens in the 32k suite, not from a paired 16k suite.
