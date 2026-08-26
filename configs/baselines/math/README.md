# Math-only OPD baselines (4-8 GPUs, global batch approximately 256)

These configs compare four implemented baselines at two student scales using
the same Math data, Qwen3-30B-A3B-Instruct-2507 teacher, optimizer, rollout,
evaluation protocol, and 200-step budget.

## Model variants

- The 20 configs directly under this directory use a Qwen3-1.7B student.
- The 20 matching configs under `qwen4b/` use a Qwen3-4B student and inherit
  the corresponding 1.7B method and resource profile.

Keeping the resource and batch profiles matched isolates student scale as the
intended difference. The Qwen3-4B profiles should still receive a short GPU
smoke test on the target H200 cluster before launching the full sweep.

## Resource profiles

| Allocation | Actor + teacher GPUs | Global batch | Samples / actor GPU |
|---:|---:|---:|---:|
| 4 | 3 + 1 | 255 | 85 |
| 5 | 4 + 1 | 256 | 64 |
| 6 | 5 + 1 | 255 | 51 |
| 7 | 6 + 1 | 258 | 43 |
| 8 | 7 + 1 | 259 | 37 |

The one-GPU teacher profile assumes an H200-class GPU, matching the current
CityU deployment. `trainer.n_gpus_per_node` is the actor data-parallel world
size, not the total Slurm allocation. Every batch is the closest integer to
256 that is divisible by that actor world size.

## Methods

- `opd`: native chosen-token OPD policy-gradient objective.
- `fire_opd`: FiRE-OPD token weighting plus exact bottom-20% trajectory
  filtering on the native OPD objective.
- `tip_topk32`: scalable TIP Soft-OR selection at `rho=0.5`, using teacher
  Top-32 renormalized reverse KL for disagreement and training.
- `eopd`: OPD plus the `tau=0.8`, `alpha=1.0`, teacher Top-16 entropy-gated
  forward-KL term.

Native full-vocabulary TIP is intentionally not expanded to batch 256. Its
canonical memory-safe config uses batch 24 and a colocated sequential
teacher/student topology; scaling its cached full-vocabulary logits by about
10x is not a credible H200 launch profile.

## Launch examples

```bash
# Validate the Qwen3-1.7B variant.
bash scripts/run_mopd.sh \
  configs/baselines/math/opd_5gpu_b256.yaml --dry-run

# Validate the matching Qwen3-4B variant.
bash scripts/run_mopd.sh \
  configs/baselines/math/qwen4b/opd_5gpu_b256.yaml --dry-run

# Submit one Qwen3-4B run through Slurm.
bash scripts/run_mopd.sh \
  configs/baselines/math/qwen4b/fire_opd_8gpu_b259.yaml --slurm
```

All 40 user-facing configs follow `{method}_{gpu_count}gpu_b{batch}.yaml`
within their model-variant location.
The `_common.yaml` and `_methods/` files are inheritance fragments, not
separate experiment cells.

## Hugging Face models

All 40 user-facing configs force-save and upload loadable Hugging Face models at global steps
50, 55, 60, 65, and 70 to the private `icemoon28/opd-checkpoints` repository.
Each config uses a unique path under `checkpoints/math/`, so methods and GPU
profiles cannot overwrite one another. Authentication is read from the
exported `HF_TOKEN`; the raw token is never stored in these YAML files. Only
the contents of `actor/huggingface/` are uploaded; optimizer, scheduler,
dataloader, critic, and other restart-only state remain local. Uploads run in a
single background queue while training continues; the trainer waits for the
queue only when training exits. Failed model-only snapshots are retried on the
next launch with the same config.
