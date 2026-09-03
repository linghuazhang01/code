# Math-only online token-selector configs

All configs use Qwen3-1.7B, teacher Top-32 reverse-KL training, fixed token
weight 4, and the strict source gate `occurrence >20` at every source-window
step. The default resource profile uses 4 GPUs (3 actor/rollout + 1
ref/teacher) and global batch size 255. The ExpandedPruned-V2 and V3 5-GPU
overlays use 4 actor/rollout GPUs + 1 ref/teacher GPU and global batch size 256.
The FullTaxonomy dual-teacher profiles use 3/4/5/6 actor GPUs for 5/6/7/8 total
GPUs, with the 30B teacher FSDP-sharded across the two-GPU ref-policy pool. The
offline-optimum configs use Metric A (`top_logp_diff`); the explicitly named
Top-32-KL variants use `top_loss`.

The three compact candidate pools use the Math-domain effective intersections
recorded in `candidate-pool-membership.csv`. FullTaxonomy uses the frozen Math
subset in `domain-subsets.csv`: 124 Control and 266 Structure token IDs.

| Target used to tune parameters | Candidate pool | Effective Math pool | Runtime parameters | Config |
|---|---|---:|---|---|
| next-step | ExpandedPruned-V2 | 89 | i1 / w1 / K29 | `a_next_step_expanded_pruned_v2_i1_w1_k29_4gpu_b255.yaml` |
| next-step, 5-GPU resource overlay | ExpandedPruned-V2 | 89 | i1 / w1 / K29 | `a_next_step_expanded_pruned_v2_i1_w1_k29_5gpu_b256.yaml` |
| next-step | Robust190 | 66 | i1 / w1 / K27 | `a_next_step_robust190_i1_w1_k27_4gpu_b255.yaml` |
| next-step | Control-44 | 40 | i1 / w1 / K8 | `a_next_step_control44_i1_w1_k8_4gpu_b255.yaml` |
| next-step | FullTaxonomy split | 124 Control + 266 Structure | i3 / w1 / K21 per type (42 total) | `a_next_step_full_taxonomy_split_i3_w1_k21_per_type_4gpu_b255.yaml` |
| next-step, Top-32 KL selector variant | FullTaxonomy split | 124 Control + 266 Structure | i1 / w1 / K21 per type (42 total) | `top32kl_next_step_full_taxonomy_split_i1_w1_k21_per_type_4gpu_b255.yaml` |
| next-step, Top-32 KL, 5-GPU Top-P variant | FullTaxonomy split | 124 Control + 266 Structure | i1 / w1 / 5% valid-token occurrence coverage | `top32kl_next_step_full_taxonomy_split_topp0p05_i1_w1_5gpu_b256.yaml` |
| next-step, Top-32 KL, 5-GPU dual-teacher Avg@4 | FullTaxonomy split | 124 Control + 266 Structure | i1 / w1 / 10% valid-token occurrence coverage | `top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_5gpu_3a2t_b258.yaml` |
| next-step, Top-32 KL, 6-GPU dual-teacher | FullTaxonomy split | 124 Control + 266 Structure | i1 / w1 / 10% valid-token occurrence coverage | `top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_6gpu_4a2t_b256.yaml` |
| next-step, Top-32 KL, 7-GPU dual-teacher | FullTaxonomy split | 124 Control + 266 Structure | i1 / w1 / 10% valid-token occurrence coverage | `top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_7gpu_5a2t_b255.yaml` |
| next-step, Top-32 KL, 8-GPU dual-teacher Avg@4 | FullTaxonomy split | 124 Control + 266 Structure | i1 / w1 / 10% valid-token occurrence coverage | `top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_8gpu_6a2t_b258.yaml` |
| next-step, Top-32 KL, 6-GPU dual-teacher | FullTaxonomy split | 124 Control + 266 Structure | i1 / w1 / 20% valid-token occurrence coverage | `top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_6gpu_4a2t_b256.yaml` |
| next-step, Top-32 KL, 7-GPU dual-teacher | FullTaxonomy split | 124 Control + 266 Structure | i1 / w1 / 20% valid-token occurrence coverage | `top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_7gpu_5a2t_b255.yaml` |
| next-step, Top-32 KL, 8-GPU dual-teacher | FullTaxonomy split | 124 Control + 266 Structure | i1 / w1 / 20% valid-token occurrence coverage | `top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_8gpu_6a2t_b258.yaml` |
| next-window | ExpandedPruned-V2 | 89 | i6 / w6 / K30 | `a_next_window_expanded_pruned_v2_i6_w6_k30_4gpu_b255.yaml` |
| next-window | Robust190 | 66 | i6 / w6 / K30 | `a_next_window_robust190_i6_w6_k30_4gpu_b255.yaml` |
| next-window | Control-44 | 40 | i7 / w7 / K8 | `a_next_window_control44_i7_w7_k8_4gpu_b255.yaml` |
| next-window | FullTaxonomy split | 124 Control + 266 Structure | i6 / w6 / K32 per type (64 total) | `a_next_window_full_taxonomy_split_i6_w6_k32_per_type_4gpu_b255.yaml` |

The FullTaxonomy rows reproduce the four-baseline offline Type-F1 optima. The
split metrics are macro-averaged over the Control/Structure cells, so the
reported mean selected count is K per type rather than the combined runtime
budget. The Top-32-KL selector variant uses the requested per-step i1/w1/K21
schedule and is not presented as a KL-grid optimum.

The 5- and 8-GPU 10% FullTaxonomy profiles use stochastic validation Avg@4 and
the bounded batched reward scorer. The 6- and 7-GPU 10% profiles retain Avg@1
and the single-sample scorer, so they are not validation-equivalent resource
variants of the 5- and 8-GPU profiles.

## ExpandedPruned-V3 unified configs

The V3 fixed-Top-K base configs contain the complete resolved YAML and have no
`extends` key. They use the Math effective taxonomy intersection: 115 unified
candidate IDs (85 Control + 30 Structure). Metric A is `top_logp_diff`, and
selected tokens receive raw weight 4 with per-domain normalization.

| Target | Offline optimum | Config filename prefix |
|---|---|---|
| next-step | i1 / w1 (pre-update source) / K25 total | `a_next_step_expanded_pruned_v3_unified_i1_w1_k25_` |
| next-window | i7 / w7 / K41 total | `a_next_window_expanded_pruned_v3_unified_i7_w7_k41_` |

Both targets provide the following resource variants:

| Total GPUs | Actor + teacher | Global batch | Filename suffix |
|---:|---:|---:|---|
| 4 | 3 + 1 | 255 | `4gpu_3a1t_b255.yaml` |
| 5 | 4 + 1 | 256 | `5gpu_4a1t_b256.yaml` |
| 6 | 4 + 2 | 256 | `6gpu_4a2t_b256.yaml` |
| 7 | 5 + 2 | 255 | `7gpu_5a2t_b255.yaml` |
| 8 | 6 + 2 | 258 | `8gpu_6a2t_b258.yaml` |

Every 4--8 GPU next-step profile also provides unified Top-P occurrence-budget
overlays. Each overlay keeps `i1 / w1`, Metric A, the 115-token V3 pool, and
the fixed selected-token weight; it replaces the active fixed `K=25` budget
with the smallest ranked prefix reaching the requested valid-token occurrence
share when eligible candidate coverage is sufficient. Shortfalls are logged.
The inherited positive K25 field remains only for schema/state compatibility
and does not cap Top-P selection.

| Target occurrence share | Config filename prefix |
|---:|---|
| 1% | `a_next_step_expanded_pruned_v3_unified_topp0p01_i1_w1_` |
| 2% | `a_next_step_expanded_pruned_v3_unified_topp0p02_i1_w1_` |
| 5% | `a_next_step_expanded_pruned_v3_unified_topp0p05_i1_w1_` |
| 10% | `a_next_step_expanded_pruned_v3_unified_topp0p1_i1_w1_` |

Append one of the five topology suffixes in the resource table above to form
the complete filename. This gives 20 Top-P overlays in total.

The 6--8 GPU variants add `actor_rollout_ref.ref.fsdp_config.fsdp_size=2`
because the teacher pool spans two GPUs. Slurm host memory is a launcher
setting rather than a YAML field; use the project rule of 100G per allocated
GPU when submitting these configs. All ten fixed-Top-K profiles and all twenty
Top-P overlays use the bounded batched Math reward scorer (32 workers with a
120-second batch deadline).

### Historical FullTaxonomy replay reference

The following metrics belong to the earlier FullTaxonomy split configs, not
ExpandedPruned-V3:

| Target | Precision | Pool Recall | Pool F1 | Type Recall | Type F1 | Mean selection share |
|---|---:|---:|---:|---:|---:|---:|
| next-step, i3/w1/K21 per type | 0.2830 | 0.5282 | 0.3646 | 0.4743 | 0.3504 | 15.71% |
| next-window, i6/w6/K32 per type | 0.2949 | 0.5885 | 0.3891 | 0.4889 | 0.3658 | not recorded in this replay table |

`next-step` and `next-window` name the offline target used to tune the
parameters. At runtime, every selection is still lagged and first applies at
the following optimizer step. The next-window configs simply retain the
selected set until the next six- or seven-step audit boundary.

The runtime budget defaults to fixed `top_k`. To select the smallest ranked
prefix whose observed occurrence counts reach a valid-token fraction instead,
override:

```yaml
audit:
  control_token_online_budget_mode: top_p
  control_token_online_top_p: 0.8
```

Here `top_p` targets the occurrence fraction already reported as
`online_control_occurrence_fraction`. After ranking token types, the selector
accumulates their observed occurrence counts until
`selected_occurrences / valid_token_count >= top_p`. Grouped taxonomy entries
form one domain candidate union for Top-P rather than independent group quotas;
`control_token_online_top_k_per_group` must be omitted or set to `null`. Top-P
is therefore not the percentage of candidate token types and not cumulative
selection-score mass. Adding a complete token type can overshoot the requested
share, while insufficient eligible coverage is reported as a shortfall.

Launch a resolved profile directly through `start.sh` in non-Slurm mode. For
example, the V3 4-GPU next-step profile is:

```bash
GPU_IDS=0,1,2,3 bash start.sh --local --config configs/token_selection/math/a_next_step_expanded_pruned_v3_unified_i1_w1_k25_4gpu_3a1t_b255.yaml
```

Additional FullTaxonomy examples:

```bash
GPU_IDS=0,1,2,3,4,5 bash start.sh --local --config configs/token_selection/math/top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_6gpu_4a2t_b256.yaml
GPU_IDS=0,1,2,3,4,5,6 bash start.sh --local --config configs/token_selection/math/top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_7gpu_5a2t_b255.yaml
GPU_IDS=0,1,2,3,4,5,6,7 bash start.sh --local --config configs/token_selection/math/top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_8gpu_6a2t_b258.yaml

GPU_IDS=0,1,2,3,4,5 bash start.sh --local --config configs/token_selection/math/top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_6gpu_4a2t_b256.yaml
GPU_IDS=0,1,2,3,4,5,6 bash start.sh --local --config configs/token_selection/math/top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_7gpu_5a2t_b255.yaml
GPU_IDS=0,1,2,3,4,5,6,7 bash start.sh --local --config configs/token_selection/math/top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_8gpu_6a2t_b258.yaml
```
