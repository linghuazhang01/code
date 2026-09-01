# Math-only online token-selector configs

All configs use Qwen3-1.7B, teacher Top-32 reverse-KL training, fixed token
weight 4, and the strict source gate `occurrence >20` at every source-window
step. The default resource profile uses 4 GPUs (3 actor/rollout + 1
ref/teacher) and global batch size 255. The explicitly named 5-GPU overlay uses
4 actor/rollout GPUs + 1 ref/teacher GPU and global batch size 256. The 6/7/8
GPU Top-P grid uses 4/5/6 actor GPUs plus 2 ref/teacher GPUs, with the 30B
teacher FSDP-sharded across the two-GPU ref-policy pool. The offline-optimum
configs use Metric A (`top_logp_diff`); the explicitly named Top-32-KL variants
use `top_loss`.

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
| next-step, Top-32 KL, 6-GPU dual-teacher | FullTaxonomy split | 124 Control + 266 Structure | i1 / w1 / 10% valid-token occurrence coverage | `top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_6gpu_4a2t_b256.yaml` |
| next-step, Top-32 KL, 7-GPU dual-teacher | FullTaxonomy split | 124 Control + 266 Structure | i1 / w1 / 10% valid-token occurrence coverage | `top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_7gpu_5a2t_b255.yaml` |
| next-step, Top-32 KL, 8-GPU dual-teacher | FullTaxonomy split | 124 Control + 266 Structure | i1 / w1 / 10% valid-token occurrence coverage | `top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_8gpu_6a2t_b258.yaml` |
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

| Target | Precision | Pool Recall | Pool F1 | Type Recall | Type F1 | Mean selection share |
|---|---:|---:|---:|---:|---:|---:|
| next-step, i3/w1/K21 per type | 0.2830 | 0.5282 | 0.3646 | 0.4743 | 0.3504 | 15.71% |
| next-window, i6/w6/K32 per type | 0.2949 | 0.5885 | 0.3891 | 0.4889 | 0.3658 | not recorded in this replay table |

`next-step` and `next-window` name the offline target used to tune the
parameters. At runtime, every selection is still lagged and first applies at
the following optimizer step. The next-window configs simply retain the
selected set until the next six- or seven-step audit boundary.

The runtime budget defaults to fixed `top_k`. To select the smallest ranked
prefix whose non-negative selection scores reach a cumulative fraction instead,
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
`control_token_online_top_k_per_group` must be omitted.

Launch any profile directly through `start.sh` in non-Slurm mode:

```bash
GPU_IDS=0,1,2,3,4,5 bash start.sh --local --config configs/token_selection/math/top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_6gpu_4a2t_b256.yaml
GPU_IDS=0,1,2,3,4,5,6 bash start.sh --local --config configs/token_selection/math/top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_7gpu_5a2t_b255.yaml
GPU_IDS=0,1,2,3,4,5,6,7 bash start.sh --local --config configs/token_selection/math/top32kl_next_step_full_taxonomy_split_topp0p1_i1_w1_8gpu_6a2t_b258.yaml

GPU_IDS=0,1,2,3,4,5 bash start.sh --local --config configs/token_selection/math/top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_6gpu_4a2t_b256.yaml
GPU_IDS=0,1,2,3,4,5,6 bash start.sh --local --config configs/token_selection/math/top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_7gpu_5a2t_b255.yaml
GPU_IDS=0,1,2,3,4,5,6,7 bash start.sh --local --config configs/token_selection/math/top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_8gpu_6a2t_b258.yaml
```
