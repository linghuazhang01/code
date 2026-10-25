# List-12 Only: terminate / look-back supplementation

Checked 2026-10-25. Qwen3-1.7B, Math-only and Code-only, native chosen-token policy-gradient OPD.
These are add-only diagnostic presets: every old list-12 ID is retained. They do not replace the old Only or Ignore presets.

## Configurations

All six leaves extend the corresponding `native_opd_list12_only_toklog_8gpu_8c_b256.yaml` in the same directory.
Only `audit.domain_control_token_ids.<domain>` and the six run/output identity fields change.

| Domain | Added candidate group | Old IDs | Candidate IDs | Already in old pool | Newly added | Final pool |
|---|---|---:|---:|---:|---:|---:|
| Math | terminate | 78 | 12 | 7 | 5 | 83 |
| Math | look-back | 78 | 38 | 1 | 37 | 115 |
| Math | both | 78 | 50 | 8 | 42 | 120 |
| Code | terminate | 119 | 12 | 2 | 10 | 129 |
| Code | look-back | 119 | 37 | 0 | 37 | 156 |
| Code | both | 119 | 49 | 2 | 47 | 166 |

Math directory: `configs/token_selection/math/taxonomy/`.
Code directory: `configs/token_selection/code/`.
Each directory has these three new leaf names:

- `native_opd_list12_only_plus_terminate_toklog_8gpu_8c_b256.yaml`
- `native_opd_list12_only_plus_lookback_toklog_8gpu_8c_b256.yaml`
- `native_opd_list12_only_plus_terminate_lookback_toklog_8gpu_8c_b256.yaml`

New run tags are `l12only-term`, `l12only-look` and `l12only-termlook`; the full run name is
`q1p7b-<domain>-opdnative-<tag>-toklog-8g8c-b256-s42`. Audit, evaluation output, local checkpoint and HF prefixes are unique.
Resume remains disabled, so these are new runs from the student base, not continuations of the failed Only checkpoints.

## Frozen candidate provenance and matching

`list12_role_supplements.json` stores the candidate IDs, original pool hashes, source CSV hashes, final counts and final pool hashes.
Hashes use sorted, comma-joined IDs without spaces. It is provenance only; runtime configuration loading does not need the analysis directory.

Candidates are exactly the domain-specific rows labeled `2a terminate` or `2b look-back` in
`analysis/token-roles-20261025/tables/roles_<domain>.csv`, with at least 0.02 occurrences per 1,000 tokens in the equal-weight
1.7B native OPD trajectory. Math used all 60 cached steps; Code used the 21 scanned steps. The classification is the existing
ID-string heuristic in that analysis, not a complete semantic inventory and not the earlier A5 pool.

Matching remains by token ID at every valid occurrence, exactly as in old list-12 Only. There is no new heading, code-fence,
comment/string or sentence-role parser. Ambiguous candidates such as `answer`, `Finally`, `think`, `Yes`, `No` and `?` are not
context-resolved. These configs therefore test supplementation of an observed lexical candidate set, not a clean intervention
on every semantic terminate or look-back occurrence. They do not redefine any Control/Structure taxonomy in `Token.md`.

EOS ID 151645 was already in both original pools and remains present in all three arms. The terminate arm does not add an EOS
signal that was previously absent. It adds 5 other IDs for Math and 10 for Code. Generation stop rules are unchanged.

## Unchanged training contract

- Eight colocated GPUs: actor, rollout and FSDP-8 teacher share every GPU; global batch and PPO minibatch 256.
- Student Qwen3-1.7B; teacher Qwen3-30B-A3B-Instruct-2507; learning rate 5e-6; data, trainer and rollout seeds 42.
- Selected valid positions weight 1, all other valid positions weight 0; no online selection or weight normalization.
- Native policy-gradient OPD, not Top-32 RKL; `token_baseline_method: none`, `loss_agg_mode: token-mean`.
- Loss aggregation still uses all valid response positions, not just selected positions. The larger pools change the amount
  and distribution of direct training signal; these six runs alone are not matched-budget proof of a semantic mechanism.
- Prompt cap 2,048 and response cap 16,384; vLLM memory utilization 0.6; teacher batching, fused statistics and compact IDs on.
- Per-token audit on, 60 training steps, save every step, HF upload only step 60 with unique public-repository prefixes.
- Existing `paper_eval.enabled: false` is inherited. A training launch still needs its post-training evaluation chained by the runner.

No training or remote synchronization is performed by adding these files.
