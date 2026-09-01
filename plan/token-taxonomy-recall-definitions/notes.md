# Execution Notes: Phase Top-200 and Recall Replay

## Frozen inputs

- Baselines: `1.7B-OPD`, `1.7B-EOPD`, `4B-OPD`, `4B-EOPD`.
- Domains: `math`, `code`, `science`.
- Global taxonomy: 175 Control + 634 Structure tokens.
- Domain-level Control/Structure sets are derived only from the frozen global taxonomy.
- Candidate pools: `ExpandedPruned-V2`, `Robust190`, `Control-44`.

## Rising/Stable contract

- Use the fixed phase boundaries recorded in `protocol.py`.
- Compute endpoint net optimization speed as
  `(gap_start - gap_end) / (end_step - start_step)`.
- Require occurrence strictly `>20` at both endpoints.
- Rank the complete eligible vocabulary, not a candidate pool; break ties by token ID.
- Select Rising and Stable Top-200 independently.
- Classify the selected cohort with the frozen domain taxonomy; all remaining tokens are Other.
- All figures and claims are explicitly about taxonomy composition **inside Top-200**.

## Recall replay contract

- Search `window=1..20`, `K=1..30`, metrics A/C/E/F.
- Selection boundaries are anchored at the earliest supported boundary and then spaced by exactly
  `window` optimizer steps.
- Source score uses only data at or before the boundary.
- A candidate must have occurrence strictly `>20` at every source step required by the metric.
- Next-step target uses endpoint speed from `t` to `t+1`.
- Next-window target uses endpoint speed from `t` to `t+window` and requires strict occurrence
  support at every intervening target snapshot.
- `Actual` is the number of Future Top-200 tokens inside the dynamic, source-eligible candidate set.
- Split mode selects Control and Structure separately; unified mode selects their union once.
- Primary pool metrics are Precision, Pool Recall, and Pool F1. Coverage metrics Pool Capacity,
  Type Recall, and Type F1 are retained to prevent small pools from appearing universally superior.
- A deployable optimum must have complete cell coverage and `full-K rate = 100%`.

## Metric formulas

- A: occurrence-weighted mean log-probability gap in the source window.
- C: occurrence-weighted mean student entropy in the source window.
- E: occurrence-weighted mean of normalized `A + C + A*C` per source step.
- F: historical endpoint optimization speed `(gap[t-window] - gap[t]) / window`.

## Evidence discipline

- Preserve baseline × domain × token-type cells before macro averaging.
- Treat the grid search as exploratory because the same four trajectories are used for tuning and
  evaluation.
- Do not attach inferential significance to four baseline trajectories.
