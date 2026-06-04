# Agentic Benchmark Data Staging

This directory stages non-code/math task domains for possible MOPD experiments.
The current goal is to keep the benchmark sources and data-preparation entrypoints
available without mixing them into the existing math/code parquet training set.

## Layout

```text
data/agentic_benchmarks/
  sources/
    alfworld/
    scienceworld/
    webshop/
  cache/
    alfworld/
    scienceworld/
    webshop/
  processed/
```

## Current Status

| Dataset | Source checkout | Commit | Local data status | Notes |
|---|---|---|---|---|
| ALFWorld | `sources/alfworld` | `aaba6870f86c5be6a08a491f32a50b906227bc3e` | `cache/alfworld` created; full download not completed | Official release download was attempted, but the GitHub release connection was too slow in this session. Use the prepare script to retry. |
| ScienceWorld | `sources/scienceworld` | `f6d8f5ec41eadfdcad23cc3ab097f0903dc1378b` | `cache/scienceworld` contains `scienceworld.jar`, `tasks.json`, and `goldpaths-all.zip` | Mirrored with `bash scripts/prepare_agentic_benchmarks.sh scienceworld`. |
| WebShop | `sources/webshop` | `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` | source staged; product/instruction data not downloaded | Use `webshop-small` first. `webshop-all` downloads the full product set and builds a larger search index. |

## Prepare Commands

From the code checkout:

```bash
bash scripts/prepare_agentic_benchmarks.sh sources
bash scripts/prepare_agentic_benchmarks.sh scienceworld
bash scripts/prepare_agentic_benchmarks.sh alfworld
bash scripts/prepare_agentic_benchmarks.sh webshop-small
```

Useful runtime paths:

```bash
export ALFWORLD_DATA=/Users/linghuazhang/Desktop/Project/OPD/code/data/agentic_benchmarks/cache/alfworld
export SCIENCEWORLD_HOME=/Users/linghuazhang/Desktop/Project/OPD/code/data/agentic_benchmarks/sources/scienceworld
export WEBSHOP_HOME=/Users/linghuazhang/Desktop/Project/OPD/code/data/agentic_benchmarks/sources/webshop
```

## MOPD Use

These benchmarks should be treated as environment/task sources first. Before
training with `verl`, add a converter that logs trajectories into a stable MOPD
format with at least:

- `prompt`
- `response` or action trace
- `extra_info.opd_teacher`
- `extra_info.domain`
- task id / split / environment metadata
- success, reward, and action-validity fields when available
