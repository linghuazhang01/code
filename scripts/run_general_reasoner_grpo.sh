#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'MSG'
Legacy General-Reasoner GRPO was removed when code/grpo was reset for M2RL-style IF/Science GRPO.

Use one of:
  scripts/run_m2rl_if_grpo.sh
  scripts/run_m2rl_science_grpo.sh
  scripts/run_m2rl_if_science_grpo.sh

The old code is backed up under temp/grpo_legacy_backup_*/grpo from the project root.
MSG
exit 2
