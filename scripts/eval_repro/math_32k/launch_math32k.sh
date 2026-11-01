#!/usr/bin/env bash
# 2026-11-01 prepared, not started: Math-only 1.7B naive / list-12 ignore / only / termlook at a 32768-token cap, seed 42 first.
# Usage: nohup bash logs/launch_math32k.sh <RUNNING hold job id> > logs/math32k_<job>.out 2>&1 &   Several cards may share the list (locks).
J="${1:?RUNNING hold job id}"
BASE="${OPD_EVAL_BASE:-${HOME}/scratch/opd}"
cd "${BASE}"
L=""
for seed in 42 1042 2042; do
  for arm in q1p7b_math_l12only q1p7b_math_opdnative q1p7b_math_termlook q1p7b_math_l12ign; do L="${L:+${L},}math4:${arm}:${seed}"; done
done
echo "$(date -Is) start math32k queue on ${J}"
srun --jobid="${J}" --overlap --ntasks=1 --cpus-per-task=8 --gres=gpu:a100:1 bash "${BASE}/run_eval_jobs_a100_v12_math32k.sh" --jobs "${L}"
echo "$(date -Is) math32k queue on ${J} rc=$?"
