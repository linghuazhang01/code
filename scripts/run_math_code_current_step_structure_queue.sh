#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export STOP_STALE_RAY="${STOP_STALE_RAY:-0}"
export GPU_IDLE_MEMORY_LIMIT_MB="${GPU_IDLE_MEMORY_LIMIT_MB:-4096}"
export MOPD_LOCAL_CONDA_ROOT="${MOPD_LOCAL_CONDA_ROOT:-/root/miniconda3}"
export MOPD_LOCAL_CONDA_ENV="${MOPD_LOCAL_CONDA_ENV:-/root/miniconda3/envs/mopd-verl}"
export PYTHON="${PYTHON:-${MOPD_LOCAL_CONDA_ENV}/bin/python}"

configs=(
  "configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_toploss_m05_c01_code_structure_fixed4_8gpu_6s2t.yaml"
  "configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c01_code_structure_fixed4_8gpu_colocated.yaml"
)

for config in "${configs[@]}"; do
  printf '[%s] starting %s on GPU_IDS=%s\n' \
    "$(date -Iseconds)" "${config}" "${GPU_IDS}"
  bash start.sh --local --foreground --config "${config}"
  printf '[%s] completed %s\n' "$(date -Iseconds)" "${config}"
done
