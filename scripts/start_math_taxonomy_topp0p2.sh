#!/usr/bin/env bash
# Launch the Math FullTaxonomy top_p=0.2 run through start.sh in local mode.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GPU_COUNT="${MOPD_GPU_COUNT:-8}"

case "${1:-}" in
  6|7|8)
    GPU_COUNT="$1"
    shift
    ;;
esac

case "${GPU_COUNT}" in
  6)
    CONFIG_NAME="top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_6gpu_4a2t_b256.yaml"
    DEFAULT_GPU_IDS="0,1,2,3,4,5"
    ;;
  7)
    CONFIG_NAME="top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_7gpu_5a2t_b255.yaml"
    DEFAULT_GPU_IDS="0,1,2,3,4,5,6"
    ;;
  8)
    CONFIG_NAME="top32kl_next_step_full_taxonomy_split_topp0p2_i1_w1_8gpu_6a2t_b258.yaml"
    DEFAULT_GPU_IDS="0,1,2,3,4,5,6,7"
    ;;
  *)
    echo "MOPD_GPU_COUNT must be 6, 7, or 8; got ${GPU_COUNT}." >&2
    exit 2
    ;;
esac

export GPU_IDS="${GPU_IDS:-${DEFAULT_GPU_IDS}}"
CONFIG_PATH="${CODE_DIR}/configs/token_selection/math/${CONFIG_NAME}"

exec bash "${CODE_DIR}/start.sh" --config "${CONFIG_PATH}" --local "$@"
