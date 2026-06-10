#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/start_general_reasoner_mopd_training.sh [start_remote_mopd_training options] [-- <hydra overrides...>]

Examples:
  scripts/start_general_reasoner_mopd_training.sh

  scripts/start_general_reasoner_mopd_training.sh --foreground --dry-run -- \
    trainer.total_training_steps=1

Environment:
  GREASONER_MOPD_CONFIG=configs/mopd_general_reasoner.yaml
  RUN_ID=greasoner_14b_mopd_<timestamp>
  GPU_IDS=0,1,2,3,4,5,6,7
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${GREASONER_MOPD_CONFIG:-configs/mopd_general_reasoner.yaml}"
RUN_ID="${RUN_ID:-greasoner_14b_mopd_$(date +%Y%m%d_%H%M%S)}"
HAS_RUN_ID=0

for arg in "$@"; do
  if [[ "${arg}" == "--run-id" ]]; then
    HAS_RUN_ID=1
    break
  fi
done

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  "${SCRIPT_DIR}/start_remote_mopd_training.sh" --help
  exit 0
fi

ARGS=("${CONFIG_PATH}")
if [[ "${HAS_RUN_ID}" == "0" ]]; then
  ARGS+=(--run-id "${RUN_ID}")
fi

exec "${SCRIPT_DIR}/start_remote_mopd_training.sh" "${ARGS[@]}" "$@"
