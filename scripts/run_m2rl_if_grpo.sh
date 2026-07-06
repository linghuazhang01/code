#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${GRPO_CONFIG:-${CODE_DIR}/grpo/configs/m2rl_if.yaml}"

exec env MOPD_CONFIG="${CONFIG_PATH}" "${SCRIPT_DIR}/run_mopd.sh" "$@"
