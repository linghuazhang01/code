#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: test_grad_configs/start.sh <config[::profile]> [launcher args...]" >&2
  exit 2
fi

config_reference="$1"
shift
profile_suffix=""
config_file="${config_reference}"
if [[ "${config_reference}" == *::* ]]; then
  profile_suffix="::${config_reference##*::}"
  config_file="${config_reference%::*}"
fi
absolute_config="$(cd "$(dirname "${config_file}")" && pwd)/$(basename "${config_file}")"

GPU_IDS="${GPU_IDS:-0,1,2}" bash \
  "$(cd "$(dirname "$0")/.." && pwd)/scripts/run_local_mopd_training.sh" \
  "${absolute_config}${profile_suffix}" "$@"
