GPU_IDS="${GPU_IDS:-0,1,2}" bash "$(cd "$(dirname "$0")/.." && pwd)/scripts/run_local_mopd_training.sh" "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
