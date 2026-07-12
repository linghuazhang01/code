#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IFBENCH_REPO="${IFBENCH_REPO:-${CODE_DIR}/../temp/IFBench}"

if [[ -f "${IFBENCH_REPO}/evaluation_lib.py" ]]; then
  printf '[ifbench-runtime] ready: %s\n' "${IFBENCH_REPO}"
  exit 0
fi

command -v git >/dev/null 2>&1 || {
  echo "git is required to prepare the official IFBench evaluator." >&2
  exit 2
}

mkdir -p "$(dirname "${IFBENCH_REPO}")"
echo "[ifbench-runtime] cloning official evaluator into ${IFBENCH_REPO}"
git clone --depth 1 https://github.com/allenai/IFBench.git "${IFBENCH_REPO}"
[[ -f "${IFBENCH_REPO}/evaluation_lib.py" ]] || {
  echo "IFBench clone completed but evaluation_lib.py is missing." >&2
  exit 2
}
