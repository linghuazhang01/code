#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IFBENCH_REPO="${IFBENCH_REPO:-${CODE_DIR}/../temp/IFBench}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "${IFBENCH_REPO}/evaluation_lib.py" ]]; then
  command -v git >/dev/null 2>&1 || {
    echo "git is required to prepare the official IFBench evaluator." >&2
    exit 2
  }
  mkdir -p "$(dirname "${IFBENCH_REPO}")"
  echo "[ifbench-runtime] cloning official evaluator into ${IFBENCH_REPO}"
  git clone --depth 1 https://github.com/allenai/IFBench.git "${IFBENCH_REPO}"
fi
[[ -f "${IFBENCH_REPO}/evaluation_lib.py" ]] || {
  echo "IFBench clone completed but evaluation_lib.py is missing." >&2
  exit 2
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "IFBench runtime Python not found: ${PYTHON_BIN}" >&2
  exit 2
}

if ! "${PYTHON_BIN}" -c 'import emoji, nltk, syllapy' >/dev/null 2>&1; then
  echo "IFBench Python dependencies are missing for ${PYTHON_BIN}." >&2
  echo "Activate the environment from scripts/setup_training_env.sh, or install:" >&2
  echo "  ${PYTHON_BIN} -m pip install -r ${IFBENCH_REPO}/requirements.txt" >&2
  exit 2
fi

NLTK_DATA_DIR="${IFBENCH_REPO}/.nltk_data"
mkdir -p "${NLTK_DATA_DIR}"
"${PYTHON_BIN}" -m nltk.downloader \
  -d "${NLTK_DATA_DIR}" \
  punkt punkt_tab stopwords averaged_perceptron_tagger_eng >/dev/null

PYTHONPATH="${IFBENCH_REPO}:${PYTHONPATH:-}" \
NLTK_DATA="${NLTK_DATA_DIR}:${NLTK_DATA:-}" \
  "${PYTHON_BIN}" -c 'import evaluation_lib'

printf '[ifbench-runtime] ready: %s (python=%s)\n' \
  "${IFBENCH_REPO}" "${PYTHON_BIN}"
