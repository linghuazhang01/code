#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/home/shuang_qiu/mopd_code}"
GOPD_REPO="${GOPD_REPO:-/home/shuang_qiu/mopd/code/data/G-OPD-Training-Data/.eval-source/G-OPD}"
GOPD_COMMIT="${GOPD_COMMIT:-37371a4c31ad7947746200d234161769191f4748}"
BASE_PYTHON="${BASE_PYTHON:-/home/shuang_qiu/env/miniconda3/envs/mopd-verl/bin/python}"
RUNTIME_ROOT="${EVALPLUS_RUNTIME_ROOT:-${REPO_ROOT}/.runtime/evalplus-gopd-${GOPD_COMMIT}}"
SOURCE_ROOT="${RUNTIME_ROOT}/source"
VENV_ROOT="${RUNTIME_ROOT}/venv"

actual_commit="$(git -C "${GOPD_REPO}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${GOPD_COMMIT}" ]]; then
  echo "G-OPD commit mismatch: expected=${GOPD_COMMIT}, actual=${actual_commit}" >&2
  exit 1
fi

mkdir -p "${RUNTIME_ROOT}"
if [[ ! -f "${SOURCE_ROOT}/evalplus/evaluate.py" ]]; then
  staging="$(mktemp -d "${RUNTIME_ROOT}/source.XXXXXX")"
  git -C "${GOPD_REPO}" archive "${GOPD_COMMIT}:code_eval/coding/evalplus" \
    | tar -x -C "${staging}"
  mv "${staging}" "${SOURCE_ROOT}"
fi

if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_ROOT}"
fi
"${VENV_ROOT}/bin/python" -m pip install --disable-pip-version-check \
  --requirement "${REPO_ROOT}/eval/evalplus_rescore_requirements.txt"

PYTHONPATH="${SOURCE_ROOT}" "${VENV_ROOT}/bin/python" - <<'PY'
from evalplus.evaluate import evaluate
from evalplus.sanitize import sanitize

assert callable(evaluate)
assert sanitize("def f():\n    return 1", entrypoint="f").startswith("def f")
print("EvalPlus runtime witness: OK")
PY

"${VENV_ROOT}/bin/python" -m pip freeze > "${RUNTIME_ROOT}/pip-freeze.txt"

cat > "${RUNTIME_ROOT}/RUNTIME_MANIFEST.txt" <<EOF
gopd_repo=${GOPD_REPO}
gopd_commit=${GOPD_COMMIT}
python=${VENV_ROOT}/bin/python
pythonpath=${SOURCE_ROOT}
requirements=${REPO_ROOT}/eval/evalplus_rescore_requirements.txt
pip_freeze=${RUNTIME_ROOT}/pip-freeze.txt
EOF
printf '%s\n' "${RUNTIME_ROOT}"
