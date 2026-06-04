#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/prepare_agentic_benchmarks.sh TARGET

Targets:
  sources        Clone missing benchmark source repositories.
  alfworld       Download ALFWorld base data into data/agentic_benchmarks/cache/alfworld.
  scienceworld   Mirror ScienceWorld runtime assets into data/agentic_benchmarks/cache/scienceworld.
  webshop-small  Run WebShop setup with the 1k-product data subset.
  webshop-all    Run WebShop setup with the full product data.

Environment knobs:
  PYTHON_VERSION=3.11
  ALFWORLD_EXTRA=0
  GIT_LFS_SKIP_SMUDGE=1

Notes:
  - WebShop setup installs Python/conda dependencies and builds a search index.
  - Prefer webshop-small before webshop-all.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || -z "${1:-}" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_ROOT="${CODE_DIR}/data/agentic_benchmarks"
SOURCE_DIR="${BENCH_ROOT}/sources"
CACHE_DIR="${BENCH_ROOT}/cache"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
export GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"

clone_if_missing() {
  local name="$1"
  local url="$2"
  local branch="$3"
  local target="${SOURCE_DIR}/${name}"

  if [[ -d "${target}/.git" ]]; then
    echo "[agentic-data] source exists: ${target}"
    return 0
  fi

  mkdir -p "${SOURCE_DIR}"
  git clone --depth 1 --branch "${branch}" "${url}" "${target}"
}

prepare_sources() {
  clone_if_missing "alfworld" "https://github.com/alfworld/alfworld.git" "master"
  clone_if_missing "scienceworld" "https://github.com/allenai/ScienceWorld.git" "main"
  clone_if_missing "webshop" "https://github.com/princeton-nlp/WebShop.git" "master"
}

prepare_alfworld() {
  prepare_sources
  mkdir -p "${CACHE_DIR}/alfworld"

  local extra_args=()
  if [[ "${ALFWORLD_EXTRA:-0}" == "1" ]]; then
    extra_args+=(--extra)
  fi

  PYTHONPATH="${SOURCE_DIR}/alfworld" \
    uv run --with requests --with tqdm --with pyyaml --python "${PYTHON_VERSION}" \
    python "${SOURCE_DIR}/alfworld/scripts/alfworld-download" \
      --data-dir "${CACHE_DIR}/alfworld" \
      "${extra_args[@]}"

  echo "[agentic-data] ALFWorld data ready: ${CACHE_DIR}/alfworld"
}

prepare_scienceworld() {
  prepare_sources
  mkdir -p "${CACHE_DIR}/scienceworld"
  cp -n "${SOURCE_DIR}/scienceworld/scienceworld/scienceworld.jar" "${CACHE_DIR}/scienceworld/"
  cp -n "${SOURCE_DIR}/scienceworld/scienceworld/tasks.json" "${CACHE_DIR}/scienceworld/"
  cp -n "${SOURCE_DIR}/scienceworld/goldpaths/goldpaths-all.zip" "${CACHE_DIR}/scienceworld/"
  echo "[agentic-data] ScienceWorld runtime assets ready: ${CACHE_DIR}/scienceworld"
}

prepare_webshop() {
  local size="$1"
  prepare_sources
  mkdir -p "${CACHE_DIR}/webshop"
  (
    cd "${SOURCE_DIR}/webshop"
    bash setup.sh -d "${size}"
  )
  echo "[agentic-data] WebShop ${size} setup ready: ${SOURCE_DIR}/webshop/data"
}

case "${1}" in
  sources)
    prepare_sources
    ;;
  alfworld)
    prepare_alfworld
    ;;
  scienceworld)
    prepare_scienceworld
    ;;
  webshop-small)
    prepare_webshop "small"
    ;;
  webshop-all)
    prepare_webshop "all"
    ;;
  *)
    usage
    exit 2
    ;;
esac
