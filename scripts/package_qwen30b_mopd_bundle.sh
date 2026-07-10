#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/package_qwen30b_mopd_bundle.sh [output.zip]

Create a data-only zip bundle with the Qwen30B four-domain data needed by the
bootstrap script. Code, models, and the Python/conda environment are
intentionally not bundled; code is cloned from Git and models are prepared on
the remote host.

Defaults:
  BUNDLE_ZIP=temp/opd_qwen30b_mopd_data_<timestamp>.zip
  DATA_DIR=$CODE_DIR/data/G-OPD-Training-Data
  EVAL_DOMAIN_DIR=$CODE_DIR/eval/domains
  REQUIRE_EVAL_DATA=1
  SPLIT_BUNDLE=0
  SPLIT_SIZE=900m

The bundle includes:
  data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet
  data/G-OPD-Training-Data/Eurus/code_train.parquet
  data/G-OPD-Training-Data/IF/train.parquet
  data/G-OPD-Training-Data/Science/train.parquet
  eval/domains/math/data/AIME24/test.parquet
  eval/domains/math/data/AIME25/test.parquet
  eval/domains/math/data/HMMT25Feb/test.parquet
  eval/domains/math/data/HMMT25Nov/test.parquet
  eval/domains/code/data/HumanEvalPlus/test.parquet
  eval/domains/code/data/MBPPPlus/test.parquet
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

BUNDLE_ZIP="${1:-${BUNDLE_ZIP:-${CODE_DIR}/temp/opd_qwen30b_mopd_data_${TIMESTAMP}.zip}}"
if [[ "${BUNDLE_ZIP}" != /* ]]; then
  BUNDLE_ZIP="${CODE_DIR}/${BUNDLE_ZIP}"
fi
DATA_DIR="${DATA_DIR:-${CODE_DIR}/data/G-OPD-Training-Data}"
EVAL_DOMAIN_DIR="${EVAL_DOMAIN_DIR:-${CODE_DIR}/eval/domains}"
REQUIRE_EVAL_DATA="${REQUIRE_EVAL_DATA:-1}"
SPLIT_BUNDLE="${SPLIT_BUNDLE:-0}"
SPLIT_SIZE="${SPLIT_SIZE:-900m}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1"
}

fail() {
  echo "$1" >&2
  exit "${2:-1}"
}

is_lfs_pointer() {
  local file_path="$1"
  [[ -f "${file_path}" ]] || return 1
  head -c 96 "${file_path}" | grep -q "version https://git-lfs.github.com/spec"
}

require_real_file() {
  local label="$1"
  local file_path="$2"
  [[ -f "${file_path}" ]] || fail "Missing ${label}: ${file_path}" 2
  if is_lfs_pointer "${file_path}"; then
    fail "${label} is still a Git LFS pointer: ${file_path}" 2
  fi
}

command -v zip >/dev/null 2>&1 || fail "zip is required to create the bundle." 2
if [[ "${SPLIT_BUNDLE}" == "1" ]]; then
  command -v split >/dev/null 2>&1 || fail "split is required for SPLIT_BUNDLE=1." 2
fi

required_train_files=(
  "DeepMath-103K/train_filtered_level6.parquet"
  "Eurus/code_train.parquet"
  "IF/train.parquet"
  "Science/train.parquet"
)
required_eval_files=(
  "math/data/AIME24/test.parquet"
  "math/data/AIME25/test.parquet"
  "math/data/HMMT25Feb/test.parquet"
  "math/data/HMMT25Nov/test.parquet"
  "code/data/HumanEvalPlus/test.parquet"
  "code/data/MBPPPlus/test.parquet"
)

zip_paths=()

add_zip_path() {
  local file_path="$1"
  case "${file_path}" in
    "${CODE_DIR}"/*)
      zip_paths+=("${file_path#"${CODE_DIR}/"}")
      ;;
    *)
      fail "Data file must be under CODE_DIR for repo-relative zip layout: ${file_path}" 2
      ;;
  esac
}

log "Validating four-domain training data."
for relative_path in "${required_train_files[@]}"; do
  file_path="${DATA_DIR}/${relative_path}"
  require_real_file "training data ${relative_path}" "${file_path}"
  add_zip_path "${file_path}"
done

if [[ "${REQUIRE_EVAL_DATA}" == "1" ]]; then
  log "Validating eval data referenced by the Qwen30B config."
  for relative_path in "${required_eval_files[@]}"; do
    file_path="${EVAL_DOMAIN_DIR}/${relative_path}"
    require_real_file "eval data ${relative_path}" "${file_path}"
    add_zip_path "${file_path}"
  done
fi

mkdir -p "$(dirname "${BUNDLE_ZIP}")"
if [[ "${SPLIT_BUNDLE}" == "1" ]]; then
  rm -f "${BUNDLE_ZIP}" "${BUNDLE_ZIP}".part-*
  log "Creating streamed split data zip parts: ${BUNDLE_ZIP}.part-*"
  (
    cd "${CODE_DIR}"
    zip -qr - "${zip_paths[@]}"
  ) | split -b "${SPLIT_SIZE}" -d -a 2 - "${BUNDLE_ZIP}.part-"
  log "Split data bundle ready:"
  ls -lh "${BUNDLE_ZIP}".part-*
else
  rm -f "${BUNDLE_ZIP}" "${BUNDLE_ZIP}".part-*
  log "Creating data zip bundle: ${BUNDLE_ZIP}"
  (
    cd "${CODE_DIR}"
    zip -qr "${BUNDLE_ZIP}" "${zip_paths[@]}"
  )
  log "Data bundle ready: ${BUNDLE_ZIP}"
  du -h "${BUNDLE_ZIP}" | awk '{print "Bundle size: " $1}'
fi
