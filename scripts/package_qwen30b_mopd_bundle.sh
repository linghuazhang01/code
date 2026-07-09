#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/package_qwen30b_mopd_bundle.sh [output.zip]

Create a zip bundle that contains the current OPD code plus the Qwen30B
four-domain training data needed by the bootstrap script. Models and the
Python/conda environment are intentionally not bundled; they are prepared on
the remote host.

Defaults:
  BUNDLE_ZIP=temp/opd_qwen30b_mopd_bundle_<timestamp>.zip
  BUNDLE_ROOT_NAME=OPD-code
  DATA_DIR=$CODE_DIR/data/G-OPD-Training-Data
  EVAL_DOMAIN_DIR=$CODE_DIR/eval/domains
  REQUIRE_EVAL_DATA=1
  KEEP_STAGING=0
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

BUNDLE_ZIP="${1:-${BUNDLE_ZIP:-${CODE_DIR}/temp/opd_qwen30b_mopd_bundle_${TIMESTAMP}.zip}}"
if [[ "${BUNDLE_ZIP}" != /* ]]; then
  BUNDLE_ZIP="${CODE_DIR}/${BUNDLE_ZIP}"
fi
BUNDLE_ROOT_NAME="${BUNDLE_ROOT_NAME:-OPD-code}"
DATA_DIR="${DATA_DIR:-${CODE_DIR}/data/G-OPD-Training-Data}"
EVAL_DOMAIN_DIR="${EVAL_DOMAIN_DIR:-${CODE_DIR}/eval/domains}"
REQUIRE_EVAL_DATA="${REQUIRE_EVAL_DATA:-1}"
KEEP_STAGING="${KEEP_STAGING:-0}"
SPLIT_BUNDLE="${SPLIT_BUNDLE:-0}"
SPLIT_SIZE="${SPLIT_SIZE:-900m}"
STAGING_DIR="${STAGING_DIR:-${CODE_DIR}/temp/bundle_staging_${TIMESTAMP}}"

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

cleanup() {
  if [[ "${KEEP_STAGING}" != "1" && -n "${STAGING_DIR:-}" && -d "${STAGING_DIR}" ]]; then
    rm -rf "${STAGING_DIR}"
  fi
}

trap cleanup EXIT

command -v rsync >/dev/null 2>&1 || fail "rsync is required to stage the bundle." 2
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

log "Validating four-domain training data."
for relative_path in "${required_train_files[@]}"; do
  require_real_file "training data ${relative_path}" "${DATA_DIR}/${relative_path}"
done

if [[ "${REQUIRE_EVAL_DATA}" == "1" ]]; then
  log "Validating eval data referenced by the Qwen30B config."
  for relative_path in "${required_eval_files[@]}"; do
    require_real_file "eval data ${relative_path}" "${EVAL_DOMAIN_DIR}/${relative_path}"
  done
fi

rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}/${BUNDLE_ROOT_NAME}" "$(dirname "${BUNDLE_ZIP}")"

log "Staging code without large runtime artifacts."
rsync -a "${CODE_DIR}/" "${STAGING_DIR}/${BUNDLE_ROOT_NAME}/" \
  --exclude '.git/' \
  --exclude '.DS_Store' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude 'temp/' \
  --exclude 'deliverables/' \
  --exclude 'dist/' \
  --exclude 'logs/' \
  --exclude 'hf_home/' \
  --exclude 'wandb/' \
  --exclude 'outputs/' \
  --exclude 'checkpoints/' \
  --exclude 'models/' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.ckpt' \
  --exclude '*.bin' \
  --exclude '*.safetensors' \
  --exclude 'data/G-OPD-Training-Data/' \
  --exclude 'eval/domains/*/data/'

log "Adding required training parquet files."
for relative_path in "${required_train_files[@]}"; do
  mkdir -p "${STAGING_DIR}/${BUNDLE_ROOT_NAME}/data/G-OPD-Training-Data/$(dirname "${relative_path}")"
  cp "${DATA_DIR}/${relative_path}" "${STAGING_DIR}/${BUNDLE_ROOT_NAME}/data/G-OPD-Training-Data/${relative_path}"
done

if [[ "${REQUIRE_EVAL_DATA}" == "1" ]]; then
  log "Adding required eval parquet files."
  for relative_path in "${required_eval_files[@]}"; do
    mkdir -p "${STAGING_DIR}/${BUNDLE_ROOT_NAME}/eval/domains/$(dirname "${relative_path}")"
    cp "${EVAL_DOMAIN_DIR}/${relative_path}" "${STAGING_DIR}/${BUNDLE_ROOT_NAME}/eval/domains/${relative_path}"
  done
fi

zip_output="${BUNDLE_ZIP}"
if [[ "${SPLIT_BUNDLE}" == "1" ]]; then
  zip_output="${STAGING_DIR}/bundle_full.zip"
  rm -f "${BUNDLE_ZIP}" "${BUNDLE_ZIP}".part-*
else
  rm -f "${BUNDLE_ZIP}"
fi

log "Creating zip bundle: ${zip_output}"
(
  cd "${STAGING_DIR}"
  zip -qr "${zip_output}" "${BUNDLE_ROOT_NAME}"
)

if [[ "${SPLIT_BUNDLE}" == "1" ]]; then
  log "Splitting bundle into ${SPLIT_SIZE} parts: ${BUNDLE_ZIP}.part-*"
  split -b "${SPLIT_SIZE}" -d -a 2 "${zip_output}" "${BUNDLE_ZIP}.part-"
  log "Split bundle ready:"
  ls -lh "${BUNDLE_ZIP}".part-*
else
  log "Bundle ready: ${BUNDLE_ZIP}"
  du -h "${BUNDLE_ZIP}" | awk '{print "Bundle size: " $1}'
fi
