#!/usr/bin/env bash
# Container setup, step 2 of 2: download the training data and the models into one directory (--dir), check them
# against Hugging Face, and link them to where the training configs look. Configs name paths relative to the code
# checkout (../mopd/models/<model>, ../mopd/code/data/...), so the script fills <parent of the checkout>/mopd/ with links:
#   mopd/models                          -> DIR/models
#   mopd/code/data/G-OPD-Training-Data   -> DIR/data/G-OPD-Training-Data   (DeepMath level-6 + Eurus code parquet)
#   mopd/code/data/eval_data             -> <checkout>/data/eval_data      (validation parquet, already in git)
# The checkout itself is not changed. Reruns skip files that are already complete. The Python comes from step 1
# (deploy/container/install_env.sh).
#
#   bash deploy/container/download_data_models.sh --dir /mnt/disk/opd_data                 # data, 1.7B + 4B, 30B teacher
#   bash deploy/container/download_data_models.sh --dir /mnt/disk/opd_data --students Qwen/Qwen3-1.7B   # one student
#   bash deploy/container/download_data_models.sh --dir /mnt/disk/opd_data --teacher ""     # no teacher
#
# Options:  --dir DIR (default: <parent of the checkout>/opd_downloads)
#           --students "Qwen/Qwen3-1.7B Qwen/Qwen3-4B"  --teacher Qwen/Qwen3-30B-A3B-Instruct-2507 ("" = none)
#           --no-data  --no-models  --python PATH (default: $PYTHON from opd_env.sh, else
#           /opt/miniforge3/envs/mopd-verl/bin/python; any Python >= 3.10 with huggingface_hub and pyarrow works)
# Network:  everything comes from Hugging Face (public repos, no token). Without a route to huggingface.co set
#           HF_ENDPOINT=https://hf-mirror.com (read-only mirror, fine for downloads); HTTPS_PROXY / HTTP_PROXY also work.
set -euo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET_ID=icemoon28/MOPD-Training-Data
TRAIN_FILES=(DeepMath-103K/train_filtered_level6.parquet Eurus/code_train.parquet)   # the only training files in configs/
LINK_ROOT="$(dirname "${CODE_DIR}")/mopd"

DIR="$(dirname "${CODE_DIR}")/opd_downloads"; STUDENTS="Qwen/Qwen3-1.7B Qwen/Qwen3-4B"; TEACHER=Qwen/Qwen3-30B-A3B-Instruct-2507
DATA=1; MODELS=1; PY="${PYTHON:-}"
while (( $# )); do
  case "$1" in
    --dir) DIR="${2:?--dir needs a directory}"; shift 2 ;;
    --students) STUDENTS="$2"; shift 2 ;;
    --teacher) TEACHER="$2"; shift 2 ;;
    --no-data) DATA=0; shift ;;
    --no-models) MODELS=0; shift ;;
    --python) PY="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
die() { log "ERROR: $*" >&2; exit 2; }
real() { (cd "$1" 2>/dev/null && pwd -P); }
link() {  # link PATH TARGET: PATH becomes a symlink to TARGET; a real file or directory there is never replaced
  local path="$1" target="$2"
  mkdir -p "$(dirname "${path}")"
  if [[ -L "${path}" || ! -e "${path}" ]]; then
    ln -sfn "${target}" "${path}"
  elif [[ "$(real "${path}")" != "$(real "${target}")" ]]; then
    die "${path} already exists and is not a link to ${target}; move it aside and rerun"
  fi
  log "  ${path} -> ${target}"
}

[[ -n "${PY}" ]] || PY=/opt/miniforge3/envs/mopd-verl/bin/python
command -v "${PY}" >/dev/null || die "no Python at ${PY}: run deploy/container/install_env.sh and source its opd_env.sh first, or pass --python"
"${PY}" -c 'import sys, huggingface_hub, pyarrow; assert sys.version_info >= (3, 10)' 2>/dev/null \
  || die "${PY} needs Python >= 3.10 with huggingface_hub and pyarrow (the mopd-verl env has them)"
[[ -d "${CODE_DIR}/data/eval_data" ]] || die "no data/eval_data under ${CODE_DIR}; run this script from the OPD code checkout"
mkdir -p "${DIR}/models" "${DIR}/data/G-OPD-Training-Data"
DIR="$(real "${DIR}")"
export HF_HOME="${HF_HOME:-${DIR}/.hf_home}"   # Hub caches next to the downloads, not on the container's root disk
export HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120

log "download dir ${DIR} ($(df -h "${DIR}" | awk 'NR == 2 {print $4}') free), endpoint ${HF_ENDPOINT:-https://huggingface.co}"
log "links for the training configs (their paths are relative to ${CODE_DIR}):"
link "${LINK_ROOT}/models" "${DIR}/models"
link "${LINK_ROOT}/code/data/G-OPD-Training-Data" "${DIR}/data/G-OPD-Training-Data"
link "${LINK_ROOT}/code/data/eval_data" "${CODE_DIR}/data/eval_data"

failed=()
if (( DATA )); then
  log "training data: ${DATASET_ID} -> ${DIR}/data/G-OPD-Training-Data"
  if ! "${PY}" - "${DATASET_ID}" "${DIR}/data/G-OPD-Training-Data" "${TRAIN_FILES[@]}" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

repo, target, files = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(target), allow_patterns=files)
for rel in files:
    path = target / rel
    if not path.is_file():
        sys.exit(f"missing after download: {path}")
    with path.open("rb") as handle:
        if handle.read(96).startswith(b"version https://git-lfs.github.com/spec"):
            sys.exit(f"{path} is a Git LFS pointer, not data")
    print(f"ok {path} rows={pq.ParquetFile(path).metadata.num_rows}")
PY
  then failed+=("training data"); fi
fi

if (( MODELS )); then
  for m in ${STUDENTS} ${TEACHER}; do
    log "model ${m} -> ${DIR}/models/${m##*/}"
    # every file is checked against the hub (size, and SHA-256 for weights); a complete folder is skipped
    "${PY}" "${CODE_DIR}/deploy/remote/fetch_hf.py" model "${m}" "${DIR}/models/${m##*/}" || failed+=("${m}")
  done
fi

log "check: the paths the configs use"
need=(../mopd/code/data/eval_data/math/AIME24/test.parquet ../mopd/code/data/eval_data/code/HumanEvalPlus/test.parquet)
if (( DATA )); then for f in "${TRAIN_FILES[@]}"; do need+=("../mopd/code/data/G-OPD-Training-Data/${f}"); done; fi
if (( MODELS )); then for m in ${STUDENTS} ${TEACHER}; do need+=("../mopd/models/${m##*/}/config.json"); done; fi
for p in "${need[@]}"; do
  if [[ -f "${CODE_DIR}/${p}" ]]; then log "  ok      ${p}"; else log "  MISSING ${p}"; failed+=("${p}"); fi
done

if (( ${#failed[@]} )); then
  log "FAILED: ${failed[*]}"
  log "fix the cause and rerun the same command (finished files are skipped)"
  exit 1
fi
log "done: data and models in ${DIR}, linked under ${LINK_ROOT}"
