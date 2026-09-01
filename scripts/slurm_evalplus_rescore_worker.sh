#!/usr/bin/env bash
#SBATCH --job-name=evalplus-rescore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/shuang_qiu/mopd_code/data/eval_data/results/logs/%x-%j.out
#SBATCH --error=/home/shuang_qiu/mopd_code/data/eval_data/results/logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 MODEL_LABEL RECORDS_JSONL OUTPUT_ROOT" >&2
  exit 2
fi

MODEL_LABEL="$1"
RECORDS_JSONL="$2"
OUTPUT_ROOT="$3"
REPO_ROOT="/home/shuang_qiu/mopd_code"
GOPD_COMMIT="37371a4c31ad7947746200d234161769191f4748"
RUNTIME_ROOT="${REPO_ROOT}/.runtime/evalplus-gopd-${GOPD_COMMIT}"
PYTHON_BIN="${RUNTIME_ROOT}/venv/bin/python"
EVALPLUS_SOURCE="${RUNTIME_ROOT}/source"
HUMANEVAL_SOURCE="${HUMANEVAL_SOURCE:-/home/shuang_qiu/mopd/code/data/G-OPD-Training-Data/.eval-source/G-OPD/code_eval/data/HumanEvalPlus.jsonl}"
MBPP_SOURCE="${MBPP_SOURCE:-/home/shuang_qiu/mopd/code/data/G-OPD-Training-Data/.eval-source/G-OPD/code_eval/data/MbppPlus.jsonl}"
HUMANEVAL_SHA256="42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"
MBPP_SHA256="b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad"

[[ -f "${RECORDS_JSONL}" ]] || { echo "Missing records: ${RECORDS_JSONL}" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing runtime: ${PYTHON_BIN}" >&2; exit 1; }
mkdir -p "$(dirname "${OUTPUT_ROOT}")" "${REPO_ROOT}/data/eval_data/results/logs"
if ! mkdir "${OUTPUT_ROOT}"; then
  echo "Refusing to reuse an existing output root: ${OUTPUT_ROOT}" >&2
  exit 1
fi

check_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Dataset hash mismatch: path=${path}, expected=${expected}, actual=${actual}" >&2
    exit 1
  fi
}
check_sha256 "${HUMANEVAL_SHA256}" "${HUMANEVAL_SOURCE}"
check_sha256 "${MBPP_SHA256}" "${MBPP_SOURCE}"

export PYTHONPATH="${EVALPLUS_SOURCE}"
export HUMANEVAL_OVERRIDE_PATH="${HUMANEVAL_SOURCE}"
export MBPP_OVERRIDE_PATH="${MBPP_SOURCE}"
export XDG_CACHE_HOME="${OUTPUT_ROOT}/cache"

run_dataset() {
  local dataset="$1"
  local source_path="$2"
  local dataset_dir="${OUTPUT_ROOT}/${dataset}"
  mkdir -p "${dataset_dir}"

  "${PYTHON_BIN}" "${REPO_ROOT}/eval/official_evalplus_rescore.py" prepare \
    --records "${RECORDS_JSONL}" \
    --source "${source_path}" \
    --dataset "${dataset}" \
    --output "${dataset_dir}/samples_sanitized.jsonl" \
    --manifest "${dataset_dir}/PREPARE_MANIFEST.json" \
    --expected-rollouts 8 \
    --parallel "${SLURM_CPUS_PER_TASK:-64}"

  "${PYTHON_BIN}" -m evalplus.evaluate \
    --dataset "${dataset}" \
    --samples "${dataset_dir}/samples_sanitized.jsonl" \
    --output_file "${dataset_dir}/eval_results.json" \
    --parallel "${SLURM_CPUS_PER_TASK:-64}" \
    --min_time_limit 10.0 \
    --gt_time_limit_factor 8.0

  "${PYTHON_BIN}" "${REPO_ROOT}/eval/official_evalplus_rescore.py" summarize \
    --results "${dataset_dir}/eval_results.json" \
    --output "${dataset_dir}/SUMMARY.json" \
    --expected-rollouts 8
  touch "${dataset_dir}/SUCCESS"
}

run_dataset humaneval "${HUMANEVAL_SOURCE}"
run_dataset mbpp "${MBPP_SOURCE}"

{
  echo "model_label=${MODEL_LABEL}"
  echo "records=${RECORDS_JSONL}"
  echo "gopd_commit=${GOPD_COMMIT}"
  echo "slurm_job_id=${SLURM_JOB_ID:-local}"
  echo "slurm_cpus=${SLURM_CPUS_PER_TASK:-64}"
  echo "min_time_limit=10.0"
  echo "gt_time_limit_factor=8.0"
  "${PYTHON_BIN}" --version
} > "${OUTPUT_ROOT}/RUN_MANIFEST.txt"
touch "${OUTPUT_ROOT}/SUCCESS"
