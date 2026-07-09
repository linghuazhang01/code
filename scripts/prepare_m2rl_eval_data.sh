#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/prepare_m2rl_eval_data.sh

Prepare GRPO-aligned M2RL evaluation parquet files for OPD validation.

Output defaults:
  eval/domains/ifbench/data/IFBench_test.parquet
  eval/domains/science/data/gpqa.parquet

Source variables:
  IF_VAL_SOURCE=/path/to/raw_if_val.parquet
  SCIENCE_VAL_SOURCE=/path/to/raw_science_val.parquet
  NEMOTRON_RL_SOURCE=/path/to/instruction_following.jsonl

Optional variables:
  PYTHON_BIN=python
  IF_EVAL_OUTPUT=eval/domains/ifbench/data/IFBench_test.parquet
  SCIENCE_EVAL_OUTPUT=eval/domains/science/data/gpqa.parquet
  M2RL_EVAL_MAX_SAMPLES=128
  IF_VAL_MAX_SAMPLES=128
  SCIENCE_VAL_MAX_SAMPLES=128
  REQUIRE_M2RL_EVAL_DATA=1

Direct IF_VAL_SOURCE / SCIENCE_VAL_SOURCE values take precedence over files
generated from NEMOTRON_RL_SOURCE.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
IF_EVAL_OUTPUT="${IF_EVAL_OUTPUT:-${CODE_DIR}/eval/domains/ifbench/data/IFBench_test.parquet}"
SCIENCE_EVAL_OUTPUT="${SCIENCE_EVAL_OUTPUT:-${CODE_DIR}/eval/domains/science/data/gpqa.parquet}"
NEMOTRON_RL_SOURCE="${NEMOTRON_RL_SOURCE:-}"
IF_VAL_SOURCE="${IF_VAL_SOURCE:-}"
SCIENCE_VAL_SOURCE="${SCIENCE_VAL_SOURCE:-}"
M2RL_EVAL_MAX_SAMPLES="${M2RL_EVAL_MAX_SAMPLES:-}"
IF_VAL_MAX_SAMPLES="${IF_VAL_MAX_SAMPLES:-${M2RL_EVAL_MAX_SAMPLES}}"
SCIENCE_VAL_MAX_SAMPLES="${SCIENCE_VAL_MAX_SAMPLES:-${M2RL_EVAL_MAX_SAMPLES}}"
REQUIRE_M2RL_EVAL_DATA="${REQUIRE_M2RL_EVAL_DATA:-0}"

export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/third_party/verl:${PYTHONPATH:-}"

mkdir -p "$(dirname "${IF_EVAL_OUTPUT}")" "$(dirname "${SCIENCE_EVAL_OUTPUT}")"

prepare_from_nemotron() {
  local source_path="$1"
  if [[ -z "${source_path}" ]]; then
    return 0
  fi
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing Nemotron RL source file: ${source_path}" >&2
    return 1
  fi

  local args=(
    "${CODE_DIR}/scripts/prepare_nemotron_rl_data.py"
    --input "${source_path}"
    --split validation
    --manifest "${CODE_DIR}/data/nemotron_rl/validation_manifest.json"
    --if-output "${IF_EVAL_OUTPUT}"
    --science-output "${SCIENCE_EVAL_OUTPUT}"
  )
  if [[ -n "${IF_VAL_MAX_SAMPLES}" ]]; then
    args+=(--if-max-samples "${IF_VAL_MAX_SAMPLES}")
  fi
  if [[ -n "${SCIENCE_VAL_MAX_SAMPLES}" ]]; then
    args+=(--science-max-samples "${SCIENCE_VAL_MAX_SAMPLES}")
  fi

  echo "Preparing IF/science eval data from Nemotron RL source: ${source_path}"
  "${PYTHON_BIN}" "${args[@]}"
}

prepare_with_m2rl_converter() {
  local source_path="$1"
  local output_path="$2"
  local rm_type="$3"
  local domain="$4"
  local max_samples="$5"

  if [[ -z "${source_path}" ]]; then
    return 0
  fi
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing ${domain} eval source file: ${source_path}" >&2
    return 1
  fi

  local args=(
    -m grpo.data.m2rl prepare
    --input "${source_path}"
    --output "${output_path}"
    --rm-type "${rm_type}"
    --domain "${domain}"
    --split validation
  )
  if [[ -n "${max_samples}" ]]; then
    args+=(--max-samples "${max_samples}")
  fi

  echo "Preparing ${domain} eval data: ${source_path} -> ${output_path}"
  "${PYTHON_BIN}" "${args[@]}"
}

validate_output() {
  local output_path="$1"
  local rm_type="$2"
  local label="$3"

  if [[ ! -f "${output_path}" ]]; then
    echo "Missing ${label} eval parquet: ${output_path}"
    return 1
  fi

  echo "Validating ${label} eval parquet: ${output_path}"
  "${PYTHON_BIN}" -m grpo.data.m2rl validate --input "${output_path}" --rm-type "${rm_type}"
}

if [[ -n "${NEMOTRON_RL_SOURCE}" ]]; then
  prepare_from_nemotron "${NEMOTRON_RL_SOURCE}"
fi

prepare_with_m2rl_converter "${IF_VAL_SOURCE}" "${IF_EVAL_OUTPUT}" ifbench if "${IF_VAL_MAX_SAMPLES}"
prepare_with_m2rl_converter "${SCIENCE_VAL_SOURCE}" "${SCIENCE_EVAL_OUTPUT}" gpqa science "${SCIENCE_VAL_MAX_SAMPLES}"

missing=0
validate_output "${IF_EVAL_OUTPUT}" ifbench IF || missing=1
validate_output "${SCIENCE_EVAL_OUTPUT}" gpqa science || missing=1

if [[ "${missing}" != "0" ]]; then
  if [[ "${REQUIRE_M2RL_EVAL_DATA}" == "1" ]]; then
    exit 1
  fi
  echo "M2RL eval data is incomplete. Set source variables above to generate it."
  exit 0
fi

echo "M2RL eval data is ready."
