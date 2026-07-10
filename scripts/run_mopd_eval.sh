#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_mopd_eval.sh [config] [--dry-run] [--model-path PATH] [--run-id RUN_ID] [--paper-eval] [-- <hydra overrides...>]

Examples:
  scripts/run_mopd_eval.sh configs/mopd_qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_6gpu_math_code_fsdp.yaml --dry-run

  scripts/run_mopd_eval.sh configs/mopd_qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_6gpu_math_code_fsdp.yaml \
    --model-path checkpoints/MOPD/run_id/global_step_200/actor \
    --run-id qwen30b_math_code_step200_eval

  scripts/run_mopd_eval.sh configs/mopd_qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_6gpu_math_code_fsdp.yaml \
    --paper-eval --paper-datasets aime24,humaneval_plus

Notes:
  - This uses the same verl entry point and config translation as training.
  - It forces trainer.val_before_train=true and trainer.val_only=true, so verl
    runs initial validation and exits before the training loop.
  - The underlying trainer still builds the train dataloader, so training data
    paths from the config must exist even though no optimization step runs.

Environment:
  VERL_RUNTIME_DIR=<vendored verl runtime dir>
  MOPD_LAUNCH_PYTHON=<python executable for this launcher, default: python3>
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${CODE_DIR}/configs/mopd_formal_audit_all_2gpu.yaml"
CONFIG_PATH="${MOPD_CONFIG:-${DEFAULT_CONFIG}}"
VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-${CODE_DIR}/third_party/verl}"
ARGS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run|--paper-eval|--paper-fail-on-error)
      ARGS+=("$1")
      shift
      ;;
    --model-path|--run-id|--output-dir|--paper-datasets|--paper-output-dir|--paper-timeout-seconds)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a value" >&2
        exit 2
      fi
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    -*)
      echo "Unknown script option: $1" >&2
      echo "Put Hydra overrides after '--'." >&2
      exit 2
      ;;
    *)
      if [[ "${CONFIG_PATH}" != "${MOPD_CONFIG:-${DEFAULT_CONFIG}}" ]]; then
        echo "Only one config path is allowed." >&2
        exit 2
      fi
      CONFIG_PATH="$1"
      shift
      ;;
  esac
done

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${CODE_DIR}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

if [[ ! -f "${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" ]]; then
  echo "Vendored verl runtime not found at '${VERL_RUNTIME_DIR}'." >&2
  echo "Expected '${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py'." >&2
  exit 2
fi

export PYTHONPATH="${CODE_DIR}:${VERL_RUNTIME_DIR}:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"
cd "${CODE_DIR}"

CMD=(--config "${CONFIG_PATH}" "${ARGS[@]}")
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  CMD+=(-- "${EXTRA_ARGS[@]}")
fi

exec "${MOPD_LAUNCH_PYTHON:-python3}" -m mopd_verl.eval_launch "${CMD[@]}"
