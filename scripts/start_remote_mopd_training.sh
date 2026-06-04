#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/start_remote_mopd_training.sh <config> [--run-id RUN_ID] [--foreground] [--tail] [--dry-run] [-- <hydra overrides...>]

Examples:
  scripts/start_remote_mopd_training.sh configs/mopd_formal_single_a800.yaml

  scripts/start_remote_mopd_training.sh configs/mopd_formal_single_a800.yaml \
    --run-id mopd_manual_test

  scripts/start_remote_mopd_training.sh configs/mopd_formal_single_a800.yaml \
    --run-id mopd_bsz128 \
    -- data.train_batch_size=128 data.val_batch_size=128 trainer.val_before_train=false

Notes:
  - Run this script on the remote host from a synced OPD-code checkout.
  - It does not SSH and does not sync files.
  - Training imports verl from this repo's third_party/verl directory.
  - By default it launches in a detached screen session.

Environment:
  REMOTE_ROOT=<parent of OPD-code>
  CONDA_SH=<auto-detected conda.sh when available>
  CONDA_ENV=mopd-verl
  PYTHON_BIN=<active python after conda activation>
  GPU_ID=0
  LOG_DIR=$CODE_DIR/logs
  STOP_STALE_RAY=1
  GPU_IDLE_MEMORY_LIMIT_MB=1000
  MOPD_STEP_PROGRESS=1
  MOPD_VLLM_GENERATE_PROGRESS=1
USAGE
}

quote() {
  printf "%q" "$1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_ROOT="${REMOTE_ROOT:-$(cd "${CODE_DIR}/.." && pwd)}"
if [[ -z "${CONDA_SH:-}" ]]; then
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
  else
    CONDA_SH=""
  fi
fi
CONDA_ENV="${CONDA_ENV:-mopd-verl}"
PYTHON_BIN="${PYTHON_BIN:-}"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/logs}"
STOP_STALE_RAY="${STOP_STALE_RAY:-1}"
GPU_IDLE_MEMORY_LIMIT_MB="${GPU_IDLE_MEMORY_LIMIT_MB:-1000}"
VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-${CODE_DIR}/third_party/verl}"

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CONFIG_ARG=""
RUN_ID=""
FOREGROUND=0
TAIL_LOG=0
DRY_RUN_FLAG=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --run-id)
      if [[ $# -lt 2 ]]; then
        echo "--run-id requires a value" >&2
        exit 2
      fi
      RUN_ID="$2"
      shift 2
      ;;
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --tail)
      TAIL_LOG=1
      shift
      ;;
    --dry-run)
      DRY_RUN_FLAG=1
      shift
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
      if [[ -n "${CONFIG_ARG}" ]]; then
        echo "Only one config path is allowed. Extra Hydra overrides must go after '--'." >&2
        exit 2
      fi
      CONFIG_ARG="$1"
      shift
      ;;
  esac
done

if [[ -z "${CONFIG_ARG}" ]]; then
  echo "<config> is required." >&2
  usage >&2
  exit 2
fi

if [[ "${CONFIG_ARG}" == /* ]]; then
  CONFIG_PATH="${CONFIG_ARG}"
  CONFIG_LABEL="$(basename "${CONFIG_ARG}")"
else
  CONFIG_PATH="${CODE_DIR}/${CONFIG_ARG}"
  CONFIG_LABEL="${CONFIG_ARG##*/}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

if [[ -z "${RUN_ID}" ]]; then
  config_stem="${CONFIG_LABEL%.*}"
  RUN_ID="${config_stem}_$(date +%Y%m%d_%H%M%S)"
fi

if [[ ! -f "${SCRIPT_DIR}/run_math_code_mopd.sh" ]]; then
  echo "Missing training wrapper: ${SCRIPT_DIR}/run_math_code_mopd.sh" >&2
  exit 2
fi

if [[ ! -f "${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" ]]; then
  echo "Vendored verl runtime not found: ${VERL_RUNTIME_DIR}" >&2
  echo "Expected ${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" >&2
  exit 2
fi

if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

export PYTHONPATH="${CODE_DIR}:${VERL_RUNTIME_DIR}:${PYTHONPATH:-}"
if ! "${PYTHON_BIN}" -c "import yaml, verl; import verl.trainer.main_ppo" >/dev/null 2>&1; then
  echo "Python environment cannot import yaml or vendored verl." >&2
  echo "PYTHON_BIN=${PYTHON_BIN}" >&2
  echo "PYTHONPATH=${PYTHONPATH}" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${CONFIG_PATH}" "${CODE_DIR}" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
code_dir = Path(sys.argv[2])
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

paths: list[str] = []
data = config.get("data") or {}
for key in ("train_files", "val_files"):
    value = data.get(key) or []
    if isinstance(value, str):
        paths.append(value)
    else:
        paths.extend(str(item) for item in value)

domain_train_files = data.get("domain_train_files") or {}
if isinstance(domain_train_files, dict):
    for value in domain_train_files.values():
        if isinstance(value, str):
            paths.append(value)
        else:
            paths.extend(str(item) for item in value)

audit = config.get("audit") or {}
value = audit.get("full_gradient_validation_files") or []
if isinstance(value, str):
    paths.append(value)
else:
    paths.extend(str(item) for item in value)

models = config.get("model") or {}
for key in ("student_path", "student_base_path", "math_teacher_path", "code_teacher_path"):
    value = models.get(key)
    if not value:
        continue
    value = str(value)
    if value.startswith(("/", "./", "../", "models/", "checkpoints/")):
        paths.append(value)

missing = []
for item in paths:
    path = Path(item)
    if not path.is_absolute():
        path = code_dir / path
    if not path.exists():
        missing.append(str(path))

if missing:
    print("Missing config data files:", file=sys.stderr)
    for item in missing:
        print(f"  {item}", file=sys.stderr)
    raise SystemExit(2)
PY

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
GPU_CSV="${LOG_DIR}/${RUN_ID}_gpu.csv"
LAUNCH_FILE="${LOG_DIR}/${RUN_ID}.launch.sh"

if [[ "${FOREGROUND}" != "1" ]]; then
  if ! command -v screen >/dev/null 2>&1; then
    echo "screen is required unless --foreground is set." >&2
    exit 2
  fi
  if screen -ls | grep -Eq "[.]${RUN_ID}[[:space:]]"; then
    echo "A screen session named '${RUN_ID}' already exists." >&2
    screen -ls >&2 || true
    exit 2
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_USED="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')"
  if [[ -z "${GPU_USED}" ]]; then
    echo "Could not read GPU ${GPU_ID} memory usage from nvidia-smi." >&2
    nvidia-smi >&2 || true
    exit 3
  fi
  if [[ "${GPU_USED:-999999}" -gt "${GPU_IDLE_MEMORY_LIMIT_MB}" ]]; then
    echo "GPU ${GPU_ID} is not idle: ${GPU_USED} MiB used." >&2
    nvidia-smi >&2 || true
    exit 3
  fi
fi

if [[ "${STOP_STALE_RAY}" == "1" ]]; then
  ray stop --force >/dev/null 2>&1 || true
fi

echo "${RUN_ID}" > "${LOG_DIR}/opd_target_run_id"
echo "${LOG_FILE}" > "${LOG_DIR}/opd_target_log"
echo "${CONFIG_PATH}" > "${LOG_DIR}/opd_target_config"
echo "${GPU_CSV}" > "${LOG_DIR}/opd_target_gpu_csv"

EXTRA_ARGS_Q=""
for arg in "${EXTRA_ARGS[@]}"; do
  EXTRA_ARGS_Q+=" $(quote "${arg}")"
done

DRY_RUN_ENV=""
if [[ "${DRY_RUN_FLAG}" == "1" ]]; then
  DRY_RUN_ENV="DRY_RUN=1 "
fi

cat > "${LAUNCH_FILE}" <<LAUNCH
#!/usr/bin/env bash
set -euo pipefail
if [[ -n $(quote "${CONDA_SH}") && -f $(quote "${CONDA_SH}") ]]; then
  source $(quote "${CONDA_SH}")
  conda activate $(quote "${CONDA_ENV}")
fi
cd $(quote "${CODE_DIR}")
export CUDA_VISIBLE_DEVICES=$(quote "${GPU_ID}")
export PYTHONUNBUFFERED=1
export PYTHONINTMAXSTRDIGITS=0
export VERL_RUNTIME_DIR=$(quote "${VERL_RUNTIME_DIR}")
export PYTHONPATH=$(quote "${CODE_DIR}"):$(quote "${VERL_RUNTIME_DIR}"):\${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-8}
export WANDB_MODE=\${WANDB_MODE:-disabled}
export USED_MODEL=\${USED_MODEL:-no_api}
export MOPD_STEP_PROGRESS=\${MOPD_STEP_PROGRESS:-1}
export MOPD_VLLM_GENERATE_PROGRESS=\${MOPD_VLLM_GENERATE_PROGRESS:-1}

( while true; do
    date '+%F %T'
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    sleep 60
  done ) > $(quote "${GPU_CSV}") 2>&1 &
GPU_MONITOR_PID=\$!
trap 'kill \${GPU_MONITOR_PID} 2>/dev/null || true' EXIT

{
  echo RUN_ID=$(quote "${RUN_ID}")
  echo CONFIG=$(quote "${CONFIG_PATH}")
  echo CODE_DIR=$(quote "${CODE_DIR}")
  echo VERL_RUNTIME_DIR=$(quote "${VERL_RUNTIME_DIR}")
  echo LOG_FILE=$(quote "${LOG_FILE}")
  echo START_TS=\$(date -Is)
  $(printf "%s" "${DRY_RUN_ENV}")MOPD_CONFIG=$(quote "${CONFIG_PATH}") PYTHON_BIN=$(quote "${PYTHON_BIN}") bash scripts/run_math_code_mopd.sh${EXTRA_ARGS_Q}
} 2>&1 | tee -a $(quote "${LOG_FILE}")
LAUNCH
chmod +x "${LAUNCH_FILE}"

echo "== Remote training launch =="
echo "CODE_DIR=${CODE_DIR}"
echo "CONFIG=${CONFIG_PATH}"
echo "VERL_RUNTIME_DIR=${VERL_RUNTIME_DIR}"
echo "RUN_ID=${RUN_ID}"
echo "LOG_FILE=${LOG_FILE}"
echo "LAUNCH_FILE=${LAUNCH_FILE}"
echo

if [[ "${FOREGROUND}" == "1" ]]; then
  exec bash "${LAUNCH_FILE}"
fi

screen -dmS "${RUN_ID}" bash "${LAUNCH_FILE}"
screen -ls
echo
echo "Follow logs:"
echo "tail -f ${LOG_FILE}"

if [[ "${TAIL_LOG}" == "1" ]]; then
  tail -f "${LOG_FILE}"
fi
