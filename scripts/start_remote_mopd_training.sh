#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/start_remote_mopd_training.sh <config> [--run-id RUN_ID] [--foreground] [--tail] [--dry-run] [-- <hydra overrides...>]

Examples:
  scripts/start_remote_mopd_training.sh configs/mopd_formal_audit_all_2gpu.yaml

  scripts/start_remote_mopd_training.sh configs/mopd_formal_audit_all_4gpu.yaml \
    --run-id mopd_manual_test

  scripts/start_remote_mopd_training.sh configs/mopd_formal_audit_off_2gpu.yaml \
    --run-id mopd_bsz128 \
    -- data.train_batch_size=128 data.val_batch_size=128 trainer.val_before_train=false

Notes:
  - Run this script on the remote host from a synced OPD-code checkout.
  - It does not SSH and does not sync files.
  - Training imports verl from this repo's third_party/verl directory.
  - By default it launches in a detached screen session.

Environment:
  REMOTE_ROOT=<parent of OPD-code>
  GPU_IDS=0,1                # comma- or space-separated visible physical GPUs
  GPU_ID=0                   # legacy alias used only when GPU_IDS is unset
  LOG_DIR=$CODE_DIR/logs
  STOP_STALE_RAY=1
  GPU_IDLE_MEMORY_LIMIT_MB=1000
  MOPD_REMOTE_CONDA_ENV=/root/miniconda3/envs/mopd-verl
  MOPD_REMOTE_CONDA_ROOT=/root/miniconda3
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
GPU_IDS="${GPU_IDS:-${GPU_ID:-0,1}}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/logs}"
STOP_STALE_RAY="${STOP_STALE_RAY:-1}"
GPU_IDLE_MEMORY_LIMIT_MB="${GPU_IDLE_MEMORY_LIMIT_MB:-1000}"
VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-${CODE_DIR}/third_party/verl}"
MOPD_REMOTE_CONDA_ENV="${MOPD_REMOTE_CONDA_ENV:-/root/miniconda3/envs/mopd-verl}"
MOPD_REMOTE_CONDA_ROOT="${MOPD_REMOTE_CONDA_ROOT:-/root/miniconda3}"

if [[ -d "${MOPD_REMOTE_CONDA_ENV}/bin" ]]; then
  export PATH="${MOPD_REMOTE_CONDA_ENV}/bin:${MOPD_REMOTE_CONDA_ROOT}/bin:${PATH:-}"
fi

GPU_IDS="${GPU_IDS//$'\t'/,}"
GPU_IDS="${GPU_IDS// /,}"
while [[ "${GPU_IDS}" == *",,"* ]]; do
  GPU_IDS="${GPU_IDS//,,/,}"
done
GPU_IDS="${GPU_IDS#,}"
GPU_IDS="${GPU_IDS%,}"
if [[ -z "${GPU_IDS}" ]]; then
  echo "GPU_IDS cannot be empty." >&2
  exit 2
fi
IFS=',' read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
VISIBLE_GPU_COUNT="${#GPU_ID_LIST[@]}"

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

if [[ ! -f "${SCRIPT_DIR}/run_mopd.sh" ]]; then
  echo "Missing training wrapper: ${SCRIPT_DIR}/run_mopd.sh" >&2
  exit 2
fi

if [[ ! -f "${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" ]]; then
  echo "Vendored verl runtime not found: ${VERL_RUNTIME_DIR}" >&2
  echo "Expected ${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" >&2
  exit 2
fi

export PYTHONPATH="${CODE_DIR}:${VERL_RUNTIME_DIR}:${PYTHONPATH:-}"

python - "${CONFIG_PATH}" "${CODE_DIR}" <<'PY'
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

models = config.get("model") or {}
for key in (
    "student_path",
    "student_base_path",
    "math_teacher_path",
    "code_teacher_path",
    "primary_teacher_path",
    "secondary_teacher_path",
    "reasoning_teacher_path",
):
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

REQUIRED_GPUS="$(python - "${CONFIG_PATH}" "${EXTRA_ARGS[@]}" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
value = (config.get("trainer") or {}).get("n_gpus_per_node", 1)

for override in sys.argv[2:]:
    if "=" not in override:
        continue
    key, raw_value = override.split("=", 1)
    if key.lstrip("+") == "trainer.n_gpus_per_node":
        value = raw_value.strip().strip("'\"")

try:
    print(int(value))
except (TypeError, ValueError) as exc:
    raise SystemExit(f"trainer.n_gpus_per_node must be an integer, got {value!r}") from exc
PY
)"

if [[ "${REQUIRED_GPUS}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  cat >&2 <<EOF
Config requests trainer.n_gpus_per_node=${REQUIRED_GPUS}, but GPU_IDS exposes only ${VISIBLE_GPU_COUNT} GPU(s): ${GPU_IDS}

Set GPU_IDS to enough physical GPUs, for example:
  GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/start_remote_mopd_training.sh ${CONFIG_ARG}

Or pass a compatible config/override set after '--'.
EOF
  exit 2
fi

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
  for gpu_id in "${GPU_ID_LIST[@]}"; do
    GPU_USED="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9')"
    if [[ -z "${GPU_USED}" ]]; then
      echo "Could not read GPU ${gpu_id} memory usage from nvidia-smi." >&2
      nvidia-smi >&2 || true
      exit 3
    fi
    if [[ "${GPU_USED:-999999}" -gt "${GPU_IDLE_MEMORY_LIMIT_MB}" ]]; then
      echo "GPU ${gpu_id} is not idle: ${GPU_USED} MiB used." >&2
      nvidia-smi >&2 || true
      exit 3
    fi
  done
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
RUN_MOPD_EXTRA_ARGS_Q=""
if [[ -n "${EXTRA_ARGS_Q}" ]]; then
  RUN_MOPD_EXTRA_ARGS_Q=" --${EXTRA_ARGS_Q}"
fi

DRY_RUN_ENV=""
if [[ "${DRY_RUN_FLAG}" == "1" ]]; then
  DRY_RUN_ENV="DRY_RUN=1 "
fi

cat > "${LAUNCH_FILE}" <<LAUNCH
#!/usr/bin/env bash
set -euo pipefail
cd $(quote "${CODE_DIR}")
if [[ -d $(quote "${MOPD_REMOTE_CONDA_ENV}")/bin ]]; then
  export PATH=$(quote "${MOPD_REMOTE_CONDA_ENV}")/bin:$(quote "${MOPD_REMOTE_CONDA_ROOT}")/bin:\${PATH:-}
fi
export CUDA_VISIBLE_DEVICES=$(quote "${GPU_IDS}")
export PYTHONUNBUFFERED=1
export PYTHONINTMAXSTRDIGITS=0
export VERL_RUNTIME_DIR=$(quote "${VERL_RUNTIME_DIR}")
export PYTHONPATH=$(quote "${CODE_DIR}"):$(quote "${VERL_RUNTIME_DIR}"):\${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-8}
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
  echo CUDA_VISIBLE_DEVICES=$(quote "${GPU_IDS}")
  echo LOG_FILE=$(quote "${LOG_FILE}")
  echo START_TS=\$(date -Is)
  echo PYTHON_BIN=\$(command -v python)
  python --version
  $(printf "%s" "${DRY_RUN_ENV}")bash scripts/run_mopd.sh $(quote "${CONFIG_PATH}")${RUN_MOPD_EXTRA_ARGS_Q}
} 2>&1 | tee -a $(quote "${LOG_FILE}")
LAUNCH
chmod +x "${LAUNCH_FILE}"

echo "== Remote training launch =="
echo "CODE_DIR=${CODE_DIR}"
echo "CONFIG=${CONFIG_PATH}"
echo "VERL_RUNTIME_DIR=${VERL_RUNTIME_DIR}"
echo "GPU_IDS=${GPU_IDS}"
echo "REQUIRED_GPUS=${REQUIRED_GPUS}"
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
