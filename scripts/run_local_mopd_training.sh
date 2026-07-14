#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_local_mopd_training.sh <config> [--run-id RUN_ID] [--foreground] [--tail] [--dry-run] [-- <hydra overrides...>]

Examples:
  scripts/run_local_mopd_training.sh configs/mopd_formal_audit_all_2gpu.yaml

  scripts/run_local_mopd_training.sh configs/mopd_formal_audit_all_4gpu.yaml \
    --run-id mopd_manual_test

  scripts/run_local_mopd_training.sh configs/mopd_formal_audit_off_2gpu.yaml \
    --run-id mopd_bsz128 \
    -- data.train_batch_size=128 data.val_batch_size=128 trainer.val_before_train=false

Notes:
  - Run this script from a local OPD-code checkout.
  - Training imports verl from this repo's third_party/verl directory.
  - By default it launches in a detached screen session.
  - When audit is enabled, JSONL files go to <config audit.output_dir>/<RUN_ID>
    unless mopd_audit.output_dir is explicitly passed after '--'.

Environment:
  LOCAL_ROOT=<parent of OPD-code>
  CONDA_ROOT=$HOME/miniconda3
  ENV_NAME=mopd-verl
  GPU_IDS=0,1                # comma- or space-separated visible physical GPUs
  GPU_ID=0                   # legacy alias used only when GPU_IDS is unset
  LOG_DIR=$CODE_DIR/logs
  STOP_STALE_RAY=1
  GPU_IDLE_MEMORY_LIMIT_MB=1000
  MOPD_LOCAL_CONDA_ENV=$CONDA_ROOT/envs/$ENV_NAME
  MOPD_LOCAL_CONDA_ROOT=$CONDA_ROOT
  MOPD_STEP_PROGRESS=1
  MOPD_VLLM_GENERATE_PROGRESS=1
USAGE
}

quote() {
  printf "%q" "$1"
}

has_hydra_override() {
  local wanted_key="$1"
  local arg key
  for arg in "${EXTRA_ARGS[@]}"; do
    [[ "${arg}" == *=* ]] || continue
    key="${arg%%=*}"
    while [[ "${key}" == +* ]]; do
      key="${key#+}"
    done
    if [[ "${key}" == "${wanted_key}" ]]; then
      return 0
    fi
  done
  return 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOCAL_ROOT="${LOCAL_ROOT:-$(cd "${CODE_DIR}/.." && pwd)}"
ENV_NAME="${ENV_NAME:-mopd-verl}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-0,1}}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/logs}"
STOP_STALE_RAY="${STOP_STALE_RAY:-1}"
GPU_IDLE_MEMORY_LIMIT_MB="${GPU_IDLE_MEMORY_LIMIT_MB:-1000}"
VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-${CODE_DIR}/third_party/verl}"
MOPD_LOCAL_CONDA_ENV="${MOPD_LOCAL_CONDA_ENV:-${CONDA_ROOT}/envs/${ENV_NAME}}"
MOPD_LOCAL_CONDA_ROOT="${MOPD_LOCAL_CONDA_ROOT:-${CONDA_ROOT}}"

if [[ -d "${MOPD_LOCAL_CONDA_ENV}/bin" ]]; then
  export PATH="${MOPD_LOCAL_CONDA_ENV}/bin:${MOPD_LOCAL_CONDA_ROOT}/bin:${PATH:-}"
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
trainer = config.get("trainer") or {}
worker_placement = config.get("worker_placement") or (config.get("actor_rollout_ref") or {}).get("worker_placement") or {}
actor_rollout = worker_placement.get("actor_rollout") or {}
ref_policy = worker_placement.get("ref_policy") or {}
trainer_gpus = trainer.get("n_gpus_per_node", 1)
trainer_nnodes = trainer.get("nnodes", 1)
separate_ref_policy = worker_placement.get("separate_ref_policy", False)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def parse_int(value, key):
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{key} must be an integer, got {value!r}") from exc
    if numeric <= 0:
        raise SystemExit(f"{key} must be positive, got {numeric!r}")
    return numeric


def parse_process_on_nodes(value, key):
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().strip("[]")
        value = [] if not cleaned else [part.strip() for part in cleaned.split(",")]
    if not isinstance(value, list) or not value:
        raise SystemExit(f"{key} must be a non-empty list of positive integers, got {value!r}")
    return [parse_int(item, f"{key}[]") for item in value]


def first_node_gpus(pool, default_gpus, key):
    process_on_nodes = parse_process_on_nodes(pool.get("process_on_nodes"), f"{key}.process_on_nodes")
    if process_on_nodes is not None:
        return process_on_nodes[0]
    return parse_int(pool.get("n_gpus_per_node", default_gpus), f"{key}.n_gpus_per_node")


def set_path(root, dotted_key, raw_value):
    parts = dotted_key.split(".")
    cursor = root
    for part in parts[:-1]:
        next_cursor = cursor.get(part)
        if not isinstance(next_cursor, dict):
            next_cursor = {}
            cursor[part] = next_cursor
        cursor = next_cursor
    cursor[parts[-1]] = raw_value.strip().strip("'\"")

for override in sys.argv[2:]:
    if "=" not in override:
        continue
    key, raw_value = override.split("=", 1)
    key = key.lstrip("+")
    value = raw_value.strip().strip("'\"")
    if key == "trainer.n_gpus_per_node":
        trainer_gpus = value
    elif key == "trainer.nnodes":
        trainer_nnodes = value
    elif key.startswith("worker_placement."):
        set_path({"worker_placement": worker_placement}, key, value)
    elif key.startswith("actor_rollout_ref.worker_placement."):
        set_path({"actor_rollout_ref": {"worker_placement": worker_placement}}, key, value)

trainer_gpus = parse_int(trainer_gpus, "trainer.n_gpus_per_node")
parse_int(trainer_nnodes, "trainer.nnodes")
separate_ref_policy = parse_bool(worker_placement.get("separate_ref_policy", separate_ref_policy))
actor_rollout = worker_placement.get("actor_rollout") or {}
ref_policy = worker_placement.get("ref_policy") or {}
required_gpus = first_node_gpus(actor_rollout, trainer_gpus, "worker_placement.actor_rollout")
if separate_ref_policy:
    required_gpus += first_node_gpus(ref_policy, trainer_gpus, "worker_placement.ref_policy")
print(required_gpus)
PY
)"

if [[ "${REQUIRED_GPUS}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  cat >&2 <<EOF
Config requests ${REQUIRED_GPUS} visible GPU(s) across worker pools, but GPU_IDS exposes only ${VISIBLE_GPU_COUNT}: ${GPU_IDS}

Set GPU_IDS to enough physical GPUs, for example:
  GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/run_local_mopd_training.sh ${CONFIG_ARG}

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

if [[ "${DRY_RUN_FLAG}" == "1" ]]; then
  echo "Dry run: skipping GPU idle check."
elif command -v nvidia-smi >/dev/null 2>&1; then
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

AUDIT_OUTPUT_DIR=""
if ! has_hydra_override "mopd_audit.output_dir"; then
  AUDIT_CONFIG_INFO="$(python - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
audit = config.get("audit") or {}
enabled = bool(audit.get("enabled", False))
output_dir = str(audit.get("output_dir") or "mopd_audit")
print(f"{int(enabled)}\t{output_dir}")
PY
)"
  AUDIT_ENABLED="${AUDIT_CONFIG_INFO%%$'\t'*}"
  AUDIT_OUTPUT_BASE="${AUDIT_CONFIG_INFO#*$'\t'}"
  if [[ "${AUDIT_ENABLED}" == "1" ]]; then
    AUDIT_OUTPUT_BASE="${AUDIT_OUTPUT_BASE%/}"
    if [[ -z "${AUDIT_OUTPUT_BASE}" ]]; then
      AUDIT_OUTPUT_BASE="mopd_audit"
    fi
    AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_BASE}/${RUN_ID}"
    EXTRA_ARGS=("++mopd_audit.output_dir=${AUDIT_OUTPUT_DIR}" "${EXTRA_ARGS[@]}")
  fi
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
if [[ -d $(quote "${MOPD_LOCAL_CONDA_ENV}")/bin ]]; then
  export PATH=$(quote "${MOPD_LOCAL_CONDA_ENV}")/bin:$(quote "${MOPD_LOCAL_CONDA_ROOT}")/bin:\${PATH:-}
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
  echo AUDIT_OUTPUT_DIR=$(quote "${AUDIT_OUTPUT_DIR:-manual_or_disabled}")
  echo START_TS=\$(date -Is)
  echo PYTHON_BIN=\$(command -v python)
  python --version
  $(printf "%s" "${DRY_RUN_ENV}")bash scripts/run_mopd.sh $(quote "${CONFIG_PATH}")${RUN_MOPD_EXTRA_ARGS_Q}
} 2>&1 | tee -a $(quote "${LOG_FILE}")
LAUNCH
chmod +x "${LAUNCH_FILE}"

echo "== Local training launch =="
echo "CODE_DIR=${CODE_DIR}"
echo "CONFIG=${CONFIG_PATH}"
echo "VERL_RUNTIME_DIR=${VERL_RUNTIME_DIR}"
echo "MOPD_LOCAL_CONDA_ENV=${MOPD_LOCAL_CONDA_ENV}"
echo "GPU_IDS=${GPU_IDS}"
echo "REQUIRED_GPUS=${REQUIRED_GPUS}"
echo "RUN_ID=${RUN_ID}"
echo "LOG_FILE=${LOG_FILE}"
echo "LAUNCH_FILE=${LAUNCH_FILE}"
if [[ -n "${AUDIT_OUTPUT_DIR}" ]]; then
  echo "AUDIT_OUTPUT_DIR=${AUDIT_OUTPUT_DIR}"
fi
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
