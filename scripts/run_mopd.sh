#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_mopd.sh [config[::profile]] [--dry-run] [--slurm] [--slurm-args <sbatch directive>]... [-- <hydra overrides...>]

Examples:
  scripts/run_mopd.sh configs/mopd_formal_audit_all_2gpu.yaml --dry-run

  scripts/run_mopd.sh \
    test_grad_configs/mopd_grad_reliability_qwen0p6b_8b_matrix.yaml::aw2_fsdp2_audit_on \
    --dry-run

  scripts/run_mopd.sh configs/mopd_formal_audit_off_2gpu.yaml -- \
    trainer.experiment_name=mopd_audit_off_manual

  # Submit as a Slurm job
  scripts/run_mopd.sh configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science.yaml --slurm

  # With extra Slurm directives (can be repeated)
  scripts/run_mopd.sh configs/... --slurm --slurm-args "--partition=gpu" --slurm-args "--time=24:00:00"

Environment:
  MOPD_CONFIG=<default config when config arg is omitted>
  VERL_RUNTIME_DIR=<vendored verl runtime dir>
  MOPD_LAUNCH_PYTHON=<python executable for this launcher, default: python3>
  SLURM_LOG_DIR=<Slurm script/log directory, default: CODE_DIR/logs/slurm>
  SLURM_EXTRA_ENV=<space-separated sbatch directives, alternative to --slurm-args>
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${CODE_DIR}/configs/mopd_formal_audit_all_2gpu.yaml"
CONFIG_PATH="${MOPD_CONFIG:-${DEFAULT_CONFIG}}"
VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-${CODE_DIR}/third_party/verl}"
DRY_RUN_FLAG=0
SLURM_FLAG=0
SLURM_EXTRA_DIRECTIVES=()
# Parse SLURM_EXTRA from env var, if set (space-separated sbatch directives)
if [[ -n "${SLURM_EXTRA_ENV:-}" ]]; then
  read -r -a SLURM_EXTRA_DIRECTIVES <<< "${SLURM_EXTRA_ENV}"
fi
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN_FLAG=1
      shift
      ;;
    --slurm)
      SLURM_FLAG=1
      shift
      ;;
    --slurm-args)
      if [[ $# -lt 2 ]]; then
        echo "--slurm-args requires a value" >&2
        exit 2
      fi
      SLURM_EXTRA_DIRECTIVES+=("$2")
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

CONFIG_PROFILE=""
if [[ "${CONFIG_PATH}" == *::* ]]; then
  CONFIG_PROFILE="${CONFIG_PATH##*::}"
  CONFIG_PATH="${CONFIG_PATH%::*}"
fi

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${CODE_DIR}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi
CONFIG_REFERENCE="${CONFIG_PATH}"
if [[ -n "${CONFIG_PROFILE}" ]]; then
  CONFIG_REFERENCE="${CONFIG_REFERENCE}::${CONFIG_PROFILE}"
fi

if [[ ! -f "${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" ]]; then
  echo "Vendored verl runtime not found at '${VERL_RUNTIME_DIR}'." >&2
  echo "Expected '${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py'." >&2
  echo "Sync or restore third_party/verl before launching training." >&2
  exit 2
fi

export PYTHONPATH="${CODE_DIR}:${VERL_RUNTIME_DIR}:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"
cd "${CODE_DIR}"

ARGS=(--config "${CONFIG_REFERENCE}")
if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN_FLAG}" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  ARGS+=(-- "${EXTRA_ARGS[@]}")
fi

if [[ "${SLURM_FLAG}" == "1" ]]; then
  # Derive the Slurm resources from the selected config's worker pools.
  SLURM_RESOURCE_LINE="$(
    "${MOPD_LAUNCH_PYTHON:-python3}" - \
      "${CONFIG_REFERENCE}" "${EXTRA_ARGS[@]}" <<'PY'
import sys

from mopd_verl.config_profiles import load_raw_config

config = load_raw_config(sys.argv[1])
placement = config.get("worker_placement") or (
    (config.get("actor_rollout_ref") or {}).get("worker_placement") or {}
)


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
        value = [] if not cleaned else [
            part.strip() for part in cleaned.split(",")
        ]
    if not isinstance(value, list) or not value:
        raise SystemExit(
            f"{key} must be a non-empty list of positive integers, "
            f"got {value!r}"
        )
    return [parse_int(item, f"{key}[]") for item in value]


def pool_gpus(pool, default, key):
    process_on_nodes = parse_process_on_nodes(
        pool.get("process_on_nodes"),
        f"{key}.process_on_nodes",
    )
    if process_on_nodes is not None:
        return process_on_nodes[0]
    return parse_int(
        pool.get("n_gpus_per_node", default),
        f"{key}.n_gpus_per_node",
    )


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
    if key in {
        "trainer.n_gpus_per_node",
        "trainer.experiment_name",
        "ray_kwargs.ray_init.num_cpus",
    }:
        set_path(config, key, raw_value)
    elif key.startswith("worker_placement."):
        set_path({"worker_placement": placement}, key, raw_value)
    elif key.startswith("actor_rollout_ref.worker_placement."):
        set_path(
            {"actor_rollout_ref": {"worker_placement": placement}},
            key,
            raw_value,
        )

trainer = config.get("trainer") or {}
actor_pool = placement.get("actor_rollout") or {}
ref_pool = placement.get("ref_policy") or {}
trainer_gpus = parse_int(
    trainer.get("n_gpus_per_node", 1),
    "trainer.n_gpus_per_node",
)
required_gpus = pool_gpus(
    actor_pool,
    trainer_gpus,
    "worker_placement.actor_rollout",
)
if parse_bool(placement.get("separate_ref_policy", False)):
    required_gpus += pool_gpus(
        ref_pool,
        trainer_gpus,
        "worker_placement.ref_policy",
    )
ray_init = (config.get("ray_kwargs") or {}).get("ray_init") or {}
required_cpus = parse_int(
    ray_init.get("num_cpus", max(8, required_gpus * 4)),
    "ray_kwargs.ray_init.num_cpus",
)
# Server has 192 CPUs.  If the config asks for fewer than 72,
# double it (minimum 64) so Ray and vLLM have enough headroom.
if required_cpus < 72:
    required_cpus = max(64, required_cpus * 2)
experiment_name = trainer.get("experiment_name", "mopd_training")
gpu_ids = ",".join(str(index) for index in range(required_gpus))
print(experiment_name, required_gpus, required_cpus, gpu_ids, sep="\t")
PY
  )"
  IFS=$'\t' read -r \
    EXPERIMENT_NAME SLURM_GPUS SLURM_CPUS SLURM_GPU_IDS \
    <<< "${SLURM_RESOURCE_LINE}"
  JOB_NAME="mopd_${EXPERIMENT_NAME}"
  SLURM_LOG_DIR="${SLURM_LOG_DIR:-${CODE_DIR}/logs/slurm}"
  mkdir -p "${SLURM_LOG_DIR}"
  SLURM_LOG="${SLURM_LOG_DIR}/${JOB_NAME}_%j.log"

  # Build safely-quoted command args for the sbatch script
  LAUNCH_ARGS=()
  for _arg in "${ARGS[@]}"; do
    LAUNCH_ARGS+=("$(printf '%q' "${_arg}")")
  done

  SBATCH_SCRIPT="${SLURM_LOG_DIR}/${JOB_NAME}_$$.sbatch"
  cat > "${SBATCH_SCRIPT}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${SLURM_LOG}
#SBATCH --error=${SLURM_LOG}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=${SLURM_GPUS}
#SBATCH --cpus-per-task=${SLURM_CPUS}
$(for _arg in "${SLURM_EXTRA_DIRECTIVES[@]}"; do echo "#SBATCH ${_arg}"; done)

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${VERL_RUNTIME_DIR}:\${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"
export CUDA_VISIBLE_DEVICES=${SLURM_GPU_IDS}

exec "${MOPD_LAUNCH_PYTHON:-python3}" -m mopd_verl.launch ${LAUNCH_ARGS[*]}
SBATCH

  chmod +x "${SBATCH_SCRIPT}"
  echo "Submitting Slurm job: ${JOB_NAME}"
  echo "Config: ${CONFIG_REFERENCE}"
  echo "Slurm script: ${SBATCH_SCRIPT}"

  SBATCH_OUTPUT="$(sbatch "${SBATCH_SCRIPT}")"
  echo "${SBATCH_OUTPUT}"

  JOB_ID="$(echo "${SBATCH_OUTPUT}" | grep -oE '[0-9]+' | head -1)"
  echo ""
  echo "============================================"
  echo "Job submitted!"
  echo "  Job ID:    ${JOB_ID}"
  echo "  Job Name:  ${JOB_NAME}"
  echo "  Log file:  ${SLURM_LOG//%j/${JOB_ID}}"
  echo "============================================"
  echo ""
  echo "View job status:"
  echo "  squeue -j ${JOB_ID}"
  echo "  scontrol show job ${JOB_ID}"
  echo ""
  echo "Follow console output:"
  echo "  tail -f ${SLURM_LOG//%j/${JOB_ID}}"
  echo ""
  echo "Cancel job:"
  echo "  scancel ${JOB_ID}"
  exit 0
fi

exec "${MOPD_LAUNCH_PYTHON:-python3}" -m mopd_verl.launch "${ARGS[@]}"
