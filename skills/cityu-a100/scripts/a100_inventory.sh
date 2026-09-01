#!/usr/bin/env bash
set -euo pipefail

require_command() {
  local command_name="$1"
  command -v "${command_name}" >/dev/null 2>&1 || {
    printf 'Missing required command: %s\n' "${command_name}" >&2
    exit 2
  }
}

extract_field() {
  local detail="$1"
  local field_name="$2"
  local token
  for token in ${detail}; do
    if [[ "${token}" == "${field_name}="* ]]; then
      printf '%s\n' "${token#*=}"
      return
    fi
  done
  printf '%s\n' ""
}

a100_count_from_gres() {
  local gres="$1"
  if [[ "${gres}" =~ gpu:a100:([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    printf '0\n'
  fi
}

a100_count_from_tres() {
  local tres="$1"
  if [[ "${tres}" =~ gres/gpu:a100=([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    printf '0\n'
  fi
}

partition_list() {
  local node="$1"
  local partitions
  partitions="$(
    sinfo -a -N -h -n "${node}" -o '%P' 2>/dev/null |
      sed 's/\*$//' |
      sort -u |
      paste -sd, -
  )"
  printf '%s\n' "${partitions:-none}"
}

require_command scontrol
require_command sinfo

printf 'snapshot=%s\n' "$(date -Is)"
printf 'node|partitions|state|registered_a100|configured_a100|allocated_a100|slurm_free_a100|capacity_requestable_a100\n'

for suffix in $(seq -w 1 12); do
  node="gpu-a100-${suffix}"
  if ! detail="$(scontrol show node -o "${node}" 2>/dev/null)"; then
    printf '%s|none|ABSENT_OR_HIDDEN|0|0|0|0|0\n' "${node}"
    continue
  fi

  gres="$(extract_field "${detail}" Gres)"
  state="$(extract_field "${detail}" State)"
  cfg_tres="$(extract_field "${detail}" CfgTRES)"
  alloc_tres="$(extract_field "${detail}" AllocTRES)"
  registered="$(a100_count_from_gres "${gres}")"
  configured="$(a100_count_from_tres "${cfg_tres}")"
  allocated="$(a100_count_from_tres "${alloc_tres}")"
  free_count=$((configured - allocated))
  if ((free_count < 0)); then
    free_count=0
  fi
  capacity_requestable="${free_count}"
  case "${state}" in
    *DOWN*|*DRAIN*|*NOT_RESPONDING*|*RESERVED*) capacity_requestable=0 ;;
  esac

  printf '%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "${node}" \
    "$(partition_list "${node}")" \
    "${state:-unknown}" \
    "${registered}" \
    "${configured}" \
    "${allocated}" \
    "${free_count}" \
    "${capacity_requestable}"
done

printf '%s\n' \
  'NOTE: slurm_free_a100 is scheduler capacity, not a CUDA health verdict.' \
  'capacity_requestable_a100 removes obvious node-state blocks, not account or health blocks.' \
  'Run the recovery-state and seeded-kernel checks inside an allocation.'
