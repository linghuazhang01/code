#!/usr/bin/env bash
set -euo pipefail
export LANG=C
export LC_ALL=C

usage() {
  cat <<'USAGE'
Usage:
  ./slurm.sh <config[::profile]> [--dry-run|--check-only]

Examples:
  ./slurm.sh code/configs/mopd_qwen1p7b_30b_a3b_instruct_2507_6gpu_math_code_science_topk32.yaml
  ./slurm.sh configs/mopd_formal_audit_all_2gpu.yaml
  ./slurm.sh test_grad_configs/mopd_grad_reliability_qwen0p6b_8b_matrix.yaml::aw2_fsdp2_audit_on

The config must live inside the local code/ directory. A normal run performs:
  1. remote environment and GPU preflight;
  2. rsync --dry-run for the selected config;
  3. config upload without --delete;
  4. SHA-256 and remote launcher validation;
  5. Slurm submission with the fixed resource directives below.

--dry-run stops after the rsync preview and does not upload or submit.
--check-only uploads and validates the config but does not submit.

Optional environment overrides:
  MOPD_REMOTE_DIR       default: /home/shuang_qiu/mopd_code
  MOPD_REMOTE_PYTHON    default: /home/shuang_qiu/env/miniconda3/envs/mopd-verl/bin/python
  MOPD_SLURM_PARTITION  default: compute
  MOPD_SLURM_MEMORY     default: 600G
  MOPD_SLURM_TIME       default: 72:00:00
  MOPD_SLURM_PRIORITY   default: 1000000 (cluster baseline is 1)
USAGE
}

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

quote_shell() {
  printf '%q' "$1"
}

remote_exec() {
  local remote_command="$1"

  MOPD_REMOTE_COMMAND="$remote_command" \
    MOPD_SSH_HELPER="$SSH_HELPER" \
    expect <<'EXPECT'
set timeout 120

proc parse_helper {helper_path} {
  set handle [open $helper_path r]
  set payload [string map {"\r\n" "\n"} [read $handle]]
  close $handle
  if {[string length $payload] > 0 && [string index $payload end] eq "\n"} {
    set payload [string range $payload 0 end-1]
  }
  set lines [split $payload "\n"]
  if {[llength $lines] != 2} {
    puts stderr "ssh.sh must contain exactly two lines."
    exit 2
  }

  set ssh_line [string trim [lindex $lines 0]]
  set password [string trimright [lindex $lines 1] "\r"]
  if {$password eq ""} {
    puts stderr "ssh.sh password line must not be empty."
    exit 2
  }
  if {[regexp {['"\\]} $ssh_line]} {
    puts stderr "ssh.sh quoting is unsupported; use plain SSH arguments."
    exit 2
  }

  set tokens [split $ssh_line]
  if {[lindex $tokens 0] ne "ssh"} {
    puts stderr "ssh.sh line 1 must start with ssh."
    exit 2
  }
  set ssh_args [list ssh]
  set target ""
  for {set index 1} {$index < [llength $tokens]} {incr index} {
    set arg [lindex $tokens $index]
    if {$target ne ""} {
      if {$index == [expr {[llength $tokens] - 1}] && $arg in {"bash" "sh" "zsh"}} {
        continue
      }
      puts stderr "Unexpected token after SSH target in ssh.sh."
      exit 2
    }
    if {$arg in {"-p" "-i" "-o" "-F" "-J" "-S" "-c" "-l"}} {
      incr index
      if {$index >= [llength $tokens]} {
        puts stderr "SSH option requires a value in ssh.sh."
        exit 2
      }
      lappend ssh_args $arg [lindex $tokens $index]
    } elseif {$arg in {"-4" "-6" "-A" "-a" "-C" "-q" "-T" "-v"}} {
      lappend ssh_args $arg
    } elseif {[string match "-*" $arg]} {
      puts stderr "Unsupported SSH option in ssh.sh: $arg"
      exit 2
    } else {
      set target $arg
      lappend ssh_args $arg
    }
  }
  if {$target eq ""} {
    puts stderr "Could not parse the SSH target from ssh.sh."
    exit 2
  }
  return [list $ssh_args $password]
}

lassign [parse_helper $env(MOPD_SSH_HELPER)] ssh_args password

set remote_command $env(MOPD_REMOTE_COMMAND)
set wrapped_command "printf '%s\\n' __MOPD_REMOTE_START__; sleep 1; $remote_command"
set password_prompts 0
log_user 0
spawn {*}$ssh_args $wrapped_command

expect {
  -re "(?i)password:" {
    incr password_prompts
    if {$password_prompts > 1} {
      puts stderr "SSH authentication failed."
      exit 255
    }
    send -- "$password\r"
    exp_continue
  }
  -re "__MOPD_REMOTE_START__\\r?\\n" {
    log_user 1
    exp_continue
  }
  -re "(?i)permission denied" {
    puts stderr "SSH authentication failed."
    exit 255
  }
  timeout {
    puts stderr "SSH command timed out."
    exit 124
  }
  eof {
    log_user 0
    catch wait result
    exit [lindex $result 3]
  }
}
EXPECT
}

rsync_config() {
  local dry_run="$1"

  MOPD_LOCAL_CONFIG="$LOCAL_CONFIG" \
    MOPD_REMOTE_PARENT="$REMOTE_CONFIG_PARENT" \
    MOPD_RSYNC_DRY_RUN="$dry_run" \
    MOPD_SSH_HELPER="$SSH_HELPER" \
    expect <<'EXPECT'
set timeout 120

proc parse_helper {helper_path} {
  set handle [open $helper_path r]
  set payload [string map {"\r\n" "\n"} [read $handle]]
  close $handle
  if {[string length $payload] > 0 && [string index $payload end] eq "\n"} {
    set payload [string range $payload 0 end-1]
  }
  set lines [split $payload "\n"]
  if {[llength $lines] != 2} {
    puts stderr "ssh.sh must contain exactly two lines."
    exit 2
  }

  set ssh_line [string trim [lindex $lines 0]]
  set password [string trimright [lindex $lines 1] "\r"]
  if {$password eq ""} {
    puts stderr "ssh.sh password line must not be empty."
    exit 2
  }
  if {[regexp {['"\\]} $ssh_line]} {
    puts stderr "ssh.sh quoting is unsupported; use plain SSH arguments."
    exit 2
  }

  set tokens [split $ssh_line]
  if {[lindex $tokens 0] ne "ssh"} {
    puts stderr "ssh.sh line 1 must start with ssh."
    exit 2
  }
  set transport [list ssh]
  set target ""
  for {set index 1} {$index < [llength $tokens]} {incr index} {
    set arg [lindex $tokens $index]
    if {$target ne ""} {
      if {$index == [expr {[llength $tokens] - 1}] && $arg in {"bash" "sh" "zsh"}} {
        continue
      }
      puts stderr "Unexpected token after SSH target in ssh.sh."
      exit 2
    }
    if {$arg in {"-p" "-i" "-o" "-F" "-J" "-S" "-c" "-l"}} {
      incr index
      if {$index >= [llength $tokens]} {
        puts stderr "SSH option requires a value in ssh.sh."
        exit 2
      }
      lappend transport $arg [lindex $tokens $index]
    } elseif {$arg in {"-4" "-6" "-A" "-a" "-C" "-q" "-T" "-v"}} {
      lappend transport $arg
    } elseif {[string match "-*" $arg]} {
      puts stderr "Unsupported SSH option in ssh.sh: $arg"
      exit 2
    } else {
      set target $arg
    }
  }
  if {$target eq ""} {
    puts stderr "Could not parse the SSH target from ssh.sh."
    exit 2
  }
  return [list $transport $target $password]
}

lassign [parse_helper $env(MOPD_SSH_HELPER)] transport target password

set args [list rsync -avz]
if {$env(MOPD_RSYNC_DRY_RUN) eq "1"} {
  lappend args "--dry-run"
}
lappend args \
  "-e" [join $transport " "] \
  $env(MOPD_LOCAL_CONFIG) \
  [format "%s:%s/" $target $env(MOPD_REMOTE_PARENT)]

set password_prompts 0
log_user 0
spawn {*}$args
expect {
  -re "(?i)password:" {
    incr password_prompts
    if {$password_prompts > 1} {
      puts stderr "SSH authentication failed during rsync."
      exit 255
    }
    send -- "$password\r"
    log_user 1
    exp_continue
  }
  -re "(?i)permission denied" {
    puts stderr "SSH authentication failed during rsync."
    exit 255
  }
  timeout {
    puts stderr "rsync timed out."
    exit 124
  }
  eof {
    log_user 0
    catch wait result
    exit [lindex $result 3]
  }
}
EXPECT
}

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "${CODE_DIR}/.." && pwd -P)"
SSH_HELPER="${ROOT_DIR}/ssh.sh"
REMOTE_DIR="${MOPD_REMOTE_DIR:-/home/shuang_qiu/mopd_code}"
REMOTE_PYTHON="${MOPD_REMOTE_PYTHON:-/home/shuang_qiu/env/miniconda3/envs/mopd-verl/bin/python}"
SLURM_PARTITION="${MOPD_SLURM_PARTITION:-compute}"
SLURM_MEMORY="${MOPD_SLURM_MEMORY:-600G}"
SLURM_TIME="${MOPD_SLURM_TIME:-72:00:00}"
SLURM_PRIORITY="${MOPD_SLURM_PRIORITY:-1000000}"
DRY_RUN=0
CHECK_ONLY=0

[[ $# -ge 1 ]] || {
  usage >&2
  exit 2
}

case "$1" in
  -h|--help)
    usage
    exit 0
    ;;
esac

CONFIG_ARGUMENT="$1"
shift
case "${1:-}" in
  --dry-run)
    DRY_RUN=1
    shift
    ;;
  --check-only)
    CHECK_ONLY=1
    shift
    ;;
esac
[[ $# -eq 0 ]] || fail "unexpected argument: $1"

CONFIG_PROFILE=""
CONFIG_PATH="$CONFIG_ARGUMENT"
if [[ "$CONFIG_PATH" == *::* ]]; then
  CONFIG_PROFILE="${CONFIG_PATH##*::}"
  CONFIG_PATH="${CONFIG_PATH%::*}"
  [[ -n "$CONFIG_PROFILE" ]] || fail "config profile cannot be empty"
fi

if [[ "$CONFIG_PATH" != /* ]]; then
  if [[ -f "${ROOT_DIR}/${CONFIG_PATH}" ]]; then
    CONFIG_PATH="${ROOT_DIR}/${CONFIG_PATH}"
  elif [[ -f "${CODE_DIR}/${CONFIG_PATH}" ]]; then
    CONFIG_PATH="${CODE_DIR}/${CONFIG_PATH}"
  else
    fail "config not found: $CONFIG_PATH"
  fi
fi

[[ -f "$CONFIG_PATH" ]] || fail "config not found: $CONFIG_PATH"
LOCAL_CONFIG="$(cd "$(dirname "$CONFIG_PATH")" && pwd -P)/$(basename "$CONFIG_PATH")"
case "$LOCAL_CONFIG" in
  "${CODE_DIR}"/*) ;;
  *) fail "config must be located inside ${CODE_DIR}" ;;
esac

[[ -f "$SSH_HELPER" ]] || fail "missing SSH helper: $SSH_HELPER"
[[ -s "$LOCAL_CONFIG" ]] || fail "config is empty: $LOCAL_CONFIG"
SSH_MODE="$(stat -f '%Lp' "$SSH_HELPER" 2>/dev/null || stat -c '%a' "$SSH_HELPER")"
[[ "$SSH_MODE" =~ ^[0-7]00$ ]] || \
  fail "ssh.sh must not be readable or writable by group/others (current mode: $SSH_MODE)"
command -v expect >/dev/null 2>&1 || fail "expect is required"
command -v rsync >/dev/null 2>&1 || fail "rsync is required"
command -v shasum >/dev/null 2>&1 || fail "shasum is required"

RELATIVE_CONFIG="${LOCAL_CONFIG#${CODE_DIR}/}"
[[ "$RELATIVE_CONFIG" =~ ^[A-Za-z0-9._/-]+$ ]] || \
  fail "config path may only contain letters, digits, '.', '_', '-', and '/'"
case "/${RELATIVE_CONFIG}/" in
  *"//"*|*"/../"*|*"/./"*) fail "config path contains an unsafe segment" ;;
esac
[[ "$CONFIG_PROFILE" =~ ^[A-Za-z0-9_.-]*$ ]] || \
  fail "config profile may only contain letters, digits, '.', '_', and '-'"
[[ "$REMOTE_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]] || \
  fail "MOPD_REMOTE_DIR must be an absolute path without spaces or shell metacharacters"
case "${REMOTE_DIR}/" in
  *"//"*|*"/../"*|*"/./"*) fail "MOPD_REMOTE_DIR contains an unsafe segment" ;;
esac
[[ "$REMOTE_PYTHON" =~ ^/[A-Za-z0-9._/-]+$ ]] || \
  fail "MOPD_REMOTE_PYTHON must be an absolute path without spaces or shell metacharacters"
case "${REMOTE_PYTHON}/" in
  *"//"*|*"/../"*|*"/./"*) fail "MOPD_REMOTE_PYTHON contains an unsafe segment" ;;
esac
[[ "$SLURM_PARTITION" =~ ^[A-Za-z0-9_.-]+$ ]] || \
  fail "MOPD_SLURM_PARTITION contains unsupported characters"
[[ "$SLURM_MEMORY" =~ ^[1-9][0-9]*[KMGTP]?$ ]] || \
  fail "MOPD_SLURM_MEMORY must look like 600G or 102400"
[[ "$SLURM_TIME" =~ ^([0-9]+-)?[0-9][0-9]?[0-9]?:[0-5][0-9]:[0-5][0-9]$ ]] || \
  fail "MOPD_SLURM_TIME must use [days-]hours:minutes:seconds"
[[ "$SLURM_PRIORITY" =~ ^[1-9][0-9]*$ ]] || \
  fail "MOPD_SLURM_PRIORITY must be a positive integer"
REMOTE_CONFIG="${REMOTE_DIR}/${RELATIVE_CONFIG}"
REMOTE_CONFIG_PARENT="$(dirname "$REMOTE_CONFIG")"
REMOTE_CONFIG_REFERENCE="$REMOTE_CONFIG"
if [[ -n "$CONFIG_PROFILE" ]]; then
  REMOTE_CONFIG_REFERENCE="${REMOTE_CONFIG_REFERENCE}::${CONFIG_PROFILE}"
fi

printf 'Config: %s\n' "$LOCAL_CONFIG"
printf 'Remote: %s\n' "$REMOTE_CONFIG_REFERENCE"
printf 'Slurm: partition=%s mem=%s time=%s priority=%s\n' \
  "$SLURM_PARTITION" "$SLURM_MEMORY" "$SLURM_TIME" "$SLURM_PRIORITY"

quoted_remote_dir="$(quote_shell "$REMOTE_DIR")"
quoted_remote_python="$(quote_shell "$REMOTE_PYTHON")"
quoted_remote_parent="$(quote_shell "$REMOTE_CONFIG_PARENT")"

printf '\n[1/5] Remote preflight\n'
remote_exec "set -eu; test -d ${quoted_remote_dir}; test -d ${quoted_remote_parent}; test -x ${quoted_remote_python}; test -x ${quoted_remote_dir}/scripts/run_mopd.sh; export PATH=$(quote_shell "$(dirname "$REMOTE_PYTHON")"):\$PATH; command -v sbatch >/dev/null; command -v ninja >/dev/null; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader; squeue -h -u \$USER -o '%i|%T|%M|%j'"

printf '\n[2/5] rsync preview\n'
rsync_config 1
if [[ "$DRY_RUN" == "1" ]]; then
  printf '\nDry-run complete; nothing was uploaded or submitted.\n'
  exit 0
fi

printf '\n[3/5] Upload config\n'
rsync_config 0

printf '\n[4/5] Verify config\n'
LOCAL_HASH="$(shasum -a 256 "$LOCAL_CONFIG" | awk '{print $1}')"
quoted_remote_config="$(quote_shell "$REMOTE_CONFIG")"
quoted_remote_reference="$(quote_shell "$REMOTE_CONFIG_REFERENCE")"
REMOTE_HASH_OUTPUT="$(remote_exec "sha256sum ${quoted_remote_config}")"
REMOTE_HASH="$(
  printf '%s\n' "$REMOTE_HASH_OUTPUT" \
    | tr -d '\r' \
    | awk 'length($1) == 64 && $1 ~ /^[[:xdigit:]]+$/ { print $1; exit }'
)"
[[ -n "$REMOTE_HASH" ]] || fail "could not parse remote SHA-256 output"
[[ "$LOCAL_HASH" == "$REMOTE_HASH" ]] || fail "local/remote SHA-256 mismatch"
remote_exec "set -eu; cd ${quoted_remote_dir}; export PYTHONPATH=${quoted_remote_dir}:${quoted_remote_dir}/third_party/verl:\${PYTHONPATH:-}; ${quoted_remote_python} -m mopd_verl.launch --config ${quoted_remote_reference} --dry-run >/dev/null"
printf 'SHA-256 verified: %s\n' "$LOCAL_HASH"
if [[ "$CHECK_ONLY" == "1" ]]; then
  printf '\nCheck-only complete; no Slurm job was submitted.\n'
  exit 0
fi

printf '\n[5/5] Submit Slurm job\n'
remote_exec "set -eu; cd ${quoted_remote_dir}; export PATH=$(quote_shell "$(dirname "$REMOTE_PYTHON")"):\$PATH; MOPD_LAUNCH_PYTHON=${quoted_remote_python} scripts/run_mopd.sh ${quoted_remote_reference} --slurm --slurm-args $(quote_shell "--partition=${SLURM_PARTITION}") --slurm-args $(quote_shell "--mem=${SLURM_MEMORY}") --slurm-args $(quote_shell "--time=${SLURM_TIME}") --slurm-args $(quote_shell "--priority=${SLURM_PRIORITY}")"
