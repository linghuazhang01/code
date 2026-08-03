#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/download_qwen1p7b_goosereason_models.sh

Download or validate the model pair used by the Qwen3-1.7B-Base to
Nemotron-Research-GooseReason-4B-Instruct distillation configs.

Environment knobs:
  MODEL_ROOT=<parent of OPD-code>/models
  PYTHON_BIN=<auto-detected python or python3>
  MODEL_BACKEND=huggingface
  DOWNLOAD_MODELS=1
  REQUIRE_MODELS=1
  STUDENT_MODEL_ID=Qwen/Qwen3-1.7B-Base
  STUDENT_DIR_NAME=Qwen3-1.7B-Base
  GOOSE_MODEL_ID=nvidia/Nemotron-Research-GooseReason-4B-Instruct
  GOOSE_DIR_NAME=Nemotron-Research-GooseReason-4B-Instruct

Use DOWNLOAD_MODELS=0 REQUIRE_MODELS=1 to validate existing checkpoints
without downloading them again.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ROOT="${MODEL_ROOT:-$(cd "${CODE_DIR}/.." && pwd)/models}"
PYTHON_BIN="${PYTHON_BIN:-}"
MODEL_BACKEND="${MODEL_BACKEND:-huggingface}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
REQUIRE_MODELS="${REQUIRE_MODELS:-1}"
STUDENT_MODEL_ID="${STUDENT_MODEL_ID:-Qwen/Qwen3-1.7B-Base}"
STUDENT_DIR_NAME="${STUDENT_DIR_NAME:-Qwen3-1.7B-Base}"
GOOSE_MODEL_ID="${GOOSE_MODEL_ID:-nvidia/Nemotron-Research-GooseReason-4B-Instruct}"
GOOSE_DIR_NAME="${GOOSE_DIR_NAME:-Nemotron-Research-GooseReason-4B-Instruct}"

for binary_flag in DOWNLOAD_MODELS REQUIRE_MODELS; do
  if [[ "${!binary_flag}" != "0" && "${!binary_flag}" != "1" ]]; then
    echo "${binary_flag} must be 0 or 1: ${!binary_flag}" >&2
    exit 2
  fi
done

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "python or python3 is required." >&2
    exit 1
  fi
fi

if [[ "${DOWNLOAD_MODELS}" == "1" || "${REQUIRE_MODELS}" == "1" ]]; then
  MODEL_ROOT="${MODEL_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  MODEL_BACKEND="${MODEL_BACKEND}" \
  DOWNLOAD_STUDENT="${DOWNLOAD_MODELS}" \
  REQUIRE_STUDENT="${REQUIRE_MODELS}" \
  STUDENT_MODEL_ID="${STUDENT_MODEL_ID}" \
  STUDENT_DIR_NAME="${STUDENT_DIR_NAME}" \
  DOWNLOAD_BASE_4B=0 \
  REQUIRE_BASE_4B=0 \
  DOWNLOAD_REASONING_BASE_14B=0 \
  DOWNLOAD_REASONING_TEACHER="${DOWNLOAD_MODELS}" \
  REQUIRE_REASONING_TEACHER="${REQUIRE_MODELS}" \
  REASONING_TEACHER_MODEL_ID="${GOOSE_MODEL_ID}" \
  REASONING_TEACHER_DIR_NAME="${GOOSE_DIR_NAME}" \
    bash "${SCRIPT_DIR}/download_mopd_models.sh"
fi

if [[ "${REQUIRE_MODELS}" != "1" ]]; then
  echo "Model compatibility validation skipped."
  exit 0
fi

student_dir="${MODEL_ROOT}/${STUDENT_DIR_NAME}"
goose_dir="${MODEL_ROOT}/${GOOSE_DIR_NAME}"

"${PYTHON_BIN}" - "${student_dir}" "${goose_dir}" <<'PY'
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing required model asset: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def token_id_map(model_dir: Path) -> dict[str, int]:
    tokenizer = load_json(model_dir / "tokenizer.json")
    token_map = dict(tokenizer["model"]["vocab"])
    for token in tokenizer.get("added_tokens", []):
        content = token["content"]
        token_id = int(token["id"])
        previous = token_map.setdefault(content, token_id)
        if previous != token_id:
            raise SystemExit(
                f"Conflicting token id in {model_dir}: {content!r}={previous}/{token_id}"
            )
    return token_map


def validate_chat_template(model_dir: Path) -> None:
    tokenizer_config = load_json(model_dir / "tokenizer_config.json")
    embedded_template = tokenizer_config.get("chat_template")
    template_path = model_dir / "chat_template.jinja"
    if embedded_template:
        return
    if template_path.is_file() and template_path.stat().st_size > 0:
        return
    raise SystemExit(f"Missing chat template for prompt rendering: {model_dir}")


def validate_weight_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Missing or empty model weight file: {path}")
    with path.open("rb") as handle:
        header = handle.read(96)
    if header.startswith(b"version https://git-lfs.github.com/spec"):
        raise SystemExit(f"Unresolved Git LFS model weight pointer: {path}")


def validate_model_weights(model_dir: Path) -> tuple[str, ...]:
    index_names = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    for index_name in index_names:
        index_path = model_dir / index_name
        if not index_path.is_file():
            continue
        index = load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SystemExit(f"Invalid or empty model weight index: {index_path}")
        shard_names = tuple(sorted({str(name) for name in weight_map.values()}))
        for shard_name in shard_names:
            validate_weight_file(model_dir / shard_name)
        return shard_names

    for single_name in ("model.safetensors", "pytorch_model.bin"):
        single_path = model_dir / single_name
        if single_path.is_file():
            validate_weight_file(single_path)
            return (single_name,)
    raise SystemExit(f"Missing model weights or weight index: {model_dir}")


student_dir = Path(sys.argv[1])
goose_dir = Path(sys.argv[2])
student_config = load_json(student_dir / "config.json")
goose_config = load_json(goose_dir / "config.json")
validate_chat_template(student_dir)
validate_chat_template(goose_dir)
student_weight_files = validate_model_weights(student_dir)
goose_weight_files = validate_model_weights(goose_dir)
student_tokens = token_id_map(student_dir)
goose_tokens = token_id_map(goose_dir)

student_vocab_size = int(student_config["vocab_size"])
goose_vocab_size = int(goose_config["vocab_size"])
if student_vocab_size != goose_vocab_size:
    raise SystemExit(
        f"Model vocabulary sizes differ: student={student_vocab_size}, "
        f"teacher={goose_vocab_size}"
    )
if student_tokens != goose_tokens:
    only_student = sorted(student_tokens.items() - goose_tokens.items())[:5]
    only_teacher = sorted(goose_tokens.items() - student_tokens.items())[:5]
    raise SystemExit(
        "Tokenizer token-to-id mappings differ; token-ID Top-K distillation is unsafe. "
        f"student_only={only_student}, teacher_only={only_teacher}"
    )
student_token_ids = set(student_tokens.values())
goose_token_ids = set(goose_tokens.values())
if min(student_token_ids) < 0 or max(student_token_ids) >= student_vocab_size:
    raise SystemExit(
        "Student tokenizer contains ids outside the model vocabulary: "
        f"min={min(student_token_ids)}, max={max(student_token_ids)}, "
        f"vocab_size={student_vocab_size}"
    )
if min(goose_token_ids) < 0 or max(goose_token_ids) >= goose_vocab_size:
    raise SystemExit(
        "Teacher tokenizer contains ids outside the model vocabulary: "
        f"min={min(goose_token_ids)}, max={max(goose_token_ids)}, "
        f"vocab_size={goose_vocab_size}"
    )

student_eos = student_config.get("eos_token_id")
goose_eos = goose_config.get("eos_token_id")
goose_generation_path = goose_dir / "generation_config.json"
goose_generation = load_json(goose_generation_path)
goose_generation_eos = goose_generation.get("eos_token_id")
for label, eos_token_id in (("student", student_eos), ("teacher", goose_eos)):
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise SystemExit(f"Invalid {label} model EOS token id: {eos_token_id!r}")
if isinstance(goose_generation_eos, int) and not isinstance(goose_generation_eos, bool):
    accepted_eos = [goose_generation_eos]
elif isinstance(goose_generation_eos, list) and goose_generation_eos and all(
    isinstance(item, int) and not isinstance(item, bool)
    for item in goose_generation_eos
):
    accepted_eos = goose_generation_eos
else:
    raise SystemExit(
        f"Invalid teacher generation EOS token ids: {goose_generation_eos!r}"
    )
if student_eos not in accepted_eos:
    raise SystemExit(
        f"Teacher generation config does not accept the student EOS: {student_eos}; "
        f"accepted={accepted_eos}"
    )
if goose_eos not in accepted_eos:
    raise SystemExit(
        f"Teacher generation config does not accept its model EOS: {goose_eos}; "
        f"accepted={accepted_eos}"
    )

print(
    "Model pair ready: "
    f"vocab={student_vocab_size}, mapped_tokens={len(student_tokens)}, "
    "token_map=identical, "
    f"student_eos={student_eos}, teacher_eos={goose_eos}, "
    f"teacher_generation_eos={accepted_eos}, "
    f"student_weight_files={len(student_weight_files)}, "
    f"teacher_weight_files={len(goose_weight_files)}"
)
PY

echo "Student: ${student_dir}"
echo "Teacher: ${goose_dir}"
