"""Upload selected verl checkpoints to the Hugging Face Hub."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_HF_TOKEN_ENV_VAR = "HF_TOKEN"
UPLOAD_RECEIPT_FILENAME = ".huggingface_upload_complete.json"
UPLOAD_KIND = "huggingface_model"


@dataclass(frozen=True)
class HuggingFaceCheckpointConfig:
    """Configuration for step-selective checkpoint uploads."""

    enabled: bool = False
    steps: tuple[int, ...] = field(default_factory=tuple)
    repo_id: str | None = None
    private: bool = True
    path_prefix: str = "checkpoints"
    token_env_var: str = DEFAULT_HF_TOKEN_ENV_VAR
    env_file: str | None = None

    def __post_init__(self) -> None:
        if any(step <= 0 for step in self.steps):
            raise ValueError(
                "huggingface_checkpoint.steps must contain positive integers."
            )
        if len(self.steps) != len(set(self.steps)):
            raise ValueError("huggingface_checkpoint.steps must not contain duplicates.")
        if self.enabled and not self.steps:
            raise ValueError(
                "huggingface_checkpoint.steps must be non-empty when uploads are enabled."
            )
        if self.enabled and not (self.repo_id or "").strip():
            raise ValueError(
                "huggingface_checkpoint.repo_id is required when uploads are enabled."
            )
        if not _valid_env_key(self.token_env_var):
            raise ValueError(
                "huggingface_checkpoint.token_env_var must be a valid environment variable name."
            )
        _validate_path_prefix(self.path_prefix)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> HuggingFaceCheckpointConfig:
        """Build and validate upload configuration from YAML/OmegaConf data."""

        raw = value or {}
        steps_raw = raw.get("steps", [])
        if isinstance(steps_raw, (str, bytes)) or not isinstance(
            steps_raw, Sequence
        ):
            raise ValueError("huggingface_checkpoint.steps must be an array of integers.")
        steps: list[int] = []
        for index, item in enumerate(steps_raw):
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(
                    "huggingface_checkpoint.steps"
                    f"[{index}] must be a positive integer."
                )
            steps.append(item)

        repo_id_raw = raw.get("repo_id")
        repo_id = None if repo_id_raw is None else str(repo_id_raw).strip()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            steps=tuple(steps),
            repo_id=repo_id,
            private=bool(raw.get("private", True)),
            path_prefix=str(raw.get("path_prefix", "checkpoints")).strip("/"),
            token_env_var=str(
                raw.get("token_env_var", DEFAULT_HF_TOKEN_ENV_VAR)
            ).strip(),
            env_file=(
                None
                if raw.get("env_file") is None
                else str(raw.get("env_file")).strip()
            ),
        )

    def includes_step(self, step: int) -> bool:
        """Return whether ``step`` must be saved and uploaded."""

        return self.enabled and step in self.steps


def checkpoint_save_required(
    *,
    step: int,
    save_freq: int,
    is_last_step: bool,
    esi_close_to_expiration: bool,
    upload_config: HuggingFaceCheckpointConfig,
) -> bool:
    """Preserve periodic saves while forcing saves for selected upload steps."""

    periodic_save = save_freq > 0 and (
        is_last_step
        or step % save_freq == 0
        or esi_close_to_expiration
    )
    return periodic_save or upload_config.includes_step(step)


def require_huggingface_token(
    config: HuggingFaceCheckpointConfig,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the Hub token without storing it in configuration or logs."""

    source = os.environ if environ is None else environ
    token = source.get(config.token_env_var, "").strip()
    if not token and config.env_file:
        token = _read_env_value(config.env_file, config.token_env_var) or ""
    if not token:
        env_file = config.env_file or "runtime.env_file (normally .env.local)"
        raise RuntimeError(
            f"Missing {config.token_env_var}. Add it to {env_file} or export it "
            "before training."
        )
    return token


def checkpoint_upload_completed(
    config: HuggingFaceCheckpointConfig,
    checkpoint_dir: str | Path,
    step: int,
) -> bool:
    """Return whether this local checkpoint has a matching upload receipt."""

    receipt_path = Path(checkpoint_dir) / UPLOAD_RECEIPT_FILENAME
    if not receipt_path.is_file():
        return False
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("step") == step
        and payload.get("repo_id") == config.repo_id
        and payload.get("path_in_repo") == _path_in_repo(config, step)
        and payload.get("upload_kind") == UPLOAD_KIND
    )


def clear_checkpoint_upload_receipt(checkpoint_dir: str | Path) -> None:
    """Remove a stale receipt before rewriting a checkpoint at the same step."""

    receipt_path = Path(checkpoint_dir) / UPLOAD_RECEIPT_FILENAME
    receipt_path.unlink(missing_ok=True)


def upload_checkpoint_to_huggingface(
    config: HuggingFaceCheckpointConfig,
    checkpoint_dir: str | Path,
    step: int,
    *,
    environ: MutableMapping[str, str] | None = None,
    api_factory: Callable[[str], Any] | None = None,
) -> str:
    """Upload only the loadable Hugging Face model for one checkpoint step."""

    if not config.includes_step(step):
        raise ValueError(f"Step {step} is not configured for Hugging Face upload.")
    local_dir = Path(checkpoint_dir)
    model_dir = _huggingface_model_dir(local_dir)

    return _upload_huggingface_model(
        config,
        local_dir,
        model_dir,
        step,
        environ=environ,
        api_factory=api_factory,
    )


def _upload_huggingface_model(
    config: HuggingFaceCheckpointConfig,
    checkpoint_dir: Path,
    model_dir: Path,
    step: int,
    *,
    environ: MutableMapping[str, str] | None,
    api_factory: Callable[[str], Any] | None,
) -> str:
    """Upload one validated model directory and write a local receipt."""

    token = require_huggingface_token(config, environ)
    if api_factory is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for checkpoint uploads."
            ) from exc
        api_factory = lambda resolved_token: HfApi(token=resolved_token)

    api = api_factory(token)
    repo_id = config.repo_id
    if repo_id is None:
        raise RuntimeError("Hugging Face repo_id was not configured.")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=config.private,
        exist_ok=True,
    )
    path_in_repo = _path_in_repo(config, step)
    result = api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        commit_message=f"Upload Hugging Face model at step {step}",
        delete_patterns=["*", "**/*"],
    )
    commit_url = str(getattr(result, "commit_url", result))
    receipt = {
        "step": step,
        "repo_id": repo_id,
        "path_in_repo": path_in_repo,
        "upload_kind": UPLOAD_KIND,
        "commit_url": commit_url,
    }
    _write_upload_receipt(checkpoint_dir, receipt)
    return commit_url


def _valid_env_key(value: str) -> bool:
    if not value or value[0].isdigit():
        return False
    return all(character == "_" or character.isalnum() for character in value)


def _validate_path_prefix(value: str) -> None:
    if not value:
        return
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "huggingface_checkpoint.path_prefix must be a relative Hub path."
        )


def _path_in_repo(config: HuggingFaceCheckpointConfig, step: int) -> str:
    step_dir = f"global_step_{step}"
    if not config.path_prefix:
        return step_dir
    return f"{config.path_prefix}/{step_dir}"


def _huggingface_model_dir(checkpoint_dir: Path) -> Path:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )

    model_dir = checkpoint_dir / "actor" / "huggingface"
    return _validate_huggingface_model_dir(model_dir)


def _validate_huggingface_model_dir(model_dir: Path) -> Path:
    """Validate one standalone Transformers model directory."""

    if not model_dir.is_dir():
        raise FileNotFoundError(
            "Hugging Face model directory does not exist: "
            f"{model_dir}. Ensure actor checkpoint save_contents includes hf_model."
        )
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"Hugging Face model config does not exist: {model_dir / 'config.json'}"
        )

    weight_files = [
        path
        for path in model_dir.iterdir()
        if path.is_file()
        and (
            path.suffix == ".safetensors"
            or (path.name.startswith("pytorch_model") and path.suffix == ".bin")
        )
    ]
    if not weight_files:
        raise FileNotFoundError(
            "Hugging Face model weights do not exist in "
            f"{model_dir}. Ensure actor checkpoint save_contents includes hf_model."
        )
    return model_dir


def _write_upload_receipt(checkpoint_dir: Path, receipt: Mapping[str, Any]) -> None:
    if not checkpoint_dir.is_dir():
        return
    receipt_path = checkpoint_dir / UPLOAD_RECEIPT_FILENAME
    temporary_receipt_path = receipt_path.with_suffix(".tmp")
    try:
        temporary_receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_receipt_path.replace(receipt_path)
    except FileNotFoundError:
        temporary_receipt_path.unlink(missing_ok=True)


def _read_env_value(path: str, wanted_key: str) -> str | None:
    env_path = Path(path).expanduser()
    if not env_path.is_absolute():
        env_path = Path.cwd() / env_path
    if not env_path.is_file():
        return None
    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(
                f"Invalid env file line {line_number} in {env_path}: "
                "expected KEY=value."
            )
        key, value = line.split("=", 1)
        if key.strip() != wanted_key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value.strip()
    return None
