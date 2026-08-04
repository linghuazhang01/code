"""Upload local evaluation metrics and rollout artifacts to Weights & Biases."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from collections.abc import MutableMapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mopd_verl.launch import _read_env_file

LOGGER = logging.getLogger(__name__)
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env.local"

SUMMARY_FIELDS = (
    "mode",
    "dataset",
    "ability",
    "sample_count",
    "scored_count",
    "accuracy",
    "avg_score",
    "unique_sample_count",
    "min_samples_per_prompt",
    "max_samples_per_prompt",
    "avg_at_k",
    "observed_pass_at_k",
    "avg_generated_tokens",
    "avg_thinking_tokens",
    "avg_answer_tokens",
    "avg_total_tokens",
    "avg_latency_seconds",
    "max_generated_tokens",
)
METRIC_FIELDS = SUMMARY_FIELDS[3:]
BASE_ARTIFACT_FILES = (
    "README.md",
    "eval_run_config.json",
    "eval_resume_config.json",
    "thinking_eval_summary.json",
    "thinking_eval_summary.csv",
)
RAW_ARTIFACT_FILES = (
    "thinking_eval_samples.jsonl",
    "thinking_eval_results.json",
    "records.jsonl",
    "prompt_response_records.jsonl",
)
REQUIRED_RAW_ARTIFACT_FILES = (
    "thinking_eval_samples.jsonl",
    "prompt_response_records.jsonl",
)


@dataclass(frozen=True)
class WandbUploadConfig:
    """Configuration for one local evaluation upload."""

    output_dir: Path
    project: str = "mopd-eval"
    entity: str | None = None
    group: str | None = None
    mode: str = "online"
    upload_raw: bool = True
    env_file: Path | None = None


@dataclass(frozen=True)
class WandbUploadResult:
    """Identifiers returned after a completed upload."""

    run_id: str
    run_url: str
    mode: str
    local_run_dir: str
    artifact_name: str
    uploaded_files: tuple[str, ...]


def apply_wandb_environment(
    env_file: Path | None,
    environ: MutableMapping[str, str] | None = None,
) -> bool:
    """Merge the existing env-file format and map legacy Wandb_Key safely."""

    target = os.environ if environ is None else environ
    for key, value in _read_env_file(str(env_file) if env_file else None).items():
        target.setdefault(key, value)
    legacy_key = target.get("Wandb_Key")
    if legacy_key and not target.get("WANDB_API_KEY"):
        target["WANDB_API_KEY"] = legacy_key
    return bool(target.get("WANDB_API_KEY"))


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    return cleaned or "eval"


def stable_wandb_run_id(local_run_id: str) -> str:
    """Create a stable W&B id so upload-only retries resume the same run."""

    digest = hashlib.sha256(local_run_id.encode("utf-8")).hexdigest()[:10]
    return f"{_safe_name(local_run_id)[:48]}-{digest}"


def _artifact_name(local_run_id: str) -> str:
    digest = hashlib.sha256(local_run_id.encode("utf-8")).hexdigest()[:10]
    return f"{_safe_name(local_run_id)[:96]}-{digest}-rollouts"


def flatten_summary_metrics(summary: Sequence[dict[str, Any]]) -> dict[str, int | float]:
    """Convert dataset summary rows to W&B scalar metric keys."""

    metrics: dict[str, int | float] = {}
    for row in summary:
        mode = _safe_name(str(row.get("mode", "unknown")))
        ability = _safe_name(str(row.get("ability", "unknown")))
        dataset = _safe_name(str(row.get("dataset", "unknown")))
        prefix = f"eval/{mode}/{ability}/{dataset}"
        for field in METRIC_FIELDS:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            metrics[f"{prefix}/{field}"] = value
    return metrics


def artifact_paths(output_dir: Path, upload_raw: bool) -> tuple[Path, ...]:
    """Return existing report files without duplicating per-domain views."""

    filenames = BASE_ARTIFACT_FILES + (RAW_ARTIFACT_FILES if upload_raw else ())
    return tuple(output_dir / name for name in filenames if (output_dir / name).is_file())


def _validate_raw_artifacts(output_dir: Path) -> None:
    missing = [
        filename
        for filename in REQUIRED_RAW_ARTIFACT_FILES
        if not (output_dir / filename).is_file() or (output_dir / filename).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"Raw rollout upload requires non-empty files in {output_dir}: {', '.join(missing)}"
        )


def _load_payload(output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "thinking_eval_results.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing evaluation report: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("summary"), list):
        raise ValueError(f"Invalid evaluation summary in {result_path}")
    return payload


def _write_status(
    config: WandbUploadConfig,
    state: str,
    *,
    error: str | None = None,
    result: WandbUploadResult | None = None,
) -> None:
    config_payload = asdict(config)
    config_payload["output_dir"] = str(config.output_dir)
    config_payload["env_file"] = str(config.env_file) if config.env_file else None
    payload: dict[str, Any] = {"state": state, "config": config_payload}
    if error:
        payload["error"] = error
    if result:
        payload["result"] = asdict(result)
    (config.output_dir / "wandb_upload_status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def upload_eval_output(config: WandbUploadConfig, wandb_module: Any | None = None) -> WandbUploadResult:
    """Upload one completed local report and return its W&B identifiers."""

    apply_wandb_environment(config.env_file)
    if wandb_module is None:
        try:
            import wandb as wandb_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError("wandb is not installed in the evaluation environment") from exc

    payload = _load_payload(config.output_dir)
    if config.upload_raw:
        _validate_raw_artifacts(config.output_dir)
    local_run_id = str(payload.get("run_id") or config.output_dir.name)
    wandb_run_id = stable_wandb_run_id(local_run_id)
    summary = payload["summary"]
    metrics = flatten_summary_metrics(summary)
    record_count = len(payload.get("records", []))
    expected_total = int(payload.get("expected_total", record_count))
    metrics["eval/record_count"] = record_count
    metrics["eval/expected_total"] = expected_total
    if expected_total > 0:
        metrics["eval/completion_ratio"] = record_count / expected_total

    init_kwargs: dict[str, Any] = {
        "project": config.project,
        "id": wandb_run_id,
        "resume": "allow",
        "name": local_run_id,
        "job_type": "evaluation",
        "mode": config.mode,
        "dir": str(config.output_dir / "wandb"),
        "tags": ["evaluation", "rollout-artifact"],
        "config": {
            "local_run_id": local_run_id,
            "model_path": payload.get("model_path", ""),
            "status": payload.get("status", ""),
            "record_source": payload.get("record_source", ""),
            "scoring_backend": payload.get("scoring_backend", ""),
            "expected_total": expected_total,
            "generation": payload.get("run_config", {}),
        },
    }
    if config.entity:
        init_kwargs["entity"] = config.entity
    if config.group:
        init_kwargs["group"] = config.group

    (config.output_dir / "wandb").mkdir(parents=True, exist_ok=True)
    run = wandb_module.init(**init_kwargs)
    if run is None:
        raise RuntimeError("wandb.init() did not return a run")

    artifact_name = _artifact_name(local_run_id)
    uploaded_paths = artifact_paths(config.output_dir, config.upload_raw)
    try:
        table_data = [[row.get(field) for field in SUMMARY_FIELDS] for row in summary]
        run.log({**metrics, "eval/summary_table": wandb_module.Table(columns=list(SUMMARY_FIELDS), data=table_data)})
        run.summary["eval_status"] = payload.get("status", "")
        run.summary["local_output_dir"] = str(config.output_dir)

        artifact = wandb_module.Artifact(
            artifact_name,
            type="evaluation-results",
            description=f"Metrics and saved rollouts for {local_run_id}",
            metadata={
                "local_run_id": local_run_id,
                "model_path": str(payload.get("model_path", "")),
                "record_count": record_count,
                "expected_total": expected_total,
                "includes_raw_rollouts": config.upload_raw,
            },
        )
        for path in uploaded_paths:
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact, aliases=["latest", str(payload.get("status", "final"))])
        run_url = str(getattr(run, "url", "") or "")
        run_settings = getattr(run, "settings", None)
        actual_mode = str(getattr(run_settings, "mode", config.mode) or config.mode)
        run_dir_value = getattr(run, "dir", None)
        run_directory = str(Path(str(run_dir_value)).parent if run_dir_value else config.output_dir / "wandb")
    finally:
        run.finish()

    return WandbUploadResult(
        run_id=str(getattr(run, "id", wandb_run_id)),
        run_url=run_url,
        mode=actual_mode,
        local_run_dir=run_directory,
        artifact_name=artifact_name,
        uploaded_files=tuple(path.name for path in uploaded_paths),
    )


def _output_dirs(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.output_dir:
        return (Path(args.output_dir),)
    root = Path(args.output_root)
    return tuple(sorted(path.parent for path in root.rglob("thinking_eval_results.json")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    paths = parser.add_mutually_exclusive_group(required=True)
    paths.add_argument("--output-dir", help="One completed evaluation directory.")
    paths.add_argument("--output-root", help="Recursively upload completed evaluations below this directory.")
    parser.add_argument("--project", default="mopd-eval")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--upload-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    output_dirs = _output_dirs(args)
    if not output_dirs:
        LOGGER.error("No thinking_eval_results.json files found under %s", args.output_root)
        return 1

    failures = 0
    for output_dir in output_dirs:
        config = WandbUploadConfig(
            output_dir=output_dir,
            project=args.project,
            entity=args.entity,
            group=args.group,
            mode=args.mode,
            upload_raw=args.upload_raw,
            env_file=Path(args.env_file) if args.env_file else None,
        )
        _write_status(config, "uploading")
        try:
            result = upload_eval_output(config)
        except Exception as exc:  # W&B failures must not invalidate local evaluation output.
            failures += 1
            _write_status(config, "upload_pending", error=str(exc))
            LOGGER.exception("W&B upload pending for %s", output_dir)
            continue
        state = {"online": "uploaded", "offline": "offline_saved", "disabled": "disabled"}.get(
            result.mode,
            "offline_saved" if not result.run_url else "uploaded",
        )
        _write_status(config, state, result=result)
        LOGGER.info("W&B state=%s for %s: %s", state, output_dir, result.run_url or result.run_id)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
