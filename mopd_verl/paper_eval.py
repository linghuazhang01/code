"""Run paper benchmark evaluation from inside patched verl validation."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _string_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def _write_status(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_eval_status.json").write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_paper_eval_from_config(config: Any, step: int, default_model_path: str) -> dict[str, float]:
    """Run the configured seven-benchmark paper eval suite at validation time."""

    paper_eval = _cfg_get(config, "paper_eval", {})
    if not bool(_cfg_get(paper_eval, "enabled", False)):
        return {}
    if step == 0 and not bool(_cfg_get(paper_eval, "run_on_initial_validation", True)):
        return {}

    output_root = Path(str(_cfg_get(paper_eval, "output_dir", "paper_eval")))
    output_dir = output_root / f"step_{step:08d}"
    script_path = _cfg_get(paper_eval, "script_path")
    model_path = str(_cfg_get(paper_eval, "model_path", None) or default_model_path)
    fail_on_error = bool(_cfg_get(paper_eval, "fail_on_error", False))

    if not script_path:
        status = {"status": "missing_script_path", "step": step, "returncode": -1}
        _write_status(output_dir, status)
        if fail_on_error:
            raise RuntimeError("paper_eval.enabled=true but paper_eval.script_path is missing.")
        return {"paper_eval/enabled": 1.0, "paper_eval/returncode": -1.0}

    datasets = _string_sequence(_cfg_get(paper_eval, "datasets", []))
    env = os.environ.copy()
    env["MODEL_PATH"] = model_path
    env["EVAL_OUTPUT_DIR"] = str(output_dir)
    env["MOPD_GLOBAL_STEP"] = str(step)
    if datasets:
        env["PAPER_EVAL_DATASETS"] = ",".join(datasets)

    timeout_seconds = int(_cfg_get(paper_eval, "timeout_seconds", 0) or 0)
    cmd = ["bash", str(script_path)]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=None if timeout_seconds <= 0 else timeout_seconds,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "paper_eval_runner.stdout").write_text(completed.stdout, encoding="utf-8")
        (output_dir / "paper_eval_runner.stderr").write_text(completed.stderr, encoding="utf-8")
        status = {
            "status": "completed",
            "command": cmd,
            "datasets": datasets,
            "model_path": model_path,
            "returncode": completed.returncode,
            "step": step,
        }
        _write_status(output_dir, status)
        if completed.returncode != 0 and fail_on_error:
            raise RuntimeError(f"paper eval failed with return code {completed.returncode}: {script_path}")
        return {
            "paper_eval/enabled": 1.0,
            "paper_eval/completed": 1.0 if completed.returncode == 0 else 0.0,
            "paper_eval/returncode": float(completed.returncode),
        }
    except subprocess.TimeoutExpired as exc:
        status = {
            "status": "timeout",
            "command": cmd,
            "datasets": datasets,
            "model_path": model_path,
            "returncode": -2,
            "step": step,
            "timeout_seconds": timeout_seconds,
        }
        _write_status(output_dir, status)
        if fail_on_error:
            raise RuntimeError(f"paper eval timed out after {timeout_seconds} seconds: {script_path}") from exc
        return {"paper_eval/enabled": 1.0, "paper_eval/completed": 0.0, "paper_eval/returncode": -2.0}
