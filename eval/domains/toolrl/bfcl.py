"""BFCL launcher for the ToolRL RLLA handler."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from eval.official_utils import OfficialEvalResult, ensure_output_dir, write_json


def run_bfcl(
    *,
    model_path: str,
    output_dir: str | Path,
    bfcl_command: str | None,
    api_base_url: str | None = None,
    api_key: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> OfficialEvalResult:
    output = ensure_output_dir(output_dir)
    handler_module = "eval.domains.toolrl.bfcl_handler.RLLAHandler"
    summary: dict[str, Any] = {
        "dataset": "bfcl",
        "model_path": model_path,
        "handler_module": handler_module,
        "status": "configured",
    }
    if not bfcl_command:
        summary.update(
            {
                "status": "not_run",
                "reason": "BFCL official harness command was not provided.",
                "example_env": {
                    "BFCL_MODEL_PATH": model_path,
                    "BFCL_OUTPUT_DIR": str(output),
                    "BFCL_HANDLER": handler_module,
                },
            }
        )
        write_json(output / "summary.json", summary)
        return OfficialEvalResult(dataset="bfcl", output_dir=output, summary=summary)

    env = os.environ.copy()
    env.update(
        {
            "BFCL_MODEL_PATH": model_path,
            "BFCL_OUTPUT_DIR": str(output),
            "BFCL_HANDLER": handler_module,
        }
    )
    if api_base_url:
        env["BFCL_API_BASE_URL"] = api_base_url
    if api_key:
        env["BFCL_API_KEY"] = api_key
    if extra_env:
        env.update(extra_env)

    command = shlex.split(bfcl_command)
    completed = subprocess.run(command, cwd=output, env=env, check=False, text=True, capture_output=True)
    (output / "bfcl_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "bfcl_stderr.log").write_text(completed.stderr, encoding="utf-8")
    summary.update(
        {
            "status": "complete" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
        }
    )
    write_json(output / "summary.json", summary)
    if completed.returncode != 0:
        raise RuntimeError(f"BFCL command failed with code {completed.returncode}. See {output}.")
    return OfficialEvalResult(dataset="bfcl", output_dir=output, summary=summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="data/eval_data/results/official_toolrl/bfcl")
    parser.add_argument("--bfcl-command", default=None)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_bfcl(
        model_path=args.model_path,
        output_dir=args.output_dir,
        bfcl_command=args.bfcl_command,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
