"""Build and run eval-only verl commands for MOPD configs."""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from mopd_verl.launch import build_command, format_command, run_command
from mopd_verl.settings import MOPDConfig, load_config


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_audit_all_2gpu.yaml"
DEFAULT_PAPER_EVAL_SCRIPT = "eval/scripts/run_paper_eval_suite.sh"


def _hydra_list(values: Sequence[str]) -> str:
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"[{quoted}]"


def _csv_values(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "mopd-eval"


def _paper_eval_prefix(config: MOPDConfig) -> str:
    return "" if config.paper_eval.enabled else "+"


def default_run_id(config: MOPDConfig) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _safe_run_id(f"{config.trainer.experiment_name}-eval-{timestamp}")


def build_eval_overrides(
    config: MOPDConfig,
    *,
    model_path: str | None = None,
    run_id: str | None = None,
    output_dir: str | None = None,
    paper_eval: bool = False,
    paper_datasets: Sequence[str] | None = None,
    paper_output_dir: str | None = None,
    paper_timeout_seconds: int = 0,
    paper_fail_on_error: bool = False,
) -> list[str]:
    """Return Hydra overrides that turn a training config into eval-only mode."""

    resolved_run_id = _safe_run_id(run_id or default_run_id(config))
    eval_output_dir = output_dir or f"eval_outputs/verl_validation/{resolved_run_id}"

    overrides = [
        f"trainer.experiment_name={resolved_run_id}",
        f"trainer.default_local_dir={eval_output_dir}/checkpoints",
        "trainer.val_before_train=True",
        "trainer.val_only=True",
        "trainer.test_freq=-1",
        "trainer.save_freq=-1",
        "trainer.total_epochs=1",
        "trainer.total_training_steps=1",
    ]
    if model_path is not None:
        overrides.append(f"actor_rollout_ref.model.path={model_path}")

    if paper_eval:
        datasets = list(paper_datasets if paper_datasets is not None else config.paper_eval.datasets)
        output_root = paper_output_dir or f"{eval_output_dir}/paper_suite"
        prefix = _paper_eval_prefix(config)
        overrides.extend(
            [
                f"{prefix}paper_eval.enabled=true",
                f"{prefix}paper_eval.script_path={config.paper_eval.script_path or DEFAULT_PAPER_EVAL_SCRIPT}",
                f"{prefix}paper_eval.model_path=null",
                f"{prefix}paper_eval.output_dir={output_root}",
                f"{prefix}paper_eval.datasets={_hydra_list(datasets)}",
                f"{prefix}paper_eval.run_on_initial_validation=true",
                f"{prefix}paper_eval.evaluate_current_checkpoint=false",
                f"{prefix}paper_eval.fail_on_error={str(paper_fail_on_error).lower()}",
                f"{prefix}paper_eval.timeout_seconds={paper_timeout_seconds}",
            ]
        )

    return overrides


def build_eval_command(
    config: MOPDConfig,
    *,
    model_path: str | None = None,
    run_id: str | None = None,
    output_dir: str | None = None,
    paper_eval: bool = False,
    paper_datasets: Sequence[str] | None = None,
    paper_output_dir: str | None = None,
    paper_timeout_seconds: int = 0,
    paper_fail_on_error: bool = False,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    eval_overrides = build_eval_overrides(
        config,
        model_path=model_path,
        run_id=run_id,
        output_dir=output_dir,
        paper_eval=paper_eval,
        paper_datasets=paper_datasets,
        paper_output_dir=paper_output_dir,
        paper_timeout_seconds=paper_timeout_seconds,
        paper_fail_on_error=paper_fail_on_error,
    )
    return build_command(config, [*eval_overrides, *(extra_args or [])])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to a MOPD YAML config.")
    parser.add_argument("--model-path", default=None, help="Optional actor model/checkpoint path to evaluate.")
    parser.add_argument("--run-id", default=None, help="Stable eval run id. Defaults to config experiment name + timestamp.")
    parser.add_argument("--output-dir", default=None, help="Eval-only output root for logs/checkpoint placeholders.")
    parser.add_argument("--paper-eval", action="store_true", help="Also run configured paper benchmark eval at step 0.")
    parser.add_argument("--paper-datasets", default=None, help="Comma-separated paper eval dataset keys.")
    parser.add_argument("--paper-output-dir", default=None, help="Paper eval output root.")
    parser.add_argument("--paper-timeout-seconds", type=int, default=0, help="Paper eval timeout; 0 disables timeout.")
    parser.add_argument("--paper-fail-on-error", action="store_true", help="Fail the eval run if paper eval exits nonzero.")
    parser.add_argument("--dry-run", action="store_true", help="Print the verl command without executing it.")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra Hydra overrides after '--'.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    extra_args = list(args.extra)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    command = build_eval_command(
        config,
        model_path=args.model_path,
        run_id=args.run_id,
        output_dir=args.output_dir,
        paper_eval=args.paper_eval,
        paper_datasets=_csv_values(args.paper_datasets),
        paper_output_dir=args.paper_output_dir,
        paper_timeout_seconds=args.paper_timeout_seconds,
        paper_fail_on_error=args.paper_fail_on_error,
        extra_args=extra_args,
    )
    if args.dry_run:
        sys.stdout.write(format_command(command) + "\n")
        return 0
    return run_command(command, config)


if __name__ == "__main__":
    raise SystemExit(main())
