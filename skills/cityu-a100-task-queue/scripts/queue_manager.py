#!/usr/bin/env python3
"""Atomic persistent LIFO queue for CityU A100 task dispatch."""

from __future__ import annotations

import argparse
import json
import shlex
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from queue_state import (
    SUPPORTED_GPU_COUNTS,
    TERMINAL_STATUSES,
    VALID_STATUSES,
    QueueTask,
    default_state_file,
    find_task,
    load_state,
    locked,
    newest_pending,
    now,
    replace_task,
    validate_task,
    write_state,
)


def _parse_gpu_counts(value: str) -> tuple[int, ...]:
    counts = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    if not counts or any(item not in SUPPORTED_GPU_COUNTS for item in counts):
        raise argparse.ArgumentTypeError("GPU counts must be 3, 4, or 3,4")
    return counts


def _print_task(task: QueueTask, as_json: bool) -> None:
    if as_json:
        print(json.dumps(task.to_mapping(), indent=2, sort_keys=True))
        return
    gpu_counts = ",".join(str(item) for item in task.gpu_counts)
    print(
        f"{task.sequence}|{task.task_id}|{task.status}|{gpu_counts}|"
        f"{task.slurm_job_id or '-'}|{task.title}"
    )


def _task_summary(task: QueueTask) -> dict[str, Any]:
    value = task.to_mapping()
    value.pop("command")
    return value


def _quoted_remote_cwd(remote_cwd: str) -> str:
    if remote_cwd == "$HOME":
        return '"$HOME"'
    if remote_cwd.startswith("$HOME/"):
        return f'"$HOME"/{shlex.quote(remote_cwd.removeprefix("$HOME/"))}'
    return shlex.quote(remote_cwd)


def _cmd_init(args: argparse.Namespace) -> int:
    with locked(args.state_file):
        state = load_state(args.state_file)
        write_state(args.state_file, state)
    print(f"queue_file={args.state_file}")
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    if "\x00" in args.command:
        raise ValueError("command must not contain NUL bytes")
    with locked(args.state_file):
        state = load_state(args.state_file)
        timestamp = now()
        sequence = state.next_sequence
        task = QueueTask(
            task_id=f"task-{sequence:06d}-{uuid.uuid4().hex[:8]}",
            sequence=sequence,
            title=args.title.strip(),
            command=args.command,
            remote_cwd=args.remote_cwd,
            gpu_counts=args.gpu_counts,
            status="pending",
            created_at=timestamp,
            updated_at=timestamp,
        )
        validate_task(task)
        updated = replace(
            state,
            next_sequence=sequence + 1,
            tasks=state.tasks + (task,),
        )
        write_state(args.state_file, updated)
    _print_task(task, args.json)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    with locked(args.state_file):
        state = load_state(args.state_file)
    statuses = set(args.statuses.split(",")) if args.statuses else set(VALID_STATUSES)
    unknown = statuses.difference(VALID_STATUSES)
    if unknown:
        raise ValueError(f"unknown statuses: {','.join(sorted(unknown))}")
    tasks = sorted(
        (task for task in state.tasks if task.status in statuses),
        key=lambda item: item.sequence,
        reverse=True,
    )
    if args.json:
        counts = {
            status: sum(task.status == status for task in state.tasks)
            for status in VALID_STATUSES
        }
        print(
            json.dumps(
                {
                    "queue_file": str(args.state_file),
                    "counts": counts,
                    "tasks": [_task_summary(task) for task in tasks],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print("sequence|task_id|status|gpu_counts|slurm_job_id|title")
    for task in tasks:
        _print_task(task, False)
    return 0


def _cmd_peek(args: argparse.Namespace) -> int:
    with locked(args.state_file):
        task = newest_pending(load_state(args.state_file), args.gpu_count)
    if task is None:
        print("NO_COMPATIBLE_PENDING_TASK")
        return 4
    _print_task(task, args.json)
    return 0


def _cmd_claim(args: argparse.Namespace) -> int:
    with locked(args.state_file):
        state = load_state(args.state_file)
        newest_task = newest_pending(state, args.gpu_count)
        if newest_task is None:
            print("NO_COMPATIBLE_PENDING_TASK")
            return 4
        task = find_task(state, args.task_id) if args.task_id else newest_task
        if task.task_id != newest_task.task_id:
            raise ValueError(
                "task_id is not the newest compatible pending task for this GPU count"
            )
        if task.status != "pending" or args.gpu_count not in task.gpu_counts:
            raise ValueError("task is not pending and compatible with this GPU count")
        if any(
            item.slurm_job_id == args.slurm_job_id and item.status in ("claimed", "running")
            for item in state.tasks
        ):
            raise ValueError(f"Slurm job {args.slurm_job_id} already owns a task")
        updated_task = replace(
            task,
            status="claimed",
            updated_at=now(),
            slurm_job_id=args.slurm_job_id,
            allocation_gpu_count=args.gpu_count,
            note=None,
        )
        write_state(args.state_file, replace_task(state, updated_task))
    _print_task(updated_task, args.json)
    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    with locked(args.state_file):
        state = load_state(args.state_file)
        task = find_task(state, args.task_id)
        if task.status != "claimed" or task.slurm_job_id != args.slurm_job_id:
            raise ValueError("start requires the matching claimed task and Slurm job")
        updated_task = replace(task, status="running", updated_at=now(), node=args.node)
        write_state(args.state_file, replace_task(state, updated_task))
    _print_task(updated_task, args.json)
    return 0


def _cmd_finish(args: argparse.Namespace) -> int:
    with locked(args.state_file):
        state = load_state(args.state_file)
        task = find_task(state, args.task_id)
        if task.status not in ("claimed", "running") or task.slurm_job_id != args.slurm_job_id:
            raise ValueError("finish requires the matching claimed or running task")
        updated_task = replace(
            task,
            status=args.result,
            updated_at=now(),
            note=args.note,
        )
        write_state(args.state_file, replace_task(state, updated_task))
    _print_task(updated_task, args.json)
    return 0


def _cmd_release(args: argparse.Namespace) -> int:
    with locked(args.state_file):
        state = load_state(args.state_file)
        task = find_task(state, args.task_id)
        if task.status != "claimed" or task.slurm_job_id != args.slurm_job_id:
            raise ValueError("release requires the matching claimed task")
        updated_task = replace(
            task,
            status="pending",
            updated_at=now(),
            slurm_job_id=None,
            allocation_gpu_count=None,
            node=None,
            note=args.note,
        )
        write_state(args.state_file, replace_task(state, updated_task))
    _print_task(updated_task, args.json)
    return 0


def _cmd_cancel(args: argparse.Namespace) -> int:
    with locked(args.state_file):
        state = load_state(args.state_file)
        task = find_task(state, args.task_id)
        if task.status != "pending":
            raise ValueError("only a pending local task can be canceled directly")
        updated_task = replace(task, status="canceled", updated_at=now(), note=args.note)
        write_state(args.state_file, replace_task(state, updated_task))
    _print_task(updated_task, args.json)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    with locked(args.state_file):
        task = find_task(load_state(args.state_file), args.task_id)
    _print_task(task, True)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    with locked(args.state_file):
        task = find_task(load_state(args.state_file), args.task_id)
    if task.status != "claimed" or task.slurm_job_id != args.slurm_job_id:
        raise ValueError("render requires the matching claimed task and Slurm job")
    print("#!/usr/bin/env bash")
    print("set -euo pipefail")
    print(f"export OPD_QUEUE_TASK_ID={shlex.quote(task.task_id)}")
    print(f"export OPD_QUEUE_SLURM_JOB_ID={shlex.quote(args.slurm_job_id)}")
    print(f"cd -- {_quoted_remote_cwd(task.remote_cwd)}")
    print(task.command)
    return 0


def _add_common_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=default_state_file(),
        help="Queue state path; defaults to repo-local gitignored runtime state.",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.set_defaults(handler=_cmd_init)

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("--title", required=True)
    add_parser.add_argument("--command", required=True)
    add_parser.add_argument("--remote-cwd", default="$HOME/scratch/opd/mopd_code")
    add_parser.add_argument("--gpu-counts", type=_parse_gpu_counts, default=(3, 4))
    _add_common_output(add_parser)
    add_parser.set_defaults(handler=_cmd_add)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--statuses", help="Comma-separated statuses to include.")
    _add_common_output(list_parser)
    list_parser.set_defaults(handler=_cmd_list)

    peek_parser = subparsers.add_parser("peek")
    peek_parser.add_argument("--gpu-count", type=int, choices=SUPPORTED_GPU_COUNTS, required=True)
    _add_common_output(peek_parser)
    peek_parser.set_defaults(handler=_cmd_peek)

    claim_parser = subparsers.add_parser("claim")
    claim_parser.add_argument("--gpu-count", type=int, choices=SUPPORTED_GPU_COUNTS, required=True)
    claim_parser.add_argument("--slurm-job-id", required=True)
    claim_parser.add_argument("--task-id")
    _add_common_output(claim_parser)
    claim_parser.set_defaults(handler=_cmd_claim)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--task-id", required=True)
    start_parser.add_argument("--slurm-job-id", required=True)
    start_parser.add_argument("--node", required=True)
    _add_common_output(start_parser)
    start_parser.set_defaults(handler=_cmd_start)

    finish_parser = subparsers.add_parser("finish")
    finish_parser.add_argument("--task-id", required=True)
    finish_parser.add_argument("--slurm-job-id", required=True)
    finish_parser.add_argument("--result", choices=TERMINAL_STATUSES, required=True)
    finish_parser.add_argument("--note")
    _add_common_output(finish_parser)
    finish_parser.set_defaults(handler=_cmd_finish)

    release_parser = subparsers.add_parser("release")
    release_parser.add_argument("--task-id", required=True)
    release_parser.add_argument("--slurm-job-id", required=True)
    release_parser.add_argument("--note")
    _add_common_output(release_parser)
    release_parser.set_defaults(handler=_cmd_release)

    cancel_parser = subparsers.add_parser("cancel")
    cancel_parser.add_argument("--task-id", required=True)
    cancel_parser.add_argument("--note")
    _add_common_output(cancel_parser)
    cancel_parser.set_defaults(handler=_cmd_cancel)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--task-id", required=True)
    show_parser.set_defaults(handler=_cmd_show)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--task-id", required=True)
    render_parser.add_argument("--slurm-job-id", required=True)
    render_parser.set_defaults(handler=_cmd_render)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
