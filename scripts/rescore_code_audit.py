#!/usr/bin/env python3
"""Post hoc Code rescoring of an audited run (response_records_<domain>.jsonl.gz).

Runs started with reward_manager=batch before mopd_verl/batched_reward.py was fixed
(fd8a0a0) logged every Code reward as 0. A run that kept per-response audit records can
be scored again offline:

  response_records row  --sample_id "<domain>:<sha256(train file path)[:16]>:<row>"-->
  training parquet row  --reward_model.ground_truth / data_source-->
  mopd_verl.batched_reward.compute_score_batched  (the training scorer, pass-all)

Nothing under the audit directory is modified. Per-step results go to --output_dir:
  step_NNNNNN.jsonl   one row per response (score, timeout, error, data source, length)
  summary.jsonl       one row per step (pass rate overall and per data source)
Finished steps are skipped on a rerun (--force recomputes them).

With --wandb_log the per-step summary is written to the original W&B run as
code_rescore/* metrics plotted against training/global_step. W&B does not allow
rewriting history rows that were already logged, so critic/score/mean and
code/reward/* keep their original (zero) values.

Needs PYTHONPATH to contain the repository root and third_party/verl, plus pyarrow and
transformers (and wandb for --wandb_log).
"""

from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

def parse_steps(spec: str | None, available: list[int]) -> list[int]:
    """'1-60', '5', '1-10,20,30-32' or None (all available)."""

    if not spec:
        return sorted(available)
    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            raise ValueError(f"Bad --steps entry {part!r}; use e.g. 1-60 or 5,7,10-12.")
        low = int(match.group(1))
        high = int(match.group(2) or low)
        wanted.update(range(low, high + 1))
    missing = sorted(wanted - set(available))
    if missing:
        raise ValueError(f"Steps without response records: {missing}")
    return sorted(wanted)


def discover_steps(audit_run_dir: Path, domain: str) -> dict[int, Path]:
    steps: dict[int, Path] = {}
    for entry in audit_run_dir.glob("step_*"):
        match = re.fullmatch(r"step_(\d+)", entry.name)
        records = entry / "jsonls" / f"response_records_{domain}.jsonl.gz"
        if match and records.is_file():
            steps[int(match.group(1))] = records
    return steps


def source_fingerprint(path_string: str) -> str:
    return hashlib.sha256(path_string.encode("utf-8")).hexdigest()[:16]


def split_sample_id(sample_id: str) -> tuple[str, str, int]:
    domain, fingerprint, row = sample_id.rsplit(":", 2)
    return domain, fingerprint, int(row)


def summarize_step(step: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["score"]) for row in rows]
    by_source: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        by_source[row["data_source"]].append(float(row["score"]))
    n = len(rows)
    return {
        "step": step,
        "n_samples": n,
        "n_pass": int(round(sum(scores))),
        "accuracy": (sum(scores) / n) if n else None,
        "reward_timeout": int(sum(row["reward_timeout"] for row in rows)),
        "reward_error": int(sum(row["reward_error"] for row in rows)),
        "mean_response_tokens": (sum(row["valid_token_count"] for row in rows) / n) if n else None,
        "by_data_source": {
            source: {"n": len(values), "accuracy": sum(values) / len(values)}
            for source, values in sorted(by_source.items())
        },
    }


RECORD_FIELDS = ("response_uid", "sample_id", "valid_token_count", "response_token_ids")


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read one response_records file keeping only the fields rescoring needs.

    Full audit records carry per-token float arrays (up to ~150 MB gzipped per step); the
    parse dominates the run time, so callers read each step once and reuse the slim rows.
    """

    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append({key: row[key] for key in RECORD_FIELDS})
    return rows


def read_all_records(paths: dict[int, Path], parse_workers: int) -> dict[int, list[dict[str, Any]]]:
    steps = sorted(paths)
    if parse_workers <= 1 or len(steps) <= 1:
        return {step: read_records(paths[step]) for step in steps}
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=min(parse_workers, len(steps))) as pool:
        return dict(zip(steps, pool.map(read_records, [paths[step] for step in steps])))


def load_ground_truth(train_parquet: str, indices: set[int]) -> dict[int, tuple[str, Any]]:
    """Return {row index: (data_source, ground_truth)} reading only the needed row groups."""

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(train_parquet)
    starts: list[int] = []
    total = 0
    for group in range(parquet.metadata.num_row_groups):
        starts.append(total)
        total += parquet.metadata.row_group(group).num_rows
    needed: dict[int, list[int]] = collections.defaultdict(list)
    for index in indices:
        if not 0 <= index < total:
            raise ValueError(f"Row index {index} outside {train_parquet} ({total} rows).")
        needed[bisect.bisect_right(starts, index) - 1].append(index)
    out: dict[int, tuple[str, Any]] = {}
    for group, group_indices in sorted(needed.items()):
        table = parquet.read_row_group(group, columns=["data_source", "reward_model"]).to_pylist()
        for index in group_indices:
            row = table[index - starts[group]]
            out[index] = (str(row["data_source"]), row["reward_model"]["ground_truth"])
    return out


def rescore_step(
    step: int,
    records: list[dict[str, Any]],
    ground_truth: dict[int, tuple[str, Any]],
    tokenizer: Any,
    workers: int,
    batch_timeout: float,
) -> list[dict[str, Any]]:
    from mopd_verl import batched_reward

    indices = [split_sample_id(record["sample_id"])[2] for record in records]
    texts = [
        tokenizer.decode(record["response_token_ids"], skip_special_tokens=True)
        for record in records
    ]
    results = batched_reward.compute_score_batched(
        data_sources=[ground_truth[i][0] for i in indices],
        solution_strs=texts,
        ground_truths=[ground_truth[i][1] for i in indices],
        extra_infos=[{} for _ in indices],
        max_workers=workers,
        batch_timeout_seconds=batch_timeout,
    )
    rows = []
    for record, index, result in zip(records, indices, results, strict=True):
        rows.append(
            {
                "step": step,
                "response_uid": record["response_uid"],
                "sample_id": record["sample_id"],
                "row_index": index,
                "data_source": ground_truth[index][0],
                "valid_token_count": record["valid_token_count"],
                "score": float(result["score"]),
                "reward_timeout": float(result["reward_timeout"]),
                "reward_error": float(result["reward_error"]),
            }
        )
    return rows


def log_to_wandb(summaries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    import wandb

    marker = Path(args.output_dir) / "wandb_logged.json"
    if marker.exists() and not args.wandb_force:
        raise SystemExit(f"{marker} exists: this output was already logged to W&B (use --wandb_force to log again).")
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        id=args.wandb_run_id,
        resume="allow",
        save_code=False,
    )
    run.define_metric("code_rescore/*", step_metric="training/global_step")
    for summary in summaries:
        payload: dict[str, Any] = {
            "training/global_step": summary["step"],
            "code_rescore/training_accuracy": summary["accuracy"],
            "code_rescore/n_pass": summary["n_pass"],
            "code_rescore/n_samples": summary["n_samples"],
            "code_rescore/reward_timeout": summary["reward_timeout"],
            "code_rescore/reward_error": summary["reward_error"],
        }
        for source, stats in summary["by_data_source"].items():
            payload[f"code_rescore/accuracy_{source}"] = stats["accuracy"]
        run.log(payload)
    run.summary["code_rescore/note"] = (
        "Code rewards logged during training (critic/score/mean, code/reward/*) were 0 because of a "
        "scoring bug (fixed in fd8a0a0); code_rescore/* is the pass rate recomputed offline from the audit "
        "response records."
    )
    run.finish()
    marker.write_text(json.dumps({"run": f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}",
                                  "steps": [s["step"] for s in summaries], "time": time.time()}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit_run_dir", required=True, help="Directory holding step_NNNNNN/ for one run tag.")
    parser.add_argument("--train_parquet", required=True, help="Training parquet the run sampled from.")
    parser.add_argument("--train_file_key", help="Path string exactly as in data.domain_train_files; "
                        "its sha256 prefix is checked against sample_id.")
    parser.add_argument("--tokenizer", required=True, help="Student tokenizer directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--domain", default="code")
    parser.add_argument("--steps", help="e.g. 1-60 or 1-10,20 (default: every step with records).")
    parser.add_argument("--workers", type=int, default=8, help="Scoring processes.")
    parser.add_argument("--parse_workers", type=int, default=4, help="Processes reading the record files.")
    parser.add_argument("--batch_timeout", type=float, default=900.0, help="Seconds per step.")
    parser.add_argument("--force", action="store_true", help="Recompute steps that already have results.")
    parser.add_argument("--wandb_log", action="store_true")
    parser.add_argument("--wandb_entity")
    parser.add_argument("--wandb_project")
    parser.add_argument("--wandb_run_id")
    parser.add_argument("--wandb_force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.wandb_log and not (args.wandb_entity and args.wandb_project and args.wandb_run_id):
        raise SystemExit("--wandb_log needs --wandb_entity, --wandb_project and --wandb_run_id.")

    available = discover_steps(Path(args.audit_run_dir), args.domain)
    if not available:
        raise SystemExit(f"No step_*/jsonls/response_records_{args.domain}.jsonl.gz under {args.audit_run_dir}.")
    steps = parse_steps(args.steps, list(available))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    todo = [s for s in steps if args.force or not (output_dir / f"step_{s:06d}.jsonl").is_file()]
    print(f"steps requested: {len(steps)}, to compute: {len(todo)}", flush=True)

    if todo:
        started_read = time.time()
        records_by_step = read_all_records({step: available[step] for step in todo}, args.parse_workers)
        print(f"read {len(records_by_step)} steps of response records in {time.time() - started_read:.0f}s", flush=True)
        wanted_rows: set[int] = set()
        fingerprints: set[str] = set()
        for step in todo:
            for record in records_by_step[step]:
                _, fingerprint, row = split_sample_id(record["sample_id"])
                fingerprints.add(fingerprint)
                wanted_rows.add(row)
        if args.train_file_key:
            expected = source_fingerprint(args.train_file_key)
            if fingerprints != {expected}:
                raise SystemExit(f"sample_id fingerprints {sorted(fingerprints)} != sha256({args.train_file_key!r})[:16]={expected}")
            print(f"source fingerprint verified: {expected}", flush=True)
        else:
            print(f"WARNING: source fingerprint not verified (sample_ids carry {sorted(fingerprints)})", flush=True)
        ground_truth = load_ground_truth(args.train_parquet, wanted_rows)
        print(f"loaded {len(ground_truth)} training rows", flush=True)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        for step in todo:
            started = time.time()
            rows = rescore_step(step, records_by_step.pop(step), ground_truth, tokenizer, args.workers, args.batch_timeout)
            temp = output_dir / f"step_{step:06d}.jsonl.tmp"
            temp.write_text("".join(json.dumps(row) + "\n" for row in rows))
            temp.replace(output_dir / f"step_{step:06d}.jsonl")
            summary = summarize_step(step, rows)
            print(
                f"step {step:3d}: {summary['n_pass']:3d}/{summary['n_samples']} pass "
                f"({summary['accuracy']:.3f}) timeouts={summary['reward_timeout']} "
                f"errors={summary['reward_error']} [{time.time() - started:.1f}s]",
                flush=True,
            )

    summaries = []
    for step in steps:
        rows = [json.loads(line) for line in (output_dir / f"step_{step:06d}.jsonl").read_text().splitlines()]
        summaries.append(summarize_step(step, rows))
    (output_dir / "summary.jsonl").write_text("".join(json.dumps(s) + "\n" for s in summaries))
    print(f"wrote {output_dir / 'summary.jsonl'} ({len(summaries)} steps)", flush=True)

    if args.wandb_log:
        log_to_wandb(summaries, args)
        print("logged code_rescore/* to W&B", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
