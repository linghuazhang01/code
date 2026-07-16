"""Generate JSON and Markdown reports for thinking-mode eval runs."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.domains.code import is_code_dataset
from eval.domains.science import is_science_dataset
from eval.domains.search import is_search_dataset
from eval.domains.toolrl import is_toolrl_dataset

LOG_PATTERN = re.compile(
    r"mode=(?P<mode>\S+) dataset=(?P<dataset>\S+) score=(?P<score>[0-9.]+|None) "
    r"generated_tokens=(?P<generated_tokens>\d+) thinking_tokens=(?P<thinking_tokens>\d+) "
    r"(?:batch_latency|latency)=(?P<latency>[0-9.]+)s"
)
COMPACT_RECORD_FIELDS = (
    "sample_id",
    "mode",
    "dataset",
    "ability",
    "score",
    "correct",
    "prompt_tokens",
    "generated_tokens",
    "thinking_tokens",
    "answer_tokens",
    "total_tokens",
    "latency_seconds",
    "generated_tokens_per_second",
    "max_new_tokens",
    "prediction",
    "completion_preview",
    "rollout_index",
    "generation_seed",
    "source",
    "record_index",
)


def _ability(dataset: str) -> str:
    normalized = dataset.lower()
    if "ifeval" in normalized or "ifbench" in normalized:
        return "if"
    if "gpqa" in normalized or "hle" in normalized or is_science_dataset(dataset):
        return "science"
    if is_code_dataset(dataset):
        return "code"
    if is_toolrl_dataset(dataset):
        return "tool"
    if is_search_dataset(dataset):
        return "search"
    return "math"


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _read_log_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for index, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        match = LOG_PATTERN.search(line)
        if not match:
            continue
        data = match.groupdict()
        score = None if data["score"] == "None" else float(data["score"])
        records.append(
            {
                "record_index": index,
                "mode": data["mode"],
                "dataset": data["dataset"],
                "ability": _ability(data["dataset"]),
                "score": score,
                "correct": None if score is None else score > 0,
                "generated_tokens": int(data["generated_tokens"]),
                "thinking_tokens": int(data["thinking_tokens"]),
                "answer_tokens": int(data["generated_tokens"]) - int(data["thinking_tokens"]),
                "latency_seconds": float(data["latency"]),
                "source": "log",
            }
        )
    return records


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    compact = {field: record[field] for field in COMPACT_RECORD_FIELDS if field in record}
    if "ability" not in compact and "dataset" in compact:
        compact["ability"] = _ability(str(compact["dataset"]))
    if "answer_tokens" not in compact and "generated_tokens" in compact and "thinking_tokens" in compact:
        compact["answer_tokens"] = int(compact["generated_tokens"]) - int(compact["thinking_tokens"])
    return compact


def _detail_record(record: dict[str, Any]) -> dict[str, Any] | None:
    if not any(record.get(field) for field in ("messages", "prompt", "completion")):
        return None
    detail = {
        "sample_id": record.get("sample_id"),
        "mode": record.get("mode"),
        "dataset": record.get("dataset"),
        "ability": record.get("ability") or _ability(str(record.get("dataset", ""))),
        "ground_truth": record.get("ground_truth"),
        "prediction": record.get("prediction"),
        "score": record.get("score"),
        "correct": record.get("correct"),
        "messages": record.get("messages"),
        "prompt": record.get("prompt"),
        "response": record.get("completion"),
        "response_preview": record.get("completion_preview"),
        "reward_metadata": record.get("reward_metadata"),
    }
    return {key: value for key, value in detail.items() if value is not None}


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [record for record in records if record.get("score") is not None]
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(scored):
        by_sample[str(record.get("sample_id") or f"record:{index}")].append(record)
    sample_sizes = [len(sample_records) for sample_records in by_sample.values()]
    return {
        "sample_count": len(records),
        "scored_count": len(scored),
        "accuracy": _mean([1.0 if record.get("correct") else 0.0 for record in scored]),
        "avg_score": _mean([float(record["score"]) for record in scored]),
        "unique_sample_count": len(by_sample),
        "min_samples_per_prompt": min(sample_sizes, default=0),
        "max_samples_per_prompt": max(sample_sizes, default=0),
        "avg_at_k": _mean(
            [
                sum(float(record["score"]) for record in sample_records) / len(sample_records)
                for sample_records in by_sample.values()
            ]
        ),
        "observed_pass_at_k": _mean(
            [float(any(record.get("correct") is True for record in sample_records)) for sample_records in by_sample.values()]
        ),
        "avg_generated_tokens": _mean([float(record.get("generated_tokens", 0)) for record in records]),
        "avg_thinking_tokens": _mean([float(record.get("thinking_tokens", 0)) for record in records]),
        "avg_answer_tokens": _mean([float(record.get("answer_tokens", 0)) for record in records]),
        "avg_total_tokens": _mean([float(record.get("total_tokens", record.get("generated_tokens", 0))) for record in records]),
        "avg_latency_seconds": _mean([float(record.get("latency_seconds", 0)) for record in records]),
        "max_generated_tokens": max([int(record.get("generated_tokens", 0)) for record in records], default=0),
    }


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    aggregates: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        mode = str(record["mode"])
        dataset = str(record["dataset"])
        ability = str(record.get("ability") or _ability(dataset))
        groups[(mode, dataset, ability)].append(record)
        aggregates[(mode, "ALL", ability)].append(record)
        aggregates[(mode, "ALL", "all")].append(record)

    rows: list[dict[str, Any]] = []
    for (mode, dataset, ability), items in {**groups, **aggregates}.items():
        row = {"mode": mode, "dataset": dataset, "ability": ability}
        row.update(_summarize_group(items))
        rows.append(row)
    return sorted(rows, key=lambda row: (row["mode"], row["dataset"], row["ability"]))


def _format_percent(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{100 * float(value):.2f}%"


def _format_number(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.1f}"


def _summary_table(rows: list[dict[str, Any]], mode: str) -> str:
    selected = [row for row in rows if row["mode"] == mode and row["dataset"] != "ALL"]
    lines = [
        "| Dataset | Domain | Records | Scored | Unique prompts | K | Accuracy | Avg@K | Observed pass@K |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| {dataset} | {ability} | {records} | {scored} | {unique} | {k} | {acc} | {avg_k} | {pass_k} |".format(
                dataset=row["dataset"],
                ability=row["ability"],
                records=row["sample_count"],
                scored=row["scored_count"],
                unique=row.get("unique_sample_count", 0),
                k=row.get("min_samples_per_prompt", 0),
                acc=_format_percent(row["accuracy"]),
                avg_k=_format_percent(row.get("avg_at_k")),
                pass_k=_format_percent(row.get("observed_pass_at_k")),
            )
        )
    return "\n".join(lines)


def write_report(payload: dict[str, Any], output_dir: Path, detail_records: list[dict[str, Any]] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "thinking_eval_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in payload["records"]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if detail_records:
        with (output_dir / "prompt_response_records.jsonl").open("w", encoding="utf-8") as handle:
            for record in detail_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    for domain in ("math", "code", "if", "science", "reasoning", "tool", "search"):
        domain_dir = output_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        domain_records = [record for record in payload["records"] if record.get("ability") == domain]
        domain_detail_records = [record for record in detail_records or [] if record.get("ability") == domain]
        domain_summary = [row for row in payload["summary"] if row.get("ability") in {domain, "all"}]
        (domain_dir / "summary.json").write_text(
            json.dumps(domain_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with (domain_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in domain_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if domain_detail_records:
            with (domain_dir / "prompt_response_records.jsonl").open("w", encoding="utf-8") as handle:
                for record in domain_detail_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = payload["summary"]
    readme = [
        f"# {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Model: `{payload['model_path']}`",
        f"- Scoring backend: `{payload['scoring_backend']}`",
        f"- Record source: `{payload['record_source']}`",
        f"- Records: `{len(payload['records'])}` / `{payload['expected_total']}`",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Generation config: `{payload.get('run_config', {})}`",
        "",
        "## Notes",
        "",
        payload["notes"] or "No extra notes.",
        "",
        "## Non-Thinking",
        "",
        _summary_table(summary, "non_thinking"),
        "",
        "## Thinking",
        "",
        _summary_table(summary, "thinking"),
        "",
        "## Files",
        "",
        "- `thinking_eval_results.json`: structured summary and records",
        "- `records.jsonl`: compact per-record metrics used for this report",
        "- `prompt_response_records.jsonl`: prompt and response details when completions are saved",
        "- `<domain>/summary.json`, `<domain>/records.jsonl`: per-domain views",
        "- `thinking_eval_samples.jsonl`: full raw evaluator output when the run has completed",
        "- `run.log`: copied remote log when available",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--status", choices=("partial", "final"), default="final")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--expected-total", type=int, default=None)
    parser.add_argument("--scoring-backend", default="verl.utils.reward_score.default_compute_score")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    run_id = args.run_id or output_dir.name
    raw_records = _read_jsonl(output_dir / "thinking_eval_samples.jsonl")
    record_source = "thinking_eval_samples.jsonl"
    if not raw_records and args.log_file:
        raw_records = _read_log_records(Path(args.log_file))
        record_source = "run.log"
    records = [_compact_record(record) for record in raw_records]
    detail_records = [record for record in (_detail_record(record) for record in raw_records) if record is not None]
    expected_total = args.expected_total if args.expected_total is not None else len(records)
    run_config_path = output_dir / "eval_run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    payload = {
        "run_id": run_id,
        "status": args.status,
        "model_path": args.model_path,
        "scoring_backend": args.scoring_backend,
        "record_source": record_source,
        "expected_total": expected_total,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": args.notes,
        "summary": summarize_records(records),
        "records": records,
        "run_config": run_config,
    }
    write_report(payload, output_dir, detail_records)


if __name__ == "__main__":
    main()
