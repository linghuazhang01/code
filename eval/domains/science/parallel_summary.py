"""Merged summaries for prompt-sharded official science evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def summarize_mmlupro_records(
    records: Sequence[Mapping[str, Any]],
    *,
    model_path: str,
) -> dict[str, Any]:
    """Recompute the official MMLU-Pro summary after merging prompt shards."""

    correct = sum(int(record.get("correct") is True) for record in records)
    prompts: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    categories: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        question_id = str(record.get("question_id", record.get("index")))
        prompts[question_id].append(record)
        categories[str(record.get("category", "unknown"))].append(record)

    def category_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        category_prompts: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in items:
            question_id = str(item.get("question_id", item.get("index")))
            category_prompts[question_id].append(item)
        category_correct = sum(int(item.get("correct") is True) for item in items)
        passed = sum(
            any(item.get("correct") is True for item in values)
            for values in category_prompts.values()
        )
        return {
            "correct": category_correct,
            "total": len(items),
            "passed_prompts": passed,
            "prompt_count": len(category_prompts),
            "accuracy": category_correct / len(items) if items else None,
            "observed_pass_at_k": (
                passed / len(category_prompts) if category_prompts else None
            ),
        }

    passed_prompts = sum(
        any(item.get("correct") is True for item in values)
        for values in prompts.values()
    )
    rollout_counts = {len(values) for values in prompts.values()}
    return {
        "dataset": "mmlupro_500_seed42",
        "model_path": model_path,
        "prompt_count": len(prompts),
        "rollouts_per_prompt": rollout_counts.pop() if len(rollout_counts) == 1 else None,
        "sample_count": len(records),
        "correct": correct,
        "accuracy": correct / len(records) if records else None,
        "passed_prompts": passed_prompts,
        "observed_pass_at_k": passed_prompts / len(prompts) if prompts else None,
        "per_category": {
            category: category_summary(items)
            for category, items in sorted(categories.items())
        },
    }
