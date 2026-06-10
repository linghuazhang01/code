"""ToolRL API-Bank official benchmark wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.domains.toolrl.common import extract_between, extract_tool_calls, score_single_tool_call
from eval.official_utils import OfficialEvalResult, ensure_output_dir, limited, load_vllm, sampling_params, write_json

LEVELS = ("1", "2", "3")


def _load_level_records(source_dir: Path, level: str, max_samples: int | None) -> list[dict[str, Any]]:
    path = source_dir / f"level-{level}-api_processed.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    return limited(records, max_samples)


def _level_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(int(record["score"] == 1) for record in records)
    return {"correct": correct, "total": total, "accuracy": correct / total if total else None}


def _generate_chat(llm: Any, messages: list[dict[str, str]], params: Any, enable_thinking: bool | None) -> str:
    if enable_thinking is None:
        request_output = llm.chat(messages, sampling_params=params)[0]
        return request_output.outputs[0].text.strip()
    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    request_output = llm.generate([prompt], params)[0]
    return request_output.outputs[0].text.strip()


def run_api_bank(
    *,
    model_path: str,
    source_dir: str | Path,
    output_dir: str | Path,
    levels: list[str],
    max_samples: int | None,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool | None,
) -> OfficialEvalResult:
    source = Path(source_dir)
    output = ensure_output_dir(output_dir)
    llm = load_vllm(model_path, tensor_parallel_size, gpu_memory_utilization, max_model_len)
    params = sampling_params(max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    results: dict[str, dict[str, Any]] = {}
    for level in levels:
        for index, data in enumerate(_load_level_records(source, level, max_samples)):
            sample_id = f"Level{level}_{index}"
            messages = [
                {"role": "system", "content": str(data["system"])},
                {"role": "user", "content": str(data["user"])},
            ]
            assistant_output = _generate_chat(llm, messages, params, enable_thinking)
            tool_calls = extract_tool_calls(assistant_output)
            result = {
                "id": sample_id,
                "level": level,
                "data": data,
                "raw_output": assistant_output,
                "thought": extract_between(assistant_output, "<think>", "</think>"),
                "tool_calls": tool_calls,
                "score": score_single_tool_call(tool_calls, data.get("answer")),
            }
            results[sample_id] = result

    records = list(results.values())
    by_level = {
        f"level_{level}": _level_summary([record for record in records if record["level"] == level])
        for level in levels
    }
    correct = sum(item["correct"] for item in by_level.values())
    total = sum(item["total"] for item in by_level.values())
    summary = {
        "dataset": "api_bank",
        "model_path": model_path,
        "sample_count": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "enable_thinking": enable_thinking,
        "levels": by_level,
    }
    write_json(output / "result.json", results)
    write_json(output / "summary.json", summary)
    return OfficialEvalResult(dataset="api_bank", output_dir=output, summary=summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-dir", default="../temp/grpo_sources/ToolRL/benchmarks/API-Bank")
    parser.add_argument("--output-dir", default="eval/results/official_toolrl/api_bank")
    parser.add_argument("--levels", nargs="+", default=list(LEVELS), choices=LEVELS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0001)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", choices=("true", "false", "auto"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_api_bank(
        model_path=args.model_path,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        levels=args.levels,
        max_samples=args.max_samples,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=None if args.enable_thinking == "auto" else args.enable_thinking == "true",
    )
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
