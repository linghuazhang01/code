"""ToolRL Bamboogle official benchmark wrapper with configurable external APIs."""

from __future__ import annotations

import argparse
import http.client
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from eval.domains.toolrl.common import extract_between, extract_tool_calls, response_text
from eval.official_utils import (
    OfficialEvalResult,
    ensure_output_dir,
    limited,
    load_vllm,
    retry_call,
    sampling_params,
    write_json,
)

BAMBOOGLE_SYSTEM_PROMPT = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls
to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
1. Name: Search
Description: Search the answer for a specific query leveraging the google search engine
Parameters: {"query": {"description": "The query to search for.", "type": "string", "default": ""}}

**Steps for Each Turn**
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

**Output Format**
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call>
{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "... ...": "... ..."}}
...
</tool_call>
<response> AI's final response </response>
```

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning.
2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields.
3. Refer to previous dialogue records and tool feedback noted as `<obs>` when available.
"""

USER_TEMPLATE = (
    "{query}\n"
    "Perform tool call if you are not sure about the answer. "
    "Make a plan to solve the problem step by step."
)

JUDGE_SYSTEM_PROMPT = """You are a helpful assistant to judge whether an answer is correct.
Given the query, ground truth answer, and predicted answer, decide if the predicted answer is semantically equivalent.

Output exactly:
- Thought: <brief thought>
- Judgment: <Yes/No>
"""

JUDGE_USER_TEMPLATE = """**Query**
{query}

**Ground Truth Answer**
{ground_truth}

**Predicted Answer**
{predicted}

Your Response:
"""


def _format_history(history: list[dict[str, str]]) -> str:
    output = "**Dialogue Records History**\n"
    for item in history:
        if item["role"] == "user":
            output += f"<user> {item['content']} </user>\n\n"
        elif item["role"] == "obs":
            output += f"<obs> {item['content']} </obs>\n\n"
        elif item["role"] == "assistant":
            output += f"{item['content']}\n"
    return output.strip()


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


def search_serper(query: str, *, api_key: str, base_url: str, num: int = 5) -> str:
    parsed = urlparse(base_url)
    host = parsed.netloc or parsed.path
    path = parsed.path if parsed.netloc and parsed.path else "/search"
    connection_cls = http.client.HTTPConnection if parsed.scheme == "http" else http.client.HTTPSConnection
    conn = connection_cls(host)
    payload = json.dumps({"q": query, "num": 15})
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    for _ in range(6):
        conn.request("POST", path, payload, headers)
        response = conn.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        if data.get("organic"):
            break
    else:
        return "Search Error: Timeout"

    lines: list[str] = []
    answer_box = data.get("answerBox") or {}
    if answer_box and answer_box.get("title"):
        if answer_box.get("answer"):
            lines.append(f"1. {answer_box['title']}\n- Answer: {answer_box['answer']}")
        elif answer_box.get("snippet"):
            lines.append(f"1. {answer_box['title']}\n- Snippet: {answer_box['snippet']}")
    for item in data.get("organic", []):
        if len(lines) >= num:
            break
        if item.get("title") and item.get("snippet"):
            lines.append(f"{len(lines) + 1}. {item['title']}\n- Snippet: {item['snippet']}")
    return "\n".join(lines).strip()


def judge_answer(
    *,
    query: str,
    ground_truth: str,
    predicted: str,
    api_key: str,
    base_url: str | None,
    model: str,
) -> tuple[int, str]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Bamboogle judge requires the `openai` package.") from exc

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(query=query, ground_truth=ground_truth, predicted=predicted),
        },
    ]
    response = retry_call(
        lambda: client.chat.completions.create(model=model, messages=messages, temperature=0, n=1)
    )
    content = response.choices[0].message.content.strip()
    judgment = content.split("- Judgment:")[-1].strip() if "- Judgment:" in content else content
    return int(judgment.startswith("Yes")), content


def run_bamboogle(
    *,
    model_path: str,
    source_file: str | Path,
    output_dir: str | Path,
    max_samples: int | None,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool | None,
    serper_api_key: str,
    serper_base_url: str,
    judge_api_key: str,
    judge_base_url: str | None,
    judge_model: str,
    round_threshold: int,
) -> OfficialEvalResult:
    output = ensure_output_dir(output_dir)
    records = limited(json.loads(Path(source_file).read_text(encoding="utf-8")), max_samples)
    llm = load_vllm(model_path, tensor_parallel_size, gpu_memory_utilization, max_model_len)
    params = sampling_params(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    results: dict[str, dict[str, Any]] = {}

    for index, data in enumerate(records):
        query = str(data["Question"])
        ground_truth = str(data["Answer"])
        history = [{"role": "user", "content": USER_TEMPLATE.format(query=query)}]
        predicted = ""
        for round_index in range(1, round_threshold + 1):
            messages = [
                {"role": "system", "content": BAMBOOGLE_SYSTEM_PROMPT},
                {"role": "user", "content": _format_history(history)},
            ]
            assistant_output = _generate_chat(llm, messages, params, enable_thinking)
            tool_calls = extract_tool_calls(assistant_output)
            if tool_calls:
                thought = extract_between(assistant_output, "<think>", "</think>")
                tool_query = str(dict(tool_calls[0].get("parameters") or {}).get("query", ""))
                search_result = search_serper(tool_query, api_key=serper_api_key, base_url=serper_base_url, num=5)
                history.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"<think> {thought} </think>\n<tool_call>\n"
                            f"{json.dumps(tool_calls[0], ensure_ascii=False)}\n</tool_call>"
                        ),
                    }
                )
                history.append(
                    {
                        "role": "obs",
                        "content": f"Search results for the first tool call query:\n{search_result}",
                    }
                )
                if round_index < round_threshold:
                    history.append(
                        {
                            "role": "user",
                            "content": "Please continue to search if needed. Otherwise answer directly.",
                        }
                    )
                continue
            predicted = response_text(assistant_output)
            history.append({"role": "assistant", "content": assistant_output})
            break

        if not predicted:
            predicted = "Round Limit Exceeded, no answer is derived."
            score, judge_raw = 0, ""
        else:
            score, judge_raw = judge_answer(
                query=query,
                ground_truth=ground_truth,
                predicted=predicted,
                api_key=judge_api_key,
                base_url=judge_base_url,
                model=judge_model,
            )
        results[str(index)] = {
            "query": query,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "history": history,
            "judge": judge_raw,
            "score": score,
        }
        write_json(output / "result.json", results)

    total = len(results)
    correct = sum(int(item["score"] == 1) for item in results.values())
    tool_calls = sum(1 for item in results.values() for event in item["history"] if event["role"] == "obs")
    summary = {
        "dataset": "bamboogle",
        "model_path": model_path,
        "sample_count": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "avg_tool_calls": tool_calls / total if total else None,
        "judge_model": judge_model,
        "enable_thinking": enable_thinking,
        "serper_base_url": serper_base_url,
        "judge_base_url": judge_base_url,
    }
    write_json(output / "summary.json", summary)
    return OfficialEvalResult(dataset="bamboogle", output_dir=output, summary=summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-file", default="../temp/grpo_sources/ToolRL/benchmarks/Bamboogle/data.json")
    parser.add_argument("--output-dir", default="eval/results/official_toolrl/bamboogle")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", choices=("true", "false", "auto"), default="auto")
    parser.add_argument("--round-threshold", type=int, default=4)
    parser.add_argument("--serper-api-key", required=True)
    parser.add_argument("--serper-base-url", default="https://google.serper.dev/search")
    parser.add_argument("--judge-api-key", required=True)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-model", default="gpt-4o")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = vars(args)
    payload["enable_thinking"] = None if args.enable_thinking == "auto" else args.enable_thinking == "true"
    result = run_bamboogle(**payload)
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
