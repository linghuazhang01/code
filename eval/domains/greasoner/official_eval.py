"""Standalone wrappers for General-Reasoner official evaluation scripts."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from eval.official_utils import (
    OfficialEvalResult,
    ensure_output_dir,
    limited,
    load_vllm,
    sampling_params,
    write_json,
    write_jsonl,
)

DATASET_CHOICES = ("mmlupro", "gpqa_d", "supergpqa", "theoremqa", "bbeh")
CHOICE_LETTERS = tuple("ABCDEFGHIJ")
LOCAL_DATA_FILES = {
    "mmlupro": Path("data/eval_data/greasoner/official/MMLU-Pro/test.parquet"),
    "gpqa_d": Path("data/eval_data/greasoner/official/GPQA-D/test.parquet"),
    "supergpqa": Path("data/eval_data/greasoner/official/SuperGPQA/test.parquet"),
    "theoremqa": Path("data/eval_data/greasoner/official/TheoremQA/test.parquet"),
    "bbeh": Path("data/eval_data/greasoner/official/BBEH/test.parquet"),
}


def _progress(message: str) -> None:
    print(f"[greasoner-eval] {message}", flush=True)


def extract_last_boxed(text: str) -> str | None:
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    return matches[-1].group(1) if matches else None


def extract_last_final_answer(text: str) -> str | None:
    patterns = (
        r"Final Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"The answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"Answer:\s*((?:[^<]|<[^<])*?)\n",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip()
    return None


def extract_solution(solution_str: str) -> str | None:
    if "<|im_start|>user" in solution_str:
        model_output = re.sub(
            r"^.*?<\|im_start\|>assistant",
            "<|im_start|>assistant",
            solution_str,
            flags=re.DOTALL,
            count=1,
        )
    elif "Assistant:" in solution_str:
        model_output = solution_str.split("Assistant:")[-1].strip()
    else:
        model_output = solution_str

    for stop_word in ("</s>", "<|im_end|>", "<|endoftext|>"):
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    return extract_last_boxed(model_output) or extract_last_final_answer(model_output)


def strip_latex(response: str) -> str:
    for wrapper in ("boxed", "text", "texttt"):
        prefix = f"{wrapper}{{"
        if prefix in response and response.endswith("}"):
            return response[:-1].split(prefix, 1)[1]
    if response.startswith("$") and response.endswith("$"):
        return response[1:-1]
    return response


def evaluate_bbeh_correctness(sample: str | None, reference: str) -> bool:
    prediction = strip_latex((sample or "").strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "").split("\n")[0]
    if prediction.endswith("."):
        prediction = prediction[:-1]
    reference_norm = reference.strip().lower().replace(", ", ",")
    if prediction == reference_norm:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference_norm
    if len(reference_norm) == 3 and reference_norm[0] == "(" and reference_norm[-1] == ")":
        return reference_norm[1] == prediction
    try:
        return float(prediction) == float(reference_norm)
    except ValueError:
        return prediction.replace("'", "") == reference_norm.replace("'", "")


def form_options(options: Iterable[str]) -> str:
    output = "Options are:\n"
    for option_text, letter in zip(options, CHOICE_LETTERS, strict=False):
        output += f"({letter}): {option_text}\n"
    return output


def get_prediction(output: str) -> str:
    solution = extract_solution(output)
    if solution is None:
        return random.choice(list(CHOICE_LETTERS))
    for option in CHOICE_LETTERS:
        if option in solution:
            return option
    return random.choice(list(CHOICE_LETTERS))


def render_prompt(tokenizer: Any, content: str, enable_thinking: bool | None) -> str:
    messages = [{"role": "user", "content": content}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_hf_dataset(dataset_name: str, *args: Any, **kwargs: Any) -> Any:
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("General-Reasoner official eval requires the `datasets` package.") from exc
    return datasets.load_dataset(dataset_name, *args, **kwargs)


def _load_local_parquet(dataset_key: str) -> list[dict[str, Any]] | None:
    path = LOCAL_DATA_FILES[dataset_key]
    if not path.exists():
        return None
    dataset = load_hf_dataset("parquet", data_files={"test": str(path)}, split="test")
    return list(dataset)


def _prepare_gpqa_d_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(0)
    prepared: list[dict[str, Any]] = []
    for entry in entries:
        choices = [
            entry["Correct Answer"],
            entry["Incorrect Answer 1"],
            entry["Incorrect Answer 2"],
            entry["Incorrect Answer 3"],
        ]
        choices = [choices[index] for index in rng.sample(range(4), 4)]
        correct_index = choices.index(entry["Correct Answer"])
        prepared.append(
            {
                **entry,
                "options": choices,
                "answer_letter": "ABCD"[correct_index],
                "category": entry.get("High-level domain", "unknown"),
            }
        )
    return prepared


def _dataset_entries(dataset_key: str) -> tuple[list[dict[str, Any]], str, str]:
    local_entries = _load_local_parquet(dataset_key)
    if dataset_key == "mmlupro":
        entries = local_entries if local_entries is not None else list(load_hf_dataset("TIGER-Lab/MMLU-Pro")["test"])
        return entries, "category", "answer"
    if dataset_key == "gpqa_d":
        if local_entries is None:
            raise FileNotFoundError(
                "GPQA-D local parquet is missing. Run "
                "`python -m eval.domains.greasoner.download_official_data --datasets gpqa_d` first."
            )
        return _prepare_gpqa_d_entries(local_entries), "category", "answer_letter"
    if dataset_key == "supergpqa":
        entries = local_entries if local_entries is not None else list(load_hf_dataset("m-a-p/SuperGPQA")["train"])
        return entries, "discipline", "answer_letter"
    if dataset_key == "theoremqa":
        entries = local_entries if local_entries is not None else list(load_hf_dataset("TIGER-Lab/TheoremQA")["test"])
        return entries, "Answer_type", "Answer"
    if dataset_key == "bbeh":
        entries = local_entries if local_entries is not None else list(load_hf_dataset("MrLight/bbeh-eval")["train"])
        return entries, "task", "answer"
    raise ValueError(f"Unsupported General-Reasoner dataset: {dataset_key}")


def _prompt_for_entry(dataset_key: str, entry: dict[str, Any]) -> str:
    if dataset_key == "theoremqa":
        return (
            f"{entry['Question']}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
    if dataset_key == "bbeh":
        return (
            f"{entry['question']}\n\n"
            "Please reason step by step, and put your final answer option within \\boxed{}."
        )
    if dataset_key == "gpqa_d":
        return (
            f"{entry['Question']}\n\n"
            f"A: {entry['options'][0]}\n"
            f"B: {entry['options'][1]}\n"
            f"C: {entry['options'][2]}\n"
            f"D: {entry['options'][3]}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}.\n"
            "Please only provide the letter of the answer in the box."
        )
    option_instruction = (
        "Please reason step by step, and put your final answer option within \\boxed{}. "
        "Only put the option letter in the box, e.g. \\boxed{A}. There is only one correct answer."
    )
    if dataset_key == "supergpqa":
        option_instruction = option_instruction.replace("option letter", "letter")
    return f"{entry['question']}\n{form_options(entry['options'])}\n\n{option_instruction}"


def judge_answer_equivalence(
    *,
    reference: str,
    prediction: str | None,
    api_key: str,
    base_url: str | None,
    model: str,
) -> tuple[bool, str]:
    if not prediction:
        return False, ""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("TheoremQA judging requires the `openai` package.") from exc

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    prompt = (
        "Judge whether the predicted answer is equivalent to the reference answer. "
        "Allow trivial formatting differences, equivalent numeric forms, and equivalent units. "
        "Return exactly Yes or No.\n\n"
        f"Reference answer: {reference}\n"
        f"Predicted answer: {prediction}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = (response.choices[0].message.content or "").strip()
    return content.lower().startswith("yes"), content


def run_dataset(
    *,
    dataset_key: str,
    model_path: str,
    output_dir: str | Path,
    max_samples: int | None,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool | None,
    judge_api_key: str | None = None,
    judge_base_url: str | None = None,
    judge_model: str = "gpt-4o",
) -> OfficialEvalResult:
    output = ensure_output_dir(Path(output_dir) / dataset_key)
    entries, category_field, answer_field = _dataset_entries(dataset_key)
    entries = limited(entries, max_samples)
    total_entries = len(entries)
    _progress(f"dataset={dataset_key} samples={total_entries} output={output}")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("General-Reasoner official eval requires the `transformers` package.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = load_vllm(model_path, tensor_parallel_size, gpu_memory_utilization, max_model_len)
    params = sampling_params(max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    prompts = [render_prompt(tokenizer, _prompt_for_entry(dataset_key, entry), enable_thinking) for entry in entries]
    _progress(f"dataset={dataset_key} generation_start")
    outputs = llm.generate(prompts, params)
    _progress(f"dataset={dataset_key} generation_done outputs={len(outputs)} scoring_start")
    records: list[dict[str, Any]] = []
    correct = 0
    per_category: dict[str, dict[str, int]] = {}
    for index, (entry, request_output) in enumerate(zip(entries, outputs, strict=True)):
        completion = request_output.outputs[0].text
        category = str(entry.get(category_field, "unknown"))
        per_category.setdefault(category, {"correct": 0, "total": 0})
        judge_raw = None
        if dataset_key == "theoremqa":
            prediction = extract_solution(completion)
            if not judge_api_key:
                raise ValueError("TheoremQA requires --judge-api-key or OPENAI_API_KEY for paper-aligned scoring.")
            is_correct, judge_raw = judge_answer_equivalence(
                reference=str(entry[answer_field]),
                prediction=prediction,
                api_key=judge_api_key,
                base_url=judge_base_url,
                model=judge_model,
            )
        elif dataset_key == "bbeh":
            prediction = extract_solution(completion)
            is_correct = evaluate_bbeh_correctness(prediction, str(entry[answer_field]))
        else:
            prediction = get_prediction(completion)
            is_correct = prediction == str(entry[answer_field])
        correct += int(is_correct)
        per_category[category]["correct"] += int(is_correct)
        per_category[category]["total"] += 1
        records.append(
            {
                "index": index,
                "dataset": dataset_key,
                "category": category,
                "prompt": prompts[index],
                "completion": completion,
                "prediction": prediction,
                "answer": entry[answer_field],
                "correct": is_correct,
                "judge": judge_raw,
                "source": entry,
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == total_entries:
            _progress(f"dataset={dataset_key} scored={index + 1}/{total_entries} correct={correct}")

    total = len(records)
    summary = {
        "dataset": dataset_key,
        "model_path": model_path,
        "sample_count": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "judge_model": judge_model if dataset_key == "theoremqa" else None,
        "per_category": {
            key: {**value, "accuracy": value["correct"] / value["total"] if value["total"] else None}
            for key, value in sorted(per_category.items())
        },
    }
    write_jsonl(output / "records.jsonl", records)
    write_json(output / "summary.json", summary)
    _progress(f"dataset={dataset_key} done accuracy={summary['accuracy']} records={output / 'records.jsonl'}")
    return OfficialEvalResult(dataset=dataset_key, output_dir=output, summary=summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="data/eval_data/results/official_greasoner")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", choices=("true", "false", "auto"), default="auto")
    parser.add_argument("--judge-api-key", default=None, help="Required for TheoremQA scoring.")
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-model", default="gpt-4o")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    enable_thinking = None if args.enable_thinking == "auto" else args.enable_thinking == "true"
    result = run_dataset(
        dataset_key=args.dataset,
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=enable_thinking,
        judge_api_key=args.judge_api_key,
        judge_base_url=args.judge_base_url,
        judge_model=args.judge_model,
    )
    print(json.dumps({"dataset": result.dataset, "summary": result.summary}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
