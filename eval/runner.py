"""Evaluate Qwen thinking-mode validation accuracy and generation cost."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

from eval.common import (
    DEFAULT_DATA_FILES,
    THINKING_MODES,
    EvalResult,
    EvalSample,
    append_sample_outputs,
    count_thinking_tokens,
    load_eval_samples,
    remove_think_block,
    summarize_results,
    write_outputs,
)
from eval.domains.scoring import SCORER_NAME, score_completion

LOGGER = logging.getLogger(__name__)


def _chat_template_kwargs(enable_thinking: bool) -> dict[str, Any]:
    return {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "enable_thinking": enable_thinking,
    }


def encode_messages(messages: list[dict[str, str]], tokenizer: Any, enable_thinking: bool) -> Any:
    try:
        return tokenizer.apply_chat_template(messages, **_chat_template_kwargs(enable_thinking))
    except TypeError:
        LOGGER.warning("Tokenizer chat template does not accept enable_thinking; falling back to /think control text.")
        fallback_messages = [dict(message) for message in messages]
        control = "/think" if enable_thinking else "/no_think"
        fallback_messages[-1]["content"] = f"{fallback_messages[-1]['content'].rstrip()}\n\n{control}"
        return tokenizer.apply_chat_template(
            fallback_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )


def encode_prompt_text(messages: list[dict[str, str]], tokenizer: Any, enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        LOGGER.warning("Tokenizer chat template does not accept enable_thinking; falling back to /think control text.")
        fallback_messages = [dict(message) for message in messages]
        control = "/think" if enable_thinking else "/no_think"
        fallback_messages[-1]["content"] = f"{fallback_messages[-1]['content'].rstrip()}\n\n{control}"
        return tokenizer.apply_chat_template(
            fallback_messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def load_model_and_tokenizer(model_path: str, torch_dtype: str, device_map: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_one(
    model: Any,
    tokenizer: Any,
    sample: EvalSample,
    *,
    mode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    score_code: bool,
    save_completion: bool,
) -> EvalResult:
    import torch

    enable_thinking = mode == "thinking"
    prompt_text = encode_prompt_text(sample.messages, tokenizer, enable_thinking)
    input_ids = encode_messages(sample.messages, tokenizer, enable_thinking).to(model.device)
    prompt_tokens = int(input_ids.shape[-1])
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        generation_kwargs.update({"do_sample": False})

    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(input_ids, **generation_kwargs)
    latency_seconds = time.perf_counter() - start_time

    generated_ids = outputs[0, prompt_tokens:]
    generated_tokens = int(generated_ids.shape[-1])
    raw_completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    score, prediction, reward_metadata = score_completion(sample, completion, score_code=score_code)
    thinking_tokens = count_thinking_tokens(raw_completion, tokenizer)
    answer_tokens = max(generated_tokens - thinking_tokens, 0)
    tokens_per_second = generated_tokens / latency_seconds if latency_seconds > 0 else 0.0
    cleaned_preview = remove_think_block(completion)[:600]
    return EvalResult(
        mode=mode,
        enable_thinking=enable_thinking,
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        ability=sample.ability,
        ground_truth=sample.ground_truth,
        prediction=prediction,
        score=score,
        correct=None if score is None else score > 0,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        thinking_tokens=thinking_tokens,
        answer_tokens=answer_tokens,
        total_tokens=prompt_tokens + generated_tokens,
        latency_seconds=latency_seconds,
        generated_tokens_per_second=tokens_per_second,
        completion_preview=cleaned_preview,
        max_new_tokens=max_new_tokens,
        messages=sample.messages if save_completion else None,
        prompt=prompt_text if save_completion else None,
        completion=completion if save_completion else None,
        reward_metadata=reward_metadata,
    )


def load_vllm_model(model_path: str, torch_dtype: str, tensor_parallel_size: int, gpu_memory_utilization: float) -> Any:
    from vllm import LLM

    return LLM(
        model=model_path,
        dtype=torch_dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )


def generate_vllm_batch(
    llm: Any,
    tokenizer: Any,
    samples: Sequence[EvalSample],
    *,
    mode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    score_code: bool,
    save_completion: bool,
) -> list[EvalResult]:
    from vllm import SamplingParams

    enable_thinking = mode == "thinking"
    prompts = [encode_prompt_text(sample.messages, tokenizer, enable_thinking) for sample in samples]
    prompt_tokens = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    batch_latency = time.perf_counter() - start_time
    results: list[EvalResult] = []
    for index, (sample, request_output, prompt_token_count) in enumerate(zip(samples, outputs, prompt_tokens)):
        output = request_output.outputs[0]
        completion = output.text
        generated_tokens = len(output.token_ids)
        score, prediction, reward_metadata = score_completion(sample, completion, score_code=score_code)
        thinking_tokens = count_thinking_tokens(completion, tokenizer)
        answer_tokens = max(generated_tokens - thinking_tokens, 0)
        tokens_per_second = generated_tokens / batch_latency if batch_latency > 0 else 0.0
        results.append(
            EvalResult(
                mode=mode,
                enable_thinking=enable_thinking,
                sample_id=sample.sample_id,
                dataset=sample.dataset,
                ability=sample.ability,
                ground_truth=sample.ground_truth,
                prediction=prediction,
                score=score,
                correct=None if score is None else score > 0,
                prompt_tokens=prompt_token_count,
                generated_tokens=generated_tokens,
                thinking_tokens=thinking_tokens,
                answer_tokens=answer_tokens,
                total_tokens=prompt_token_count + generated_tokens,
                latency_seconds=batch_latency,
                generated_tokens_per_second=tokens_per_second,
                completion_preview=remove_think_block(completion)[:600],
                max_new_tokens=max_new_tokens,
                messages=sample.messages if save_completion else None,
                prompt=prompts[index] if save_completion else None,
                completion=completion if save_completion else None,
                reward_metadata=reward_metadata,
            )
        )
    return results


def resolve_max_new_tokens(
    sample: EvalSample,
    mode: str,
    mode_token_limits: dict[str, int],
    ability_token_limits: dict[tuple[str, str], int | None],
) -> int:
    ability_limit = ability_token_limits.get((mode, sample.ability))
    if ability_limit is not None:
        return ability_limit
    return mode_token_limits[mode]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="../models/Qwen3-4B", help="HF model id or local model directory.")
    parser.add_argument("--data-files", nargs="+", default=list(DEFAULT_DATA_FILES), help="Validation parquet files.")
    parser.add_argument("--output-dir", default="eval/results/qwen3_4b_thinking", help="Output directory.")
    parser.add_argument("--modes", nargs="+", default=list(THINKING_MODES), choices=THINKING_MODES)
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    parser.add_argument("--max-new-tokens-thinking", type=int, default=32768)
    parser.add_argument("--max-new-tokens-non-thinking", type=int, default=8192)
    parser.add_argument("--max-new-tokens-thinking-math", type=int, default=None)
    parser.add_argument("--max-new-tokens-thinking-code", type=int, default=None)
    parser.add_argument("--max-new-tokens-non-thinking-math", type=int, default=None)
    parser.add_argument("--max-new-tokens-non-thinking-code", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="transformers")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--score-code", action="store_true", help="Execute code validation rewards when code data is included.")
    parser.add_argument("--save-completions", action="store_true", help="Store full completions in samples JSONL.")
    parser.add_argument("--skip-missing-data-files", action="store_true", help="Skip missing validation parquet files.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    LOGGER.info("Using scoring backend: %s", SCORER_NAME)
    samples = load_eval_samples(
        args.data_files,
        max_samples_per_dataset=args.max_samples_per_dataset,
        skip_missing=args.skip_missing_data_files,
    )
    if not samples:
        raise ValueError("No validation samples loaded.")

    results: list[EvalResult] = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    incremental_samples_path = output_dir / "thinking_eval_samples.jsonl"
    if incremental_samples_path.exists():
        incremental_samples_path.unlink()
    mode_token_limits = {
        "thinking": args.max_new_tokens_thinking,
        "non_thinking": args.max_new_tokens_non_thinking,
    }
    ability_token_limits = {
        ("thinking", "math"): args.max_new_tokens_thinking_math,
        ("thinking", "code"): args.max_new_tokens_thinking_code,
        ("non_thinking", "math"): args.max_new_tokens_non_thinking_math,
        ("non_thinking", "code"): args.max_new_tokens_non_thinking_code,
    }
    if args.backend == "vllm":
        llm = load_vllm_model(args.model_path, args.torch_dtype, args.tensor_parallel_size, args.gpu_memory_utilization)
        tokenizer = llm.get_tokenizer()
        for mode in args.modes:
            for start in range(0, len(samples), args.batch_size):
                batch = samples[start : start + args.batch_size]
                batch_max_new_tokens = max(
                    resolve_max_new_tokens(sample, mode, mode_token_limits, ability_token_limits) for sample in batch
                )
                LOGGER.info(
                    "Evaluating backend=vllm mode=%s samples=%d-%d/%d max_new_tokens=%d",
                    mode,
                    start + 1,
                    start + len(batch),
                    len(samples),
                    batch_max_new_tokens,
                )
                batch_results = generate_vllm_batch(
                    llm,
                    tokenizer,
                    batch,
                    mode=mode,
                    max_new_tokens=batch_max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    score_code=args.score_code,
                    save_completion=args.save_completions,
                )
                results.extend(batch_results)
                append_sample_outputs(batch_results, output_dir)
                for result in batch_results:
                    LOGGER.info(
                        "mode=%s dataset=%s score=%s generated_tokens=%d thinking_tokens=%d batch_latency=%.2fs",
                        mode,
                        result.dataset,
                        result.score,
                        result.generated_tokens,
                        result.thinking_tokens,
                        result.latency_seconds,
                    )
    else:
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.torch_dtype, args.device_map)
        for mode in args.modes:
            for index, sample in enumerate(samples, start=1):
                max_new_tokens = resolve_max_new_tokens(sample, mode, mode_token_limits, ability_token_limits)
                LOGGER.info(
                    "Evaluating mode=%s sample=%d/%d dataset=%s id=%s max_new_tokens=%d",
                    mode,
                    index,
                    len(samples),
                    sample.dataset,
                    sample.sample_id,
                    max_new_tokens,
                )
                result = generate_one(
                    model,
                    tokenizer,
                    sample,
                    mode=mode,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    score_code=args.score_code,
                    save_completion=args.save_completions,
                )
                results.append(result)
                append_sample_outputs([result], output_dir)
                LOGGER.info(
                    "mode=%s dataset=%s score=%s generated_tokens=%d thinking_tokens=%d latency=%.2fs",
                    mode,
                    sample.dataset,
                    result.score,
                    result.generated_tokens,
                    result.thinking_tokens,
                    result.latency_seconds,
                )

    write_outputs(results, output_dir)
    print(json.dumps(summarize_results(results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
