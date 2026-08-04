"""Evaluate Qwen thinking-mode validation accuracy and generation cost."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

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
    rollout_index: int = 0,
    generation_seed: int | None = None,
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
        if generation_seed is not None:
            torch.manual_seed(generation_seed)
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
        correct=None if score is None else score == 1.0,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        thinking_tokens=thinking_tokens,
        answer_tokens=answer_tokens,
        total_tokens=prompt_tokens + generated_tokens,
        latency_seconds=latency_seconds,
        generated_tokens_per_second=tokens_per_second,
        completion_preview=cleaned_preview,
        rollout_index=rollout_index,
        generation_seed=generation_seed,
        max_new_tokens=max_new_tokens,
        messages=sample.messages if save_completion else None,
        prompt=prompt_text if save_completion else None,
        completion=completion if save_completion else None,
        reward_metadata=reward_metadata,
        sample_metadata=sample.sample_metadata if save_completion else None,
    )


def load_vllm_model(
    model_path: str,
    torch_dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    *,
    max_model_len: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    enforce_eager: bool = False,
    enable_chunked_prefill: bool | None = None,
) -> Any:
    from vllm import LLM

    optional_kwargs: dict[str, Any] = {}
    if max_model_len is not None:
        optional_kwargs["max_model_len"] = max_model_len
    if max_num_batched_tokens is not None:
        optional_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if max_num_seqs is not None:
        optional_kwargs["max_num_seqs"] = max_num_seqs
    if enable_chunked_prefill is not None:
        optional_kwargs["enable_chunked_prefill"] = enable_chunked_prefill

    with _temporary_vllm_v1_chunked_prefill_setting(
        enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    ):
        return LLM(
            model=model_path,
            dtype=torch_dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
            **optional_kwargs,
        )


@contextmanager
def _temporary_vllm_v1_chunked_prefill_setting(
    enable_chunked_prefill: bool | None,
    *,
    max_num_batched_tokens: int | None,
    max_num_seqs: int | None,
    engine_args_cls: Any | None = None,
) -> Iterator[None]:
    """Make vLLM V1 honor an explicit disabled chunked-prefill setting."""
    if enable_chunked_prefill is not False:
        yield
        return
    if max_num_batched_tokens is None or max_num_seqs is None:
        raise ValueError(
            "Disabling chunked prefill on vLLM V1 requires explicit "
            "max_num_batched_tokens and max_num_seqs."
        )
    if engine_args_cls is None:
        from vllm.engine.arg_utils import EngineArgs

        engine_args_cls = EngineArgs

    original = engine_args_cls._set_default_args

    def _set_default_args_and_disable(
        self: Any,
        usage_context: Any,
        model_config: Any,
    ) -> None:
        original(self, usage_context, model_config)
        self.enable_chunked_prefill = False

    engine_args_cls._set_default_args = _set_default_args_and_disable
    try:
        yield
    finally:
        engine_args_cls._set_default_args = original


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
    rollout_index: int = 0,
    generation_seed: int | None = None,
) -> list[EvalResult]:
    from vllm import SamplingParams

    enable_thinking = mode == "thinking"
    prompts = [encode_prompt_text(sample.messages, tokenizer, enable_thinking) for sample in samples]
    prompt_tokens = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=generation_seed,
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
                correct=None if score is None else score == 1.0,
                prompt_tokens=prompt_token_count,
                generated_tokens=generated_tokens,
                thinking_tokens=thinking_tokens,
                answer_tokens=answer_tokens,
                total_tokens=prompt_token_count + generated_tokens,
                latency_seconds=batch_latency,
                generated_tokens_per_second=tokens_per_second,
                completion_preview=remove_think_block(completion)[:600],
                rollout_index=rollout_index,
                generation_seed=generation_seed,
                max_new_tokens=max_new_tokens,
                messages=sample.messages if save_completion else None,
                prompt=prompts[index] if save_completion else None,
                completion=completion if save_completion else None,
                reward_metadata=reward_metadata,
                sample_metadata=sample.sample_metadata if save_completion else None,
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
    parser.add_argument("--output-dir", default="data/eval_data/results/qwen3_4b_thinking", help="Output directory.")
    parser.add_argument("--modes", nargs="+", default=list(THINKING_MODES), choices=THINKING_MODES)
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    parser.add_argument(
        "--sample-offset-per-dataset",
        type=int,
        default=0,
        help="Skip this many rows in every parquet before applying the sample limit.",
    )
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
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--score-code", action="store_true", help="Execute code validation rewards when code data is included.")
    parser.add_argument("--save-completions", action="store_true", help="Store full completions in samples JSONL.")
    parser.add_argument("--skip-missing-data-files", action="store_true", help="Skip missing validation parquet files.")
    parser.add_argument("--num-samples", type=int, default=1, help="Rollouts per prompt; use 32 for GRPO AIME Avg@32.")
    parser.add_argument("--seed", type=int, default=42, help="Base generation seed for reproducible sampling.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a validated thinking_eval_samples.jsonl prefix instead of overwriting it.",
    )
    return parser.parse_args()


def load_incremental_results(path: Path) -> list[EvalResult]:
    results: list[EvalResult] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                results.append(EvalResult(**payload))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Invalid resume record at {path}:{line_number}") from exc
    return results


def validate_resume_config(
    existing_config: dict[str, Any],
    requested_config: dict[str, Any],
) -> None:
    """Reject resume attempts that would mix incompatible run settings."""

    existing = dict(existing_config)
    requested = dict(requested_config)
    existing.pop("resume", None)
    requested.pop("resume", None)
    existing.setdefault("sample_offset_per_dataset", 0)
    requested.setdefault("sample_offset_per_dataset", 0)
    if existing == requested:
        return

    differing_keys = sorted(
        key
        for key in set(existing) | set(requested)
        if existing.get(key) != requested.get(key)
    )
    differences = ", ".join(
        f"{key}={existing.get(key)!r}->{requested.get(key)!r}"
        for key in differing_keys[:8]
    )
    raise ValueError(f"Resume configuration differs from the original run: {differences}")


def validate_output_directory(output_dir: Path, *, resume: bool) -> None:
    """Prevent accidental replacement or mixing of an existing evaluation."""

    if resume or not output_dir.exists():
        return
    existing_entries = sorted(
        path.name for path in output_dir.iterdir() if path.name != "run.log"
    )
    if existing_entries:
        preview = ", ".join(existing_entries[:8])
        raise FileExistsError(
            f"Refusing to overwrite non-empty evaluation directory {output_dir}: {preview}. "
            "Use a new output directory, or pass --resume with identical settings."
        )


def validate_resume_prefix(
    results: Sequence[EvalResult],
    samples: Sequence[EvalSample],
    modes: Sequence[str],
    num_samples: int,
) -> None:
    sample_count = len(samples)
    expected_count = sample_count * len(modes) * num_samples
    if len(results) > expected_count:
        raise ValueError(
            f"Resume file contains {len(results)} records, but this run expects only {expected_count}."
        )

    for result_index, result in enumerate(results):
        mode_index, mode_remainder = divmod(result_index, num_samples * sample_count)
        rollout_index, sample_index = divmod(mode_remainder, sample_count)
        expected_sample = samples[sample_index]
        expected_mode = modes[mode_index]
        if (
            result.mode != expected_mode
            or result.rollout_index != rollout_index
            or result.sample_id != expected_sample.sample_id
            or result.dataset != expected_sample.dataset
        ):
            raise ValueError(
                "Resume file is not a strict prefix of the requested run at "
                f"record {result_index + 1}: expected "
                f"mode={expected_mode!r}, rollout={rollout_index}, "
                f"sample_id={expected_sample.sample_id!r}, dataset={expected_sample.dataset!r}; "
                f"found mode={result.mode!r}, rollout={result.rollout_index}, "
                f"sample_id={result.sample_id!r}, dataset={result.dataset!r}."
            )


def completed_samples_for_rollout(
    completed_results: int,
    *,
    mode_index: int,
    rollout_index: int,
    sample_count: int,
    num_samples: int,
) -> int:
    rollout_offset = (mode_index * num_samples + rollout_index) * sample_count
    return min(max(completed_results - rollout_offset, 0), sample_count)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    LOGGER.info("Using scoring backend: %s", SCORER_NAME)
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("--temperature must be a finite non-negative number.")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1.")
    if args.num_samples > 1 and args.temperature <= 0:
        raise ValueError("--num-samples > 1 requires --temperature > 0.")
    if args.sample_offset_per_dataset < 0:
        raise ValueError("--sample-offset-per-dataset must be non-negative.")

    output_dir = Path(args.output_dir)
    validate_output_directory(output_dir, resume=args.resume)
    samples = load_eval_samples(
        args.data_files,
        max_samples_per_dataset=args.max_samples_per_dataset,
        sample_offset_per_dataset=args.sample_offset_per_dataset,
        skip_missing=args.skip_missing_data_files,
    )
    if not samples:
        raise ValueError("No validation samples loaded.")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "eval_run_config.json"
    if args.resume:
        if not run_config_path.is_file():
            raise FileNotFoundError(
                f"--resume requires the original run configuration: {run_config_path}"
            )
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        validate_resume_config(existing_config, vars(args))
        (output_dir / "eval_resume_config.json").write_text(
            json.dumps(vars(args), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        run_config_path.write_text(
            json.dumps(vars(args), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    incremental_samples_path = output_dir / "thinking_eval_samples.jsonl"
    if args.resume:
        if not incremental_samples_path.exists():
            raise FileNotFoundError(
                f"--resume requires an existing incremental result file: {incremental_samples_path}"
            )
        results = load_incremental_results(incremental_samples_path)
        validate_resume_prefix(results, samples, args.modes, args.num_samples)
        LOGGER.info(
            "Resuming from %d/%d validated result records.",
            len(results),
            len(samples) * len(args.modes) * args.num_samples,
        )
    else:
        results = []
    if incremental_samples_path.exists() and not args.resume:
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
        llm = load_vllm_model(
            args.model_path,
            args.torch_dtype,
            args.tensor_parallel_size,
            args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            enable_chunked_prefill=args.enable_chunked_prefill,
        )
        tokenizer = llm.get_tokenizer()
        for mode_index, mode in enumerate(args.modes):
            for rollout_index in range(args.num_samples):
                resume_start = completed_samples_for_rollout(
                    len(results),
                    mode_index=mode_index,
                    rollout_index=rollout_index,
                    sample_count=len(samples),
                    num_samples=args.num_samples,
                )
                if resume_start == len(samples):
                    continue
                if resume_start % args.batch_size != 0:
                    raise ValueError(
                        "vLLM resume point must align with --batch-size because results are "
                        f"committed one batch at a time; got {resume_start} completed samples "
                        f"with batch_size={args.batch_size}."
                    )
                for start in range(resume_start, len(samples), args.batch_size):
                    generation_seed = args.seed + rollout_index * len(samples) + start
                    batch = samples[start : start + args.batch_size]
                    batch_max_new_tokens = max(
                        resolve_max_new_tokens(sample, mode, mode_token_limits, ability_token_limits)
                        for sample in batch
                    )
                    LOGGER.info(
                        "Evaluating backend=vllm mode=%s rollout=%d samples=%d-%d/%d max_new_tokens=%d",
                        mode,
                        rollout_index,
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
                        rollout_index=rollout_index,
                        generation_seed=generation_seed,
                    )
                    results.extend(batch_results)
                    append_sample_outputs(batch_results, output_dir)
                    for result in batch_results:
                        LOGGER.info(
                            "mode=%s dataset=%s score=%s generated_tokens=%d "
                            "thinking_tokens=%d batch_latency=%.2fs",
                            mode,
                            result.dataset,
                            result.score,
                            result.generated_tokens,
                            result.thinking_tokens,
                            result.latency_seconds,
                        )
    else:
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.torch_dtype, args.device_map)
        for mode_index, mode in enumerate(args.modes):
            for rollout_index in range(args.num_samples):
                resume_start = completed_samples_for_rollout(
                    len(results),
                    mode_index=mode_index,
                    rollout_index=rollout_index,
                    sample_count=len(samples),
                    num_samples=args.num_samples,
                )
                for index, sample in enumerate(samples[resume_start:], start=resume_start + 1):
                    generation_seed = args.seed + rollout_index * len(samples) + index - 1
                    max_new_tokens = resolve_max_new_tokens(sample, mode, mode_token_limits, ability_token_limits)
                    LOGGER.info(
                        "Evaluating mode=%s rollout=%d sample=%d/%d dataset=%s id=%s max_new_tokens=%d",
                        mode,
                        rollout_index,
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
                        rollout_index=rollout_index,
                        generation_seed=generation_seed,
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
