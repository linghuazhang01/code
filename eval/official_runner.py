"""Run standalone official eval suites by domain and dataset."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.domains.greasoner.official_eval import DATASET_CHOICES as GREASONER_DATASETS
from eval.domains.greasoner.official_eval import run_dataset as run_greasoner_dataset
from eval.domains.toolrl.api_bank import LEVELS as API_BANK_LEVELS
from eval.domains.toolrl.api_bank import run_api_bank
from eval.domains.toolrl.bamboogle import run_bamboogle
from eval.domains.toolrl.bfcl import run_bfcl
from eval.official_utils import OfficialEvalResult, ensure_output_dir, write_json

TOOLRL_DATASETS = ("api_bank", "bfcl", "bamboogle")
DOMAIN_TO_DATASETS = {
    "greasoner": GREASONER_DATASETS,
    "toolrl": TOOLRL_DATASETS,
}


def _default_output_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval/results/official_{timestamp}"


def _env_or_arg(value: str | None, env_name: str) -> str | None:
    return value or os.environ.get(env_name)


def _resolve_datasets(domains: list[str], datasets: list[str]) -> list[str]:
    if "all" in datasets:
        selected: list[str] = []
        for domain in domains:
            selected.extend(DOMAIN_TO_DATASETS[domain])
        return selected
    allowed = {dataset for domain in domains for dataset in DOMAIN_TO_DATASETS[domain]}
    unknown = sorted(set(datasets) - allowed)
    if unknown:
        raise ValueError(f"Datasets {unknown} are not valid for domains {domains}. Allowed: {sorted(allowed)}")
    return datasets


def _write_index(output_dir: Path, results: list[OfficialEvalResult]) -> None:
    payload = {
        "output_dir": str(output_dir),
        "results": [
            {
                "dataset": result.dataset,
                "output_dir": str(result.output_dir),
                "summary": result.summary,
            }
            for result in results
        ],
    }
    write_json(output_dir / "official_eval_summary.json", payload)


def run_selected(args: argparse.Namespace) -> list[OfficialEvalResult]:
    output_root = ensure_output_dir(args.output_dir)
    datasets = _resolve_datasets(args.domains, args.datasets)
    results: list[OfficialEvalResult] = []
    enable_thinking = None if args.enable_thinking == "auto" else args.enable_thinking == "true"

    for dataset in datasets:
        dataset_output = output_root / dataset
        if dataset in GREASONER_DATASETS:
            results.append(
                run_greasoner_dataset(
                    dataset_key=dataset,
                    model_path=args.model_path,
                    output_dir=output_root,
                    max_samples=args.max_samples,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature if args.temperature is not None else 0.0,
                    top_p=args.top_p,
                    enable_thinking=enable_thinking,
                    judge_api_key=_env_or_arg(args.judge_api_key, "OPENAI_API_KEY"),
                    judge_base_url=_env_or_arg(args.judge_base_url, "OPENAI_BASE_URL"),
                    judge_model=args.judge_model,
                )
            )
        elif dataset == "api_bank":
            results.append(
                run_api_bank(
                    model_path=args.model_path,
                    source_dir=args.api_bank_dir,
                    output_dir=dataset_output,
                    levels=args.api_bank_levels,
                    max_samples=args.max_samples,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len or 4096,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature if args.temperature is not None else 0.0001,
                    top_p=args.top_p,
                    enable_thinking=enable_thinking,
                )
            )
        elif dataset == "bfcl":
            results.append(
                run_bfcl(
                    model_path=args.model_path,
                    output_dir=dataset_output,
                    bfcl_command=args.bfcl_command,
                    api_base_url=_env_or_arg(args.api_base_url, "BFCL_API_BASE_URL"),
                    api_key=_env_or_arg(args.api_key, "BFCL_API_KEY"),
                )
            )
        elif dataset == "bamboogle":
            serper_api_key = _env_or_arg(args.serper_api_key, "SERPER_API_KEY")
            judge_api_key = _env_or_arg(args.judge_api_key, "OPENAI_API_KEY")
            if not serper_api_key:
                raise ValueError("Bamboogle requires --serper-api-key or SERPER_API_KEY.")
            if not judge_api_key:
                raise ValueError("Bamboogle requires --judge-api-key or OPENAI_API_KEY.")
            results.append(
                run_bamboogle(
                    model_path=args.model_path,
                    source_file=args.bamboogle_file,
                    output_dir=dataset_output,
                    max_samples=args.max_samples,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len or 4096,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature if args.temperature is not None else 0.001,
                    top_p=args.top_p,
                    enable_thinking=enable_thinking,
                    serper_api_key=serper_api_key,
                    serper_base_url=args.serper_base_url,
                    judge_api_key=judge_api_key,
                    judge_base_url=_env_or_arg(args.judge_base_url, "OPENAI_BASE_URL"),
                    judge_model=args.judge_model,
                    round_threshold=args.bamboogle_rounds,
                )
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    _write_index(output_root, results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", default=["greasoner", "toolrl"], choices=tuple(DOMAIN_TO_DATASETS))
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default=_default_output_dir())
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature. If omitted, each official wrapper uses its own default.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", choices=("true", "false", "auto"), default="auto")
    parser.add_argument("--api-base-url", default=None, help="Generic external API base URL, currently passed to BFCL.")
    parser.add_argument("--api-key", default=None, help="Generic external API key, currently passed to BFCL.")
    parser.add_argument("--api-bank-dir", default="../temp/grpo_sources/ToolRL/benchmarks/API-Bank")
    parser.add_argument("--api-bank-levels", nargs="+", default=list(API_BANK_LEVELS), choices=API_BANK_LEVELS)
    parser.add_argument("--bfcl-command", default=None, help="External BFCL harness command. Receives BFCL_* env vars.")
    parser.add_argument("--bamboogle-file", default="../temp/grpo_sources/ToolRL/benchmarks/Bamboogle/data.json")
    parser.add_argument("--bamboogle-rounds", type=int, default=4)
    parser.add_argument("--serper-api-key", default=None)
    parser.add_argument("--serper-base-url", default="https://google.serper.dev/search")
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-model", default="gpt-4o")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_selected(args)
    print(
        json.dumps(
            [{"dataset": result.dataset, "summary": result.summary} for result in results],
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
