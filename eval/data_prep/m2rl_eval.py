"""Prepare the instruction-following and science benchmarks used by M2RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Image, load_dataset


DEFAULT_OUTPUT_ROOT = Path("data/eval_data")
DEFAULT_GPQA_CSV = Path("data/eval_data/science/GPQA/_sources/gpqa_diamond.csv")
GPQA_PROMPT = (
    "Answer the following multiple choice question. The last line of your response should be "
    "of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
    "Think step by step before answering.\n\n{question}\n\n"
    "A) {a}\nB) {b}\nC) {c}\nD) {d}"
)
HF_REVISIONS = {
    "allenai/IFBench_test": "2e8a48de45ff3bf41242f927254ca81b59ca3ae2",
    "google/IFEval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    "cais/hle": "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29",
}


def _verl_row(
    *,
    sample_id: str,
    prompt: str,
    domain: str,
    data_source: str,
    ground_truth: Any,
    metadata: Mapping[str, Any],
    reward_style: str = "rule",
) -> dict[str, Any]:
    extra_info = {
        **metadata,
        "domain": domain,
        "source_domain": domain,
        "opd_teacher": domain,
        "sample_id": sample_id,
        "split": "test",
    }
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": domain,
        "reward_model": {"style": reward_style, "ground_truth": ground_truth},
        "extra_info": extra_info,
    }


def _if_rows(dataset_name: str, benchmark: str) -> list[dict[str, Any]]:
    dataset = load_dataset(dataset_name, split="train", revision=HF_REVISIONS[dataset_name])
    rows: list[dict[str, Any]] = []
    for position, example in enumerate(dataset):
        key = str(example.get("key", position))
        prompt = str(example["prompt"])
        rows.append(
            _verl_row(
                sample_id=f"if:{benchmark}:{key}",
                prompt=prompt,
                domain="if",
                data_source=f"m2rl_{benchmark}",
                ground_truth="",
                metadata={
                    "benchmark": benchmark,
                    "rm_type": "ifbench",
                    "prompt_text": prompt,
                    "record_id": key,
                    "instruction_id_list": example["instruction_id_list"],
                    "kwargs": example["kwargs"],
                },
            )
        )
    return rows


def _gpqa_rows(csv_path: Path, seed: int) -> list[dict[str, Any]]:
    frame = pd.read_csv(csv_path)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for position, example in frame.iterrows():
        choices = [
            str(example["Correct Answer"]).strip(),
            str(example["Incorrect Answer 1"]).strip(),
            str(example["Incorrect Answer 2"]).strip(),
            str(example["Incorrect Answer 3"]).strip(),
        ]
        rng.shuffle(choices)
        correct_letter = "ABCD"[choices.index(str(example["Correct Answer"]).strip())]
        prompt = GPQA_PROMPT.format(
            question=str(example["Question"]).strip(),
            a=choices[0],
            b=choices[1],
            c=choices[2],
            d=choices[3],
        )
        rows.append(
            _verl_row(
                sample_id=f"science:gpqa_diamond:{position}",
                prompt=prompt,
                domain="science",
                data_source="m2rl_gpqa_diamond",
                ground_truth=correct_letter,
                metadata={
                    "benchmark": "gpqa_diamond",
                    "rm_type": "gpqa",
                    "choices": choices,
                    "correct_letter": correct_letter,
                    "valid_letters": list("ABCD"),
                    "shuffle_seed": seed,
                },
            )
        )
    return rows


def _hle_examples() -> Iterable[dict[str, Any]]:
    dataset = load_dataset(
        "cais/hle",
        split="test",
        streaming=True,
        revision=HF_REVISIONS["cais/hle"],
    )
    dataset = dataset.cast_column("image_preview", Image(decode=False))
    dataset = dataset.cast_column("rationale_image", Image(decode=False))
    yield from dataset


def _hle_rows() -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    total = 0
    for example in _hle_examples():
        total += 1
        if example.get("image"):
            continue
        sample_id = str(example["id"])
        rows.append(
            _verl_row(
                sample_id=f"science:hle:{sample_id}",
                prompt=str(example["question"]),
                domain="science",
                data_source="m2rl_hle",
                ground_truth=str(example["answer"]),
                metadata={
                    "benchmark": "hle",
                    "rm_type": "hle",
                    "answer_type": example.get("answer_type"),
                    "category": example.get("category"),
                    "raw_subject": example.get("raw_subject"),
                    "has_image": False,
                    "text_only_subset": True,
                },
                reward_style="judge_required",
            )
        )
    return rows, total


def _write_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(output_root: Path, gpqa_csv: Path, seed: int) -> dict[str, Any]:
    ifbench_rows = _if_rows("allenai/IFBench_test", "ifbench")
    ifeval_rows = _if_rows("google/IFEval", "ifeval")
    gpqa_rows = _gpqa_rows(gpqa_csv, seed)
    hle_rows, hle_total = _hle_rows()

    outputs = {
        "ifbench": output_root / "if" / "IFBench" / "test.parquet",
        "ifeval": output_root / "if" / "IFEval" / "test.parquet",
        "gpqa_diamond": output_root / "science" / "GPQA" / "test.parquet",
        "hle_text_only": output_root / "science" / "HLE" / "test.parquet",
    }
    row_groups = {
        "ifbench": ifbench_rows,
        "ifeval": ifeval_rows,
        "gpqa_diamond": gpqa_rows,
        "hle_text_only": hle_rows,
    }
    for name, path in outputs.items():
        _write_parquet(row_groups[name], path)

    manifest = {
        "protocol": "M2RL paper benchmark families: IF={IFEval, IFBench}; Science={HLE, GPQA-Diamond}",
        "comparability": {
            "ifeval": "paper-aligned benchmark data; strict prompt scoring",
            "ifbench": "paper-aligned benchmark data; prompt-level scoring must match the selected strict/loose protocol",
            "gpqa_diamond": "paper-aligned 198-question Diamond split",
            "hle_text_only": "inference input only; not directly comparable with M2RL full-HLE results",
        },
        "sources": {
            "ifbench": "allenai/IFBench_test:train",
            "ifeval": "google/IFEval:train",
            "gpqa_diamond": str(gpqa_csv),
            "hle": "cais/hle:test",
        },
        "counts": {name: len(rows) for name, rows in row_groups.items()},
        "hle_official_total": hle_total,
        "hle_image_rows_excluded": hle_total - len(hle_rows),
        "gpqa_choice_shuffle_seed": seed,
        "gpqa_source_sha256": _sha256(gpqa_csv),
        "hf_revisions": HF_REVISIONS,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "limitations": [
            "HLE is restricted to text-only rows because the OPD Qwen3-4B evaluation path is text-only.",
            "HLE output is inference-only: current verl rewards do not implement the official exact-match/judge protocol.",
            "Do not add science/HLE/test.parquet to verl val_files or compare it with M2RL full-HLE scores.",
        ],
    }
    manifest_path = output_root / "m2rl_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpqa-csv", type=Path, default=DEFAULT_GPQA_CSV)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.gpqa_csv.is_file():
        raise FileNotFoundError(f"GPQA-Diamond CSV not found: {args.gpqa_csv}")
    manifest = prepare(args.output_root, args.gpqa_csv, args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
