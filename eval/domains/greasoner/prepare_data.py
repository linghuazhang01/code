"""Prepare General-Reasoner eval parquet files under code/eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mopd_verl.general_reasoner_data import (
    DEFAULT_DATASET_NAME,
    general_reasoner_to_verl_parquet,
    prepare_general_reasoner_hf_dataset,
)

DEFAULT_OUTPUT_DIR = Path("data/eval_data/greasoner/WebInstructVerified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local WebInstruct JSON/JSONL/parquet test split.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "test.parquet")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--from-hf", action="store_true")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.from_hf:
        counts = prepare_general_reasoner_hf_dataset(
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            test_max_samples=args.max_samples,
        )
        print(json.dumps({"counts": counts, "output_dir": str(args.output_dir)}, sort_keys=True))
        return 0

    if args.input is None:
        raise SystemExit("--input is required unless --from-hf is set")
    count = general_reasoner_to_verl_parquet(
        args.input,
        args.output,
        split=args.split,
        max_samples=args.max_samples,
    )
    print(json.dumps({"count": count, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
