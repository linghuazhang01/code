#!/usr/bin/env python
"""Split deterministic evaluation subsets from the four OPD training domains."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from eval.data_prep.training_holdout import (  # noqa: E402
    DEFAULT_DOMAIN_SPECS,
    DomainSpec,
    HoldoutConfig,
    create_all_holdouts,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=CODE_ROOT / "data/G-OPD-Training-Data",
        help="Root containing DeepMath-103K, Eurus, IF, and Science.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CODE_ROOT / "data/eval_training_data",
        help="Destination for <domain>/test.parquet and manifest.json.",
    )
    parser.add_argument(
        "--eval-size", type=int, default=1_000, help="Target eval rows per domain."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Stable hash selection seed."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1_024, help="Streaming parquet batch size."
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=[spec.name for spec in DEFAULT_DOMAIN_SPECS],
        default=[spec.name for spec in DEFAULT_DOMAIN_SPECS],
    )
    parser.add_argument(
        "--write-remainders",
        action="store_true",
        help="Also write <remainder-root>/<domain>/train.parquet for leakage-free future training.",
    )
    parser.add_argument(
        "--remainder-root",
        type=Path,
        default=CODE_ROOT / "data/training_data_split",
        help="Destination root for train remainders.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing generated outputs."
    )
    return parser.parse_args(argv)


def _select_specs(domain_names: Sequence[str]) -> tuple[DomainSpec, ...]:
    requested = set(domain_names)
    return tuple(spec for spec in DEFAULT_DOMAIN_SPECS if spec.name in requested)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = HoldoutConfig(
        eval_size=args.eval_size,
        seed=args.seed,
        batch_size=args.batch_size,
        write_remainder=args.write_remainders,
        overwrite=args.overwrite,
    )
    create_all_holdouts(
        data_root=args.data_root,
        output_root=args.output_root,
        specs=_select_specs(args.domains),
        config=config,
        remainder_root=args.remainder_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
