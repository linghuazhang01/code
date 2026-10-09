#!/usr/bin/env python3
"""Score a sapd_c01 Math4 seed-42 run and compare it rollout by rollout with the references.

References (bundled in ./reference):
  eager_20260902  job 196, eager, DP1, gpu_memory 0.85  -> Math4 avg@8 26.04 (the archived number)
  cg_20261008     CUDA graphs, DP4, gpu_memory 0.50     -> Math4 avg@8 24.06

Usage:
  python3 compare_with_reference.py --suite SUITE_ROOT       # a finished suite
  python3 compare_with_reference.py --records a.jsonl[.gz]   # any records file
  python3 compare_with_reference.py --check-manifest SUITE_ROOT/suite_manifest.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCES = {
    "eager_20260902 (26.04)": HERE / "reference" / "c01_s42_eager_20260902_records.jsonl.gz",
    "cg_20261008 (24.06)": HERE / "reference" / "c01_s42_cg_20261008_records.jsonl.gz",
}
REFERENCE_MANIFEST = HERE / "reference" / "reference_manifest_20260902.json"
DATASETS = ("AIME2024", "AIME2025", "HMMT25Feb", "HMMT25Nov")


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def load(path: Path) -> dict[tuple[str, int], dict]:
    out = {}
    with _open(path) as handle:
        for line in handle:
            r = json.loads(line)
            out[(r["sample_id"], int(r["rollout_index"]))] = r
    return out


def is_correct(r: dict) -> bool:
    return str(r["correct"]).lower() == "true"


def score(records: dict) -> dict[str, float]:
    by_ds: dict[str, dict[str, list[bool]]] = {}
    for (sid, _), r in records.items():
        by_ds.setdefault(r["dataset"], {}).setdefault(sid, []).append(is_correct(r))
    res = {}
    for ds in DATASETS:
        probs = by_ds.get(ds, {})
        flat = [c for v in probs.values() for c in v]
        res[ds] = 100 * sum(flat) / len(flat) if flat else float("nan")
        res[ds + "_pass"] = 100 * sum(any(v) for v in probs.values()) / len(probs) if probs else float("nan")
    res["Math4_avg@8"] = statistics.mean(res[d] for d in DATASETS)
    res["Math4_pass@8"] = statistics.mean(res[d + "_pass"] for d in DATASETS)
    return res


def common_prefix_frac(a: str, b: str) -> float:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(1, len(a), len(b))


def compare(run: dict, ref: dict) -> dict:
    keys = sorted(set(run) & set(ref))
    return {
        "n": len(keys),
        "identical_text": sum(run[k]["response"] == ref[k]["response"] for k in keys),
        "first200_same": sum(run[k]["response"][:200] == ref[k]["response"][:200] for k in keys),
        "median_prefix": statistics.median(common_prefix_frac(run[k]["response"], ref[k]["response"]) for k in keys),
        "correct_agree": sum(is_correct(run[k]) == is_correct(ref[k]) for k in keys) / len(keys),
    }


def check_manifest(path: Path) -> int:
    run = json.loads(path.read_text())
    ref = json.loads(REFERENCE_MANIFEST.read_text())
    fields = ("dataset", "source_start", "source_end_exclusive", "num_samples", "generation_seed")
    want = [tuple(t[f] for f in fields) for t in ref["tasks"]]
    got = [tuple(t[f] for f in fields) for t in run["tasks"]]
    problems = []
    if got != want:
        problems.append(f"shard/seed layout differs ({len(got)} vs {len(want)} tasks)")
    for key in ("batch_size", "max_model_len", "max_num_batched_tokens", "max_num_seqs", "enforce_eager", "enable_chunked_prefill"):
        if run["execution"].get(key) != ref["execution"].get(key):
            problems.append(f"execution.{key}: run={run['execution'].get(key)} ref={ref['execution'].get(key)}")
    for key in ("base_seed", "max_new_tokens", "temperature", "top_p", "math_samples"):
        if run["generation"].get(key) != ref["generation"].get(key):
            problems.append(f"generation.{key}: run={run['generation'].get(key)} ref={ref['generation'].get(key)}")
    if problems:
        print("[manifest] MISMATCH vs reference job 196:\n  " + "\n  ".join(problems))
        return 1
    print(f"[manifest] OK: {len(got)} shards, seeds and generation/execution settings match job 196")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path)
    ap.add_argument("--records", type=Path)
    ap.add_argument("--check-manifest", type=Path)
    args = ap.parse_args()
    if args.check_manifest:
        return check_manifest(args.check_manifest)
    if args.suite:
        found = sorted(args.suite.glob("*/math/prompt_response_records.jsonl"))
        if len(found) != 1:
            sys.exit(f"expected one merged records file under {args.suite}/*/math/, found {len(found)}")
        path = found[0]
    elif args.records:
        path = args.records
    else:
        ap.error("--suite or --records is required")

    run = load(path)
    print(f"run records: {path} ({len(run)} rollouts)")
    rows = [("THIS RUN", score(run))] + [(name, score(load(p))) for name, p in REFERENCES.items()]
    print(f"\n{'':24s}" + "".join(f"{d:>11s}" for d in DATASETS) + f"{'avg@8':>9s}{'pass@8':>9s}")
    for name, s in rows:
        print(f"{name:24s}" + "".join(f"{s[d]:11.2f}" for d in DATASETS) + f"{s['Math4_avg@8']:9.2f}{s['Math4_pass@8']:9.2f}")

    print("\nrollout-level agreement with each reference (same sample_id + rollout_index):")
    for name, p in REFERENCES.items():
        c = compare(run, load(p))
        print(f"  vs {name:24s} n={c['n']} identical_text={c['identical_text']} "
              f"first200_same={c['first200_same']} median_prefix={c['median_prefix']:.3f} "
              f"correct_agree={c['correct_agree']:.3f}")
    print("\nReading: identical_text close to 960 vs eager_20260902 = bitwise reproduction of 26.04.\n"
          "first200_same ~700+ but few identical = same weights and seeds, numerics drifted (GPU/kernel/version).\n"
          "first200_same near 0 = different seeds or different weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
