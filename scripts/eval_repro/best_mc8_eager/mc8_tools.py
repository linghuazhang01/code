#!/usr/bin/env python3
"""Score an mc8 suite and compare it rollout by rollout with the best-model references.

mc8 = AIME24, AIME25, HMMT25Feb, HMMT25Nov, HumanEval+, MBPP+, LCB v5, LCB v6 (K=8).
Scores follow plan/math-baseline-multiseed-20261008/collect_mc8.py: Math = merged avg@8;
HumanEval+/MBPP+ = official EvalPlus plus.accuracy; LCB = share of passing graded entries.
Math4 / Code4 are unweighted means of four datasets; Macro8 = (Math4 + Code4) / 2.

References are stored as fingerprints, not raw text: per rollout a SHA-1 of the full response
and of its first 200 characters, the length and the correctness flag.

Usage:
  python3 mc8_tools.py compare --suite SUITE_ROOT
  python3 mc8_tools.py check-manifest --manifest SUITE_ROOT/suite_manifest.json
  python3 mc8_tools.py fingerprint --suite SUITE_ROOT --out f.jsonl.gz [--scores-out s.json]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATH = ["AIME2024", "AIME2025", "HMMT25Feb", "HMMT25Nov"]
COLUMNS = MATH + ["HumanEval+", "MBPP+", "LCB_v5", "LCB_v6"]
REFERENCES = {
    "eager_20260926 (31.66)": "mc8_k8_s60_mopd_c01_cstruct_20260926",
    "cg_20261003 (31.45)": "mc8_k8_s60_mopd_c01_cstruct_cg_20261003",
}
REFERENCE_MANIFEST = HERE / "reference" / "reference_manifest_20260926.json"


def suite_scores(suite: str) -> dict[str, float]:
    scores = {}
    results = glob.glob(os.path.join(suite, "*", "math", "thinking_eval_results.json"))
    if len(results) == 1:
        with open(results[0]) as handle:
            for row in json.load(handle)["summary"]:
                if row["dataset"] in MATH:
                    scores[row["dataset"]] = 100 * row["avg_at_k"]
    for name, key in (("humaneval", "HumanEval+"), ("mbpp", "MBPP+")):
        path = os.path.join(suite, "official_evalplus", name, "SUMMARY.json")
        if os.path.exists(path):
            with open(path) as handle:
                scores[key] = 100 * json.load(handle)["metrics"]["plus"]["accuracy"]
    for version in ("v5", "v6"):
        files = glob.glob(os.path.join(suite, "*", "lcb", version, "*_eval_all.json"))
        if len(files) == 1:
            with open(files[0]) as handle:
                graded = [bool(v) for row in json.load(handle) for v in row["graded_list"]]
            if graded:
                scores[f"LCB_{version}"] = 100 * sum(graded) / len(graded)
    if all(c in scores for c in COLUMNS):
        scores["Math4"] = sum(scores[c] for c in COLUMNS[:4]) / 4
        scores["Code4"] = sum(scores[c] for c in COLUMNS[4:]) / 4
        scores["Macro8"] = (scores["Math4"] + scores["Code4"]) / 2
    return scores


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _row(key: str, dataset: str, text: str, correct) -> dict:
    return {"key": key, "dataset": dataset, "len": len(text), "h": _h(text), "h200": _h(text[:200]),
            "correct": None if correct is None else str(correct).lower() == "true"}


def fingerprints(suite: str) -> list[dict]:
    rows = []
    for domain in ("math", "code"):
        for path in glob.glob(os.path.join(suite, "*", domain, "prompt_response_records.jsonl")):
            with open(path) as handle:
                for line in handle:
                    r = json.loads(line)
                    # Math correctness is final; Code correctness here is the in-loop sandbox flag.
                    rows.append(_row(f"{r['dataset']}|{r['sample_id']}|{r['rollout_index']}", r["dataset"],
                                     r["response"], r["correct"]))
    for version in ("v5", "v6"):
        for path in glob.glob(os.path.join(suite, "*", "lcb", version, "*_eval_all.json")):
            with open(path) as handle:
                for q in json.load(handle):
                    for i, text in enumerate(q["output_list"]):
                        rows.append(_row(f"LCB_{version}|{q['question_id']}|{i}", f"LCB_{version}", text,
                                         q["graded_list"][i]))
    return rows


def _load_fp(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt") as handle:
        return {r["key"]: r for r in map(json.loads, handle)}


def cmd_fingerprint(args) -> int:
    rows = fingerprints(str(args.suite))
    with gzip.open(args.out, "wt") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rollouts -> {args.out}")
    if args.scores_out:
        Path(args.scores_out).write_text(json.dumps(suite_scores(str(args.suite)), indent=1) + "\n")
    return 0


def cmd_check_manifest(args) -> int:
    run = json.loads(Path(args.manifest).read_text())
    ref = json.loads(REFERENCE_MANIFEST.read_text())
    fields = ("dataset", "task_type", "source_start", "source_end_exclusive", "num_samples", "generation_seed")
    want = [tuple(t[f] for f in fields) for t in ref["tasks"]]
    got = [tuple(t[f] for f in fields) for t in run["tasks"]]
    problems = []
    if got != want:
        diff = sum(a != b for a, b in zip(got, want)) + abs(len(got) - len(want))
        problems.append(f"shard/seed layout differs ({len(got)} vs {len(want)} tasks, {diff} mismatches)")
    for key in ("batch_size", "max_model_len", "max_num_batched_tokens", "max_num_seqs", "enforce_eager",
                "enable_chunked_prefill", "shards_per_dataset", "min_rows_per_shard"):
        if run["execution"].get(key) != ref["execution"].get(key):
            problems.append(f"execution.{key}: run={run['execution'].get(key)} ref={ref['execution'].get(key)}")
    for key in ("base_seed", "max_new_tokens", "temperature", "top_p", "math_samples", "code_samples"):
        if run["generation"].get(key) != ref["generation"].get(key):
            problems.append(f"generation.{key}: run={run['generation'].get(key)} ref={ref['generation'].get(key)}")
    if problems:
        print("[manifest] MISMATCH vs reference 2026-09-26 eager suite:\n  " + "\n  ".join(problems))
        return 1
    print(f"[manifest] OK: {len(got)} shards; seeds and generation/execution settings match the 31.66 eager suite")
    return 0


def cmd_compare(args) -> int:
    suite = str(args.suite)
    ref_scores = json.loads((HERE / "reference" / "reference_scores.json").read_text())
    rows = [("THIS RUN", suite_scores(suite))] + [(name, ref_scores[name]) for name in REFERENCES]
    print(f"{'':24s}" + "".join(f"{c:>11s}" for c in COLUMNS) + "".join(f"{c:>8s}" for c in ("Math4", "Code4", "Macro8")))
    for name, s in rows:
        cells = "".join(f"{s[c]:11.2f}" if c in s else f"{'NA':>11s}" for c in COLUMNS)
        tail = "".join(f"{s[c]:8.2f}" if c in s else f"{'NA':>8s}" for c in ("Math4", "Code4", "Macro8"))
        print(f"{name:24s}{cells}{tail}")

    run = {r["key"]: r for r in fingerprints(suite)}
    print(f"\nrollout-level agreement ({len(run)} rollouts in this run):")
    for name, tag in REFERENCES.items():
        ref = _load_fp(HERE / "reference" / f"{tag}.fp.jsonl.gz")
        print(f"  vs {name}")
        groups = {}
        for key in set(run) & set(ref):
            groups.setdefault(run[key]["dataset"], []).append(key)
        for ds in sorted(groups):
            keys = groups[ds]
            same = sum(run[k]["h"] == ref[k]["h"] for k in keys)
            same200 = sum(run[k]["h200"] == ref[k]["h200"] for k in keys)
            both = [k for k in keys if run[k]["correct"] is not None and ref[k]["correct"] is not None]
            agree = sum(run[k]["correct"] == ref[k]["correct"] for k in both) / len(both) if both else float("nan")
            print(f"    {ds:14s} n={len(keys):5d} identical_text={same:5d} first200_same={same200:5d} correct_agree={agree:.3f}")
    print("\nReading: identical_text ~ n vs eager_20260926 = bitwise reproduction of 31.66;\n"
          "first200_same high but identical low = same weights/seeds, numerics drifted (GPU/kernels/versions);\n"
          "first200_same ~ 0 = different seeds or weights. Code correct_agree uses the in-loop sandbox flag.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compare"); p.add_argument("--suite", type=Path, required=True)
    p = sub.add_parser("check-manifest"); p.add_argument("--manifest", type=Path, required=True)
    p = sub.add_parser("fingerprint"); p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--scores-out", type=Path)
    args = ap.parse_args()
    return {"compare": cmd_compare, "check-manifest": cmd_check_manifest, "fingerprint": cmd_fingerprint}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
