#!/usr/bin/env python3
"""Reference export, plan check and comparison for run_eager_vs_cg.sh.

  python3 eval_modes.py export-ref  --suite SUITE_ROOT --out reference/<tag>.ref.jsonl.gz
  python3 eval_modes.py check-plan  --manifest SUITE_ROOT/suite_manifest.json --job mc8:tip:42 --mode eager
  python3 eval_modes.py compare     --output-root DIR --run-id ID [--jobs ...] [--out-dir DIR]

export-ref  (run once on the CityU server) stores what compare needs from a finished suite: the shard
            and seed layout, the generation / execution settings, per-problem correctness of the 8 rollouts
            (Math: rollout score; HumanEval+ / MBPP+: official EvalPlus base and plus pass; LCB v5 / v6:
            official graded_list) and, per Math rollout, a SHA-1 of the completion text.
check-plan  fails unless a freshly planned suite has the reference's shard / seed layout and sampling
            settings and the requested execution mode, so a run never starts on a different seed map.
compare     reads the finished eager and cg suites of each job under --output-root and the bundled
            references (cg_pool = the CityU 3-seed CUDA-graph suite of the same model and seed;
            eager_0926 = the 2026-09-26 CityU eager suite, TIP seed 42 only) and writes compare.md /
            compare.tsv: dataset scores, Math4 / Code4 / Macro8, paired differences with a 95% bootstrap CI
            over problems (resampled within each dataset; aggregates are unweighted means of dataset means,
            as in the paper tables) and, for Math, the share of byte-identical completions and of rollouts
            with the same correctness.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REF_DIR = HERE / "reference"
MATH = ["aime24", "aime25", "hmmt25feb", "hmmt25nov"]
CODE = ["humaneval_plus", "mbpp_plus", "lcb_v5", "lcb_v6"]
LABEL = {"aime24": "AIME2024", "aime25": "AIME2025", "hmmt25feb": "HMMT25Feb", "hmmt25nov": "HMMT25Nov",
         "humaneval_plus": "HumanEval+", "mbpp_plus": "MBPP+", "lcb_v5": "LCB_v5", "lcb_v6": "LCB_v6"}
MODES = ("cg", "eager")
# Jobs chosen by the author on 2026-10-11, with the CityU reference suites of each.
JOBS = {
    "math4:exopd:2042": {"cg_pool": "math4_k8_s60_exopd_cg_s2042_20261008"},
    "math4:exopd:1042": {"cg_pool": "math4_k8_s60_exopd_cg_s1042_20261008"},
    "math4:exopd:42": {"cg_pool": "math4_k8_s60_exopd_cg_s42_20261008"},
    "mc8:eopd:2042": {"cg_pool": "mc8_k8_s60_eopd_cg_s2042_20261008"},
    "mc8:eopd:1042": {"cg_pool": "mc8_k8_s60_eopd_cg_s1042_20261004"},
    "mc8:tip:42": {"cg_pool": "mc8_k8_s60_tip_cg_s42_20261008", "eager_0926": "mc8_k8_s60_tip_20260926"},
}
DEFAULT_JOBS = ",".join(JOBS)
PAIRS = [("eager", "cg"), ("cg", "cg_pool"), ("eager", "cg_pool"), ("eager", "eager_0926"), ("cg_pool", "eager_0926")]
LAYOUT_FIELDS = ("dataset", "task_type", "source_start", "source_end_exclusive", "num_samples", "generation_seed")
GENERATION_KEYS = ("base_seed", "max_new_tokens", "temperature", "top_p", "math_samples", "code_samples")
EXECUTION_KEYS = ("batch_size", "max_model_len", "max_num_batched_tokens", "shards_per_dataset",
                  "min_rows_per_shard", "enable_chunked_prefill")


def fresh_tag(job: str, mode: str, run_id: str) -> str:
    setting, name, seed = job.split(":")
    return f"evmode_{setting}_k8_s60_{name}_{mode}_s{seed}_{run_id}"


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def read_suite(root: Path) -> dict | None:
    """Per-problem correctness and Math rollout hashes of a finished suite (None if unfinished)."""
    manifest_path = root / "suite_manifest.json"
    if not manifest_path.exists() or not (root / "COMPLETED_AT_UTC").exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    acc: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    text: dict[str, list] = {}
    for task in manifest["tasks"]:
        if task["domain"] != "math":
            continue
        for line in (Path(task["output_dir"]) / "thinking_eval_samples.jsonl").open():
            r = json.loads(line)
            pid = str(r["sample_id"])
            acc[task["dataset"]][pid].append(float(r["score"]))
            text[f"{task['dataset']}|{pid}|{r['rollout_index']}"] = [_h(r["completion"]), float(r["score"])]
    official = manifest.get("official_evalplus") or "official_evalplus"
    for ds, sub in (("humaneval_plus", "humaneval"), ("mbpp_plus", "mbpp")):
        path = root / official / sub / "eval_results.json"
        if not path.exists():
            continue
        for task_id, rows in json.loads(path.read_text())["eval"].items():
            acc[ds][str(task_id)] = [float(r["base_status"] == "pass" and r["plus_status"] == "pass") for r in rows]
        summary = json.loads((root / official / sub / "SUMMARY.json").read_text())["metrics"]["plus"]["accuracy"]
        mean = float(np.mean([np.mean(v) for v in acc[ds].values()]))
        assert abs(mean - summary) < 1e-9, (str(root), ds, mean, summary)
    for rel in ("v5", "v6"):
        files = glob.glob(str(root / "*" / "lcb" / rel / "*_eval_all.json"))
        if len(files) == 1:
            for r in json.loads(Path(files[0]).read_text()):
                acc["lcb_" + rel][str(r["question_id"])] = [float(bool(v)) for v in r["graded_list"]]
    for ds, rows in acc.items():
        bad = [k for k, v in rows.items() if len(v) != 8]
        assert not bad, (str(root), ds, len(bad), "problems without 8 rollouts")
    return {
        "tag": manifest["run_tag"],
        "layout": [[t[f] for f in LAYOUT_FIELDS] for t in manifest["tasks"]],
        "generation": {k: manifest["generation"].get(k) for k in GENERATION_KEYS},
        "execution": manifest["execution"],
        "acc": {ds: dict(rows) for ds, rows in acc.items()},
        "text": text,
    }


def write_ref(data: dict, out: Path) -> None:
    ex = data["execution"]
    meta = {"type": "meta", "tag": data["tag"], "layout": data["layout"], "generation": data["generation"],
            "execution": {k: ex.get(k) for k in EXECUTION_KEYS + ("enforce_eager", "gpu_memory", "max_num_seqs", "worker_count")}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as handle:
        handle.write(json.dumps(meta) + "\n")
        for ds, rows in data["acc"].items():
            for pid, values in rows.items():
                handle.write(json.dumps({"type": "problem", "ds": ds, "pid": pid, "c": values}) + "\n")
        for key, (digest, score) in data["text"].items():
            handle.write(json.dumps({"type": "rollout", "key": key, "h": digest, "c": score}) + "\n")


def read_ref(tag: str) -> dict | None:
    path = REF_DIR / f"{tag}.ref.jsonl.gz"
    if not path.exists():
        return None
    data = {"acc": collections.defaultdict(dict), "text": {}}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            r = json.loads(line)
            if r["type"] == "meta":
                data.update({k: r[k] for k in ("tag", "layout", "generation", "execution")})
            elif r["type"] == "problem":
                data["acc"][r["ds"]][r["pid"]] = r["c"]
            else:
                data["text"][r["key"]] = [r["h"], r["c"]]
    data["acc"] = dict(data["acc"])
    return data


def scores(acc: dict, datasets: list[str]) -> dict[str, float]:
    row = {ds: 100 * float(np.mean([np.mean(v) for v in acc[ds].values()])) for ds in datasets}
    row["Math4"] = float(np.mean([row[ds] for ds in MATH]))
    if CODE[0] in datasets:
        row["Code4"] = float(np.mean([row[ds] for ds in CODE]))
        row["Macro8"] = float(np.mean([row[ds] for ds in MATH + CODE]))
    return row


def paired(a: dict, b: dict, datasets: list[str], boot: int, rng) -> dict:
    """a - b in pp with a 95% bootstrap CI, resampling problems within each dataset."""
    draws = {}
    for ds in datasets:
        keys = sorted(set(a[ds]) & set(b[ds]))
        assert len(keys) == len(a[ds]) == len(b[ds]), (ds, len(keys), len(a[ds]), len(b[ds]))
        d = np.array([np.mean(a[ds][k]) - np.mean(b[ds][k]) for k in keys])
        draws[ds] = (d.mean(), d[rng.integers(0, len(d), size=(boot, len(d)))].mean(axis=1))
    groups = {"Math4": MATH}
    if CODE[0] in datasets:
        groups.update(Code4=CODE, Macro8=MATH + CODE)
    out = {}
    for name, sel in groups.items():
        dist = 100 * np.mean([draws[ds][1] for ds in sel], axis=0)
        out[name] = (100 * float(np.mean([draws[ds][0] for ds in sel])),
                     float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5)))
    return out


def rollout_match(a: dict, b: dict):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    return (len(keys), 100 * sum(a[k][0] == b[k][0] for k in keys) / len(keys),
            100 * sum(a[k][1] == b[k][1] for k in keys) / len(keys))


def parse_jobs(text: str) -> list[str]:
    jobs = [j for j in text.split(",") if j]
    unknown = [j for j in jobs if j not in JOBS]
    if unknown or not jobs:
        sys.exit(f"unknown job(s) {unknown}; choose from {DEFAULT_JOBS}")
    return jobs


def cmd_export_ref(args) -> int:
    data = read_suite(Path(args.suite))
    if data is None:
        sys.exit(f"{args.suite} is not a finished suite")
    write_ref(data, Path(args.out))
    print(f"{data['tag']}: {sum(len(v) for v in data['acc'].values())} problems, {len(data['text'])} Math rollouts -> {args.out}")
    return 0


def cmd_check_plan(args) -> int:
    job = parse_jobs(args.job)[0]
    ref = read_ref(JOBS[job]["cg_pool"])
    if ref is None:
        sys.exit(f"missing reference for {job}: {REF_DIR}/{JOBS[job]['cg_pool']}.ref.jsonl.gz")
    run = json.loads(Path(args.manifest).read_text())
    problems = []
    layout = [[t[f] for f in LAYOUT_FIELDS] for t in run["tasks"]]
    if layout != ref["layout"]:
        diff = sum(a != b for a, b in zip(layout, ref["layout"])) + abs(len(layout) - len(ref["layout"]))
        problems.append(f"shard/seed layout differs from {ref['tag']}: {len(layout)} vs {len(ref['layout'])} tasks, {diff} mismatches")
    for key in GENERATION_KEYS:
        if run["generation"].get(key) != ref["generation"].get(key):
            problems.append(f"generation.{key}: run={run['generation'].get(key)} ref={ref['generation'].get(key)}")
    for key in EXECUTION_KEYS:
        if run["execution"].get(key) != ref["execution"].get(key):
            problems.append(f"execution.{key}: run={run['execution'].get(key)} ref={ref['execution'].get(key)}")
    if run["execution"]["enforce_eager"] != (args.mode == "eager"):
        problems.append(f"execution.enforce_eager={run['execution']['enforce_eager']} for mode {args.mode}")
    if problems:
        print(f"[check-plan] {job} {args.mode}: MISMATCH\n  " + "\n  ".join(problems))
        return 1
    ex = run["execution"]
    print(f"[check-plan] {job} {args.mode}: OK, {len(layout)} shards with the seeds of {ref['tag']}; "
          f"enforce_eager={ex['enforce_eager']} gpu_memory={ex['gpu_memory']} max_num_seqs={ex['max_num_seqs']} workers={ex['worker_count']}")
    return 0


def cmd_compare(args) -> int:
    rng = np.random.default_rng(0)
    md = ["# Eager vs CUDA-graph evaluation", "",
          f"Output root `{args.output_root}`, run id `{args.run_id}`. Differences are row − column in pp with a 95% CI "
          f"from a per-problem bootstrap ({args.boot} resamples within each dataset). Math rollouts: 'identical' = "
          "byte-identical completion for the same problem and rollout index, 'agree' = same correctness. "
          "cg_pool / eager_0926 are the CityU reference suites (bundled in reference/).", ""]
    tsv = ["job\trun\ttag\tsettings\t" + "\t".join(LABEL[d] for d in MATH + CODE) + "\tMath4\tCode4\tMacro8"]
    for job in parse_jobs(args.jobs):
        setting = job.split(":")[0]
        datasets = MATH + (CODE if setting == "mc8" else [])
        cols = datasets + (["Math4", "Code4", "Macro8"] if setting == "mc8" else ["Math4"])
        runs = {mode: (fresh_tag(job, mode, args.run_id), read_suite(Path(args.output_root) / fresh_tag(job, mode, args.run_id)))
                for mode in MODES}
        runs.update({name: (tag, read_ref(tag)) for name, tag in JOBS[job].items()})
        md += [f"## {job}", "", "| run | settings | " + " | ".join(LABEL.get(c, c) for c in cols) + " |",
               "|---|---|" + "---|" * len(cols)]
        for name, (tag, data) in runs.items():
            if data is None:
                md.append(f"| {name} | `{tag}` not finished | " + " | ".join("–" for _ in cols) + " |")
                continue
            ex = data["execution"]
            settings = f"{'eager' if ex['enforce_eager'] else 'CG'} {ex['gpu_memory']}/{ex['max_num_seqs']}"
            row = scores(data["acc"], datasets)
            md.append(f"| {name} | {settings} | " + " | ".join(f"{row[c]:.2f}" for c in cols) + " |")
            tsv.append("\t".join([job, name, tag, settings] + [f"{row[d]:.2f}" if d in row else "" for d in MATH + CODE]
                                 + [f"{row[k]:.3f}" if k in row else "" for k in ("Math4", "Code4", "Macro8")]))
        md.append("")
        for x, y in PAIRS:
            if x not in runs or y not in runs or runs[x][1] is None or runs[y][1] is None:
                continue
            diff = paired(runs[x][1]["acc"], runs[y][1]["acc"], datasets, args.boot, rng)
            line = "; ".join(f"{k} {p:+.2f} [{lo:+.2f}, {hi:+.2f}]" for k, (p, lo, hi) in diff.items())
            match = rollout_match(runs[x][1]["text"], runs[y][1]["text"])
            if match:
                line += f"; Math rollouts n={match[0]}, identical {match[1]:.1f}%, agree {match[2]:.1f}%"
            md.append(f"- {x} − {y}: {line}")
        md.append("")
    text = "\n".join(md) + "\n"
    print(text)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "compare.md").write_text(text)
        (out / "compare.tsv").write_text("\n".join(tsv) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export-ref")
    p.add_argument("--suite", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check-plan")
    p.add_argument("--manifest", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--output-root", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--jobs", default=DEFAULT_JOBS)
    p.add_argument("--out-dir", default="")
    p.add_argument("--boot", type=int, default=10000)
    args = parser.parse_args()
    return {"export-ref": cmd_export_ref, "check-plan": cmd_check_plan, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
