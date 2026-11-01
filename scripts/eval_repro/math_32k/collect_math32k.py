"""Math4 budget curves for the 32k re-evaluation of the Math-only 1.7B naive / list-12 ignore / only / termlook checkpoints.

Runs on Burgundy from ~/scratch/opd with any python3 (standard library only). For each arm and evaluation seed it reads
<suite>/<ckpt>/math/thinking_eval_results.json. "Within B" counts a sample as correct only if it is correct and finished before
B tokens; Math is the mean of the four dataset accuracies, as in eval.md. Paired contrasts bootstrap problems (2000 draws).
--suite-format and --budgets point the same code at the 16k suites; those reproduce the budget curves of 2026-11-01.
Writes <out>/budget_curves.csv and <out>/contrasts.csv."""
import argparse, csv, glob, json, os, random
from collections import defaultdict

ARMS = ["q1p7b_math_opdnative", "q1p7b_math_l12ign", "q1p7b_math_l12only", "q1p7b_math_termlook"]
SEEDS = [42, 1042, 2042]
CONTRASTS = [("q1p7b_math_l12only", "q1p7b_math_opdnative"), ("q1p7b_math_l12ign", "q1p7b_math_opdnative"),
             ("q1p7b_math_termlook", "q1p7b_math_opdnative"), ("q1p7b_math_l12only", "q1p7b_math_l12ign")]


def load(root, fmt, arm, seed):
    suite = os.path.join(root, fmt.format(arm=arm, seed=seed))
    found = glob.glob(os.path.join(suite, "*", "math", "thinking_eval_results.json"))
    if not found or not os.path.exists(os.path.join(suite, "COMPLETED_AT_UTC")):
        return None
    assert len(found) == 1, found
    return json.load(open(found[0]))["records"]


def within(r, budget):
    return float(bool(r["correct"]) and r["generated_tokens"] < budget)


def macro(records, value):
    by = defaultdict(list)
    for r in records:
        by[r["dataset"]].append(value(r))
    assert len(by) == 4, sorted(by)
    return 100 * sum(sum(v) / len(v) for v in by.values()) / 4


def per_problem(records, budget):
    by = defaultdict(list)
    for r in records:
        by[(r["dataset"], r["sample_id"])].append(within(r, budget))
    return {k: sum(v) / len(v) for k, v in by.items()}


def paired(a, b, seed=0, reps=2000):
    """Mean difference of a and b over problems, weighted to the macro Math score, with a 95% problem bootstrap."""
    keys = sorted(set(a) & set(b)); groups = defaultdict(list)
    for k in keys:
        groups[k[0]].append(a[k] - b[k])
    mean = lambda g: 100 * sum(sum(v) / len(v) for v in g) / len(g)
    rng = random.Random(seed); draws = sorted(mean([rng.choices(v, k=len(v)) for v in groups.values()]) for _ in range(reps))
    return mean(groups.values()), draws[int(0.025 * reps)], draws[int(0.975 * reps)], len(keys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="a100eval_outputs_mc")
    p.add_argument("--suite-format", default="math4_32k_k8_s60_{arm}_a100_cg_s{seed}_20261101")
    p.add_argument("--budgets", default="4000,8000,12000,16384,24000,32768")
    p.add_argument("--out", default="math32k_collect")
    args = p.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]; os.makedirs(args.out, exist_ok=True)

    data, rows = {}, []
    for arm in ARMS:
        for seed in SEEDS:
            records = load(args.root, args.suite_format, arm, seed)
            if records is None:
                print("not complete:", arm, seed); continue
            data[(arm, seed)] = records
            row = dict(arm=arm, seed=seed, records=len(records), score=macro(records, lambda r: float(r["score"])),
                       cap=records[0]["max_new_tokens"], at_cap_pct=100 * sum(r["generated_tokens"] >= r["max_new_tokens"] for r in records) / len(records),
                       past_16k_pct=100 * sum(r["generated_tokens"] >= 16384 for r in records) / len(records))
            row.update({f"within_{b}": macro(records, lambda r: within(r, b)) for b in budgets})
            rows.append(row)
    if not rows:
        return
    for arm in ARMS:
        have = [r for r in rows if r["arm"] == arm]
        if have:
            mean = lambda k: sum(r[k] for r in have) / len(have)
            print(f"{arm:24s} n={len(have)} score {mean('score'):5.2f} at cap {mean('at_cap_pct'):4.1f}% past 16k {mean('past_16k_pct'):4.1f}% within",
                  " ".join(f"{b // 1000}k {mean(f'within_{b}'):5.2f}" for b in budgets))
    with open(os.path.join(args.out, "budget_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    out = []
    top, mid = budgets[-1], 16384 if 16384 in budgets else budgets[-1]
    for arm in ARMS:  # what the longer budget adds inside one arm
        for seed in SEEDS:
            if (arm, seed) in data and top != mid:
                d, lo, hi, n = paired(per_problem(data[(arm, seed)], top), per_problem(data[(arm, seed)], mid))
                out.append(dict(a=f"{arm} within {top}", b=f"within {mid}", seed=seed, diff=d, lo=lo, hi=hi, problems=n))
    for x, y in CONTRASTS:
        for budget in (mid, top) if top != mid else (top,):
            for seed in SEEDS:
                if (x, seed) in data and (y, seed) in data:
                    d, lo, hi, n = paired(per_problem(data[(x, seed)], budget), per_problem(data[(y, seed)], budget))
                    out.append(dict(a=x, b=y, seed=seed, budget=budget, diff=d, lo=lo, hi=hi, problems=n))
            pa, pb = [[per_problem(data[(k, s)], budget) for s in SEEDS if (x, s) in data and (y, s) in data] for k in (x, y)]
            if pa:
                avg = lambda ps: {k: sum(p[k] for p in ps) / len(ps) for k in ps[0]}
                d, lo, hi, n = paired(avg(pa), avg(pb))
                out.append(dict(a=x, b=y, seed="pooled", budget=budget, diff=d, lo=lo, hi=hi, problems=n))
                print(f"{x} - {y} within {budget}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}] ({len(pa)} seeds)")
    fields = ["a", "b", "seed", "budget", "diff", "lo", "hi", "problems"]
    with open(os.path.join(args.out, "contrasts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)


if __name__ == "__main__":
    main()
