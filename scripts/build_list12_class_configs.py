"""Native OPD ignore / only configs on the seven list-12 classes and their full sets, Math-only and Code-only (Qwen3-1.7B, author 2026-11-04).

Classes: the seven classes of the 2026-11-03 group-meeting slides (推理推力, 推理阻力, 结果结构, 解题结构, 虚词, 内容, 版面).
l12 pools: analysis-output/list12-partition-20261101/tables/classes.json, the list-12 IDs of each class.
full pools: analysis-output/seven-class-fullset-20261104/tables/classes.json, every ID of each class that passes the list-12 occurrence gate
(M = 20, measurable in Rising in >= 7 of 8 runs and in both phases in >= 6 of 8) without the speed rule; each contains the l12 pool.
ignore = pool weight 0, others 1; only = pool weight 1, others 0; no normalisation, as in the list-12 ignore / only runs.
Topologies 4gpu_4c and 8gpu_8c, global batch 256, seed 42. Needs the analysis directory behind analysis-output.
Writes configs/token_selection/list12_classes.json and configs/token_selection/{math,code}/list12_class/ (two parents and 56 leaves per domain)."""
import hashlib, json, textwrap
from pathlib import Path
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
PARTITION = CODE / "analysis-output/list12-partition-20261101/tables"
FULLSET = CODE / "analysis-output/seven-class-fullset-20261104/tables"
SELECTION = CODE / "configs/token_selection"
CLASSES = {"push": "推理推力", "brake": "推理阻力", "result": "结果结构", "solve": "解题结构", "func": "虚词", "content": "内容", "layout": "版面"}
# domain: (folder of the equal-weight topology parents, seen from list12_class/)
DOMAINS = {"math": "../taxonomy", "code": ".."}
TOPOLOGIES = {"4gpu_4c": (4, "4g4c"), "8gpu_8c": (8, "8g8c")}
# pool set: sentence naming one pool
SETS = {"l12": "the list-12 class {cls} ({n} IDs, {pos}% of positions)",
        "full": "the full set of class {cls} ({n} IDs passing the list-12 occurrence gate, {pos}% of positions)"}
# arm: (run tag, pool weight, other weight, sentence)
ARMS = {"ignore": ("ign", 0.0, 1.0, "ignores every position of {pool} with weight 0; all other valid positions keep weight 1"),
        "only": ("only", 1.0, 0.0, "learns only positions of {pool} with weight 1; all other valid positions get weight 0")}


def sha(ids):
    return hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest()


def parent(dom, topo, gpus):
    return "\n".join([
        f"# Qwen3-1.7B {dom}-only native OPD (\"OPD naive\") with fixed position weights on one of the seven classes and per-token logging, {gpus} colocated GPUs,",
        f"# global batch 256. Not launchable by itself: native_opd_{{l12,full}}_<class>_{{ignore,only}}_toklog_{topo}_b256.yaml set the pool, the two weights and the run names.",
        "# Same weighting as the list-12 ignore / only runs: fixed per-position weights, no online selection, no normalisation.",
        f"# Written by scripts/build_list12_class_configs.py. Same-topology equal-weight reference: {DOMAINS[dom]}/native_opd_toklog_{topo}_b256.yaml.",
        f"extends: {DOMAINS[dom]}/_native_opd_toklog_{topo}_b256.yaml", "audit:", "  control_token_loss_weighting_enabled: true",
        "  control_token_online_selection_enabled: false", "  control_token_normalize_per_domain: false", "  control_token_phase_gate_enabled: false",
        "  control_token_span_weighting_enabled: false", "  control_token_speed_weighting_enabled: false", "  control_token_ids: []", ""])


def leaf(dom, topo, gpus, run, sentence, ids, pool_w, other_w):
    return "\n".join([
        f"# Qwen3-1.7B {dom}-only native OPD that {sentence}.",
        f"# {gpus} colocated GPUs, global batch 256, seed 42, 60 steps, per-token logging on. Pool record: ../../list12_classes.json (scripts/build_list12_class_configs.py).",
        f"extends: _native_opd_l12class_toklog_{topo}_b256.yaml", "runtime:", f"  wandb_run_id: {run}", "audit:", "  domain_control_token_ids:",
        textwrap.fill(", ".join(map(str, ids)), 120, initial_indent=f"    {dom}: [", subsequent_indent="      ") + "]",
        f"  control_token_loss_weight: {pool_w}", f"  control_token_other_weight: {other_w}", f"  output_dir: audit/{run}",
        "paper_eval:", f"  output_dir: eval_outputs/paper_suite/{run}", "huggingface_checkpoint:", f"  path_prefix: checkpoints/{dom}/{run}",
        "trainer:", f"  experiment_name: {run}", f"  default_local_dir: checkpoints/MOPD/{run}", ""])


def main():
    classes = json.load(open(PARTITION / "classes.json"))
    stats = pd.read_csv(PARTITION / "class_stats.csv").set_index(["domain", "cls"])
    list12 = json.load(open(CODE / "analysis-output/token-set-toklog-20261019/tables/d1_token_list_ids.json"))
    full = json.load(open(FULLSET / "classes.json"))
    fstats = pd.read_csv(FULLSET / "fullset_counts.csv").set_index(["domain", "cls"])
    record = {"source": "analysis/list12-partition-20261101/tables/classes.json", "full_source": "analysis/seven-class-fullset-20261104/tables/classes.json",
              "date": "2026-11-04", "domains": {}}
    for dom in DOMAINS:
        ids = {k: sorted(map(int, classes[dom][c])) for k, c in CLASSES.items()}
        union = sorted(i for v in ids.values() for i in v)
        assert len(set(union)) == len(union) and union == sorted(map(int, list12[dom]["shared"]))
        record["domains"][dom] = {"list12_n_ids": len(union), "list12_sha256": sha(union), "classes": {
            k: dict(cls=CLASSES[k], n_ids=len(v), pos_pct=round(float(stats.loc[(dom, CLASSES[k]), "pos_pct"]), 2), sha256=sha(v), ids=v) for k, v in ids.items()}}
        fids = {k: sorted(map(int, full[dom][c])) for k, c in CLASSES.items()}
        assert all(set(ids[k]) <= set(fids[k]) for k in CLASSES)
        record["domains"][dom]["full_classes"] = {
            k: dict(cls=CLASSES[k], n_ids=len(v), pos_pct=round(float(fstats.loc[(dom, CLASSES[k]), "full_pos"]), 2), sha256=sha(v), ids=v) for k, v in fids.items()}
        out = SELECTION / dom / "list12_class"; out.mkdir(exist_ok=True)
        for topo, (gpus, short) in TOPOLOGIES.items():
            (out / f"_native_opd_l12class_toklog_{topo}_b256.yaml").write_text(parent(dom, topo, gpus))
            for name, key in (("l12", "classes"), ("full", "full_classes")):
                for k, c in record["domains"][dom][key].items():
                    desc = SETS[name].format(cls=c["cls"], n=c["n_ids"], pos=c["pos_pct"])
                    for arm, (tag, pool_w, other_w, sentence) in ARMS.items():
                        run = f"q1p7b-{dom}-opdnative-{name}-{k}-{tag}-toklog-{short}-b256-s42"
                        (out / f"native_opd_{name}_{k}_{arm}_toklog_{topo}_b256.yaml").write_text(
                            leaf(dom, topo, gpus, run, sentence.format(pool=desc), c["ids"], pool_w, other_w))
        print(dom, len(list(out.glob("native_opd_*.yaml"))), "leaves")
    (SELECTION / "list12_classes.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
