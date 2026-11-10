"""Qwen3-4B native OPD ignore configs on six of the seven token classes, Math-only and Code-only (author 2026-11-10).

The 4B twins of the 1.7B ignore runs already trained (Token_Set.md): full 推理推力 / 推理阻力 / 结果结构 / 解题结构 and list-12 虚词 / 版面.
内容 is left out (author 2026-11-10: "除了内容词之外"). Pools come from configs/token_selection/list12_classes.json (written by
scripts/build_list12_class_configs.py; full = every ID of the class passing the list-12 occurrence gate, counted on 8 runs per domain,
1.7B and 4B). ignore = pool weight 0, every other valid position weight 1, no normalisation, as in the 1.7B runs.
Topologies: 2gpu_2c (two runs side by side on 4 GPUs) and 4gpu_4c (fallback if 4B does not fit on 2 GPUs), global batch 256, seed 42.
Equal-weight reference: the 3-GPU naive run q4b-{math,code}-opdnative-toklog-3g3c-b255-s42 (no 2- or 4-GPU equal-weight 4B run).
Writes configs/qwen4b_suite/{math,code}/_opd_native_toklog_2gpu_2c_b256.yaml and configs/qwen4b_suite/{math,code}/list12_class/."""
import json, textwrap
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
SUITE = CODE / "configs/qwen4b_suite"
RECORD = json.loads((CODE / "configs/token_selection/list12_classes.json").read_text())["domains"]
# (pool set, class): the six 1.7B ignore runs, 内容 left out
LEAVES = [("full", "push"), ("full", "brake"), ("full", "result"), ("full", "solve"), ("l12", "func"), ("l12", "layout")]
POOLS = {"l12": "classes", "full": "full_classes"}
SETS = {"l12": "the list-12 class {cls} ({n} IDs, {pos}% of positions)",
        "full": "the full set of class {cls} ({n} IDs passing the list-12 occurrence gate, {pos}% of positions)"}
TOPOLOGIES = {"2gpu_2c": (2, "2g2c"), "4gpu_4c": (4, "4g4c")}


def equal_weight_2gpu(dom):
    return "\n".join([
        f"# Qwen3-4B {dom}-only native OPD (\"OPD naive\") with per-token logging, equal weight, 2 colocated GPUs (actor, vLLM rollout and FSDP-2",
        "# teacher share both GPUs), global batch 256. Not launchable by itself: list12_class/_opd_native_l12class_toklog_2gpu_2c_b256.yaml extends it",
        "# (author 2026-11-10: 4B class ignore runs, two at a time on 4 GPUs). Base: opd_native_toklog_3gpu_3c.yaml (the equal-weight naive run",
        f"# q4b-{dom}-opdnative-toklog-3g3c-b255-s42); this file changes only the topology (3 -> 2 colocated GPUs, batch 255 -> 256, since 255 does",
        "# not split over 2 GPUs). No 2-GPU 4B run has been trained before. Written by scripts/build_q4b_list12_class_configs.py.",
        "extends: opd_native_toklog_3gpu_3c.yaml", "runtime:", "  slurm_allocation_gpus: 2", "data:", "  train_batch_size: 256", "actor:",
        "  ppo_mini_batch_size: 256", "worker_placement:", "  separate_ref_policy: false", "  actor_rollout:", "    n_gpus_per_node: 2", "    nnodes: 1",
        "trainer:", "  n_gpus_per_node: 2", "ray_kwargs:", "  ray_init:", "    num_cpus: 16",
        "# Lists replace the parent's list, so the reference FSDP size is restated here.", "extra_overrides:",
        "- custom_reward_function.path=mopd_verl/mixed_reward.py", "- custom_reward_function.name=compute_score_batched",
        "- reward_model.reward_manager=batch", "- trainer.resume_mode=disable",
        "- actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]", "- actor_rollout_ref.ref.fsdp_config.fsdp_size=2", ""])


def parent(dom, topo, gpus):
    return "\n".join([
        f"# Qwen3-4B {dom}-only native OPD (\"OPD naive\") with fixed position weights on one token class and per-token logging, {gpus} colocated GPUs,",
        f"# global batch 256. Not launchable by itself: opd_native_{{l12,full}}_<class>_ignore_toklog_{topo}_b256.yaml set the pool, the two weights and the run names.",
        "# Same weighting as the 1.7B class ignore runs: fixed per-position weights, no online selection, no normalisation.",
        f"# Written by scripts/build_q4b_list12_class_configs.py. Equal-weight reference: ../opd_native_toklog_3gpu_3c.yaml (q4b-{dom}-opdnative-toklog-3g3c-b255-s42).",
        f"extends: ../_opd_native_toklog_{topo}_b256.yaml", "audit:", "  control_token_loss_weighting_enabled: true",
        "  control_token_online_selection_enabled: false", "  control_token_normalize_per_domain: false", "  control_token_phase_gate_enabled: false",
        "  control_token_span_weighting_enabled: false", "  control_token_speed_weighting_enabled: false", "  control_token_ids: []", ""])


def leaf(dom, topo, gpus, run, desc, ids):
    return "\n".join([
        f"# Qwen3-4B {dom}-only native OPD that ignores every position of {desc} with weight 0; all other valid positions keep weight 1.",
        f"# {gpus} colocated GPUs, global batch 256, seed 42, 60 steps, per-token logging on. Pool record: ../../../token_selection/list12_classes.json.",
        f"extends: _opd_native_l12class_toklog_{topo}_b256.yaml", "runtime:", f"  wandb_run_id: {run}", "audit:", "  domain_control_token_ids:",
        textwrap.fill(", ".join(map(str, ids)), 120, initial_indent=f"    {dom}: [", subsequent_indent="      ") + "]",
        "  control_token_loss_weight: 0.0", "  control_token_other_weight: 1.0", f"  output_dir: audit/{run}",
        "paper_eval:", f"  output_dir: eval_outputs/paper_suite/{run}", "huggingface_checkpoint:", f"  path_prefix: checkpoints/qwen4b/{dom}/{run}",
        "trainer:", f"  experiment_name: {run}", f"  default_local_dir: checkpoints/MOPD/{run}", ""])


def main():
    for dom in ("math", "code"):
        (SUITE / dom / "_opd_native_toklog_2gpu_2c_b256.yaml").write_text(equal_weight_2gpu(dom))
        out = SUITE / dom / "list12_class"; out.mkdir(exist_ok=True)
        for topo, (gpus, short) in TOPOLOGIES.items():
            (out / f"_opd_native_l12class_toklog_{topo}_b256.yaml").write_text(parent(dom, topo, gpus))
            for pool, k in LEAVES:
                c = RECORD[dom][POOLS[pool]][k]
                run = f"q4b-{dom}-opdnative-{pool}-{k}-ign-toklog-{short}-b256-s42"
                desc = SETS[pool].format(cls=c["cls"], n=c["n_ids"], pos=c["pos_pct"])
                (out / f"opd_native_{pool}_{k}_ignore_toklog_{topo}_b256.yaml").write_text(leaf(dom, topo, gpus, run, desc, c["ids"]))
        print(dom, len(list(out.glob("opd_native_*.yaml"))), "leaves")


if __name__ == "__main__":
    main()
