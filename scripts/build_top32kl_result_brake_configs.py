"""Write the Qwen3-1.7B Top-32 RKL OPD configs that up-weight 结果结构 ∪ 推理阻力 ×4 (R1, 2026-11-17).

Pool per domain = full_classes.result ∪ full_classes.brake from configs/token_selection/list12_classes.json
(the same full sets as the native_opd_full_{result,brake}_* ignore runs). A position is up-weighted when the
sampled response token's ID is in the pool; weights are then divided by their mean over the domain's valid
positions (control_token_normalize_per_domain), the same fixed path as the SAPD Fixed4 profile. Each R1 leaf
extends the matched equal-weight reference of its topology, so the loss (topk_renormalized_reverse_kl, k=32),
data, seed and per-token logging are unchanged.

Topologies (all colocated: actor, vLLM rollout and the FSDP teacher share every GPU):
- 3gpu_3c, batch 255: reference uniform_top32kl_toklog_3gpu_3c_b255.yaml (exists, has run).
- 4gpu_4c / 8gpu_8c, batch 256: references uniform_top32kl_toklog_{4gpu_4c,8gpu_8c}_b256.yaml are written here;
  they extend the 3gpu_3c reference and change only the topology (GPU count, batch 255 -> 256 by the 4/8-GPU
  batch rule, FSDP size, Ray CPUs; same values as the native _native_opd_toklog_{4gpu_4c,8gpu_8c}_b256.yaml)
  and the run names.

Usage: python scripts/build_top32kl_result_brake_configs.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "configs/token_selection/list12_classes.json"
WEIGHT = 4.0
TARGETS = {
    "math": ROOT / "configs/token_selection/math/taxonomy",
    "code": ROOT / "configs/token_selection/code",
}
BASE = "uniform_top32kl_toklog_3gpu_3c_b255.yaml"
# topology -> (n_gpus, batch, short tag, ray cpus); 3gpu_3c uses the existing reference.
TOPOLOGIES = {
    "3gpu_3c": (3, 255, "3g3c", None),
    "4gpu_4c": (4, 256, "4g4c", 32),
    "8gpu_8c": (8, 256, "8g8c", 48),
}


def pool(domain: str) -> tuple[list[int], dict]:
    full = json.loads(RECORD.read_text())["domains"][domain]["full_classes"]
    ids = sorted(set(full["result"]["ids"]) | set(full["brake"]["ids"]))
    meta = {
        "n_result": full["result"]["n_ids"],
        "n_brake": full["brake"]["n_ids"],
        "pos_pct": round(full["result"]["pos_pct"] + full["brake"]["pos_pct"], 2),
        "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    }
    return ids, meta


def wrap(ids: list[int], indent: str = "      ", width: int = 118) -> str:
    lines, line = [], ""
    for i, value in enumerate(ids):
        piece = f"{value}" + (", " if i < len(ids) - 1 else "")
        if len(indent) + len(line) + len(piece.rstrip()) > width:
            lines.append(indent + line.rstrip())
            line = ""
        line += piece
    lines.append(indent + line.rstrip())
    return "\n".join(lines).lstrip()


def reference_name(topo: str) -> str:
    n, batch, _, _ = TOPOLOGIES[topo]
    return BASE if topo == "3gpu_3c" else f"uniform_top32kl_toklog_{topo}_b{batch}.yaml"


def r1_name(topo: str) -> str:
    _, batch, _, _ = TOPOLOGIES[topo]
    return f"top32kl_result_brake_fixed4_toklog_{topo}_b{batch}.yaml"


def run_names(domain: str, topo: str) -> tuple[str, str]:
    _, batch, tag, _ = TOPOLOGIES[topo]
    if topo == "3gpu_3c":
        return f"q1p7b-{domain}-top32kl-opd-toklog-3g3c-b255", f"q1p7b-{domain}-top32kl-resbrake-fixed4-toklog-3g3c-b255-s42"
    return (
        f"q1p7b-{domain}-top32kl-opd-toklog-{tag}-b{batch}-s42",
        f"q1p7b-{domain}-top32kl-resbrake-fixed4-toklog-{tag}-b{batch}-s42",
    )


def names_block(domain: str, run: str) -> str:
    return f"""  output_dir: audit/{run}
paper_eval:
  output_dir: eval_outputs/paper_suite/{run}
huggingface_checkpoint:
  path_prefix: checkpoints/{domain}/{run}
"""


def render_reference(domain: str, topo: str) -> str:
    n, batch, _, cpus = TOPOLOGIES[topo]
    run, _ = run_names(domain, topo)
    return f"""# Qwen3-1.7B {domain}-only equal-weight Top-32 RKL OPD with per-token logging, {n} colocated GPUs (actor, vLLM rollout and
# FSDP-{n} teacher share every GPU), global batch {batch}, seed 42, 60 steps. Same-topology reference for
# {r1_name(topo)} (R1, 2026-11-17). Base: {BASE} (the equal-weight run
# q1p7b-{domain}-top32kl-opd-toklog-3g3c-b255); this file changes only the topology (3 -> {n} colocated GPUs, batch
# 255 -> {batch} by the {n}-GPU batch rule, FSDP size, Ray CPUs) and the run names.
# Written by scripts/build_top32kl_result_brake_configs.py; do not edit by hand.
extends: {BASE}
runtime:
  slurm_allocation_gpus: {n}
  wandb_run_id: {run}
  wandb_resume: never
data:
  train_batch_size: {batch}
actor:
  ppo_mini_batch_size: {batch}
worker_placement:
  separate_ref_policy: false
  actor_rollout:
    n_gpus_per_node: {n}
    nnodes: 1
ray_kwargs:
  ray_init:
    num_cpus: {cpus}
# Lists replace the parent's list, so the reference FSDP size is restated here.
extra_overrides:
- custom_reward_function.path=mopd_verl/mixed_reward.py
- custom_reward_function.name=compute_score_batched
- reward_model.reward_manager=batch
- trainer.resume_mode=disable
- actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]
- actor_rollout_ref.ref.fsdp_config.fsdp_size={n}
audit:
{names_block(domain, run)}trainer:
  n_gpus_per_node: {n}
  experiment_name: {run}
  default_local_dir: checkpoints/MOPD/{run}
"""


def render_r1(domain: str, topo: str) -> str:
    ids, meta = pool(domain)
    _, run = run_names(domain, topo)
    parent = reference_name(topo)
    return f"""# Qwen3-1.7B {domain}-only equal-weight Top-32 RKL OPD with 结果结构 ∪ 推理阻力 up-weighted ×4 (R1, 2026-11-17).
# Pool = full 结果结构 ({meta['n_result']} IDs) ∪ full 推理阻力 ({meta['n_brake']} IDs) = {len(ids)} IDs, {meta['pos_pct']}% of positions in the
# equal-weight runs; same full sets as native_opd_full_{{result,brake}}_* (list12_classes.json, sha256 of the sorted
# union {meta['sha256'][:16]}). Trigger = the sampled token's ID is in the pool; pool positions get weight 4, others 1,
# then weights are divided by their mean over valid {domain} positions (mean one, SAPD Fixed4 path). Everything else is
# the matched reference {parent}.
# Written by scripts/build_top32kl_result_brake_configs.py; do not edit by hand.
extends: {parent}
runtime:
  wandb_run_id: {run}
  wandb_resume: never
audit:
  control_token_loss_weighting_enabled: true
  control_token_online_selection_enabled: false
  control_token_normalize_per_domain: true
  control_token_phase_gate_enabled: false
  control_token_span_weighting_enabled: false
  control_token_speed_weighting_enabled: false
  control_token_ids: []
  domain_control_token_ids:
    {domain}: [{wrap(ids)}]
  control_token_loss_weight: {WEIGHT}
  control_token_other_weight: 1.0
{names_block(domain, run)}trainer:
  experiment_name: {run}
  default_local_dir: checkpoints/MOPD/{run}
"""


def main() -> None:
    for domain, folder in TARGETS.items():
        for topo in TOPOLOGIES:
            if topo != "3gpu_3c":
                path = folder / reference_name(topo)
                path.write_text(render_reference(domain, topo))
                print(path.relative_to(ROOT))
            path = folder / r1_name(topo)
            path.write_text(render_r1(domain, topo))
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
