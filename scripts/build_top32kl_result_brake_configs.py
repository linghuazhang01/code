"""Write the Qwen3-1.7B Top-32 RKL OPD configs that up-weight 结果结构 ∪ 推理阻力 ×4 (R1, 2026-11-17).

Pool per domain = full_classes.result ∪ full_classes.brake from configs/token_selection/list12_classes.json
(the same full sets as the native_opd_full_{result,brake}_* ignore runs). A position is up-weighted when the
sampled response token's ID is in the pool; weights are then divided by their mean over the domain's valid
positions (control_token_normalize_per_domain), the same fixed path as the SAPD Fixed4 profile. Each leaf
extends the matched equal-weight reference uniform_top32kl_toklog_3gpu_3c_b255.yaml, so the loss
(topk_renormalized_reverse_kl, k=32), data, topology (3 colocated GPUs, batch 255), seed and per-token logging
are unchanged.

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
STEM = "top32kl_result_brake_fixed4_toklog_3gpu_3c_b255"
PARENT = "uniform_top32kl_toklog_3gpu_3c_b255.yaml"


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


def render(domain: str) -> str:
    ids, meta = pool(domain)
    run = f"q1p7b-{domain}-top32kl-resbrake-fixed4-toklog-3g3c-b255-s42"
    return f"""# Qwen3-1.7B {domain}-only equal-weight Top-32 RKL OPD with 结果结构 ∪ 推理阻力 up-weighted ×4 (R1, 2026-11-17).
# Pool = full 结果结构 ({meta['n_result']} IDs) ∪ full 推理阻力 ({meta['n_brake']} IDs) = {len(ids)} IDs, {meta['pos_pct']}% of positions in the
# equal-weight runs; same full sets as native_opd_full_{{result,brake}}_* (list12_classes.json, sha256 of the sorted
# union {meta['sha256'][:16]}). Trigger = the sampled token's ID is in the pool; pool positions get weight 4, others 1,
# then weights are divided by their mean over valid {domain} positions (mean one, SAPD Fixed4 path). Everything else is
# the matched reference {PARENT}.
# Written by scripts/build_top32kl_result_brake_configs.py; do not edit by hand.
extends: {PARENT}
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
  output_dir: audit/{run}
paper_eval:
  output_dir: eval_outputs/paper_suite/{run}
huggingface_checkpoint:
  path_prefix: checkpoints/{domain}/{run}
trainer:
  experiment_name: {run}
  default_local_dir: checkpoints/MOPD/{run}
"""


def main() -> None:
    for domain, folder in TARGETS.items():
        path = folder / f"{STEM}.yaml"
        path.write_text(render(domain))
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
