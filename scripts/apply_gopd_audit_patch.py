#!/usr/bin/env python3
"""Verify that a G-OPD checkout uses the reviewed MOPD/verl integration.

Automatic patching of a pristine dp_actor.py is retired because a textual patch
cannot guarantee that production training and gradient audit use one loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_MARKERS: dict[str, tuple[str, ...]] = {
    "verl/verl/trainer/main_ppo.py": (
        'REF_POLICY_POOL_ID = "ref_policy_pool"',
        "self.mapping[Role.RefPolicy] = _ref_pool_id(config)",
    ),
    "verl/verl/trainer/ppo/ray_trainer.py": (
        "MOPDAuditLogger",
        "create_domain_batch_sampler(",
        "self.mopd_audit_logger.log_training_step(",
        "self.mopd_audit_logger.full_gradient_meta(",
    ),
    "verl/verl/utils/dataset/rl_dataset.py": (
        "annotate_hf_dataset_domain",
        "domain_for_data_file",
    ),
    "verl/verl/workers/actor/dp_actor.py": (
        "DomainGradientAudit",
        "build_actor_micro_batch_loss(",
        "audit.run_before_training(",
        "audit.compare_training_gradient()",
    ),
    "verl/verl/workers/fsdp_workers.py": (
        'self.config.ref.fsdp_config if self.role == "ref"',
        "teacher_model_device",
    ),
}


def verify_integration(gopd_dir: Path) -> tuple[Path, ...]:
    """Return verified files, raising before any mutation when integration is absent."""

    verified: list[Path] = []
    for relative_path, markers in REQUIRED_MARKERS.items():
        path = gopd_dir / relative_path
        if not path.is_file():
            raise RuntimeError(f"Reviewed integration file is missing: {path}")
        text = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise RuntimeError(
                "Automatic patching of a pristine checkout is retired. "
                f"Use the reviewed vendored verl integration; {path} is missing: {missing}."
            )
        verified.append(path)
    return tuple(verified)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gopd_dir", help="Directory containing verl/verl/.")
    args = parser.parse_args()

    verified = verify_integration(Path(args.gopd_dir).resolve())
    for path in verified:
        print(f"verified: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
