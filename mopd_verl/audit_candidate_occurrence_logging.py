"""Per-step candidate-position audit: loss and applied weight per occurrence.

A *candidate position* is one valid response token (``response_mask == 1``)
in the current step's rollout batch whose token ID belongs to that rollout's
domain candidate set (the up-weighting candidate whitelist).  It is keyed by
``(response_uid, response_position)`` with the same ``response_uid`` scheme
as ``response_records_{domain}.jsonl[.gz]``.

The log replaces the full response log for position-level analyses: per step
and domain it stores every candidate position, a deterministic uniform sample
of non-candidate valid positions as a reference population, and exact
domain totals over all valid positions.
"""

from __future__ import annotations

import gzip
import json
import math
import zlib
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from mopd_verl.audit_io import step_jsonl_dir
from mopd_verl.audit_response_logging import normalize_response_compression
from mopd_verl.tensorboard_tags import safe_name


CANDIDATE_OCCURRENCE_SCHEMA_VERSION = 1
CANDIDATE_OCCURRENCE_FILE_PREFIX = "candidate_occurrences_"
CANDIDATE_OCCURRENCE_MANIFEST = "candidate_occurrence_manifest.json"
APPLIED_TOKEN_WEIGHT_KEY = "mopd_applied_token_weight"
FLOAT_SIGNIFICANT_DIGITS = 6
SELECTED_RELATIVE_TOLERANCE = 1e-6

POSITION_ARRAY_FIELDS = (
    "response_position",
    "relative_position",
    "token_id",
    "is_candidate",
    "configured_token_loss",
    "abs_logp_gap",
    "student_entropy",
    "teacher_entropy",
    "teacher_student_cross_entropy",
    "applied_weight",
    "selected",
    "position_eligible",
)

FIELD_DOCS = {
    "schema_version": "Candidate-occurrence schema version.",
    "step": "Optimizer step (global_steps) of the rollout batch.",
    "domain": "Teacher domain label of the response.",
    "response_uid": (
        "Experiment-scoped response key 'step=<step>:batch_index=<i>'; "
        "identical to response_records_{domain}.jsonl[.gz]."
    ),
    "sample_id": "Source prompt key (not unique; one prompt has n rollouts).",
    "batch_index": "Row index in the driver batch at audit time.",
    "valid_token_count": "Number of response_mask=1 positions in the response.",
    "row_base_weight": (
        "Applied weight shared by the response's ordinary (non-candidate) "
        "valid positions: the most frequent applied weight among them. null "
        "when applied weights are unavailable or the response has no "
        "non-candidate valid position."
    ),
    "response_position": "Index on the padded response axis.",
    "relative_position": "response_position / valid_token_count.",
    "token_id": "Sampled response token ID at the position.",
    "is_candidate": (
        "true: token ID is in the domain candidate set (all such positions "
        "are logged). false: uniform reference sample of non-candidate "
        "valid positions."
    ),
    "configured_token_loss": (
        "l_raw: the production configured token loss (unweighted) returned "
        "by the actor update, before any token up-weighting."
    ),
    "abs_logp_gap": (
        "|teacher chosen-token logp - student (old_log_probs) chosen-token "
        "logp|."
    ),
    "student_entropy": "Student full-vocabulary entropy at the position.",
    "teacher_entropy": "Domain teacher entropy at the position.",
    "teacher_student_cross_entropy": (
        "Top-k teacher-student cross-entropy (support/k/tail/temperature in "
        "manifest.signals.teacher_student_cross_entropy)."
    ),
    "applied_weight": (
        "Per-position token multiplier actually used in the production "
        "backward (DomainGradientAudit.training_gradient_mask: after gamma "
        "and mean-one normalisation; excludes the token-mean loss scale and "
        "the domain-budget loss scale). null when not available on this "
        "path (see manifest.applied_weight)."
    ),
    "selected": (
        "true iff applied_weight > row_base_weight (relative tolerance "
        f"{SELECTED_RELATIVE_TOLERANCE}), i.e. the position was in the "
        "up-weighted set this step. null when applied_weight or "
        "row_base_weight is null."
    ),
    "position_eligible": (
        "mopd_control_position_mask at the position (Token V4/V5 position "
        "eligibility); null when the batch carries no such mask."
    ),
}


@dataclass(frozen=True)
class CandidateOccurrenceBatch:
    """Driver-side tensors aligned with ``response_mask`` ([batch, response])."""

    step: int
    domains: tuple[str, ...]
    sample_ids: tuple[str, ...]
    response_token_ids: Any
    response_mask: Any
    student_log_prob: Any
    teacher_log_prob: Any
    configured_token_loss: Any
    student_entropy: Any | None = None
    teacher_entropy: Any | None = None
    teacher_student_cross_entropy: Any | None = None
    applied_token_weight: Any | None = None
    control_position_mask: Any | None = None
    applied_weight_path: str = "unknown"
    applied_weight_unavailable_reason: str | None = None
    signals: dict[str, Any] = field(default_factory=dict)


def normalize_reference_fraction(value: Any) -> float:
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(
            "candidate_occurrence_reference_fraction must be in [0, 1], "
            f"got {value!r}."
        )
    return fraction


def reference_sampling_seed(step: int, domain: str) -> int:
    """Deterministic per-(step, domain) seed for the reference sample."""

    return (int(step) * 1_000_003 + zlib.crc32(str(domain).encode("utf-8"))) % (
        2**63 - 1
    )


def _round(value: float) -> float | None:
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return float(f"{numeric:.{FLOAT_SIGNIFICANT_DIGITS}g}")


def _float_values(matrix: Any | None, row: int, columns: Any) -> list | None:
    if matrix is None:
        return None
    return [_round(value) for value in matrix[row, columns].tolist()]


def _validate(name: str, matrix: Any | None, reference: Any) -> None:
    if matrix is None:
        return
    if tuple(getattr(matrix, "shape", ())) != tuple(reference.shape):
        raise ValueError(
            f"{name} must match response_mask shape, got "
            f"{tuple(getattr(matrix, 'shape', ()))} versus "
            f"{tuple(reference.shape)}."
        )


def _cpu(matrix: Any | None, dtype: Any) -> Any | None:
    if matrix is None:
        return None
    return matrix.detach().to(device="cpu", dtype=dtype)


def _row_base_weight(weights: Any, ordinary: Any) -> float | None:
    import torch

    values = weights[ordinary]
    if int(values.numel()) == 0:
        return None
    unique, counts = torch.unique(values, return_counts=True)
    return float(unique[int(torch.argmax(counts))])


def _reference_mask(
    noncandidate: Any,
    fraction: float,
    seed: int,
) -> Any:
    """Uniform sample without replacement of round(fraction * N) positions."""

    import torch

    flat = noncandidate.reshape(-1)
    available = torch.nonzero(flat, as_tuple=False).flatten()
    count = int(round(fraction * int(available.numel())))
    chosen = torch.zeros_like(flat)
    if count > 0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        order = torch.randperm(int(available.numel()), generator=generator)
        chosen[available[order[:count]]] = True
    return chosen.reshape(noncandidate.shape)


def _open(stack: ExitStack, path: Path, compression: str) -> TextIO:
    if compression == "gzip":
        return stack.enter_context(
            gzip.open(path, mode="wt", encoding="utf-8", compresslevel=6)
        )
    return stack.enter_context(path.open("w", encoding="utf-8"))


def write_candidate_occurrence_log(
    *,
    output_dir: str | Path,
    batch: CandidateOccurrenceBatch,
    domain_candidate_token_ids: Mapping[str, tuple[int, ...]],
    candidate_source: str,
    reference_fraction: float,
    compression: str,
) -> dict[str, dict[str, Any]]:
    """Write one gzip JSONL per domain plus a step manifest; return totals."""

    import torch

    compression = normalize_response_compression(compression)
    reference_fraction = normalize_reference_fraction(reference_fraction)
    mask = batch.response_mask.detach().to("cpu").bool()
    batch_size = int(mask.shape[0])
    if len(batch.domains) != batch_size or len(batch.sample_ids) != batch_size:
        raise ValueError("domains and sample_ids need one entry per response.")
    for name in (
        "response_token_ids",
        "student_log_prob",
        "teacher_log_prob",
        "configured_token_loss",
        "student_entropy",
        "teacher_entropy",
        "teacher_student_cross_entropy",
        "applied_token_weight",
        "control_position_mask",
    ):
        _validate(name, getattr(batch, name), mask)

    ids = _cpu(batch.response_token_ids, torch.long)
    loss = _cpu(batch.configured_token_loss, torch.float64)
    gap = (
        _cpu(batch.teacher_log_prob, torch.float64)
        - _cpu(batch.student_log_prob, torch.float64)
    ).abs()
    student_entropy = _cpu(batch.student_entropy, torch.float64)
    teacher_entropy = _cpu(batch.teacher_entropy, torch.float64)
    cross_entropy = _cpu(batch.teacher_student_cross_entropy, torch.float64)
    weights = _cpu(batch.applied_token_weight, torch.float64)
    eligible = _cpu(batch.control_position_mask, torch.bool)

    directory = step_jsonl_dir(output_dir, batch.step, create=True)
    suffix = ".jsonl.gz" if compression == "gzip" else ".jsonl"
    totals: dict[str, dict[str, Any]] = {}
    files: dict[str, str] = {}
    domains = list(dict.fromkeys(batch.domains))
    with ExitStack() as stack:
        for domain in domains:
            rows = [i for i, label in enumerate(batch.domains) if label == domain]
            row_index = torch.tensor(rows, dtype=torch.long)
            valid = mask[row_index]
            candidate_ids = torch.tensor(
                sorted({int(t) for t in domain_candidate_token_ids.get(domain, ())}),
                dtype=torch.long,
            )
            candidate = valid & torch.isin(ids[row_index], candidate_ids)
            noncandidate = valid & ~candidate
            seed = reference_sampling_seed(batch.step, domain)
            reference = _reference_mask(noncandidate, reference_fraction, seed)
            logged = candidate | reference

            safe_domain = safe_name(domain) or "unknown"
            name = f"{CANDIDATE_OCCURRENCE_FILE_PREFIX}{safe_domain}{suffix}"
            files[domain] = name
            handle = _open(stack, directory / name, compression)
            selected_candidates = 0
            selected_noncandidate_valid = 0
            selected_defined = weights is not None
            for local, row in enumerate(rows):
                row_valid = valid[local]
                valid_count = int(row_valid.sum())
                columns = torch.nonzero(logged[local], as_tuple=False).flatten()
                base = (
                    None
                    if weights is None
                    else _row_base_weight(weights[row], noncandidate[local])
                )
                applied = None
                selected: list | None = None
                if weights is not None:
                    applied = [_round(v) for v in weights[row, columns].tolist()]
                    if base is not None:
                        threshold = base + abs(base) * SELECTED_RELATIVE_TOLERANCE
                        row_selected = weights[row].gt(threshold) & row_valid
                        selected = row_selected[columns].tolist()
                        selected_candidates += int(
                            (row_selected & candidate[local]).sum()
                        )
                        selected_noncandidate_valid += int(
                            (row_selected & noncandidate[local]).sum()
                        )
                    else:
                        selected = [None] * int(columns.numel())
                        if bool(candidate[local].any()):
                            selected_defined = False
                positions = [int(c) for c in columns.tolist()]
                record = {
                    "schema_version": CANDIDATE_OCCURRENCE_SCHEMA_VERSION,
                    "step": int(batch.step),
                    "domain": domain,
                    "response_uid": f"step={int(batch.step)}:batch_index={row}",
                    "sample_id": batch.sample_ids[row],
                    "batch_index": row,
                    "valid_token_count": valid_count,
                    "row_base_weight": None if base is None else _round(base),
                    "response_position": positions,
                    "relative_position": [
                        _round(p / valid_count) if valid_count else None
                        for p in positions
                    ],
                    "token_id": ids[row, columns].tolist(),
                    "is_candidate": candidate[local, columns].tolist(),
                    "configured_token_loss": _float_values(loss, row, columns),
                    "abs_logp_gap": _float_values(gap, row, columns),
                    "student_entropy": _float_values(student_entropy, row, columns),
                    "teacher_entropy": _float_values(teacher_entropy, row, columns),
                    "teacher_student_cross_entropy": _float_values(
                        cross_entropy, row, columns
                    ),
                    "applied_weight": applied,
                    "selected": selected,
                    "position_eligible": (
                        None if eligible is None else eligible[row, columns].tolist()
                    ),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            domain_loss = loss[row_index]
            valid_float = valid.double()
            noncandidate_count = int(noncandidate.sum())
            reference_count = int(reference.sum())
            totals[domain] = {
                "response_count": len(rows),
                "valid_token_count": int(valid.sum()),
                "candidate_position_count": int(candidate.sum()),
                "noncandidate_valid_position_count": noncandidate_count,
                "reference_position_count": reference_count,
                "reference_realized_fraction": (
                    reference_count / noncandidate_count
                    if noncandidate_count
                    else 0.0
                ),
                "reference_seed": seed,
                "sum_configured_token_loss": float(
                    (domain_loss * valid_float).sum()
                ),
                "sum_configured_token_loss_candidates": float(
                    (domain_loss * candidate.double()).sum()
                ),
                "sum_applied_weight": (
                    None
                    if weights is None
                    else float((weights[row_index] * valid_float).sum())
                ),
                "sum_weighted_configured_token_loss": (
                    None
                    if weights is None
                    else float((weights[row_index] * domain_loss * valid_float).sum())
                ),
                "selected_candidate_position_count": (
                    selected_candidates if selected_defined else None
                ),
                "selected_noncandidate_valid_position_count": (
                    selected_noncandidate_valid if weights is not None else None
                ),
            }

    manifest = _manifest(
        batch,
        compression=compression,
        domain_candidate_token_ids=domain_candidate_token_ids,
        candidate_source=candidate_source,
        reference_fraction=reference_fraction,
        files=files,
        totals=totals,
    )
    manifest_path = directory / CANDIDATE_OCCURRENCE_MANIFEST
    temporary = directory / f".{CANDIDATE_OCCURRENCE_MANIFEST}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(manifest_path)
    return totals


def _manifest(
    batch: CandidateOccurrenceBatch,
    *,
    compression: str,
    domain_candidate_token_ids: Mapping[str, tuple[int, ...]],
    candidate_source: str,
    reference_fraction: float,
    files: dict[str, str],
    totals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    weights_available = batch.applied_token_weight is not None
    return {
        "schema_version": CANDIDATE_OCCURRENCE_SCHEMA_VERSION,
        "step": int(batch.step),
        "definition": (
            "A candidate position is one valid response token "
            "(response_mask=1) in this step's rollout batch whose token ID is "
            "in that rollout's domain candidate set (candidate_token_ids."
            "by_domain); it is identified by (response_uid, "
            "response_position). Every candidate position is logged; "
            "non-candidate valid positions are logged only as a uniform "
            "reference sample (is_candidate=false)."
        ),
        "record_scope": (
            "One JSONL row per response of the domain (rows with no logged "
            "position carry empty arrays); per-position arrays are aligned."
        ),
        "array_alignment": (
            "Unnest response_position and every other per-position array by "
            "the same index; null in place of an array means the signal is "
            "unavailable for the whole step."
        ),
        "per_position_fields": list(POSITION_ARRAY_FIELDS),
        "fields": FIELD_DOCS,
        "files_by_domain": files,
        "compression": compression,
        "float_precision": f"{FLOAT_SIGNIFICANT_DIGITS} significant digits",
        "identity": {
            "primary_key": "response_uid",
            "token_key": ["response_uid", "response_position"],
            "matches": "response_records_{domain}.jsonl[.gz] of the same step",
        },
        "candidate_token_ids": {
            "source": candidate_source,
            "by_domain": {
                str(domain): [int(t) for t in sorted(set(token_ids))]
                for domain, token_ids in domain_candidate_token_ids.items()
            },
        },
        "reference_sample": {
            "population": "valid non-candidate positions of the domain",
            "fraction": reference_fraction,
            "method": (
                "uniform without replacement: round(fraction * N) positions "
                "chosen by torch.randperm over the row-major flattened "
                "[domain responses x padded response] mask"
            ),
            "seed": "(step * 1000003 + crc32(domain)) mod (2**63 - 1)",
            "deterministic_given": "step, domain, and driver batch order",
            "inverse_probability_weight": (
                "noncandidate_valid_position_count / reference_position_count"
            ),
        },
        "applied_weight": {
            "available": weights_available,
            "path": batch.applied_weight_path,
            "unavailable_reason": (
                None if weights_available else batch.applied_weight_unavailable_reason
            ),
            "source": (
                "actor update_policy auxiliary output "
                f"'{APPLIED_TOKEN_WEIGHT_KEY}' = DomainGradientAudit."
                "training_gradient_mask(micro_batch) captured at backward time "
                "(final PPO epoch), or 1.0 on valid positions when no token "
                "weighting is active"
            ),
            "excludes": [
                "token-mean loss scale (global_token_mean_loss_scales)",
                "domain-budget loss scale (mopd_domain_loss_scale)",
                "same-batch adaptive-neighborhood multiplier",
            ],
        },
        "signals": batch.signals,
        "available_metrics": {
            "student_entropy": batch.student_entropy is not None,
            "teacher_entropy": batch.teacher_entropy is not None,
            "teacher_student_cross_entropy": (
                batch.teacher_student_cross_entropy is not None
            ),
            "applied_weight": weights_available,
            "position_eligible": batch.control_position_mask is not None,
        },
        "totals_scope": (
            "Exact sums over every valid position (response_mask=1) of the "
            "domain, not only logged positions."
        ),
        "totals_by_domain": totals,
    }


def iter_candidate_positions(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one flat dict per logged position from a domain file."""

    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            count = len(row["response_position"])
            scalars = {
                key: value
                for key, value in row.items()
                if key not in POSITION_ARRAY_FIELDS
            }
            for index in range(count):
                position = dict(scalars)
                for key in POSITION_ARRAY_FIELDS:
                    values = row.get(key)
                    position[key] = None if values is None else values[index]
                yield position
