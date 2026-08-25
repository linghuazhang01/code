"""Complete response-level, per-token audit records."""

from __future__ import annotations

import gzip
import json
import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from mopd_verl.audit_io import step_jsonl_dir
from mopd_verl.tensorboard_tags import safe_name


RESPONSE_AUDIT_SCHEMA_VERSION = 3
SUPPORTED_COMPRESSIONS = {"gzip", "none"}


@dataclass(frozen=True)
class ResponseAuditBatch:
    """Tensor-aligned inputs needed to serialize complete responses."""

    step: int
    learning_rate: float
    domains: tuple[str, ...]
    sample_ids: tuple[str, ...]
    response_token_ids: Any
    response_mask: Any
    student_log_prob: Any
    teacher_log_prob: Any
    configured_token_loss: Any
    configured_token_loss_mask: Any
    student_entropy: Any | None
    teacher_entropy: Any | None
    teacher_student_cross_entropy: Any | None
    configured_token_loss_name: str
    configured_token_loss_epoch_reduction: str
    configured_token_loss_epoch_count: int
    cross_entropy_scope: str
    cross_entropy_support_source: str
    cross_entropy_topk: int
    cross_entropy_include_tail: bool
    cross_entropy_temperature: float
    signal_timing: str
    tokenizer_name_or_path: str | None
    tokenizer_vocab_size: int | None
    adaptive_center_mask: Any | None
    adaptive_threshold_pass_mask: Any | None
    adaptive_neighborhood: dict[str, object] | None


def normalize_response_compression(value: str) -> str:
    """Validate and normalize the configured response JSONL compression."""

    compression = str(value).strip().lower()
    if compression not in SUPPORTED_COMPRESSIONS:
        raise ValueError(
            "response_level_compression must be one of "
            f"{sorted(SUPPORTED_COMPRESSIONS)}, got {value!r}."
        )
    return compression


def _validate_matrix(name: str, matrix: Any, reference: Any) -> None:
    if matrix is None:
        return
    if not hasattr(matrix, "shape") or tuple(matrix.shape) != tuple(reference.shape):
        matrix_shape = getattr(matrix, "shape", None)
        raise ValueError(
            f"{name} must match response_mask shape, got {matrix_shape} "
            f"versus {tuple(reference.shape)}."
        )


def _finite_float_list(values: Any) -> list[float | None]:
    output: list[float | None] = []
    for value in values.detach().float().cpu().tolist():
        numeric = float(value)
        output.append(numeric if math.isfinite(numeric) else None)
    return output


def _masked_float_list(matrix: Any | None, index: int, valid: Any) -> list[float | None] | None:
    if matrix is None:
        return None
    return _finite_float_list(matrix[index][valid])


def _response_uid(batch: ResponseAuditBatch, index: int) -> str:
    """Return the experiment-scoped primary key for one response occurrence."""

    return f"step={batch.step}:batch_index={index}"


def _response_uids(batch: ResponseAuditBatch) -> tuple[str, ...]:
    """Build and validate the response-occurrence primary keys."""

    response_uids = tuple(
        _response_uid(batch, index) for index in range(len(batch.domains))
    )
    if len(set(response_uids)) != len(response_uids):
        raise ValueError("response_uid must be unique within an experiment step.")
    return response_uids


def _response_row(
    batch: ResponseAuditBatch,
    index: int,
    response_uid: str,
    control_token_ids: frozenset[int],
) -> dict[str, Any]:
    import torch

    valid = batch.response_mask[index].detach().bool().cpu()
    positions = torch.nonzero(valid, as_tuple=False).flatten()
    token_ids = (
        batch.response_token_ids[index].detach().long().cpu()[valid].tolist()
    )
    if batch.adaptive_center_mask is None:
        control_mask = [
            int(token_id) in control_token_ids for token_id in token_ids
        ]
        adaptive_center_mask = None
    else:
        control_mask = (
            batch.adaptive_center_mask[index]
            .detach()
            .bool()
            .cpu()[valid]
            .tolist()
        )
        adaptive_center_mask = control_mask
    adaptive_threshold_pass_mask = (
        None
        if batch.adaptive_threshold_pass_mask is None
        else (
            batch.adaptive_threshold_pass_mask[index]
            .detach()
            .bool()
            .cpu()[valid]
            .tolist()
        )
    )
    response_positions = [int(position) for position in positions.tolist()]
    control_positions = [
        response_positions[offset]
        for offset, is_control in enumerate(control_mask)
        if is_control
    ]
    adaptive_threshold_pass_positions = (
        None
        if adaptive_threshold_pass_mask is None
        else [
            response_positions[offset]
            for offset, passed in enumerate(adaptive_threshold_pass_mask)
            if passed
        ]
    )
    return {
        "schema_version": RESPONSE_AUDIT_SCHEMA_VERSION,
        "step": batch.step,
        "domain": batch.domains[index],
        "response_uid": response_uid,
        "sample_id": batch.sample_ids[index],
        "batch_index": index,
        "learning_rate": batch.learning_rate,
        "padded_response_length": int(batch.response_mask.shape[-1]),
        "valid_token_count": len(token_ids),
        "response_positions": response_positions,
        "response_token_ids": [int(token_id) for token_id in token_ids],
        "control_token_mask": control_mask,
        "control_token_positions": control_positions,
        "adaptive_center_mask": adaptive_center_mask,
        "adaptive_threshold_pass_mask": adaptive_threshold_pass_mask,
        "adaptive_threshold_pass_positions": (
            adaptive_threshold_pass_positions
        ),
        "student_token_log_prob": _masked_float_list(
            batch.student_log_prob,
            index,
            valid,
        ),
        "teacher_token_log_prob": _masked_float_list(
            batch.teacher_log_prob,
            index,
            valid,
        ),
        "configured_token_loss": _masked_float_list(
            batch.configured_token_loss,
            index,
            valid,
        ),
        "configured_token_loss_mask": _masked_float_list(
            batch.configured_token_loss_mask,
            index,
            valid,
        ),
        "student_entropy": _masked_float_list(
            batch.student_entropy,
            index,
            valid,
        ),
        "teacher_entropy": _masked_float_list(
            batch.teacher_entropy,
            index,
            valid,
        ),
        "teacher_student_cross_entropy": _masked_float_list(
            batch.teacher_student_cross_entropy,
            index,
            valid,
        ),
    }


def _open_response_file(
    stack: ExitStack,
    path: Path,
    compression: str,
) -> TextIO:
    if compression == "gzip":
        return stack.enter_context(
            gzip.open(path, mode="wt", encoding="utf-8", compresslevel=1)
        )
    return stack.enter_context(path.open("w", encoding="utf-8"))


def _manifest(
    batch: ResponseAuditBatch,
    compression: str,
    global_control_token_ids: tuple[int, ...],
    domain_control_token_ids: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    available = {
        "student_entropy": batch.student_entropy is not None,
        "teacher_entropy": batch.teacher_entropy is not None,
        "teacher_student_cross_entropy": (
            batch.teacher_student_cross_entropy is not None
        ),
        "adaptive_center_mask": batch.adaptive_center_mask is not None,
        "adaptive_threshold_pass_mask": (
            batch.adaptive_threshold_pass_mask is not None
        ),
    }
    return {
        "schema_version": RESPONSE_AUDIT_SCHEMA_VERSION,
        "step": batch.step,
        "record_scope": "one JSONL row per complete valid response",
        "array_alignment": (
            "All per-token arrays align with response_token_ids; "
            "response_positions maps them to the padded response axis."
        ),
        "compression": compression,
        "signal_timing": batch.signal_timing,
        "response_count": len(batch.domains),
        "domains": sorted(set(batch.domains)),
        "identity": {
            "primary_key": "response_uid",
            "primary_key_scope": "experiment",
            "primary_key_unique": True,
            "source_sample_key": "sample_id",
            "source_sample_key_unique": False,
            "token_key": ["response_uid", "response_position"],
            "token_array_unnest": {
                "response_position": "response_positions",
                "alignment": (
                    "Unnest response_positions and every per-token array by "
                    "the same array index."
                ),
            },
        },
        "tokenizer": {
            "name_or_path": batch.tokenizer_name_or_path,
            "vocab_size": batch.tokenizer_vocab_size,
        },
        "control_token_ids": {
            "global": [int(token_id) for token_id in global_control_token_ids],
            "by_domain": {
                str(domain): [int(token_id) for token_id in token_ids]
                for domain, token_ids in domain_control_token_ids.items()
            },
        },
        "available_metrics": available,
        "configured_token_loss": {
            "name": batch.configured_token_loss_name,
            "epoch_reduction": batch.configured_token_loss_epoch_reduction,
            "epoch_count": batch.configured_token_loss_epoch_count,
        },
        "adaptive_neighborhood": batch.adaptive_neighborhood,
        "teacher_student_cross_entropy": {
            "scope": batch.cross_entropy_scope,
            "support_source": batch.cross_entropy_support_source,
            "topk": batch.cross_entropy_topk,
            "include_tail": batch.cross_entropy_include_tail,
            "temperature": batch.cross_entropy_temperature,
        },
        "derived_metrics": {
            "student_token_nll": "-student_token_log_prob",
            "teacher_token_nll": "-teacher_token_log_prob",
            "signed_logp_gap": (
                "teacher_token_log_prob - student_token_log_prob"
            ),
            "absolute_logp_gap": "abs(signed_logp_gap)",
        },
    }


def write_response_audit(
    *,
    output_dir: str | Path,
    batch: ResponseAuditBatch,
    global_control_token_ids: tuple[int, ...],
    domain_control_token_ids: dict[str, tuple[int, ...]],
    compression: str,
) -> None:
    """Stream a full response batch into per-domain JSONL shards."""

    compression = normalize_response_compression(compression)
    reference = batch.response_mask
    if len(batch.domains) != int(reference.shape[0]):
        raise ValueError("domains must contain one entry per response.")
    if len(batch.sample_ids) != int(reference.shape[0]):
        raise ValueError("sample_ids must contain one entry per response.")
    response_uids = _response_uids(batch)
    for name, matrix in (
        ("response_token_ids", batch.response_token_ids),
        ("student_log_prob", batch.student_log_prob),
        ("teacher_log_prob", batch.teacher_log_prob),
        ("configured_token_loss", batch.configured_token_loss),
        ("configured_token_loss_mask", batch.configured_token_loss_mask),
        ("student_entropy", batch.student_entropy),
        ("teacher_entropy", batch.teacher_entropy),
        ("teacher_student_cross_entropy", batch.teacher_student_cross_entropy),
        ("adaptive_center_mask", batch.adaptive_center_mask),
        (
            "adaptive_threshold_pass_mask",
            batch.adaptive_threshold_pass_mask,
        ),
    ):
        _validate_matrix(name, matrix, reference)

    directory = step_jsonl_dir(output_dir, batch.step, create=True)
    suffix = ".jsonl.gz" if compression == "gzip" else ".jsonl"
    handles: dict[str, TextIO] = {}
    with ExitStack() as stack:
        for index, domain in enumerate(batch.domains):
            safe_domain = safe_name(domain) or "unknown"
            if safe_domain not in handles:
                path = directory / f"response_records_{safe_domain}{suffix}"
                handles[safe_domain] = _open_response_file(
                    stack,
                    path,
                    compression,
                )
            control_ids = frozenset(
                (*global_control_token_ids, *domain_control_token_ids.get(domain, ()))
            )
            row = _response_row(batch, index, response_uids[index], control_ids)
            handles[safe_domain].write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )

    manifest_path = directory / "response_manifest.json"
    temporary_path = directory / ".response_manifest.json.tmp"
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _manifest(
                batch,
                compression,
                global_control_token_ids,
                domain_control_token_ids,
            ),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    temporary_path.replace(manifest_path)
