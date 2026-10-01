"""Build tokenizer-aligned, position-aware Structure masks for Token V4/V5."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from mopd_verl.domain_gradient.code_positions import (
    code_statement_spans,
    fenced_code_blocks,
)
from mopd_verl.domain_gradient.token_taxonomy_registry import (
    TOKEN_TAXONOMY_VERSIONS,
    normalize_token_taxonomy_version,
)

STRUCTURE_POSITION_PROFILE = "token_v4_v5_fourbaseline_r3_answer_format"

_MATH_ANYWHERE_IDS = frozenset({79075, 151645})
_MATH_LABEL_IDS = frozenset({1590, 4226, 13023, 16688, 19357, 21806, 73877})
_CODE_LABEL_IDS = frozenset({1590, 4226, 9217, 11822, 13023, 16688, 19357, 21806})
_CODE_EOS_ID = 151645
_CODE_FENCE_IDS = frozenset({41233, 54275, 73594})
_CODE_LANGUAGE_IDS = frozenset({12669})

_ANSWER_HEADING = re.compile(
    r"(?im)^[ \t]*#{1,6}[ \t]+[^\n]*?\b(?:Final|Answer|Conclusion)\b[^\n]*$"
)
_ANSWER_BOLD_LABEL = re.compile(
    r"(?im)(?P<marker>\*\*|__)[ \t]*"
    r"(?P<label>Final(?:[ \t]+Answer)?|Answer|Conclusion)[ \t]*:?[ \t]*(?P=marker)"
)
_ANSWER_LABEL_WORD = re.compile(r"\b(?:Final|Answer|Conclusion)\b", re.IGNORECASE)
_BOXED_LINE = re.compile(r"(?m)^[^\r\n]*\\boxed(?![A-Za-z])[^\r\n]*\r?$")


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _domain_labels(batch: Any, rows: int) -> list[str]:
    values = batch.non_tensor_batch
    for key in ("domain", "source_domain", "ability", "data_source"):
        if key in values:
            labels = [str(value) for value in values[key]]
            if len(labels) != rows:
                raise ValueError("Structure mask requires one domain label per row.")
            return labels
    raise ValueError("Structure mask requires a domain label.")


def _token_offsets(
    tokenizer: Any,
    token_ids: Sequence[int],
    token_surfaces: Mapping[int, str] | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    """Preserve generated token boundaries, including non-canonical sequences."""

    pieces = []
    for token_id in token_ids:
        if token_surfaces is not None and token_id in token_surfaces:
            pieces.append(token_surfaces[token_id])
        else:
            pieces.append(
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
    offsets = []
    cursor = 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return "".join(pieces), offsets


def _overlaps(span: tuple[int, int], targets: Sequence[tuple[int, int]]) -> bool:
    start, end = span
    return start < end and any(start < right and end > left for left, right in targets)


def _answer_spans(
    text: str, *, include_boxed_lines: bool = False
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    patterns = (_ANSWER_HEADING, _ANSWER_BOLD_LABEL)
    if include_boxed_lines:
        patterns += (_BOXED_LINE,)
    for pattern in patterns:
        for container in pattern.finditer(text):
            start, end = (
                container.span("label")
                if pattern is _ANSWER_BOLD_LABEL
                else container.span()
            )
            for label in _ANSWER_LABEL_WORD.finditer(text[start:end]):
                spans.append((start + label.start(), start + label.end()))
    return spans


def control_position_mask(
    token_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    labels: Sequence[str],
    tokenizer: Any,
    token_surfaces: Mapping[int, str] | None = None,
) -> torch.Tensor:
    """Code Control is eligible only outside every fenced block, even unclosed."""
    if token_ids.shape != valid_mask.shape or len(labels) != token_ids.shape[0]:
        raise ValueError("Control mask requires aligned IDs, mask, and labels.")
    result = valid_mask.bool().clone()
    for row, domain in enumerate(labels):
        if domain == "math":
            continue
        if domain != "code":
            raise ValueError(f"Unsupported Control domain: {domain!r}")
        columns = valid_mask[row].bool().nonzero().flatten()
        ids = token_ids[row, columns].tolist()
        text, offsets = _token_offsets(tokenizer, ids, token_surfaces)
        excluded = [(block.opening[0], block.end) for block in fenced_code_blocks(text)]
        for offset, span in enumerate(offsets):
            if _overlaps(span, excluded):
                result[row, columns[offset]] = False
    return result


def validated_control_position_mask(
    batch: Any,
    config: Any,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Use the exact same eligibility mask for scoring and weight application."""
    version = normalize_token_taxonomy_version(
        _cfg_get(config, "token_taxonomy_version", "legacy")
    )
    if version not in TOKEN_TAXONOMY_VERSIONS:
        return valid_mask
    if str(_cfg_get(config, "structure_token_position_profile", "none")) == "none":
        return valid_mask
    mask = batch.batch.get("mopd_control_position_mask")
    if mask is None:
        raise ValueError("Token V4/V5 requires mopd_control_position_mask.")
    mask = mask.detach().bool().to(valid_mask.device)
    if mask.shape != valid_mask.shape or (mask & ~valid_mask).any():
        raise ValueError(
            "Control position mask must align with valid response positions."
        )
    return mask


def structure_position_mask(
    token_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    labels: Sequence[str],
    tokenizer: Any,
    domain_structure_token_ids: Mapping[str, Sequence[int]],
    token_surfaces: Mapping[int, str] | None = None,
) -> torch.Tensor:
    """Return fixed-Structure positions, failing closed on parse mismatch."""

    if token_ids.shape != valid_mask.shape or len(labels) != token_ids.shape[0]:
        raise ValueError("Structure mask requires aligned IDs, mask, and labels.")
    result = torch.zeros_like(valid_mask, dtype=torch.bool)
    for row, domain in enumerate(labels):
        valid_columns = valid_mask[row].bool().nonzero().flatten()
        if not valid_columns.numel():
            continue
        row_ids = [int(value) for value in token_ids[row, valid_columns].tolist()]
        allowed = {int(value) for value in domain_structure_token_ids[domain]}
        for offset, token_id in enumerate(row_ids):
            if domain == "math" and token_id in allowed & _MATH_ANYWHERE_IDS:
                result[row, valid_columns[offset]] = True
            if domain == "code" and token_id in allowed and token_id == _CODE_EOS_ID:
                result[row, valid_columns[offset]] = True
        text, offsets = _token_offsets(tokenizer, row_ids, token_surfaces)
        if domain == "math":
            targets = _answer_spans(text, include_boxed_lines=True)
            for offset, (token_id, span) in enumerate(
                zip(row_ids, offsets, strict=True)
            ):
                if token_id in allowed & _MATH_LABEL_IDS and _overlaps(span, targets):
                    result[row, valid_columns[offset]] = True
            continue
        if domain != "code":
            raise ValueError(f"Unsupported Structure domain: {domain!r}")
        all_blocks = fenced_code_blocks(text)
        code_spans = [(block.opening[0], block.end) for block in all_blocks]
        label_spans = _answer_spans(text)
        for offset, (token_id, span) in enumerate(zip(row_ids, offsets, strict=True)):
            if (
                token_id in allowed & _CODE_LABEL_IDS
                and _overlaps(span, label_spans)
                and not _overlaps(span, code_spans)
            ):
                result[row, valid_columns[offset]] = True
        blocks = [block for block in all_blocks if block.closing]
        if not blocks:
            continue
        block = blocks[-1]
        opening, closing = block.opening, block.closing
        body, excluded = code_statement_spans(text, opening[1], closing[0])
        fence_spans = (opening, closing)
        for offset, (token_id, span) in enumerate(zip(row_ids, offsets, strict=True)):
            accepted = (
                (token_id in allowed & _CODE_FENCE_IDS and _overlaps(span, fence_spans))
                or (
                    token_id in allowed & _CODE_LANGUAGE_IDS
                    and _overlaps(span, (opening,))
                )
                or (
                    token_id in allowed
                    and token_id
                    not in (
                        _CODE_FENCE_IDS
                        | _CODE_LANGUAGE_IDS
                        | _CODE_LABEL_IDS
                        | {_CODE_EOS_ID}
                    )
                    and _overlaps(span, body)
                    and not _overlaps(span, excluded)
                )
            )
            if accepted:
                result[row, valid_columns[offset]] = True
    return result & valid_mask.bool()


def attach_structure_position_mask(batch: Any, tokenizer: Any, config: Any) -> None:
    """Attach the mask before DP balancing so it follows ordinary reordering."""

    if not bool(_cfg_get(config, "structure_token_loss_weighting_enabled", False)):
        return
    profile = str(_cfg_get(config, "structure_token_position_profile", "none"))
    if profile == "none":
        return
    if profile != STRUCTURE_POSITION_PROFILE:
        raise ValueError(f"Unsupported Structure position profile: {profile!r}")
    ids = _cfg_get(config, "domain_structure_token_ids", {})
    if not isinstance(ids, Mapping):
        raise TypeError("domain_structure_token_ids must be a mapping.")
    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    labels = _domain_labels(batch, int(responses.shape[0]))
    unique_ids = torch.unique(responses[response_mask.bool()]).detach().cpu().tolist()
    token_surfaces = {
        int(token_id): tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in unique_ids
    }
    batch.batch["mopd_structure_position_mask"] = structure_position_mask(
        responses,
        response_mask,
        labels,
        tokenizer,
        ids,
        token_surfaces,
    )
    batch.batch["mopd_control_position_mask"] = control_position_mask(
        responses,
        response_mask,
        labels,
        tokenizer,
        token_surfaces,
    )
