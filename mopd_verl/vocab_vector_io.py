"""Compact JSON encoding helpers for token-ID vocabulary vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SPARSE_TOKEN_ID_DICT = "sparse_token_id_dict"


def _nonzero_tensor_items(tensor: Any) -> tuple[list[int], list[Any]]:
    flat = tensor.detach().reshape(-1).cpu()
    indices = (flat != 0).nonzero(as_tuple=False).flatten()
    if int(indices.numel()) == 0:
        return [], []
    return indices.tolist(), flat.index_select(0, indices).tolist()


def tensor_to_sparse_int_dict(tensor: Any) -> dict[str, int]:
    """Encode nonzero tensor entries as ``token_id -> integer value``."""

    indices, values = _nonzero_tensor_items(tensor)
    return {
        str(int(index)): int(value)
        for index, value in zip(indices, values, strict=True)
    }


def tensor_to_sparse_float_dict(tensor: Any) -> dict[str, float]:
    """Encode nonzero tensor entries as ``token_id -> float value``."""

    indices, values = _nonzero_tensor_items(tensor)
    return {
        str(int(index)): float(value)
        for index, value in zip(indices, values, strict=True)
    }


def dense_vocab_vector(
    values: Sequence[Any] | Mapping[Any, Any],
    *,
    vocab_size: int | None = None,
) -> tuple[float, ...]:
    """Decode either legacy dense lists or sparse token-ID dictionaries."""

    if isinstance(values, Mapping):
        parsed_items = [
            (int(token_id), float(value)) for token_id, value in values.items()
        ]
        resolved_size = vocab_size
        if resolved_size is None:
            resolved_size = (
                max((token_id for token_id, _ in parsed_items), default=-1) + 1
            )
        if resolved_size < 0:
            raise ValueError("vocab_size must be non-negative")
        dense = [0.0] * resolved_size
        for token_id, value in parsed_items:
            if token_id < 0 or token_id >= resolved_size:
                raise ValueError(
                    f"Sparse token id {token_id} is outside vocab_size={resolved_size}."
                )
            dense[token_id] = value
        return tuple(dense)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        dense_values = tuple(float(value) for value in values)
        if vocab_size is not None and len(dense_values) != vocab_size:
            raise ValueError(
                f"Dense vector length {len(dense_values)} does not match vocab_size={vocab_size}."
            )
        return dense_values
    raise TypeError("Expected a dense sequence or sparse token-ID mapping.")
