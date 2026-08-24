"""Label and sample metadata helpers for full-gradient audit."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


_TEACHER_LABEL_KEY = "opd_teacher"
_DOMAIN_LABEL_KEYS = ("domain", "source_domain", "ability", "data_source")


def _non_tensor_list(value: Any, length: int, default: Any = None) -> list[Any]:
    if length <= 0:
        return []
    if value is None:
        return [default for _ in range(length)]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            items = [value.item()]
        else:
            items = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]

    if not items:
        return [default for _ in range(length)]
    if len(items) == 1 and length > 1:
        return [items[0] for _ in range(length)]
    if len(items) < length:
        return items + [default for _ in range(length - len(items))]
    if len(items) > length:
        return items[:length]
    return items


def _label_from_extra_info(extra_info: Any) -> Any:
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            return None
    if not isinstance(extra_info, dict):
        return None
    for key in _DOMAIN_LABEL_KEYS:
        value = extra_info.get(key)
        if value is not None:
            return value
    return None


def _labels_from_mapping(mapping: dict[str, Any], batch_size: int) -> list[str]:
    for key in _DOMAIN_LABEL_KEYS:
        labels = _non_tensor_list(mapping.get(key), batch_size)
        if not all(label is None for label in labels):
            return [str(label if label is not None else "unknown") for label in labels]
    extra_infos = _non_tensor_list(mapping.get("extra_info"), batch_size)
    labels = [_label_from_extra_info(extra_info) for extra_info in extra_infos]
    if not all(label is None for label in labels):
        return [str(label if label is not None else "unknown") for label in labels]
    return ["unknown" for _ in range(batch_size)]
