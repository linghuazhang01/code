"""Persistent, teacher-only performance settings and Hydra serialization."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class TeacherPerformanceConfig:
    """Requested caps; the worker checks topology, model and available memory.

    expandable_segments independently enables the allocator on dedicated CUDA
    teachers before model loading, even when enabled is false or world size > 1.
    Setting it false leaves the process allocator unchanged.
    """

    enabled: bool = True
    topk_logprob_chunk_size: int = 1024
    max_micro_batch_size: int = 32
    max_tokens: int = 57344
    moe_dispatch: str = "stable_sort"
    memory_margin_gib: float = 12.0
    expandable_segments: bool = True
    fused_statistics: bool = False
    compact_topk_ids: bool = False
    adaptive_topk_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("teacher_performance.enabled must be a boolean")
        for name in ("expandable_segments", "fused_statistics", "compact_topk_ids"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"teacher_performance.{name} must be a boolean")
        bounds = {
            "topk_logprob_chunk_size": 1024,
            "max_micro_batch_size": 32,
            "max_tokens": 57344,
        }
        for name, upper in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(
                    f"teacher_performance.{name} must be an integer in 1..{upper}"
                )
        adaptive_chunk = self.adaptive_topk_chunk_size
        if adaptive_chunk is not None:
            if (
                type(adaptive_chunk) is not int
                or not 1 <= adaptive_chunk <= self.topk_logprob_chunk_size
            ):
                raise ValueError(
                    "teacher_performance.adaptive_topk_chunk_size must be null "
                    "or an integer in 1..topk_logprob_chunk_size"
                )
        if self.moe_dispatch not in {"stable_sort", "stock"}:
            raise ValueError(
                "teacher_performance.moe_dispatch must be stable_sort or stock"
            )
        margin = self.memory_margin_gib
        if isinstance(margin, bool) or not isinstance(margin, (int, float)):
            raise ValueError("teacher_performance.memory_margin_gib must be numeric")
        if not math.isfinite(margin) or margin < 12:
            raise ValueError(
                "teacher_performance.memory_margin_gib must be finite and >=12"
            )


def parse_teacher_performance(raw: Mapping[str, Any]) -> TeacherPerformanceConfig:
    """Reject misspelled or unsupported fields instead of silently ignoring them."""
    return TeacherPerformanceConfig(**dict(raw))


def teacher_performance_overrides(config: TeacherPerformanceConfig) -> list[str]:
    """Pass the teacher settings independently of student training microbatches."""
    return [
        f"+actor_rollout_ref.ref.teacher_performance.{key}="
        f"{str(value).lower() if isinstance(value, bool) else 'null' if value is None else value}"
        for key, value in asdict(config).items()
    ]
