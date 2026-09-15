"""Memory-aware teacher batching and TopK postprocessing defaults.

The guard is a conservative heuristic, not an OOM guarantee. Dedicated
multi-rank reference workers synchronize batch boundaries before entering the
model, while unsupported worker layouts retain their original path, including
their collective call ordering. The independent expandable_segments setting
applies before model construction on dedicated CUDA teachers, including
multi-rank workers with batching disabled.
"""

import functools
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from mopd_verl.teacher_performance_config import parse_teacher_performance
from mopd_verl.teacher_performance_moe import install_moe_dispatch


@dataclass(frozen=True)
class _Tuning:
    chunk: int
    max_sequences: int
    max_tokens: int
    margin_bytes: int
    adaptive_chunk: int | None = None


def token_capacity(available: int, vocab_size: int, chunk: int) -> int:
    """Budget four BF16 logits copies plus four FP32 chunk workspaces."""
    return max(0, (available - chunk * vocab_size * 16) // (vocab_size * 8))


def _available(tuning: _Tuning) -> int:
    free, _ = torch.cuda.mem_get_info()
    cache = max(0, torch.cuda.memory_reserved() - torch.cuda.memory_allocated())
    return int(free + cache - tuning.margin_bytes)


def _distributed_min(value: int, world_size: int) -> int:
    """Reduce a small integer on the reference process group when required."""

    if world_size <= 1:
        return int(value)
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed teacher batching requires an initialized process group."
        )
    if torch.distributed.get_world_size() != world_size:
        raise RuntimeError(
            "Distributed teacher batching received a stale process-group size."
        )
    value_tensor = torch.tensor(
        [int(value)],
        dtype=torch.int64,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    torch.distributed.all_reduce(value_tensor, op=torch.distributed.ReduceOp.MIN)
    return int(value_tensor.item())


def _lengths(data: Any) -> list[int]:
    key = "ref_attention_mask" if "ref_attention_mask" in data.batch else "attention_mask"
    return [int(value) for value in data.batch[key].sum(-1).detach().cpu().tolist()]


def _concat(parts: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    if any(not isinstance(part, tuple) or len(part) != len(parts[0]) for part in parts):
        raise RuntimeError("Teacher returned inconsistent output tuples")
    output = []
    for column in zip(*parts):
        if all(value is None for value in column):
            output.append(None)
        elif all(torch.is_tensor(value) for value in column):
            output.append(torch.cat(column, dim=0))
        else:
            raise RuntimeError("Teacher output optional tensors are inconsistent")
    return tuple(output)


def next_group(lengths: list[int], start: int, rows: int, token_limit: int) -> list[int]:
    """Contiguous groups preserve all row-aligned tensor and non-tensor fields."""
    stop, tokens = start, 0
    while stop < len(lengths) and stop - start < rows:
        if stop > start and tokens + lengths[stop] > token_limit:
            break
        tokens += lengths[stop]
        stop += 1
    return list(range(start, stop))


def _batch_wrapper(
    original: Callable[..., Any], tuning: _Tuning, vocab_size: int,
    distributed_world_size: int = 1,
) -> Callable[..., Any]:
    @functools.wraps(original)
    def compute(*args: Any, **kwargs: Any) -> Any:
        data = kwargs.get("data", args[0] if args else None)
        local_size = 0 if data is None else len(data)
        if distributed_world_size > 1:
            minimum_size = _distributed_min(local_size, distributed_world_size)
            maximum_size = -_distributed_min(-local_size, distributed_world_size)
            if minimum_size != maximum_size:
                logging.getLogger(__name__).warning(
                    "Teacher memory guard: retaining original batching for uneven "
                    "rank-local batch sizes (%d..%d)",
                    minimum_size,
                    maximum_size,
                )
                return original(*args, **kwargs)
        if data is None or len(data) == 0:
            return original(*args, **kwargs)
        lengths = _lengths(data)
        locally_supported = int(
            not any(length <= 0 for length in lengths)
            and "multi_modal_inputs" not in data.non_tensor_batch
        )
        if distributed_world_size > 1:
            supported = _distributed_min(locally_supported, distributed_world_size)
        else:
            supported = locally_supported
        if not supported:
            return original(*args, **kwargs)
        parts, group_stats, start = [], [], 0
        while start < len(lengths):
            available = _available(tuning)
            candidate_chunks = [tuning.chunk]
            if (
                tuning.adaptive_chunk is not None
                and tuning.adaptive_chunk < tuning.chunk
            ):
                candidate_chunks.append(tuning.adaptive_chunk)
            candidates = []
            for chunk in candidate_chunks:
                capacity = min(
                    tuning.max_tokens,
                    token_capacity(available, vocab_size, chunk),
                )
                local_indices = next_group(
                    lengths,
                    start,
                    tuning.max_sequences,
                    capacity,
                )
                group_size = _distributed_min(
                    len(local_indices),
                    distributed_world_size,
                )
                candidates.append((group_size, chunk, capacity))
            group_size, chunk, capacity = max(
                candidates,
                key=lambda candidate: (candidate[0], candidate[1]),
            )
            indices = list(range(start, start + group_size))
            chunk_is_safe = int(
                sum(lengths[index] for index in indices) <= capacity
            )
            chunk_is_safe = _distributed_min(
                chunk_is_safe,
                distributed_world_size,
            )
            piece = data.select_idxs(indices)
            piece.meta_info = dict(data.meta_info)
            piece.meta_info.update(micro_batch_size=len(indices), use_dynamic_bsz=False)
            effective_chunk = chunk if chunk_is_safe else 16
            piece.meta_info["topk_logprob_chunk_size"] = effective_chunk
            piece.meta_info["teacher_memory_guarded_chunk"] = True
            group_stats.append(
                {
                    "rows": len(indices),
                    "tokens": sum(lengths[index] for index in indices),
                    "chunk": effective_chunk,
                    "guard_fallback": not bool(chunk_is_safe),
                }
            )
            piece_args = (piece, *args[1:]) if args else args
            piece_kwargs = dict(kwargs)
            if "data" in piece_kwargs:
                piece_kwargs["data"] = piece
            parts.append(original(*piece_args, **piece_kwargs))
            start += len(indices)
        stats = {
            "rows": len(lengths),
            "tokens": sum(lengths),
            "micro_batches": len(group_stats),
            "groups": tuple(group_stats),
        }
        setattr(compute, "_teacher_batch_stats", stats)
        logging.getLogger(__name__).info("Teacher batch plan: %s", stats)
        return _concat(parts)
    return compute


def _chunk_wrapper(
    original: Callable[..., Any], tuning: _Tuning, vocab_size: int,
) -> Callable[..., Any]:
    signature = inspect.signature(original)

    @functools.wraps(original)
    def forward(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        micro_batch = bound.arguments["micro_batch"]
        if "multi_modal_inputs" in micro_batch:
            return original(*args, **kwargs)
        # Explicit call-site overrides keep their original meaning.
        requested_chunk = bound.arguments.get("topk_logprob_chunk_size")
        memory_guarded_chunk = bool(
            bound.arguments.get("teacher_memory_guarded_chunk", False)
        )
        if requested_chunk is not None and not memory_guarded_chunk:
            return original(*args, **kwargs)
        key = "ref_attention_mask" if "ref_attention_mask" in micro_batch else "attention_mask"
        tokens = int(micro_batch[key].sum().item())
        effective_chunk = (
            tuning.chunk if requested_chunk is None else int(requested_chunk)
        )
        capacity = token_capacity(_available(tuning), vocab_size, effective_chunk)
        if tokens <= min(capacity, tuning.max_tokens):
            bound.arguments["topk_logprob_chunk_size"] = effective_chunk
        else:
            # An oversized single row cannot be split without changing attention.
            # Retain the original chunk, rather than reject otherwise valid work.
            logging.getLogger(__name__).warning(
                "Teacher memory guard: retaining original chunk16 for %d tokens (capacity=%d)",
                tokens, capacity,
            )
            bound.arguments["topk_logprob_chunk_size"] = None
        return original(*bound.args, **bound.kwargs)
    return forward


def configure_teacher_allocator(
    config: Mapping[str, Any], *, teacher_model_device: str, dedicated_teacher: bool,
) -> bool:
    """Apply the allocator opt-in before loading a dedicated GPU teacher model.

    This setting is independent of performance.enabled and world size. A false
    result leaves the process allocator unchanged, including environment settings.
    """
    requested = parse_teacher_performance(config)
    if not requested.expandable_segments or not dedicated_teacher:
        return False
    if teacher_model_device not in {"gpu", "cuda"} or not torch.cuda.is_available():
        return False
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    return True


def configure_teacher_performance(
    policy: Any, config: Mapping[str, Any], *, world_size: int,
    teacher_model_device: str, dedicated_teacher: bool = False,
    fsdp_size: int | None = None,
) -> dict[str, Any]:
    """Install policy-local defaults once, after the reference model is built."""
    if not config.get("enabled", False):
        return {"enabled": False, "reason": "disabled"}
    if hasattr(policy, "_teacher_performance_config"):
        return policy._teacher_performance_config
    reasons = []
    if world_size != 1 and not dedicated_teacher:
        reasons.append("multi-rank collective ordering")
    if teacher_model_device not in {"gpu", "cuda"} or not torch.cuda.is_available():
        reasons.append("CPU/offloaded or non-CUDA teacher")
    if policy.ulysses_sequence_parallel_size != 1 or not policy.use_remove_padding:
        reasons.append("requires remove-padding and sequence parallel size 1")
    if policy.use_fused_kernels or policy.actor_optimizer is not None:
        reasons.append("requires unfused reference-only execution")
    model_config = policy.actor_module.config
    if getattr(model_config, "torch_dtype", None) != torch.bfloat16:
        reasons.append("memory estimate validated only for BF16 model")
    if reasons:
        logging.getLogger(__name__).warning("Teacher performance fallback: %s", "; ".join(reasons))
        return {"enabled": False, "reason": "; ".join(reasons)}
    requested = parse_teacher_performance(config)
    tuning = _Tuning(
        chunk=requested.topk_logprob_chunk_size,
        max_sequences=requested.max_micro_batch_size,
        max_tokens=requested.max_tokens,
        margin_bytes=int(requested.memory_margin_gib * 1024**3),
        adaptive_chunk=requested.adaptive_topk_chunk_size,
    )
    mode = requested.moe_dispatch
    vocab = int(model_config.vocab_size)
    replicated_teacher = (
        dedicated_teacher
        and world_size > 1
        and fsdp_size == 1
    )
    batch_sync_world_size = (
        world_size if dedicated_teacher and not replicated_teacher else 1
    )
    policy._forward_micro_batch = _chunk_wrapper(
        policy._forward_micro_batch, tuning, vocab
    )
    policy.compute_log_prob = _batch_wrapper(
        policy.compute_log_prob,
        tuning,
        vocab,
        distributed_world_size=batch_sync_world_size,
    )
    blocks = install_moe_dispatch(policy.actor_module, mode)
    state = {
        "enabled": True,
        "chunk": tuning.chunk,
        "max_micro_batch_size": tuning.max_sequences,
        "max_tokens": tuning.max_tokens,
        "adaptive_topk_chunk_size": tuning.adaptive_chunk,
        "fused_statistics": requested.fused_statistics,
        "compact_topk_ids": requested.compact_topk_ids,
        "memory_margin_bytes": tuning.margin_bytes,
        "moe_blocks": blocks,
        "distributed_batch_sync": batch_sync_world_size > 1,
        "replicated_teacher": replicated_teacher,
        "fsdp_size": fsdp_size,
        "world_size": world_size,
    }
    policy._teacher_performance_config = state
    logging.getLogger(__name__).info("Teacher performance configured: %s", state)
    return state
