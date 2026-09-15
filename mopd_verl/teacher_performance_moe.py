"""Instance-local dispatch optimization for the validated Qwen3 teacher."""

import functools
import hashlib
import inspect
import logging
from importlib.metadata import version

import torch
from torch import nn
from torch.nn import functional as F

EXPECTED_SOURCE_SHA_BY_VERSION = {
    # Exact inspect.getsource() hashes for Qwen3MoeSparseMoeBlock.forward.
    "4.51.3": "c4ae8784c667994091405761b588d9d73f8b4290f5bd4d43dcd5ea3842dc63cb",
    "4.57.6": "21b88637319fb0e3b5bbb87d98d1f04104d5ce176b9270486d638e6114b1b29a",
}


def qwen3_moe_forward_stable_sort(
    self: nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the dense expert mask and per-expert where with stable grouping.

    The flattened routing slots use [top_k, num_tokens] order. Stable sorting
    preserves exactly the lexicographic order of HF torch.where for each expert.
    The compatible Qwen3 teacher path was validated against the original forward.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    num_tokens = batch_size * sequence_length
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, self.top_k, dim=-1
    )
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    rank_major_experts = selected_experts.transpose(0, 1).reshape(-1)
    permutation = torch.argsort(rank_major_experts, stable=True)
    # minlength fixes the desired output length, but installed CUDA bincount
    # may still synchronize internally to find its maximum. Profile before use.
    counts = torch.bincount(
        rank_major_experts, minlength=self.num_experts
    ).tolist()
    start = 0
    for expert_idx, count in enumerate(counts):
        if count == 0:
            continue
        positions = permutation[start : start + count]
        start += count
        top_x = positions.remainder(num_tokens)
        idx = torch.div(positions, num_tokens, rounding_mode="floor")
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = (
            self.experts[expert_idx](current_state)
            * routing_weights[top_x, idx, None]
        )
        final_hidden_states.index_add_(
            0, top_x, current_hidden_states.to(hidden_states.dtype)
        )
    final_hidden_states = final_hidden_states.reshape(
        batch_size, sequence_length, hidden_dim
    )
    return final_hidden_states, router_logits


def install_moe_dispatch(model: nn.Module, mode: str) -> int:
    """Patch only the benchmarked HF implementation; leave other models intact."""
    if mode == "stock":
        return 0
    blocks = [block for block in model.modules()
              if type(block).__name__ == "Qwen3MoeSparseMoeBlock"]
    if not blocks:
        logging.getLogger(__name__).info("Teacher MoE dispatch: no compatible Qwen3 blocks")
        return 0
    try:
        transformers_version = version("transformers")
        expected_source_sha = EXPECTED_SOURCE_SHA_BY_VERSION.get(transformers_version)
        compatible = expected_source_sha is not None and len(blocks) == 48
        compatible = compatible and all(
            type(block).__module__ == "transformers.models.qwen3_moe.modeling_qwen3_moe"
            and "forward" not in vars(block)
            and hashlib.sha256(
                inspect.getsource(type(block).forward).encode()
            ).hexdigest()
            == expected_source_sha
            for block in blocks
        )
    except (OSError, TypeError):
        compatible = False
    if not compatible:
        logging.getLogger(__name__).warning(
            "Teacher MoE stable_sort fallback: HF version/source or block layout differs"
        )
        return 0
    for block in blocks:
        block.forward = functools.partial(qwen3_moe_forward_stable_sort, block)
    return len(blocks)
