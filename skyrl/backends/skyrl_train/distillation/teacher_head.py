# The teacher's unembedding, hosted on the student rank, so teacher hidden states can be turned into
# logits there. Costs `hidden_size` per token on the wire instead of `vocab_size`, ~59x less at
# Qwen3-4B. Expects post-final-norm hidden states, what the teacher's own head consumes.

from contextlib import contextmanager
from typing import Iterator, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_base import (
    materialize_full_tensor,
)


def _find_output_embedding(model: nn.Module) -> nn.Module:
    """The module projecting hidden states to vocab logits."""
    # Take the first level that yields a head.
    node, seen = model, set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        getter = getattr(node, "get_output_embeddings", None)
        head = getter() if callable(getter) else None
        if head is not None and getattr(head, "weight", None) is not None:
            return head
        node = getattr(node, "model", None) or getattr(node, "base_model", None)
    raise ValueError(f"{type(model).__name__} exposes no output embeddings")


def extract_lm_head(model: nn.Module) -> Float[torch.Tensor, "vocab hidden"]:
    """Unembedding matrix as a detached CPU tensor. Handles tied embeddings; collective under FSDP."""
    return materialize_full_tensor(_find_output_embedding(model).weight).detach().cpu()


@contextmanager
def capture_hidden_states(model: nn.Module) -> Iterator[List[torch.Tensor]]:
    """Post-final-norm hidden states feeding the output embedding, one entry per forward call."""
    captured: List[torch.Tensor] = []
    handle = _find_output_embedding(model).register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach())
    )
    try:
        yield captured
    finally:
        handle.remove()


class TeacherHead(nn.Module):
    """Kept beside the policy model, not as a submodule: the weight must stay out of the optimizer,
    FSDP's sharding plan, and checkpoints."""

    weight: Float[torch.Tensor, "vocab hidden"]

    def __init__(self, weight: Float[torch.Tensor, "vocab hidden"], dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.register_buffer("weight", weight.detach().to(dtype), persistent=False)

    @torch.no_grad()
    def forward(self, hidden_states: Float[torch.Tensor, "batch seq hidden"]) -> Float[torch.Tensor, "batch seq vocab"]:
        return F.linear(hidden_states.to(self.weight.dtype), self.weight)
