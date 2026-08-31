# Analytic KL between a student and a frozen teacher, for on-policy distillation -- either direction,
# over the full vocab or a top-k subset of it. Replaces the sampled-token estimator
# (``compute_approx_kl`` on the decoded token), whose gradient reaches only one logit per position.

from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from jaxtyping import Float, Integer

from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean

# Teacher-side width: the full vocab, or k when distilling only a top-k subset of it.
TeacherValues = Float[torch.Tensor, "batch seq width"]
TeacherIndices = Integer[torch.Tensor, "batch seq width"]


def _per_token_kl(student_logits, teacher_values, reverse, teacher_indices, teacher_head):
    if teacher_head is not None:
        teacher_values = teacher_head(teacher_values)
    selected = student_logits if teacher_indices is None else student_logits.gather(-1, teacher_indices)
    log_q = F.log_softmax(selected.float(), dim=-1)
    log_p = F.log_softmax(teacher_values.float(), dim=-1)
    if reverse:
        return (log_q.exp() * (log_q - log_p)).sum(-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def kl_loss(
    student_logits: Float[torch.Tensor, "batch seq vocab"],
    teacher_values: TeacherValues,
    mask: Float[torch.Tensor, "batch seq"],
    reverse: bool = True,
    teacher_indices: Optional[TeacherIndices] = None,
    global_normalization_factor: Optional[torch.Tensor] = None,
    teacher_head: Optional[Callable] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Masked-mean KL between the student and a frozen teacher.

    With ``teacher_indices=None`` this is the exact full-vocab KL and ``teacher_values`` is
    ``[batch, seq, vocab]``. Otherwise both are ``[batch, seq, k]`` and the KL is taken over the
    teacher's top-k, renormalized -- which biases the loss *value* low, increasingly at high entropy,
    but leaves the gradient direction nearly intact. ``teacher_values`` may be logits or log-probs;
    they differ by a constant, which the renormalizing softmax discards.

    Args:
        student_logits: ``[batch, seq, vocab]``, requires grad.
        teacher_values: teacher logits/log-probs, detached. Width as above.
        mask: ``[batch, seq]``, 1 for supervised tokens.
        reverse: ``KL(student || teacher)`` when True (the on-policy distillation objective),
            ``KL(teacher || student)`` when False.
        teacher_indices: ``[batch, seq, k]`` vocab indices, or ``None`` for full vocab.
        global_normalization_factor: global valid-token count to use as the masked-mean denominator,
            for a correct reduction across micro-batches / DP ranks. ``None`` uses the local mean.
        teacher_head: projects ``teacher_values`` to logits. When set, ``teacher_values`` holds the
            teacher's ``[batch, seq, hidden]`` states and the projection runs inside each chunk, so
            full-vocab teacher logits are only ever materialized one chunk at a time.
        chunk_size: split the sequence into chunks of this size, each gradient-checkpointed, to bound
            peak memory. Numerically identical to computing the whole sequence at once.
    """
    assert not teacher_values.requires_grad, "distillation KL: teacher tensor must be detached"
    args = (reverse, teacher_indices, teacher_head)

    if chunk_size is None or student_logits.shape[1] <= chunk_size:
        per_token_kl = _per_token_kl(student_logits, teacher_values, *args)
        if global_normalization_factor is not None:
            return (per_token_kl * mask).sum() / global_normalization_factor.clamp(min=1.0)
        return masked_mean(per_token_kl, mask)

    masked_sum = 0.0
    for start in range(0, student_logits.shape[1], chunk_size):
        sl = slice(start, start + chunk_size)
        idx_chunk = None if teacher_indices is None else teacher_indices[:, sl]

        def chunk_masked_sum(s, t, m, _idx=idx_chunk):
            return (_per_token_kl(s, t, reverse, _idx, teacher_head) * m).sum()

        masked_sum = masked_sum + checkpoint.checkpoint(
            chunk_masked_sum, student_logits[:, sl], teacher_values[:, sl], mask[:, sl], use_reentrant=False
        )

    denom = global_normalization_factor if global_normalization_factor is not None else mask.sum()
    return masked_sum / denom.clamp(min=1.0)
