import pytest
import torch
import torch.nn.functional as F

from skyrl.backends.skyrl_train.utils.distillation_utils import kl_loss


@pytest.mark.parametrize("reverse", [True, False])
def test_matches_torch_kl_div(reverse):
    # Independent oracle: F.kl_div(input, target, log_target=True) is
    # sum_v target_v * (log target_v - input_v), with both given as log-probs.
    torch.manual_seed(0)
    student = torch.randn(2, 4, 7, requires_grad=True)
    teacher = torch.randn(2, 4, 7)

    got = kl_loss(student, teacher, torch.ones(2, 4), reverse=reverse)

    log_q = F.log_softmax(student.detach(), dim=-1)
    log_p = F.log_softmax(teacher, dim=-1)
    args = (log_p, log_q) if reverse else (log_q, log_p)
    expected = F.kl_div(*args, log_target=True, reduction="none").sum(-1).mean()

    assert torch.allclose(got.detach(), expected, atol=1e-6)
    got.backward()
    assert torch.isfinite(student.grad).all()


def test_topk_supervises_only_the_selected_columns():
    # The defining property of the truncated path: gradient lands on the teacher's top-k and nowhere
    # else, and the value matches the KL of the two distributions renormalized over that set.
    torch.manual_seed(3)
    student = torch.randn(2, 3, 20, requires_grad=True)
    teacher = torch.randn(2, 3, 20)
    vals, idx = teacher.topk(4, dim=-1)

    loss = kl_loss(student, vals, torch.ones(2, 3), teacher_indices=idx)
    loss.backward()

    log_q = F.log_softmax(student.detach().gather(-1, idx), dim=-1)
    log_p = F.log_softmax(vals, dim=-1)
    expected = (log_q.exp() * (log_q - log_p)).sum(-1).mean()
    assert torch.allclose(loss.detach(), expected, atol=1e-6)

    touched = torch.zeros_like(student, dtype=torch.bool).scatter_(-1, idx, True)
    assert (student.grad[~touched] == 0).all()
    assert (student.grad[touched] != 0).any()
