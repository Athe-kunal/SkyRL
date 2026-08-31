"""Teacher -> student distillation across non-colocated ref and policy actor groups.

Run with:
uv run --isolated --extra dev --extra fsdp -- pytest tests/backends/skyrl_train/gpu/gpu_ci/test_distillation_e2e.py -v

Needs 2 GPUs: the ref (teacher) and policy (student) get one each, so the teacher signal really
crosses between actor groups rather than being produced and consumed in one process.
"""

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import loss_fn_outputs_to_tensor
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker, RefWorker
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg

SMALL = "Qwen/Qwen2.5-1.5B-Instruct"  # hidden 1536
LARGE = "Qwen/Qwen3-4B-Instruct-2507"  # hidden 2560, same 151936 vocab
HIDDEN = {SMALL: 1536, LARGE: 2560}


def get_test_config(student: str, teacher: str, topk, teacher_unembedding: bool) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = student
    cfg.trainer.ref.model.path = teacher
    cfg.trainer.strategy = "fsdp"
    cfg.trainer.logger = "console"
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.placement.ref_num_gpus_per_node = 1
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.algorithm.policy_loss_type = "none"
    cfg.trainer.algorithm.distillation.enabled = True
    cfg.trainer.algorithm.distillation.topk = topk
    cfg.trainer.algorithm.distillation.teacher_unembedding = teacher_unembedding
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    # One GPU per model, so the optimizer state of a 4B student has to live on the host.
    cfg.trainer.policy.fsdp_config.cpu_offload = True
    cfg.trainer.ref.fsdp_config.cpu_offload = False
    validate_cfg(cfg)
    return cfg


def make_group(worker_cls, cfg, path):
    group = PPORayActorGroup(
        cfg.trainer,
        num_nodes=1,
        num_gpus_per_node=1,
        ray_actor_type=worker_cls,
        pg=None,
        num_gpus_per_actor=1,
        colocate_all=False,
        sequence_parallel_size=1,
    )
    ray.get(group.async_init_model(path))
    return group


def make_batch(batch_size=2, seq_len=16, num_actions=8) -> TrainingInputBatch:
    torch.manual_seed(0)
    batch = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 100, (batch_size, seq_len)),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=int),
            "action_log_probs": 0.4 * torch.ones((batch_size, num_actions)),
            "base_action_log_probs": 0.3 * torch.ones((batch_size, num_actions)),
            "advantages": 0.6 * torch.ones((batch_size, num_actions)),
            "returns": 0.5 * torch.ones((batch_size, num_actions)),
            "values": 0.5 * torch.ones((batch_size, num_actions)),
            "loss_mask": torch.ones((batch_size, num_actions), dtype=int),
            "response_mask": torch.ones((batch_size, num_actions), dtype=int),
        }
    )
    batch.metadata = {"response_length": num_actions, "global_step": 0}
    return batch


@pytest.mark.parametrize(
    "student, teacher, topk, teacher_unembedding",
    [
        (SMALL, LARGE, 64, False),
        (SMALL, LARGE, None, False),
        (SMALL, LARGE, None, True),
        # Reversed: the teacher's hidden size need not match the student's, only the vocab must.
        (LARGE, SMALL, None, True),
    ],
    ids=["topk", "full_vocab_logits", "full_vocab_unembedding", "full_vocab_unembedding_reversed"],
)
def test_teacher_signal_crosses_actor_groups(ray_init_fixture, student, teacher, topk, teacher_unembedding):
    """The ref emits a per-token teacher signal on its own GPU, it reaches the policy on another
    GPU, and the distillation loss there produces a finite non-zero gradient."""
    cfg = get_test_config(student, teacher, topk, teacher_unembedding)
    policy = make_group(PolicyWorker, cfg, student)
    ref = make_group(RefWorker, cfg, teacher)

    if teacher_unembedding:
        weights = ray.get(ref.async_run_ray_method("pass_through", "get_teacher_lm_head"))
        assert tuple(weights[0].shape) == (151936, HIDDEN[teacher])
        ray.get(policy.async_run_ray_method("pass_through", "set_teacher_head", weights[0]))

    batch = make_batch()
    ref_out = ray.get(ref.async_run_ray_method("mesh", "forward", batch))[0]
    assert ref_out.tensors is not None, "ref emitted no teacher tensors"

    teacher_values = ref_out.tensors["teacher_values"]
    expected_width = topk or (HIDDEN[teacher] if teacher_unembedding else 151936)
    assert tuple(teacher_values.shape) == (2, 8, expected_width)
    assert torch.isfinite(teacher_values.float()).all()
    # The ref still produces its usual logprobs alongside the teacher signal.
    assert loss_fn_outputs_to_tensor(ref_out.loss_fn_outputs, key="logprobs").shape == (2, 8)

    for key, tensor in ref_out.tensors.items():
        batch[key] = tensor

    status = ray.get(policy.async_run_ray_method("mesh", "forward_backward", batch))[0]
    kl = status.metrics["distillation_kl"]
    assert kl > 0, f"expected a positive KL between different models, got {kl}"

    grad_norm = ray.get(policy.async_run_ray_method("pass_through", "optim_step"))[0]
    assert grad_norm is not None and grad_norm > 0, f"no gradient reached the policy: {grad_norm}"
