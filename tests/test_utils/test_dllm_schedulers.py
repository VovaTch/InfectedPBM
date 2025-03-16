import torch

from utils.sample_schedulers.linear import (
    LinearEntropyBatchSampleScheduler,
    LinearSampleScheduler,
)


def test_linear_random_scheduler() -> None:
    num_steps = 100
    p_ber = 0.5
    scheduler = LinearSampleScheduler(num_steps)
    prev_mask = torch.bernoulli(torch.ones((3, 64, 4)) * p_ber)
    for step in range(num_steps):
        mask = scheduler.sample(step, prev_mask, torch.ones((3, 64, 4)))
        assert mask.shape == prev_mask.shape
        prev_mask = mask


def test_linear_entropy_batch_scheduler() -> None:
    scheduler = LinearEntropyBatchSampleScheduler(batch_length=16, steps_per_batch=5)
    prev_mask = torch.bernoulli(torch.ones((64)) * 0.0).bool()
    logits = torch.randn((64, 8))
    for step in range(20):
        mask = scheduler.sample(step, prev_mask, logits)
        assert mask.shape == prev_mask.shape
        prev_mask = mask
