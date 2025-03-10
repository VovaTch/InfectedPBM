import math

import torch
import torch.nn.functional as F

from .base import SampleScheduler


class LinearSampleScheduler(SampleScheduler):
    """
    Simple mask that performs random covering based on Bernoulli distribution, with
    p linear decreasing from 1 to 0.
    """

    def __init__(self, num_steps: int) -> None:
        """
        Initialize the scheduler with the given number of steps.

        Args:
            num_steps (int): The total number of steps for the scheduler.
        """
        super().__init__()
        self._num_steps = num_steps

    def sample(
        self, step: int, prev_mask: torch.Tensor, _: torch.Tensor
    ) -> torch.Tensor:
        if step > self._num_steps:
            raise ValueError("Step must be less than the number of steps.")
        cover_probability = 1 - step / self._num_steps
        new_mask = torch.rand_like(prev_mask) < cover_probability
        return new_mask


class LinearEntropyBatchSampleScheduler(SampleScheduler):
    """
    A linear sample scheduler that performs samples from batch, like the best
    one described in the LLaDa paper.
    """

    def __init__(self, batch_length: int, steps_per_batch: int) -> None:
        """
        Initialize the scheduler with the given batch length and steps per batch.

        Args:
            batch_length (int): The length of each batch.
            steps_per_batch (int): The number of steps in each batch.
        """
        super().__init__()
        self._batch_length = batch_length
        self._steps_per_batch = steps_per_batch

    def sample(
        self, step: int, prev_mask: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        new_mask = prev_mask.clone()

        attended_logits = logits[
            step // self._steps_per_batch : step // self._steps_per_batch
            + self._batch_length
        ]
        attended_masks = new_mask[
            step // self._steps_per_batch : step // self._steps_per_batch
            + self._batch_length
        ].clone()
        attended_entropy = (
            -F.softmax(attended_logits, dim=-1) * F.log_softmax(attended_logits, dim=-1)
        ).sum(dim=-1)
        _, sorted_indices = attended_entropy.sort(descending=True)
        tokens_to_uncover = math.ceil(
            step % self._steps_per_batch * (self._batch_length / self._steps_per_batch)
        )
        attended_masks[sorted_indices[:tokens_to_uncover]] = True
        new_mask[
            step // self._steps_per_batch : step // self._steps_per_batch
            + self._batch_length
        ] = attended_masks

        return new_mask
