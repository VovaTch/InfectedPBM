from abc import ABC, abstractmethod

import torch


class SampleScheduler(ABC):

    @property
    @abstractmethod
    def num_steps(self) -> int:
        """
        Returns the number of steps in the sampling process.

        Returns:
            int: The number of steps.
        """
        ...

    @abstractmethod
    def sample(
        self, step: int, prev_mask: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples a new mask based on the current step, previous mask, and logits.

        Args:
            step (int): The current step in the sampling process.
            prev_mask (torch.Tensor): The previous mask tensor.
            logits (torch.Tensor): The logits tensor used for sampling.

        Returns:
            torch.Tensor: The newly sampled mask tensor.
        """
        ...
