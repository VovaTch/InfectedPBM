from __future__ import annotations
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TokenDiffusionModel(nn.Module, ABC):
    """
    Base class for token generation diffusion models, specific implementations carry the architecture.
    """

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        conditional: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Abstract method to perform the forward pass of the model.

        Args:
            input (torch.Tensor): The input tensor to the model.
            conditional (torch.Tensor | None): An optional tensor for conditional input.
            mask (torch.Tensor): A tensor used to mask certain parts of the input.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        ...
