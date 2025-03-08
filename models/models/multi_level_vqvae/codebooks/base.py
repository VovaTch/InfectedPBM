from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class CodeBook(ABC, nn.Module):

    @abstractmethod
    def embed_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Embeds the given indices using the codebook.

        Args:
            indices (torch.Tensor): The indices to be embedded.

        Returns:
            torch.Tensor: The embedded representation of the indices.
        """
        ...

    @abstractmethod
    def apply_codebook(
        self, x_in: torch.Tensor, code_sg: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies a codebook to the input tensor.

        Args:
            x_in (torch.Tensor): The input tensor.
            code_sg (bool, optional): Whether to use codebook for stochastic gradient. Defaults to False.

        Returns:
        *   tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed tensor and the codebook indices.
            reconstructed_tensor size BS x num_codebooks x idx_slice_size x token_dim
            indices size BS x idx_slice_size x num_codebooks
        """
        ...

    @abstractmethod
    def update_usage(self, min_enc: torch.Tensor) -> None:
        """
        Update the usage of the model based on the minimum encoding.

        Args:
            min_enc (torch.Tensor): The minimum encoding.

        Returns:
            None
        """
        ...

    @abstractmethod
    def reset_usage(self) -> None:
        """
        Resets the usage of the object.
        """
        ...

    @abstractmethod
    def random_restart(self) -> float:
        """
        Performs a random restart for the optimization algorithm.

        Returns:
            float: The average number of dead codes
        """
        ...
