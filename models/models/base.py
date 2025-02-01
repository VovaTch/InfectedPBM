from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Tokenizer(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor x into a compressed representation.

        Args:
            x (torch.Tensor): The input tensor to be encoded.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        ...

    @abstractmethod
    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor `x` into indices.

        Args:
            x (torch.Tensor): The input tensor to be encoded.

        Returns:
            torch.Tensor: The encoded tensor with indices.
        """
        ...

    @abstractmethod
    def from_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the input tensor `x` from indices.

        Args:
            x (torch.Tensor): The input tensor to be decoded.

        Returns:
            torch.Tensor: The decoded tensor.
        """
        ...

    @abstractmethod
    def decode(
        self, z_e: torch.Tensor, origin_shape: tuple[int, int, int] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Decodes the given latent tensor `z_e` into the original image representation.

        Args:
            z_e (torch.Tensor): The latent tensor to be decoded.
            origin_shape (tuple[int, int, int] | None, optional): The shape of the original image. Defaults to None.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: A tuple containing the decoded image tensor and additional
            information.
        """
        ...
