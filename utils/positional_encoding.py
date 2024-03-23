import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """
    A module that computes sinusoidal position embeddings for transformer models.
    """

    def __init__(self, dim: int) -> None:
        """
        Initializes the PositionalEncoding class.

        Args:
            dim (int): The dimension of the positional encoding.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to the input tensor.

        Args:
            time (torch.Tensor): The input tensor representing time.

        Returns:
            torch.Tensor: The tensor after applying positional encoding.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.stack((embeddings.sin(), embeddings.cos()), dim=2).view(
            -1, self.dim
        )
        return embeddings


def apply_pos_encoding(
    x: torch.Tensor, pos_encoding: SinusoidalPositionEmbeddings
) -> torch.Tensor:
    """
    Applies positional encoding to the input tensor.

    Args:
        x (torch.Tensor): The input tensor.
        pos_encoding (SinusoidalPositionEmbeddings): The positional encoding function.

    Returns:
        torch.Tensor: The input tensor with positional encoding added.
    """
    pos_range = torch.arange(0, x.shape[1], 1).to(x.device)
    pos_enc_ind = pos_encoding(pos_range)
    pos_enc = pos_enc_ind.unsqueeze(0).repeat((x.shape[0], 1, 1))
    return x + pos_enc
