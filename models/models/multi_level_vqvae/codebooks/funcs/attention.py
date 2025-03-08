"""
This is a variation of Omkar's code implementation of his attention quantizer.
"""

import torch
import torch.nn as nn


class AttentionQuantizer(nn.Module):
    """
    Attention quantizer, according to Omkar this works well for quantizing images, so I wanted to test this in another
    setting. The issue is that the input this module is not really quantized, but it is a linear combination of the
    rest of the codebooks.
    """

    def __init__(self, num_tokens: int, token_dim: int, nheads: int) -> None:
        """
        Initializes an instance of the AttentionQuantizer class.

        Args:
            num_tokens (int): The number of tokens in the codebook.
            token_dim (int): The dimension of each token.
            nheads (int): The number of attention heads.
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.token_dim = token_dim

        self.codebook = nn.Embedding(self.num_tokens, self.token_dim)
        self.codebook.weight.data.uniform_(
            -1.0 / self.num_tokens, 1.0 / self.num_tokens
        )

        self.mha = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=nheads, dropout=0.1, batch_first=True
        )

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Args:
            queries (torch.Tensor): The input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, channels, height, width).
        """
        b, c, h, w = queries.shape

        z = queries.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(b, h * w, self.token_dim)

        kv = torch.repeat_interleave(
            self.codebook.weight.unsqueeze(0), repeats=b, dim=0
        )
        out, _ = self.mha(z_flattened, kv, kv, need_weights=False)

        out = out.permute(0, 2, 1).contiguous().reshape(b, c, h, w)
        return out
