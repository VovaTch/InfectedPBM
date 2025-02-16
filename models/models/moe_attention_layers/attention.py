from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashRotarySelfAttention(nn.Module):
    """
    Simple Flash Attention module with rotary embeddings.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        pos_embedding: nn.Module | None = None,
        dropout: float = 0.1,
    ) -> None:
        """
        Initializes the attention layer.

        Args:
            hidden_dim (int): The dimension of the hidden layer.
            n_heads (int): The number of attention heads.
            pos_embedding (nn.Module | None, optional): The positional embedding module. Defaults to None.
            dropout (float, optional): The dropout rate. Defaults to 0.1.

        """
        super().__init__()
        self._hidden_dim = hidden_dim
        self._n_heads = n_heads
        self._head_dim = hidden_dim // n_heads
        self._scaling = self._head_dim**-0.5
        self._pos_embedding = pos_embedding

        self.qvk_linear = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass for the FlashRotaryAttention module.

        Args:
            x (torch.Tensor): tensor size BS x L x C

        Returns:
            torch.Tensor: tensor size BS x L x C
        """
        batch, seq, hidden_dim = x.size()
        k, q, v = self.qvk_linear.forward(x).chunk(3, dim=-1)  # (BS x L x C) x 3

        # Apply rotary embedding if applicable
        if self._pos_embedding is not None:
            k, q = self._pos_embedding(k.unsqueeze(2), q.unsqueeze(2))

        k = (
            k.contiguous()
            .view(batch, seq, self._n_heads, self._head_dim)
            .transpose(1, 2)
        )  # (BS x H x L x C) -> (BS x H x L x C)
        q = (
            q.contiguous()
            .view(batch, seq, self._n_heads, self._head_dim)
            .transpose(1, 2)
        )  # (BS x H x L x C) -> (BS x H x L x C)
        v = (
            v.contiguous()
            .view(batch, seq, self._n_heads, self._head_dim)
            .transpose(1, 2)
        )  # (BS x H x L x C) -> (BS x H x L x C)

        attention_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            scale=self._scaling,
            dropout_p=self.dropout.p,
            attn_mask=mask,
            is_causal=mask is None,
        )
        attention_output = self.dropout.forward(attention_output)

        x = (
            attention_output.transpose(1, 2).contiguous().view(batch, seq, hidden_dim)
        )  # (BS x H x L x C) -> (BS x L x C)

        x = self.proj(x)
        return self.dropout(x)
