import torch
import torch.nn as nn


class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, hidden_dim: int, max_seq_length: int = 32768) -> None:
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError("The hidden dimension must be divisible by 2")

        self._hidden_dim = hidden_dim
        self._max_seq_length = max_seq_length

        inv_frequency = 1.0 / (
            1e4 ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim)
        )
        self.register_buffer("inv_frequency", inv_frequency)
        self.register_buffer("position_ids", torch.arange(max_seq_length).float())

    def _compute_rope_embeddings(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sin and cos embeddings for rotary position encoding.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)

        Returns:
            Tuple of (sin, cos) tensors for position embedding
        """
        seq_len = x.shape[1]
        position_ids = self.position_ids[:seq_len].to(x.device)  # type: ignore

        theta = torch.outer(position_ids, self.inv_frequency.to(x.device))  # type: ignore

        sin_pos = torch.sin(theta)
        cos_pos = torch.cos(theta)

        sin_pos = torch.repeat_interleave(sin_pos, 2, dim=-1)
        cos_pos = torch.repeat_interleave(cos_pos, 2, dim=-1)

        return sin_pos, cos_pos

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate the input tensor by 90 degrees along the last dimension.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: rotated tensor
        """
        # Split the tensor along the last dimension into [a, b, a, b...] kind of form along the last dim
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary positional embeddings for the input tensor.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, num_heads, head_dim)

        Returns:
            Tuple of (q, k) rotated along the last dimension in pairs
        """
        sin_pos, cos_pos = self._compute_rope_embeddings(q)

        sin_pos = sin_pos.reshape(q.shape[-3:]).unsqueeze(0)  # (1, seq_len, 1, dim)
        cos_pos = cos_pos.reshape(q.shape[-3:]).unsqueeze(0)  # (1, seq_len, 1, dim)

        q_embed = (q * cos_pos) + (self._rotate_half(q) * sin_pos)
        k_embed = (k * cos_pos) + (self._rotate_half(k) * sin_pos)

        return q_embed, k_embed
