from typing import Literal

import torch
import torch.nn as nn

from ...moe_attention_layers.attention import FlashRotarySelfAttention
from ...moe_attention_layers.moe_transformer_block import (
    MoETransformerDecoderBlock,
)
from ...moe_attention_layers.rotary_embeddings import (
    RotaryPositionalEmbeddings,
)
from .attention_stft import ISTFT


class MixtureOfExpertsRotaryStftDecoder(nn.Module):
    """
    Initializes a Mixture of Experts Transformer Decoder with rotary positional embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_experts: int,
        top_k_gating: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        max_seq_len: int = 32768,
        ff_hidden_dim: int = 2048,
        norm_type: Literal["rmsnorm", "layernorm"] = "layernorm",
        dropout: float = 0.1,
        expansion_factor: int = 1,
        padding: Literal["center", "same"] = "same",
        use_causal: bool = True,
    ) -> None:
        """
        Initializes the MoE-STFT decoder.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layers. Must be even.
            num_heads (int): Number of attention heads in the transformer.
            num_layers (int): Number of transformer layers.
            num_experts (int): Number of experts in the MoE layer.
            top_k_gating (int): Number of top experts to use in MoE gating.
            n_fft (int): Number of FFT components.
            hop_length (int): Hop length for STFT.
            win_length (int): Window length for STFT.
            max_seq_len (int, optional): Maximum sequence length. Defaults to 32768.
            ff_hidden_dim (int, optional): Dimension of the feed-forward hidden layer. Defaults to 2048.
            norm_type (Literal["rmsnorm", "layernorm"], optional): Type of normalization to use.
                Defaults to "layernorm".
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            expansion_factor (int, optional): The expansion factor for the initial convolutional layer.
                Default is 1.
            padding (Literal["center", "same"], optional): Padding type for STFT. Defaults to "same".
            use_causal (bool, optional): Whether to use causal attention. Defaults to True.

        Raises:
            ValueError: If `hidden_dim` is not even.
            ValueError: If `hop_length` is greater than `win_length`.
        """
        super().__init__()

        # Transformer parameters
        if hidden_dim % 2 != 0:
            raise ValueError("Hidden dimension must be even.")
        if hop_length > win_length:
            raise ValueError("Hop length must be less than or equal to window length.")
        self._hidden_dim = hidden_dim
        self._input_dim = input_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout
        self._num_experts = num_experts
        self._top_k_gating = top_k_gating
        self._max_seq_len = max_seq_len
        self._ff_hidden_dim = ff_hidden_dim
        self._norm_type = norm_type
        self._use_causal = use_causal

        # STFT parameters
        self._before_istft_dim = n_fft // 2 + 1
        self.istft = ISTFT(n_fft, hop_length, win_length, padding)
        self._in_projection = nn.ConvTranspose1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=expansion_factor,
            stride=expansion_factor,
        )
        self._before_istft_projection = nn.Linear(
            hidden_dim, self._before_istft_dim * 2
        )

        # Transformers
        rotary_embedding = RotaryPositionalEmbeddings(hidden_dim, max_seq_len)
        self._transformer_layers = nn.ModuleList()
        for idx in range(self._num_layers):
            attention = FlashRotarySelfAttention(
                hidden_dim,
                num_heads,
                pos_embedding=rotary_embedding if idx == 0 else None,
                dropout=dropout,
            )
            transformer_decoder_layer = MoETransformerDecoderBlock(
                hidden_dim=hidden_dim,
                attention=attention,
                norm_type=norm_type,
                num_experts=num_experts,
                top_k_gating=top_k_gating,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout,
            )
            self._transformer_layers.append(transformer_decoder_layer)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MixtureOfExpertsRotaryStftDecoder module.

        Args:
            z (torch.Tensor): input tensor, size BS x C x Len

        Returns:
            torch.Tensor: output tensor, size BS x 1 x OutLen
        """
        z = self._in_projection.forward(z).transpose(1, 2).contiguous()
        for transformer_layer in self._transformer_layers:
            z = transformer_layer(
                z,
                mask=(
                    None
                    if self._use_causal
                    else torch.ones(
                        (
                            z.shape[0],
                            self._num_heads,
                            z.shape[1],
                            z.shape[1],
                        ),
                        dtype=torch.bool,
                        device=z.device,
                    )
                ),
            )
        z = self._before_istft_projection(z)
        z = z[..., : self._before_istft_dim] + 1j * z[..., self._before_istft_dim :]
        z = self.istft(z.transpose(1, 2).contiguous())
        return z.unsqueeze(1)
