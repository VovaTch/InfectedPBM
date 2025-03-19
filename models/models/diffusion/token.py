from __future__ import annotations

import torch
import torch.nn as nn

from utils.positional_encoding import SinusoidalPositionEmbeddings, apply_pos_encoding

from .base import TokenDiffusionModel


class TokenDiffusionTransformer(TokenDiffusionModel):
    """
    Simple transformer model for token diffusion.
    """

    def __init__(
        self,
        vocab_size: int,
        cond_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initializes the token model with the given parameters.

        Args:
            vocab_size (int): The size of the vocabulary.
            cond_size (int): The size of the conditional vocabulary.
            hidden_size (int): The size of the hidden layers.
            num_layers (int): The number of transformer decoder layers.
            num_heads (int): The number of attention heads in each transformer decoder layer.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self._vocab_size = vocab_size
        self._cond_size = cond_size

        self._num_layers = num_layers
        self._num_heads = num_heads
        self._hidden_size = hidden_size
        self._dropout = dropout

        # Main token embeddings
        self._token_embedding = nn.Embedding(
            vocab_size + 1, hidden_size
        )  # +1 for masked inputs
        self._pos_embeddings = SinusoidalPositionEmbeddings(hidden_size)

        # Conditional token embeddings
        self._cond_embeddings = nn.Embedding(
            cond_size + 1, hidden_size
        )  # +1 for no conditional

        # Create transformer decoder layers
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
            norm_first=True,
        )
        self._transformer_decoder = nn.TransformerDecoder(
            transformer_decoder_layer, num_layers=num_layers
        )
        self._post_transformer_proj = nn.Linear(hidden_size, vocab_size + 1, bias=False)

        # Weight sharing
        self._post_transformer_proj.weight = self._token_embedding.weight

    def forward(
        self,
        input: torch.Tensor,
        conditional: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward method for the transformer model.

        Args:
            input (torch.Tensor): index tensor, size (batch_size, seq_len, num_codebooks)
            conditional (torch.Tensor | None): conditional tensor indices, size (batch_size, cond_len).
                Defaults to None, which means no conditionals.
            mask (torch.Tensor | None): mask tensor for the input for generation, size (batch_size, seq_len).
                Defaults to None, which means all of it is true, .

        Returns:
            torch.Tensor: _description_
        """

        batch_size, seq_len, num_codebooks = input.shape

        # Set conditionals
        if conditional is None:
            cond_embedding = self._cond_embeddings.forward(
                torch.ones((input.shape[0], 1), device=input.device).int()
                * self._cond_size
            )
        else:
            cond_embedding = self._cond_embeddings.forward(conditional)

        if cond_embedding.dim() == 2:
            cond_embedding = cond_embedding.unsqueeze(1)

        # Set mask
        if mask is None:
            mask = torch.ones(
                (input.shape[0], input.shape[1]), device=input.device
            ).bool()
        mask = mask.clone()

        masked_inputs = input.clone()
        masked_inputs[mask] = self._vocab_size

        # Get embeddings + positional encodings
        token_embedding = self._token_embedding.forward(masked_inputs).flatten(
            start_dim=1, end_dim=2
        )
        token_embedding = apply_pos_encoding(token_embedding, self._pos_embeddings)

        # Transformer
        t_outputs = self._transformer_decoder.forward(token_embedding, cond_embedding)

        # Project to vocab size
        output = self._post_transformer_proj.forward(t_outputs)
        return output.reshape(
            batch_size, seq_len, num_codebooks, self._vocab_size + 1
        ).contiguous()
