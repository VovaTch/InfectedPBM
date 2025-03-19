from __future__ import annotations

import torch
import torch.nn as nn

from .base import Tokenizer


class InnerTokenizer(nn.Module):
    """
    Inner tokenizer module that takes a token sequence and compresses it via a tokenizer.
    """

    def __init__(
        self, tokenizer: Tokenizer, outer_vocab_size: int, emb_dim: int
    ) -> None:
        """
        Initializes the inner model with the specified tokenizer, vocabulary size, and embedding dimension.

        Args:
            tokenizer (Tokenizer): The tokenizer instance used for text processing.
            outer_vocab_size (int): The size of the outer vocabulary.
            emb_dim (int): The dimensionality of the embedding vectors.
        """
        super().__init__()
        self._tokenizer = tokenizer
        self._outer_vocab_size = outer_vocab_size
        self._outer_vocab_embedding = nn.Embedding(outer_vocab_size, emb_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor using the outer vocabulary embedding and tokenizer.

        Args:
            x (torch.Tensor): The input tensor to be encoded.

        Returns:
            torch.Tensor: The encoded tensor after applying the outer vocabulary embedding
                            and the tokenizer.
        """
        emb = self._outer_vocab_embedding(x)
        return self._tokenizer.encode(emb)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenizes the input tensor using the outer vocabulary embedding and tokenizer.

        Args:
            x (torch.Tensor): Input tensor to be tokenized.

        Returns:
            torch.Tensor: Tokenized representation of the input tensor.
        """
        emb = self._outer_vocab_embedding(x)
        return self._tokenizer.tokenize(emb)

    def from_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts a tensor of tokens into their corresponding representations.

        Args:
            x (torch.Tensor): A tensor containing token indices.

        Returns:
            torch.Tensor: A tensor containing the representations of the tokens.
        """
        return self._tokenizer.from_tokens(x)

    def decode(
        self, z_e: torch.Tensor, origin_shape: tuple[int, int, int] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Decodes the given encoded tensor `z_e` back into its original representation.

        Args:
            z_e (torch.Tensor): The encoded tensor to be decoded.
            origin_shape (tuple[int, int, int] | None, optional): The original shape of the data
                before encoding. If provided, it will be used to reshape the decoded output.
                Defaults to None.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: A tuple containing the decoded tensor
            and a dictionary of additional tensors related to the decoding process.
        """
        return self._tokenizer.decode(z_e, origin_shape)

    @property
    def tokenizer(self) -> Tokenizer:
        """
        Provides access to the tokenizer instance.

        Returns:
            Tokenizer: The tokenizer associated with this object.
        """
        return self._tokenizer

    @property
    def last_layer(self) -> nn.Module:
        """
        Returns the last layer of the tokenizer module.

        This property provides access to the last layer of the `_tokenizer`
        attribute, which is expected to be an instance of a module containing
        a `last_layer` attribute.

        Returns:
            nn.Module: The last layer of the tokenizer.
        """
        return self._tokenizer.last_layer

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the outputs of the tokenizer
            and the input embeddings. The dictionary includes:
                - "input_embedding": The embedding tensor for the input tokens.
                - Other keys and values as produced by the tokenizer.
        """
        emb = self._outer_vocab_embedding(x)
        total_outputs = self._tokenizer(emb)
        total_outputs.update({"input_embedding": emb})
        return total_outputs
