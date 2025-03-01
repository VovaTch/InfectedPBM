from typing import Protocol
import torch
import torch.nn as nn

from loss.aggregators import LossAggregator
from .decoder.base import Decoder
from ..base import Tokenizer


class CodeBook(Protocol):
    def embed_codebook(self, indices: torch.Tensor) -> torch.Tensor: ...


class InterfaceVQ1D(Protocol):
    def __call__(
        self, z_e: torch.Tensor, extract_losses: bool = False
    ) -> dict[str, torch.Tensor]: ...

    @property
    def vq_codebook(self) -> CodeBook: ...


class MultiLvlVQVariationalAutoEncoder(Tokenizer):
    """
    VQ VAE that takes a music sample and converts it into latent space, hopefully faithfully reconstructing it later.
    This latent space is then used for the lowest level sample generation in a DiT like fashion.
    """

    def __init__(
        self,
        input_channels: int,
        encoder: nn.Module,
        decoder: Decoder,
        vq_module: InterfaceVQ1D,
        loss_aggregator: LossAggregator | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the MultiLevelVQVAE model.

        Args:
            encoder (nn.Module): The encoder module.
            decoder (nn.Module): The decoder module.
            vq_module (InterfaceVQ1D): The vector quantization module.
            loss_aggregator (LossAggregator | None, optional): The loss aggregator module. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(**kwargs)

        # Parse arguments
        self.loss_aggregator = loss_aggregator
        self.encoder = encoder
        self.decoder = decoder
        self.vq_module = vq_module
        self.input_channels = input_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor using the VQ-VAE encoder.

        Args:
            x (torch.Tensor): The input tensor to be encoded.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        x_reshaped = (
            x.reshape((x.shape[0], -1, self.input_channels))
            .permute((0, 2, 1))
            .contiguous()
        )
        z_e = self.encoder(x_reshaped.float())
        return z_e

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor `x` and returns the indices of the quantized vectors.

        Args:
            x (torch.Tensor): The input tensor to be encoded.

        Returns:
            torch.Tensor: The indices of the quantized vectors.
        """
        z_e = self.encode(x)
        return self.vq_module(z_e)["indices"]

    def from_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decodes the input tensor `indices` into the output tensor `x_out`.

        Args:
            indices (torch.Tensor): The input tensor of indices of shape (batch_size, seq_len, 1).

        Returns:
            torch.Tensor: The decoded tensor of shape (batch_size, seq_len, input_channels).
        """
        z_q = self.vq_module.vq_codebook.embed_codebook(indices)
        quantized_outputs = z_q.transpose(1, 2).contiguous()
        x_out = self.decoder(quantized_outputs.transpose(1, 2).contiguous())
        return x_out

    def decode(
        self, z_e: torch.Tensor, origin_shape: tuple[int, int, int] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Decodes the input latent tensor `z_e` into the output tensor `x_out`.

        Args:
            z_e (torch.Tensor): The input latent tensor of shape (batch_size, latent_dim).
            origin_shape (tuple[int, int, int] | None, optional): The original shape of the input tensor.
                If None, it is inferred from the shape of `z_e`. Defaults to None.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: A tuple containing the output tensor `x_out` and a dictionary
                `total_output` that includes the output of the VQ module and other intermediate results.
        """
        vq_block_output = self.vq_module(z_e, extract_losses=True)
        x_out = self.decoder(vq_block_output["v_q"][:, -1, ...])

        if origin_shape is None:
            origin_shape = (int(z_e.shape[0]), self.input_channels, -1)

        x_out = x_out.permute((0, 2, 1)).contiguous().reshape(origin_shape)

        total_output = {**vq_block_output, "slice": x_out}

        return x_out, total_output

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass of the Multi-Level VQ-VAE model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the output tensors.
        """
        # origin_shape = x.shape
        z_e = self.encode(x)
        _, total_output = self.decode(z_e)  # type: ignore

        total_output.update({"z_e": z_e})
        return total_output

    @property
    def last_layer(self) -> nn.Module:
        """
        Returns the last layer of the model.

        Returns:
            nn.Module: The last layer of the model.
        """
        return self.decoder.last_layer
