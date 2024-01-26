from __future__ import annotations
from typing import Protocol, Self, TYPE_CHECKING
from omegaconf import DictConfig

import torch
import torch.nn as nn

from .blocks import VQ1D
from .encoder import Encoder1D
from .decoder import Decoder1D
from common import registry
from ..base import Codebook, Tokenizer

if TYPE_CHECKING:
    from loss.aggregators import LossAggregator


class InterfaceVQ1D(Protocol):
    """
    Interface for the VQ1D class.
    """

    def __call__(
        self, z_e: torch.Tensor, extract_losses: bool = False
    ) -> dict[str, torch.Tensor]:
        ...

    @property
    def vq_codebook(self) -> Codebook:
        ...


@registry.register_model("multi_lvl_vqvae")
class MultiLvlVQVariationalAutoEncoder(Tokenizer):
    """
    VQ VAE that takes a music sample and converts it into latent space, hopefully faithfully reconstructing it later.
    This latent space is then used for the lowest level sample generation in a DiT like fashion.
    """

    def __init__(
        self,
        input_channels: int,
        encoder: nn.Module,
        decoder: nn.Module,
        vq_module: InterfaceVQ1D,
        loss_aggregator: "LossAggregator | None" = None,
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
        x_reshaped = x.reshape((x.shape[0], -1, self.input_channels)).permute((0, 2, 1))
        z_e = self.encoder(x_reshaped)
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
        z_q = z_q.permute((0, 2, 1))
        x_out = self.decoder(z_q)
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
        x_out = self.decoder(vq_block_output["v_q"])

        if origin_shape is None:
            origin_shape = (int(z_e.shape[0]), self.input_channels, -1)

        x_out = x_out.permute((0, 2, 1)).reshape(origin_shape)

        total_output = {**vq_block_output, "output": x_out}

        return x_out, total_output

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass of the Multi-Level VQ-VAE model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the output tensors.
        """
        origin_shape = x.shape
        z_e = self.encode(x)
        _, total_output = self.decode(z_e, origin_shape=tuple(origin_shape))  # type: ignore

        loss_target = {"music_slice": x, "z_e": z_e}

        if self.loss_aggregator is not None:
            total_loss = self.loss_aggregator(total_output, loss_target)
            total_output.update(
                {"total_loss": total_loss.total, **total_loss.individuals}
            )

        return total_output

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Create an instance of the class from a configuration dictionary.

        Args:
            cfg (DictConfig): The configuration dictionary.

        Returns:
            Self: An instance of the class.
        """

        model_cfg: DictConfig = cfg.model
        hidden_size: int = model_cfg.hidden_size
        latent_depth: int = model_cfg.latent_depth
        channel_change_dim_list: list[int] = model_cfg.channel_dim_change_list

        # Encoder parameter initialization
        input_channels: int = model_cfg.input_channels
        encoder_kernel_size: int = model_cfg.encoder_kernel_size
        num_res_block_conv: int = model_cfg.num_res_block_conv
        dilation_factor: int = model_cfg.dilation_factor
        encoder_dim_change_kernel_size: int = model_cfg.encoder_dim_change_kernel_size
        activation_type: str = model_cfg.activation_type

        # Decoder parameter initialization
        decoder_kernel_size: int = model_cfg.decoder_kernel_size
        decoder_dim_change_kernel_add: int = model_cfg.decoder_dim_change_kernel_add

        # VQ parameter initialization
        vocabulary_size: int = model_cfg.vocabulary_size

        # loss aggregator
        if cfg.loss.aggregator.type != "none":
            loss_aggregator = registry.get_loss_aggregator(
                cfg.loss.aggregator.type
            ).from_cfg(cfg)
        else:
            loss_aggregator = None

        # dim change lists
        (
            encoder_channel_list,
            encoder_dim_changes,
            decoder_channel_list,
            decoder_dim_changes,
        ) = cls._compute_dim_change_lists(
            hidden_size,
            latent_depth,
            channel_change_dim_list,
        )

        encoder = Encoder1D(
            channel_list=encoder_channel_list,
            dim_change_list=encoder_dim_changes,
            input_channels=input_channels,
            kernel_size=encoder_kernel_size,
            num_res_block_conv=num_res_block_conv,
            dilation_factor=dilation_factor,
            dim_change_kernel_size=encoder_dim_change_kernel_size,
            activation_type=activation_type,
        )

        decoder = Decoder1D(
            channel_list=decoder_channel_list,
            dim_change_list=decoder_dim_changes,
            input_channels=input_channels,
            kernel_size=decoder_kernel_size,
            dim_add_kernel_add=decoder_dim_change_kernel_add,
            num_res_block_conv=num_res_block_conv,
            dilation_factor=dilation_factor,
            activation_type=activation_type,
        )

        vq_module = VQ1D(latent_depth, num_tokens=vocabulary_size)

        return cls(input_channels, encoder, decoder, vq_module, loss_aggregator)

    @staticmethod
    def _compute_dim_change_lists(
        hidden_size: int, latent_depth: int, channel_dim_change_list: list[int]
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Compute the lists of channel and dimension changes for the encoder and decoder.

        Args:
            hidden_size (int): The hidden size of the VQ-VAE model.
            latent_depth (int): The depth of the latent space.
            channel_dim_change_list (list[int]): The list of channel dimension changes.

        Returns:
            tuple[list[int], list[int], list[int], list[int]]: A tuple containing the lists of encoder channel changes,
                encoder dimension changes, decoder channel changes, and decoder dimension changes.
        """
        encoder_channel_list = [
            hidden_size * (2 ** (idx + 1))
            for idx in range(len(channel_dim_change_list))
        ] + [latent_depth]
        encoder_dim_changes = channel_dim_change_list
        decoder_channel_list = [latent_depth] + [
            hidden_size * (2 ** (idx + 1))
            for idx in reversed(range(len(channel_dim_change_list)))
        ]
        decoder_dim_changes = list(reversed(channel_dim_change_list))
        return (
            encoder_channel_list,
            encoder_dim_changes,
            decoder_channel_list,
            decoder_dim_changes,
        )
