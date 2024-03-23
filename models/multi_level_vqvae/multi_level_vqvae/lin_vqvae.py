from __future__ import annotations
from typing_extensions import Self

import numpy as np
from omegaconf import DictConfig

from common import registry
from models.multi_level_vqvae.blocks import VQ1D
from models.multi_level_vqvae.decoder.simple import ExpandingMLPDecoder
from models.multi_level_vqvae.encoder import Encoder1D
from .ml_vqvae import MultiLvlVQVariationalAutoEncoder


@registry.register_model("lin_vqvae")
class LinearDecVariationalAutoEncoder(MultiLvlVQVariationalAutoEncoder):
    """
    Variation of the multi-level VQVAE that uses
    an MLP decoder instead of 1D convolution decoder.
    """

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Create an instance of the class using the provided configuration.

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

        # VQ parameter initialization
        vocabulary_size: int = model_cfg.vocabulary_size
        num_codebooks: int = model_cfg.num_codebooks

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
            _,
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

        # Decoder parameters
        projected_input_size: int = model_cfg.input_size
        dec_input_size = projected_input_size // np.prod(channel_change_dim_list) * latent_depth  # type: ignore
        num_layers = model_cfg.decoder_num_layers
        decoder = ExpandingMLPDecoder(
            hidden_dim=hidden_size,
            input_len=int(dec_input_size),
            output_multiplier=int(np.prod(decoder_dim_changes)),
            num_layers=num_layers,
            activation_type=activation_type,
            output_channels=input_channels,
        )

        vq_module = VQ1D(
            latent_depth, num_tokens=vocabulary_size, num_codebooks=num_codebooks
        )

        return cls(input_channels, encoder, decoder, vq_module, loss_aggregator)
