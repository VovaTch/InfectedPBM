from __future__ import annotations
from typing_extensions import Self

from omegaconf import DictConfig

from common import registry
from models.multi_level_vqvae.blocks import VQ1D
from models.multi_level_vqvae.decoder.ripple import (
    RippleDecoder,
    RippleDecoderParameters,
)
from models.multi_level_vqvae.encoder import Encoder1D
from .ml_vqvae import MultiLvlVQVariationalAutoEncoder


@registry.register_model("ripl_vqvae")
class RippleVQVariationalAutoEncoder(MultiLvlVQVariationalAutoEncoder):

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
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

        # Encoder dim change list parameters
        encoder_channel_list, encoder_dim_changes = cls._compute_dim_change_lists(
            hidden_size, latent_depth, channel_change_dim_list
        )

        # Decoder parameters
        projected_input_size: int = model_cfg.input_size
        dec_input_size = projected_input_size // np.prod(channel_change_dim_list) * latent_depth  # type: ignore
        ripl_parameters = RippleDecoderParameters(
            input_dim=int(dec_input_size),
            hidden_dim=model_cfg.decoder_params.hidden_dim,
            mlp_num_layers=model_cfg.decoder_params.mlp_num_layers,
            output_dim=projected_input_size,
            ripl_hidden_dim=model_cfg.decoder_params.ripl_hidden_dim,
            ripl_num_layers=model_cfg.decoder_params.ripl_num_layers,
            ripl_coordinate_multipler=model_cfg.decoder_params.ripl_coordinate_multipler,
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
        decoder = RippleDecoder(ripl_parameters)

        # VQ parameter initialization
        vocabulary_size: int = model_cfg.vocabulary_size
        num_codebooks: int = model_cfg.num_codebooks

        vq_module = VQ1D(
            latent_depth, num_tokens=vocabulary_size, num_codebooks=num_codebooks
        )

        # loss aggregator
        if cfg.loss.aggregator.type != "none":
            loss_aggregator = registry.get_loss_aggregator(
                cfg.loss.aggregator.type
            ).from_cfg(cfg)
        else:
            loss_aggregator = None

        return cls(input_channels, encoder, decoder, vq_module, loss_aggregator)

    @staticmethod
    def _compute_dim_change_lists(
        hidden_size: int, latent_depth: int, channel_dim_change_list: list[int]
    ) -> tuple[list[int], list[int]]:

        encoder_channel_list = [
            hidden_size * (2 ** (idx + 1))
            for idx in range(len(channel_dim_change_list))
        ] + [latent_depth]
        encoder_dim_changes = channel_dim_change_list
        return encoder_channel_list, encoder_dim_changes
