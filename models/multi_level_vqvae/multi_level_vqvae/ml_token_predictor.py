from __future__ import annotations
from typing import TYPE_CHECKING
from typing_extensions import Self

from omegaconf import DictConfig
import torch
import torch.nn as nn

from common import registry
from models.multi_level_vqvae.blocks import VQ1D
from models.multi_level_vqvae.decoder.conv import Decoder1D
from models.multi_level_vqvae.encoder import Encoder1D
from utils.waveform_tokenization import dequantize_waveform_256
from .ml_vqvae import MultiLvlVQVariationalAutoEncoder
from .interface import InterfaceVQ1D

if TYPE_CHECKING:
    from loss.aggregators import LossAggregator


@registry.register_model("token_predictor")
class TokenPredictorVQVAE(MultiLvlVQVariationalAutoEncoder):

    def __init__(
        self,
        input_channels: int,
        conv_out_channels: int,
        encoder: nn.Module,
        decoder: nn.Module,
        vq_module: InterfaceVQ1D,
        loss_aggregator: "LossAggregator | None" = None,
        **kwargs,
    ) -> None:
        super().__init__(
            input_channels, encoder, decoder, vq_module, loss_aggregator, **kwargs
        )
        self.activation = nn.GELU()
        self.conv_out_channels = conv_out_channels
        self.output_projection = nn.Linear(conv_out_channels, 256 * self.input_channels)

    def decode(
        self, z_e: torch.Tensor, origin_shape: tuple[int, int, int] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        if origin_shape is None:
            origin_shape = (int(z_e.shape[0]), self.conv_out_channels, -1)

        x_pre_cls, total_outputs = super().decode(z_e, origin_shape)
        logits_out = (
            self.output_projection(
                self.activation(x_pre_cls).transpose(1, 2).contiguous()
            )
            .transpose(1, 2)
            .contiguous()
        )
        logits_out = logits_out.view((x_pre_cls.shape[0], 256, self.input_channels, -1))

        x_out_q = torch.argmax(logits_out, dim=1)
        x_out = dequantize_waveform_256(x_out_q)
        total_outputs["pred_logits"] = logits_out
        total_outputs["slice"] = x_out
        return x_out, total_outputs

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
        conv_out_channels: int = model_cfg.conv_out_channels
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
            input_channels=conv_out_channels,
            kernel_size=decoder_kernel_size,
            dim_add_kernel_add=decoder_dim_change_kernel_add,
            num_res_block_conv=num_res_block_conv,
            dilation_factor=dilation_factor,
            activation_type=activation_type,
        )

        vq_module = VQ1D(
            latent_depth, num_tokens=vocabulary_size, num_codebooks=num_codebooks
        )

        return cls(
            input_channels,
            conv_out_channels,
            encoder,
            decoder,
            vq_module,
            loss_aggregator,
        )
