from typing_extensions import Self

from omegaconf import DictConfig
import torch

from common import registry
from models.multi_level_vqvae.blocks import VQ1D
from models.multi_level_vqvae.decoder.transformer import (
    TransformerMusicDecoder,
    TransformerParameters,
)
from models.multi_level_vqvae.encoder import Encoder1D
from models.multi_level_vqvae.multi_level_vqvae.ml_token_predictor import (
    TokenPredictorVQVAE,
)
from utils.waveform_tokenization import dequantize_waveform_256


@registry.register_model("atn_token_predictor")
class AtnTokenPredictorVQVAE(TokenPredictorVQVAE):

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
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
        slice_length: int = model_cfg.slice_length

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
            _,
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

        decoder_params = TransformerParameters.from_cfg(cfg)
        decoder = TransformerMusicDecoder(
            encoder_channel_list[-1], decoder_params, 256, slice_length
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

    def decode(
        self, z_e: torch.Tensor, origin_shape: tuple[int, int, int] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        vq_block_output = self.vq_module(z_e, extract_losses=True)
        # logits_out = self.decoder(vq_block_output["v_q"][:, -1, ...])
        logits_out = self.decoder(
            vq_block_output["v_q"].flatten(start_dim=1, end_dim=-2)
        )

        if origin_shape is None:
            origin_shape = (int(z_e.shape[0]), self.input_channels, -1)

        logits_out = logits_out.view((z_e.shape[0], 256, self.input_channels, -1))

        total_outputs = {**vq_block_output}

        x_out_q = torch.argmax(logits_out, dim=1).view(origin_shape)
        x_out = dequantize_waveform_256(x_out_q)
        total_outputs["pred_logits"] = logits_out
        total_outputs["slice"] = x_out
        return x_out, total_outputs
