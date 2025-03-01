from typing import Protocol

import torch
import torch.nn as nn

from loss.aggregators import LossAggregator
from models.models.multi_level_vqvae.decoder.base import Decoder
from utils.waveform_tokenization import dequantize_waveform_256
from .ml_vqvae import MultiLvlVQVariationalAutoEncoder


class CodeBook(Protocol):
    def embed_codebook(self, indices: torch.Tensor) -> torch.Tensor: ...


class InterfaceVQ1D(Protocol):
    def __call__(
        self, z_e: torch.Tensor, extract_losses: bool = False
    ) -> dict[str, torch.Tensor]: ...

    @property
    def vq_codebook(self) -> CodeBook: ...


class TokenPredictorVQVAE(MultiLvlVQVariationalAutoEncoder):

    def __init__(
        self,
        input_channels: int,
        conv_out_channels: int,
        encoder: nn.Module,
        decoder: Decoder,
        vq_module: InterfaceVQ1D,
        loss_aggregator: LossAggregator | None = None,
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
