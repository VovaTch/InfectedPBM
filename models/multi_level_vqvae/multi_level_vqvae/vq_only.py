from typing import TYPE_CHECKING, Any
from typing_extensions import Self

from omegaconf import DictConfig
import torch
import torch.nn as nn

from common import registry
from models.base import Tokenizer
from models.multi_level_vqvae.blocks import VQ1D
from models.multi_level_vqvae.multi_level_vqvae.interface import InterfaceVQ1D

if TYPE_CHECKING:
    from loss.aggregators import LossAggregator


@registry.register_model("vq_only")
class VQOnlyTokenizer(Tokenizer):
    def __init__(
        self,
        input_channels: int,
        slice_length: int,
        latent_depth: int,
        vq_module: InterfaceVQ1D,
        loss_aggregator: "LossAggregator | None" = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.slice_length = slice_length
        self.latent_depth = latent_depth
        self.vq_module = vq_module
        self.loss_aggregator = loss_aggregator
        self.encoder_projection = nn.Linear(slice_length, latent_depth)
        self.decoder_projection = nn.Linear(latent_depth, slice_length)
        self.activation = nn.ReLU()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_projection(x)
        x = self.activation(x)
        return x.permute((0, 2, 1)).contiguous()

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encode(x)
        return self.vq_module(z_e)["indices"]

    def from_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.vq_module.vq_codebook.embed_codebook(indices)
        x_out = z_q.transpose(1, 2).contiguous()
        return x_out

    def decode(
        self, z_e: torch.Tensor, origin_shape: tuple[int, int, int] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        vq_block_output = self.vq_module(z_e, extract_losses=True)
        x_out = vq_block_output["v_q"][:, -1, ...].transpose(1, 2).contiguous()
        x_out = self.activation(x_out)
        x_out = self.decoder_projection(x_out)

        total_output = {**vq_block_output, "slice": x_out}

        return x_out, total_output

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # origin_shape = x.shape
        z_e = self.encode(x)
        _, total_output = self.decode(z_e)  # type: ignore

        total_output.update({"z_e": z_e})
        return total_output

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        model_cfg: DictConfig = cfg.model

        # VQ parameter initialization
        vocabulary_size: int = model_cfg.vocabulary_size
        num_codebooks: int = model_cfg.num_codebooks
        slice_length: int = model_cfg.slice_length
        latent_depth: int = model_cfg.latent_depth
        input_channels: int = model_cfg.input_channels

        # loss aggregator
        if cfg.loss.aggregator.type != "none":
            loss_aggregator = registry.get_loss_aggregator(
                cfg.loss.aggregator.type
            ).from_cfg(cfg)
        else:
            loss_aggregator = None

        vq_module = VQ1D(
            latent_depth, num_tokens=vocabulary_size, num_codebooks=num_codebooks
        )

        return cls(
            input_channels, slice_length, latent_depth, vq_module, loss_aggregator
        )
