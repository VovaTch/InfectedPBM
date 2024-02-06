from dataclasses import dataclass
from mimetypes import init
from typing_extensions import Self

from omegaconf import DictConfig
import torch
import torch.nn as nn

from common import registry
from .blocks import Res1DBlockReverse


class Decoder1D(nn.Module):
    def __init__(
        self,
        channel_list: list[int],
        dim_change_list: list[int],
        input_channels: int = 1,
        kernel_size: int = 5,
        dim_add_kernel_add: int = 12,
        num_res_block_conv: int = 3,
        dilation_factor: int = 3,
        activation_type: str = "gelu",
    ) -> None:
        """
        Initializes the Decoder module of the multi-level VQ-VAE architecture.

        Args:
            channel_list (list[int]): List of channel sizes for each layer of the decoder.
            dim_change_list (list[int]): List of dimension change sizes for each layer of the decoder.
            input_channels (int, optional): Number of input channels. Defaults to 1.
            kernel_size (int, optional): Kernel size for the convolutional layers. Defaults to 5.
            dim_add_kernel_add (int, optional): Kernel size for the dimension change layers. Defaults to 12.
            num_res_block_conv (int, optional): Number of residual blocks in the convolutional layers. Defaults to 3.
            dilation_factor (int, optional): Dilation factor for the convolutional layers. Defaults to 3.
            activation_type (str, optional): Activation function type. Defaults to "gelu".
        """
        super().__init__()
        if len(channel_list) != len(dim_change_list) + 1:
            raise ValueError(
                "The channel list length must be greater than the dimension change list by 1"
            )

        self.activation = registry.get_activation_function(activation_type)

        # Create the module lists for the architecture
        self.end_conv = nn.Conv1d(
            channel_list[-1], input_channels, kernel_size=3, padding=1
        )
        self.conv_list = nn.ModuleList(
            [
                Res1DBlockReverse(
                    channel_list[idx],
                    num_res_block_conv,
                    dilation_factor,
                    kernel_size,
                    activation_type,
                )
                for idx in range(len(dim_change_list))
            ]
        )
        if dim_add_kernel_add % 2 != 0:
            raise ValueError("dim_add_kernel_size must be an even number.")

        self.dim_change_list = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    channel_list[idx],
                    channel_list[idx + 1],
                    kernel_size=dim_change_list[idx] + dim_add_kernel_add,
                    stride=dim_change_list[idx],
                    padding=dim_add_kernel_add // 2,
                )
                for idx in range(len(dim_change_list))
            ]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for _, (conv, dim_change) in enumerate(
            zip(self.conv_list, self.dim_change_list)
        ):
            z = conv(z)
            z = dim_change(z)
            z = self.activation(z)

        x_out = self.end_conv(z)

        return x_out


@dataclass
class RippleDecoderParameters:
    input_dim: int
    hidden_dim: int
    mlp_num_layers: int
    output_dim: int
    ripl_hidden_dim: int
    ripl_num_layers: int

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        return cls(
            input_dim=cfg.model.input_dim,
            hidden_dim=cfg.model.hidden_dim,
            mlp_num_layers=cfg.model.mlp_num_layers,
            output_dim=cfg.model.output_dim,
            ripl_hidden_dim=cfg.model.ripl_hidden_dim,
            ripl_num_layers=cfg.model.ripl_num_layers,
        )


class RippleDecoder(nn.Module):
    def __init__(
        self,
        dec_params: RippleDecoderParameters,
    ) -> None:
        super().__init__()
        self.dec_params = dec_params

        ripple_weight_dim = self._compute_ripple_weight_dim(
            self.dec_params.ripl_hidden_dim, self.dec_params.ripl_num_layers
        )
