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

        ripple_weight_dim = self._compute_mlp_output_dim()
        if self.dec_params.mlp_num_layers < 0:
            raise ValueError(
                f"MLP number of hidden layers must be non negative, got {self.dec_params.mlp_num_layers}"
            )
        self.mlp = nn.Sequential(
            [nn.Linear(1, self.dec_params.hidden_dim)]
            + [nn.Linear(self.dec_params.hidden_dim, self.dec_params.hidden_dim)]
            * self.dec_params.mlp_num_layers
            + [nn.Linear(self.dec_params.hidden_dim, ripple_weight_dim)]
        )

    @staticmethod
    def _compute_ripple_weight_dim(in_dim: int, out_dim: int) -> int:
        """
        Computes the dimension of the ripple weights for the decoder.

        Args:
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.

        Returns:
            int: The dimension of the ripple weights.
        """
        num_weights = 2 * in_dim * out_dim
        num_biases = (in_dim + 1) * out_dim
        return num_weights + num_biases

    def _compute_mlp_output_dim(self) -> int:
        """
        Computes the output dimension of the MLP network in the decoder.

        Returns:
            int: The output dimension of the MLP network.
        """
        in_dim = self._compute_ripple_weight_dim(1, self.dec_params.ripl_hidden_dim)
        out_dim = self._compute_ripple_weight_dim(self.dec_params.ripl_hidden_dim, 1)
        middle_dim = (
            self._compute_ripple_weight_dim(
                self.dec_params.ripl_hidden_dim, self.dec_params.ripl_hidden_dim
            )
            * self.dec_params.mlp_num_layers
        )
        return in_dim + out_dim + middle_dim

    def _split_mlp_output_to_ripple_layers(
        self, mlp_output: torch.Tensor
    ) -> nn.ModuleList:
        """
        Splits the output of the MLP into ripple layers.

        Args:
            mlp_output (torch.Tensor): The output tensor of the MLP.

        Returns:
            nn.ModuleList: A list of tensors representing the ripple layers.
        """
        in_dim = self._compute_ripple_weight_dim(1, self.dec_params.ripl_hidden_dim)
        out_dim = self._compute_ripple_weight_dim(self.dec_params.ripl_hidden_dim, 1)
        middle_dims = [
            self._compute_ripple_weight_dim(
                self.dec_params.ripl_hidden_dim, self.dec_params.ripl_hidden_dim
            )
        ] * self.dec_params.mlp_num_layers

        # Compute the splitting via a running index
        ripple_weight_layers = nn.ModuleList()
        running_idx = 0
        ripple_weight_layers.append(mlp_output[..., :in_dim])
        running_idx += in_dim
        for middle_dim in middle_dims:
            ripple_weight_layers.append(
                mlp_output[..., running_idx : running_idx + middle_dim]
            )
            running_idx += middle_dim
        ripple_weight_layers.append(
            mlp_output[..., running_idx : running_idx + out_dim]
        )
        return ripple_weight_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ripple decoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        mlp_output = self.mlp(x)
        ripple_weight_layers = self._split_mlp_output_to_ripple_layers(mlp_output)
