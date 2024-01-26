import torch
import torch.nn as nn

from common import registry
from .blocks import Res1DBlock, ConvDownsample


class Encoder1D(nn.Module):
    """
    Encoder class for the level 1 auto-encoder, this is constructed in a VAE manner.
    """

    def __init__(
        self,
        channel_list: list[int],
        dim_change_list: list[int],
        input_channels: int = 1,
        kernel_size: int = 5,
        num_res_block_conv: int = 3,
        dilation_factor: int = 3,
        dim_change_kernel_size: int = 5,
        activation_type: str = "gelu",
    ) -> None:
        """
        Initializes the Encoder module of the Multi-Level VQ-VAE.

        Args:
            channel_list (list[int]): List of channel sizes for each layer of the encoder.
            dim_change_list (list[int]): List of downsample divide factors for each layer of the encoder.
            input_channels (int, optional): Number of input channels. Defaults to 1.
            kernel_size (int, optional): Kernel size for the initial convolutional layer. Defaults to 5.
            num_res_block_conv (int, optional): Number of residual blocks in each convolutional layer. Defaults to 3.
            dilation_factor (int, optional): Dilation factor for the residual blocks. Defaults to 3.
            dim_change_kernel_size (int, optional): Kernel size for the dimension change convolutional layers. Defaults to 5.
            activation_type (str, optional): Activation function type. Defaults to "gelu".
        """
        super().__init__()
        if len(channel_list) != len(dim_change_list) + 1:
            raise ValueError(
                "The channel list length must be greater than the dimension change list by 1"
            )

        self.last_dim = channel_list[-1]
        self.activation = registry.get_activation_function(activation_type)

        # Create the module lists for the architecture
        self.init_conv = nn.Conv1d(
            input_channels,
            channel_list[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv_list = nn.ModuleList(
            [
                Res1DBlock(
                    channel_list[idx],
                    num_res_block_conv,
                    dilation_factor,
                    kernel_size,
                    activation_type,
                )
                for idx in range(len(dim_change_list))
            ]
        )
        self.dim_change_list = nn.ModuleList(
            [
                ConvDownsample(
                    kernel_size=dim_change_kernel_size,
                    downsample_divide=dim_change_param,
                    in_dim=channel_list[idx],
                    out_dim=channel_list[idx + 1],
                )
                for idx, dim_change_param in enumerate(dim_change_list)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded tensor.
        """
        x = self.init_conv(x)

        for idx, (conv, dim_change) in enumerate(
            zip(self.conv_list, self.dim_change_list)
        ):
            x = conv(x)
            x = dim_change(x)

            if idx != len(self.dim_change_list) - 1:
                x = self.activation(x)

        return x
