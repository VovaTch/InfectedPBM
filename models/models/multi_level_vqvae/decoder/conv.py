import torch
import torch.nn as nn

from ..blocks import Res1DBlockReverse


class TransparentLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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
        activation_fn: nn.Module = nn.GELU(),
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
            activation_fn (nn.Module, optional): Activation function to be used. Defaults to nn.GELU().
        """
        super().__init__()
        if len(channel_list) != len(dim_change_list) + 1:
            raise ValueError(
                "The channel list length must be greater than the dimension change list by 1"
            )

        self.activation = activation_fn

        # Create the module lists for the architecture
        self.end_conv = nn.Conv1d(
            channel_list[-1], input_channels, kernel_size=3, padding=1
        )
        self.conv_list = nn.ModuleList(
            [TransparentLayer()]
            + [
                Res1DBlockReverse(
                    channel_list[idx],
                    num_res_block_conv,
                    dilation_factor,
                    kernel_size,
                    activation_fn,
                )
                for idx in range(1, len(dim_change_list))
            ]
        )

        if dim_add_kernel_add % 2 != 0:
            raise ValueError("dim_add_kernel_size must be an even number.")

        self.dim_change_list = nn.ModuleList(
            [nn.Linear(channel_list[0], channel_list[1] * dim_change_list[0])]
            + [
                nn.ConvTranspose1d(
                    channel_list[idx],
                    channel_list[idx + 1],
                    kernel_size=dim_change_list[idx] + dim_add_kernel_add,
                    stride=dim_change_list[idx],
                    padding=dim_add_kernel_add // 2,
                )
                for idx in range(1, len(dim_change_list))
            ]
        )
        self.required_post_channel_size = channel_list[1]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for idx, (conv, dim_change) in enumerate(
            zip(self.conv_list, self.dim_change_list)
        ):
            if idx == 0:  # Avoid wasting parameters on dilations.
                z = conv(z.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                z = (
                    dim_change(z.transpose(1, 2).contiguous())
                    .transpose(1, 2)
                    .contiguous()
                )
                z = z.reshape(z.shape[0], self.required_post_channel_size, -1)
            else:
                z = conv(z)
                z = dim_change(z)
            z = self.activation(z)

        x_out = self.end_conv(z)

        return x_out
