import torch
import torch.nn as nn


class Res1DBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_res_conv: int,
        dilation_factor: int,
        kernel_size: int,
        activation_fn: nn.Module = nn.GELU(),
    ) -> None:
        """
        Residual 1D block module.

        Args:
            num_channels (int): Number of input and output channels.
            num_res_conv (int): Number of residual convolutional layers.
            dilation_factor (int): Dilation factor for the convolutional layers.
            kernel_size (int): Kernel size for the convolutional layers.
            activation_fn (Module, optional): Activation function to be used. Defaults to nn.GELU().
        """
        super().__init__()

        self.activation = activation_fn

        # Create conv, activation, norm blocks
        self.res_block_modules = nn.ModuleList([])
        for idx in range(num_res_conv):
            # Keep output dimension equal to input dim
            dilation = dilation_factor**idx
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

            if idx != num_res_conv - 1:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation,
                        nn.BatchNorm1d(num_channels),
                    )
                )

            else:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation,
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Res1DBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x_init = x.clone()
        for seq_module in self.res_block_modules:
            x = seq_module(x)

        return self.activation(x + x_init)


class Res1DBlockReverse(Res1DBlock):

    def __init__(
        self,
        num_channels: int,
        num_res_conv: int,
        dilation_factor: int,
        kernel_size: int,
        activation_fn: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__(
            num_channels, num_res_conv, dilation_factor, kernel_size, activation_fn
        )

        # Create conv, activation, norm blocks
        self.res_block_modules = nn.ModuleList([])
        for idx in range(num_res_conv):
            # Keep output dimension equal to input dim
            dilation = dilation_factor ** (num_res_conv - idx - 1)
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

            if idx != num_res_conv - 1:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation,
                        nn.BatchNorm1d(num_channels),
                    )
                )

            else:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation,
                    )
                )
