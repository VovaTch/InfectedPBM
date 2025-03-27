import torch
import torch.nn as nn


class Res2DBlock(nn.Module):

    def __init__(
        self,
        num_channels: int,
        num_res_conv: int,
        kernel_size: tuple[int, int],
        activation_fn: nn.Module = nn.GELU(),
        dilation_factor: int = 1,
    ) -> None:
        super().__init__()

        self._activation_fn = activation_fn

        layers = []
        for idx in range(num_res_conv - 1):
            dilation = dilation_factor**idx
            padding_x = (
                kernel_size[0] + (kernel_size[0] - 1) * (dilation - 1) - 1
            ) // 2
            padding_y = (
                kernel_size[1] + (kernel_size[1] - 1) * (dilation - 1) - 1
            ) // 2
            layers.append(
                nn.Conv2d(
                    num_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=(padding_x, padding_y),
                )
            )

            layers.append(activation_fn)
            layers.append(nn.BatchNorm2d(num_channels))

        padding_x = (
            kernel_size[0]
            + (kernel_size[0] - 1) * (dilation_factor ** (num_res_conv - 1) - 1)
            - 1
        ) // 2
        padding_y = (
            kernel_size[1]
            + (kernel_size[1] - 1) * (dilation_factor ** (num_res_conv - 1) - 1)
            - 1
        ) // 2
        layers.append(
            nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size=kernel_size,
                dilation=dilation_factor ** (num_res_conv - 1),
                padding=(padding_x, padding_y),
            )
        )
        layers.append(nn.BatchNorm2d(num_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_init = x.clone()
        x = self.layers(x)
        return self._activation_fn(x + x_init)
