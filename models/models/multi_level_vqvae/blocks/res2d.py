import torch
import torch.nn as nn


class Res2DBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_res_conv: int,
        kernel_size: tuple[int, int],
        activation_fn: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()

        self._activation_fn = activation_fn

        layers = []
        for _ in range(num_res_conv - 1):
            layers.append(
                nn.Conv2d(
                    num_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                )
            )

            layers.append(activation_fn)
            layers.append(nn.BatchNorm2d(num_channels))

        layers.append(
            nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size=kernel_size,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            )
        )
        layers.append(nn.BatchNorm2d(num_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_init = x.clone()
        x = self.layers(x)
        return self._activation_fn(x + x_init)
