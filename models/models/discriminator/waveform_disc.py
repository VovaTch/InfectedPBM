import torch
import torch.nn as nn

from models.models.multi_level_vqvae.blocks.res1d import Res1DBlock

from .base import Discriminator


class WaveformDiscriminator(Discriminator):
    """
    A discriminator class for waveform data that uses 1d convolutions.
    """

    def __init__(
        self,
        channel_list: list[int],
        dim_change_list: list[int],
        kernel_size: int,
        num_res_block_conv: int,
        dilation_factor: int,
        input_channels: int = 1,
        activation_fn: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()
        if len(channel_list) != len(dim_change_list) + 1:
            raise ValueError(
                "The length of `channel_list` must be one greater than the length of `dim_change_list`."
            )

        self.last_dim = channel_list[-1]
        self._activation = activation_fn

        self._init_conv = nn.Conv1d(
            input_channels,
            channel_list[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self._conv_list = nn.ModuleList(
            [
                Res1DBlock(
                    channel_list[idx],
                    num_res_block_conv,
                    dilation_factor,
                    kernel_size,
                    activation_fn,
                )
                for idx in range(len(dim_change_list))
            ]
        )
        self._dim_change_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channel_list[idx],
                        channel_list[idx + 1],
                        kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.MaxPool1d(
                        kernel_size=dim_change_list[idx], stride=dim_change_list[idx]
                    ),
                )
                for idx in range(len(dim_change_list))
            ]
        )
        self._last_conv = nn.Conv1d(
            channel_list[-1],
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._init_conv(x)

        for _, (conv, dim_change) in enumerate(
            zip(self._conv_list, self._dim_change_list)
        ):
            x = conv(x)
            x = dim_change(x)

        x = self._last_conv(x)
        return {"logits": x.transpose(1, 2).contiguous()}

    @property
    def last_layer(self) -> nn.Module:
        return self._last_conv
