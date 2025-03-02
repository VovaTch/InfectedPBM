import torch
import torch.nn as nn

from models.models.multi_level_vqvae.blocks.res2d import Res2DBlock


class EncoderConv2D(nn.Module):
    def __init__(
        self,
        channel_list: list[int],
        dim_change_list: list[int],
        kernel_size: int,
        num_res_block_conv: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        activation_fn: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()

        if len(channel_list) != len(dim_change_list) + 1:
            raise ValueError(
                "The length of `channel_list` must be one greater than the length of `dim_change_list`."
            )

        # Set STFT parameters
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length

        layers = []
        for idx in range(len(dim_change_list)):
            layers.append(
                Res2DBlock(
                    channel_list[idx],
                    num_res_block_conv,
                    (kernel_size, 1),
                    activation_fn,
                )
            )
            layers.append(
                nn.Conv2d(
                    channel_list[idx],
                    channel_list[idx + 1],
                    kernel_size,
                    stride=dim_change_list[idx],
                    padding=kernel_size // dim_change_list[idx],
                )
            )
        layers.append(
            Res2DBlock(
                channel_list[-1], num_res_block_conv, (kernel_size, 1), activation_fn
            )
        )
