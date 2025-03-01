import torch
import torch.nn as nn


class EncoderConv2D(nn.Module):
    def __init__(
        self,
        channel_list: list[int],
        dim_change_list: list[int],
        input_channels: int,
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
