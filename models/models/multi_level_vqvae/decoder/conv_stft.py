from typing import Literal
import torch
import torch.nn as nn

from models.models.multi_level_vqvae.blocks.res1d import Res1DBlockReverse

from .attention_stft import ISTFT


class StftDecoder1D(nn.Module):
    def __init__(
        self,
        channel_list: list[str],
        dim_change_list: list[str],
        input_channels: int,
        kernel_size: int,
        dim_add_kernel_add: int,
        num_res_block_conv: int,
        dilation_factor: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        activation_fn: nn.Module = nn.GELU(),
        padding: Literal["center", "same"] = "same",
    ):
        super().__init__()
        if len(channel_list) != len(dim_change_list) + 1:
            raise ValueError(
                "The channel list length must be greater than the dimension change list by 1"
            )

        self._activation = activation_fn
        self._enc_conv = nn.Conv1d(2, input_channels, kernel_size=3, padding=1)
        self._istft = ISTFT(n_fft, hop_length, win_length, padding)
        self._before_istft_dim = n_fft // 2 + 1
        self._proj_before_istft = nn.Linear(
            channel_list[-1], self._before_istft_dim * 2
        )

        self.conv_list = nn.ModuleList(
            [nn.Identity()]
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

    @property
    def last_layer(self) -> nn.Module:
        return self._enc_conv

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for idx, (conv, dim_change) in zip(self.conv_list, self.dim_change_list):
            if idx == 0:
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
            if idx != len(self.conv_list) - 1:
                z = self._activation(z)

        z = z.transpose(1, 2).contiguous()
        before_split = self._proj_before_istft(z)

        # Create separate tensors for real and imaginary parts
        real_part = before_split[..., : self._before_istft_dim].clone()
        imag_part = before_split[..., self._before_istft_dim :].clone()

        # Combine into complex tensor
        complex_z = torch.complex(real_part, imag_part).to(z.device)

        # Continue with ISTFT
        output_z = self.istft(complex_z.transpose(1, 2).contiguous())
        output_z = self.last_layer(output_z.unsqueeze(1))
        return output_z
