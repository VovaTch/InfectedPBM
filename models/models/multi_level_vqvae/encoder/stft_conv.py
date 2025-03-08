import torch
import torch.nn as nn

from models.models.multi_level_vqvae.blocks.res2d import Res2DBlock


class EncoderConv2D(nn.Module):
    """
    A convolutional encoder that uses an STFT for the first layer, the blocks inside
    are 2D resnets.
    """

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
        """
        Initializes the STFT convolutional encoder.

        Args:
            channel_list (list[int]): List of channel sizes for each layer.
            dim_change_list (list[int]): List of dimension changes (strides) for each convolutional layer.
            kernel_size (int): Size of the convolutional kernels.
            num_res_block_conv (int): Number of residual blocks in each Res2DBlock.
            n_fft (int): Number of FFT points for the STFT.
            hop_length (int): Hop length for the STFT.
            win_length (int): Window length for the STFT.
            activation_fn (nn.Module, optional): Activation function to use. Defaults to nn.GELU().

        Raises:
            ValueError: If the length of `channel_list` is not one greater than the length of `dim_change_list`.
        """
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

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the STFT convolutional encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, time_steps).

        Returns:
            torch.Tensor: Output tensor after applying STFT and convolutional layers,
                          with shape (batch_size, num_channels, H * W).
        """
        window = torch.hann_window(self._win_length).to(x.device)
        x = torch.stft(
            x.flatten(start_dim=1, end_dim=2),
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._win_length,
            return_complex=False,
            window=window,
        )
        x = x.permute((0, 3, 1, 2)).contiguous()
        x = x[..., : x.shape[2] - 1, : x.shape[3] - 1]
        x = self.layers(x)  # returns size of (batch_size, num_channels, H, W)
        x = x.transpose(2, 3).flatten(start_dim=2, end_dim=3)
        return x
