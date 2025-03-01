import torch
import torch.nn as nn

from .base import Discriminator


class StftDiscriminator(Discriminator):

    def __init__(
        self,
        channel_list: list[int],
        n_fft: int,
        hop_length: int,
        win_length: int,
        activation_fn: nn.Module = nn.LeakyReLU(0.1),
        stride: int = 2,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()

        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length

        layers = []
        for idx in range(len(channel_list) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=channel_list[idx],
                    out_channels=channel_list[idx + 1],
                    kernel_size=(kernel_size, 1),
                    stride=stride,
                    padding=kernel_size // stride,
                )
            )
            layers.append(activation_fn)
            layers.append(nn.BatchNorm2d(channel_list[idx + 1]))

        layers.append(
            nn.Conv2d(
                channel_list[-1],
                1,
                kernel_size=(3, 1),
                stride=1,
                padding=1,
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        window = torch.hann_window(self._win_length).to(x.device)
        x = torch.stft(
            x.squeeze(1),
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._win_length,
            return_complex=True,
            window=window,
        )
        x = torch.view_as_real(x)
        x = x.permute((0, 3, 1, 2)).contiguous()
        x = self.model(x)
        x = x.permute((0, 2, 3, 1)).flatten(start_dim=1, end_dim=2).contiguous()
        return {"logits": x}

    @property
    def last_layer(self) -> nn.Module:
        return self.model[-1]
