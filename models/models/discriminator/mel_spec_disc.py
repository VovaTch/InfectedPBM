import torch
import torch.nn as nn

from models.mel_spec_converters.base import MelSpecConverter

from .base import Discriminator


class MelSpecDiscriminator(Discriminator):
    """
    A discriminator that starts with a Mel Spectrogram converter and uses 2D convolutions.
    """

    def __init__(
        self,
        channel_list: list[int],
        stride: int,
        kernel_size: int,
        mel_spec_converter: MelSpecConverter,
        post_mel_fn: nn.Module = nn.Identity(),
        activation_fn: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()

        layers = []
        channel_list.insert(0, 1)
        self._post_mel_fn = post_mel_fn
        for idx in range(len(channel_list) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=channel_list[idx],
                    out_channels=channel_list[idx + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            layers.append(activation_fn)
            layers.append(nn.BatchNorm2d(channel_list[idx + 1]))

        layers.append(
            nn.Conv2d(
                channel_list[-1],
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
        )
        self.model = nn.Sequential(*layers)
        self._mel_spec_converter = mel_spec_converter

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._mel_spec_converter.convert(x)
        x = self._post_mel_fn(x)
        x = self.model(x)
        x = x.permute((0, 2, 3, 1)).flatten(start_dim=1, end_dim=2).contiguous()
        return {"logits": x}
