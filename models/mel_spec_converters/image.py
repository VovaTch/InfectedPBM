from dataclasses import asdict

import torch
from torchaudio.transforms import MelSpectrogram

from .base import MelSpecConverter
from utils.containers import MelSpecParameters


class ScaledImageMelSpecConverter(MelSpecConverter):
    """
    A class representing a scaled image Mel spectrogram converter.

    Args:
        mel_spec_params (MelSpecParameters): The parameters for the Mel spectrogram.

    Attributes:
        mel_spec (MelSpectrogram): The Mel spectrogram.

    Methods:
        convert(slice: torch.Tensor) -> torch.Tensor: Converts a slice of input tensor to a scaled Mel spectrogram.
        from_cfg(cls: type[_T], cfg: dict[str, Any]) -> _T: Creates an instance of the class from a configuration dictionary.
    """

    mel_spec: MelSpectrogram

    def __init__(self, mel_spec_params: MelSpecParameters) -> None:
        """
        Initialize the MelSpecConverter object.

        Args:
            mel_spec_params (MelSpecParameters): The parameters for the Mel spectrogram conversion.

        Returns:
            None
        """
        self.mel_spec_params = mel_spec_params
        self.mel_spec = MelSpectrogram(**asdict(mel_spec_params))

    def convert(self, slice: torch.Tensor) -> torch.Tensor:
        """
        Converts a slice of input tensor to a scaled Mel spectrogram.

        Args:
            slice (torch.Tensor): The input tensor slice.

        Returns:
            torch.Tensor: The scaled Mel spectrogram.
        """
        self.mel_spec = self.mel_spec.to(slice.device)
        output = self.mel_spec(slice)
        scaled_output = torch.tanh(output)

        # Create the repeat tensor
        scaled_output = torch.cat(
            [
                scaled_output.clone(),
                scaled_output.clone(),
                scaled_output.clone(),
            ],
            dim=-3,
        )

        return scaled_output
