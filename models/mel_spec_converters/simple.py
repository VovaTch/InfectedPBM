from dataclasses import asdict
import torch
from torchaudio.transforms import MelSpectrogram

from .base import MelSpecConverter
from utils.containers import MelSpecParameters


class SimpleMelSpecConverter(MelSpecConverter):
    """
    A simple class for converting audio slices to Mel spectrograms.

    Args:
        mel_spec_params (MelSpecParameters): Parameters for Mel spectrogram computation.

    Attributes:
        mel_spec (MelSpectrogram): Mel spectrogram instance.

    Methods:
        convert(slice: torch.Tensor) -> torch.Tensor: Converts an audio slice to a Mel spectrogram.
        from_cfg(cls: type[_T], cfg: dict[str, Any]) -> _T: Creates an instance of SimpleMelSpecConverter from a configuration dictionary.
    """

    mel_spec: MelSpectrogram

    def __init__(self, mel_spec_params: MelSpecParameters) -> None:
        """
        Initialize the MelSpecConverter object.

        Args:
            mel_spec_params (MelSpecParameters): The parameters for the Mel spectrogram conversion.
        """
        self.mel_spec_params = mel_spec_params
        self.mel_spec = MelSpectrogram(**asdict(mel_spec_params))

    def convert(self, slice: torch.Tensor) -> torch.Tensor:
        """
        Converts an audio slice to a Mel spectrogram.

        Args:
            slice (torch.Tensor): Input audio slice.

        Returns:
            torch.Tensor: Converted Mel spectrogram.
        """
        self.mel_spec = self.mel_spec.to(slice.device)
        output = self.mel_spec(slice.float())
        return output
