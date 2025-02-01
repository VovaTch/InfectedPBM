from abc import ABC, abstractmethod

import torch
from torchaudio.transforms import MelSpectrogram

from utils.containers import MelSpecParameters


class MelSpecConverter(ABC):
    """
    Base class for converters from music slices to mel spectrograms
    """

    mel_spec: MelSpectrogram

    @abstractmethod
    def __init__(self, mel_spec_params: MelSpecParameters) -> None:
        """Mel spectrogram converter constructor

        Args:
            mel_spec_params (MelSpecParameters): Mel spectrogram parameter object
        """
        ...

    @abstractmethod
    def convert(self, slice: torch.Tensor) -> torch.Tensor:
        """
        Convert a torch tensor representing a music slice to a mel spectrogram

        Args:
            slice (torch.Tensor): Music slice

        Returns:
            torch.Tensor: Mel spectrogram
        """
        ...
