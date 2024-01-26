from dataclasses import asdict
from typing import Any, Protocol, Self

import torch
from torchaudio.transforms import MelSpectrogram

from utils.containers import MelSpecParameters
from common import registry


class MelSpecConverter(Protocol):
    mel_spec: MelSpectrogram

    def __init__(self, mel_spec_params: MelSpecParameters) -> None:
        """Mel spectrogram converter constructor

        Args:
            mel_spec_params (MelSpecParameters): Mel spectrogram parameter object
        """
        ...

    def convert(self, slice: torch.Tensor) -> torch.Tensor:
        """
        Convert a torch tensor representing a music slice to a mel spectrogram

        Args:
            slice (torch.Tensor): Music slice

        Returns:
            torch.Tensor: Mel spectrogram
        """
        ...

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> Self:
        """
        Utility method to parse mel spectrogram converter parameters from a configuration dictionary

        Args:
            cfg (dict[str, Any]): configuration dictionary

        Returns:
            MelSpecConverter: mel spectrogram converter object
        """
        ...


@registry.register_mel_spec_converter("simple")
class SimpleMelSpecConverter:
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

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> Self:
        """
        Creates an instance of SimpleMelSpecConverter from a configuration dictionary.

        Args:
            cfg (dict[str, Any]): Configuration dictionary.

        Returns:
            _T: Instance of SimpleMelSpecConverter.
        #"""
        mel_spec_params = MelSpecParameters.from_cfg(cfg)
        return cls(mel_spec_params)


@registry.register_mel_spec_converter("scaled")
class ScaledImageMelSpecConverter:
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

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> Self:
        """
        Creates an instance of the class from a configuration dictionary.

        Args:
            cfg (dict[str, Any]): The configuration dictionary.

        Returns:
            _T: An instance of the class.
        """
        mel_spec_params = MelSpecParameters.from_cfg(cfg)
        return cls(mel_spec_params)
