from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import torch
import torch.nn as nn

from .base import LossComponent

if TYPE_CHECKING:
    from models.mel_spec_converters import MelSpecConverter


@dataclass
class MelSpecLoss(LossComponent):
    """
    This loss is a reconstruction loss of a mel spectrogram, convert the inputs into a spectrogram and
    compute reconstruction loss
    """

    name: str
    weight: float
    pred_key: str
    ref_key: str
    base_loss: nn.Module

    # Loss-specific parameters
    mel_spec_converter: "MelSpecConverter"
    transform_func: Callable[[torch.Tensor], torch.Tensor] = (
        lambda x: torch.tanh(x) * 2 - 1
    )
    lin_start: float = 1.0
    lin_end: float = 1.0
    differentiable: bool = True

    def _mel_spec_and_process(self, x: torch.Tensor) -> torch.Tensor:
        """
        To prepare the mel spectrogram loss, everything needs to be prepared.

        Args:
            x (torch.Tensor): Input, will be flattened

        Returns:
            torch.Tensor: mel spectrogram of the input
        """
        lin_vector = torch.linspace(
            self.lin_start,
            self.lin_end,
            self.mel_spec_converter.mel_spec.n_mels,
        )
        eye_mat = torch.diag(lin_vector).to(x.device)
        mel_out = self.mel_spec_converter.convert(x.flatten(start_dim=0, end_dim=1))
        mel_out = self.transform_func(eye_mat @ mel_out)
        return mel_out

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward method for the loss

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: Loss
        """
        pred_slice = estimation[self.pred_key]
        target_slice = target[self.ref_key]

        self.mel_spec_converter.mel_spec = self.mel_spec_converter.mel_spec.to(
            pred_slice.device
        )

        return self.base_loss(
            self._mel_spec_and_process(pred_slice),
            self._mel_spec_and_process(target_slice),
        )


@dataclass
class MelSpecDiffusionLoss(MelSpecLoss):
    """
    Mel-spectrogram loss object, used for diffusion models, where you predict the denoised output
    and compute the loss for it. Inherits from the regular mel spec loss object
    """

    # Diffusion specific
    noise_pred_key: str = "noise_pred"
    noise_ref_key: str = "noisy_slice"
    noise_scale_key: str = "noise_scale"

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward method for the loss. Computes the estimated final slice given noised slice and predicted noise.

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: Loss
        """

        target_slice = target[self.ref_key]
        noisy_slice = target[self.noise_ref_key]
        noise_scale = target[self.noise_scale_key]
        noise_pred = estimation[self.noise_pred_key]
        estimated_slice = (noisy_slice - (1.0 - noise_scale) ** 0.5 * noise_pred) / (
            noise_scale**0.5
        )

        return self.base_loss(
            self._mel_spec_and_process(estimated_slice),
            self._mel_spec_and_process(target_slice),
        )
