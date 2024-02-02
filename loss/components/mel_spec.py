from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING
from typing_extensions import Self

import torch
import torch.nn as nn

from common import registry

if TYPE_CHECKING:
    from models.mel_spec_converters import MelSpecConverter


@registry.register_loss_component("mel_spec")
@dataclass
class MelSpecLoss:
    """
    This loss is a reconstruction loss of a mel spectrogram, convert the inputs into a spectrogram and
    compute reconstruction loss
    """

    name: str
    weight: float
    base_loss: nn.Module

    # Loss-specific parameters
    mel_spec_converter: "MelSpecConverter"
    transform_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.tanh(x)
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
        pred_slice = estimation["slice"]
        target_slice = target["slice"]

        self.mel_spec_converter.mel_spec = self.mel_spec_converter.mel_spec.to(
            pred_slice.device
        )

        return self.base_loss(
            self._mel_spec_and_process(pred_slice),
            self._mel_spec_and_process(target_slice),
        )

    @classmethod
    def from_cfg(cls, name: str, loss_cfg: dict[str, Any]) -> Self:
        """
        Utility method to parse mel spectrogram loss parameters from a configuration dictionary

        Args:
            name (str): loss name
            loss_cfg (DictConfig): configuration dictionary

        Returns:
            MelSpecLoss: mel spectrogram loss object
        """
        # Create mel spec converter
        mel_spec_converter = registry.get_mel_spec_converter("simple").from_cfg(
            loss_cfg["melspec_params"]
        )
        loss_module = registry.get_loss_module(loss_cfg.get("base_loss", "mse"))
        transform_func = registry.get_transform_function(
            loss_cfg.get("transform_func", "tanh")
        )
        lin_start = loss_cfg.get("lin_start", 1.0)
        lin_end = loss_cfg.get("lin_end", 1.0)

        # Create mel-spec loss
        return cls(
            name,
            loss_cfg.get("weight", 1.0),
            loss_module,
            mel_spec_converter,
            transform_func=transform_func,
            lin_start=lin_start,
            lin_end=lin_end,
        )


@registry.register_loss_component("mel_spec_diff")
@dataclass
class MelSpecDiffusionLoss(MelSpecLoss):
    """
    Mel-spectrogram loss object, used for diffusion models, where you predict the denoised output
    and compute the loss for it. Inherits from the regular mel spec loss object
    """

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

        target_slice = target["slice"]
        noisy_slice = target["noisy_slice"]
        noise_scale = target["noise_scale"]
        noise_pred = estimation["noise_pred"]
        estimated_slice = (noisy_slice - (1.0 - noise_scale) ** 0.5 * noise_pred) / (
            noise_scale**0.5
        )

        return self.base_loss(
            self._mel_spec_and_process(estimated_slice),
            self._mel_spec_and_process(target_slice),
        )

    @classmethod
    def from_cfg(cls, name: str, loss_cfg: dict[str, Any]) -> Self:
        """
        Utility method to parse diffusion mel spectrogram loss parameters from a configuration dictionary

        Args:
            name (str): loss name
            loss_cfg (DictConfig): configuration dictionary

        Returns:
            MelSpecDiffusionLoss: diffusion mel spectrogram loss object
        """
        # Create mel spec converter
        mel_spec_converter = registry.get_mel_spec_converter("simple").from_cfg(
            loss_cfg["melspec_params"]
        )
        loss_module = registry.get_loss_module(loss_cfg.get("base_loss", "mse"))
        transform_func = registry.get_transform_function(
            loss_cfg.get("transform_func", "tanh")
        )
        lin_start = loss_cfg.get("lin_start", 1.0)
        lin_end = loss_cfg.get("lin_end", 1.0)

        # Create mel-spec loss
        return cls(
            name,
            loss_cfg.get("weight", 1.0),
            loss_module,
            mel_spec_converter,
            transform_func=transform_func,
            lin_start=lin_start,
            lin_end=lin_end,
        )
