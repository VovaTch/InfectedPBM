from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class RecLoss:
    """Reconstruction loss for sound-waves"""

    name: str
    weight: float
    base_loss: nn.Module
    pred_key: str
    ref_key: str
    differentiable: bool = True

    # Loss-specific parameters
    transform_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    phase_parameter: int = 1

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

        return self._phased_loss(
            self.transform_func(pred_slice),
            self.transform_func(target_slice),
            phase_parameter=self.phase_parameter,
        )

    def _phased_loss(
        self, estimation: torch.Tensor, target: torch.Tensor, phase_parameter: int = 10
    ) -> torch.Tensor:
        """
        Utility method for computing reconstruction loss for slices that are delayed or premature,
        the reconstruction loss doesn't need often to be exact one-to-one. Computes the minimum of those losses.

        Args:
            estimation (torch.Tensor): Network estimation
            target (torch.Tensor): Ground truth reference
            phase_parameter (int, optional): How much to consider delay or prematureness. Defaults to 10.

        Returns:
            torch.Tensor: Computed loss
        """
        loss_vector = torch.zeros(phase_parameter * 2).to(estimation.device)
        for idx in range(phase_parameter):
            if idx == 0:
                loss_vector[idx * 2] = self.base_loss(estimation, target)
                loss_vector[idx * 2 + 1] = loss_vector[idx * 2] + 1e-6
                continue

            loss_vector[idx * 2] = self.base_loss(
                estimation[:, :, idx:], target[:, :, :-idx]
            )
            loss_vector[idx * 2 + 1] = self.base_loss(
                estimation[:, :, :-idx], target[:, :, idx:]
            )

        return loss_vector.min()


@dataclass
class EdgeRecLoss:
    """
    Edge reconstruction loss, focuses on the slice edges to prevent as much as possible parasite
    sounds from stitching slices together.
    """

    name: str
    weight: float
    base_loss: nn.Module
    pred_key: str
    ref_key: str
    differentiable: bool = True

    # Loss-specific parameters
    transform_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    edge_power: float = 1.0

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward method for this loss

        Args:
            estimation (dict[str, torch.Tensor]): Estimation dictionary, expects a "slice" key
            target (dict[str, torch.Tensor]): Target dictionary, expects a "slice" key

        Returns:
            torch.Tensor: Loss output
        """

        pred_slice = estimation[self.pred_key]
        batch_size, channels, length = pred_slice.shape
        target_slice = target[self.ref_key]

        linspace_slice = torch.arange(0, length).to(pred_slice.device) / (length - 1)
        weight_slice = (
            4 * (1 - linspace_slice * (1 - linspace_slice)) ** self.edge_power
        )
        weight_slice = weight_slice.repeat(batch_size, channels, 1)

        return self.base_loss(pred_slice * weight_slice, target_slice * weight_slice)


@dataclass
class NoisePredLoss:
    """
    Basic loss for reconstructing noise, used in diffusion
    """

    name: str
    weight: float
    pred_key: str
    ref_key: str
    base_loss: nn.Module

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
        noise = target[self.pred_key]
        noise_pred = estimation[self.ref_key]

        return self.base_loss(noise, noise_pred)


@dataclass
class DiffReconstructionLoss:
    """
    Predicts at any given time-step what is the final signal and compares it to the ground truth
    """

    name: str
    weight: float
    base_loss: nn.Module
    ref_key: str
    noise_pred_key: str
    noise_ref_key: str
    noise_scale_key: str
    differentiable: bool = True

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
        target_slice = target[self.ref_key]
        noisy_slice = target[self.noise_ref_key]
        noise_scale = target[self.noise_scale_key]
        noise_pred = estimation[self.noise_pred_key]
        estimated_slice = (noisy_slice - (1.0 - noise_scale) ** 0.5 * noise_pred) / (
            noise_scale**0.5
        )

        return self.base_loss(
            estimated_slice,
            target_slice,
        )
