from __future__ import annotations
from typing import Any, Callable, Protocol, Self
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

from common import registry


registry.loss_modules.update(
    {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "huber": nn.SmoothL1Loss(),
        "bce": nn.BCEWithLogitsLoss(),
        "ce": nn.CrossEntropyLoss(),
        "focal": sigmoid_focal_loss,  # type: ignore
    }
)


class LossComponent(Protocol):
    """
    Loss component object protocol

    Fields:
        name (str): loss name
        weight (float): loss relative weight for computation (e.g. weighted sum)
        base_loss (nn.Module): base loss module specific for the loss
    """

    name: str
    weight: float
    base_loss: nn.Module
    differentiable: bool

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Call method for outputting the loss

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: loss
        """
        ...

    @classmethod
    def from_cfg(cls, name: str, loss_cfg: dict[str, Any]) -> Self:
        """
        Utility method to parse loss parameters from a configuration dictionary

        Args:
            name (str): loss name
            loss_cfg (dict[str, Any]): configuration dictionary

        Returns:
            LossComponent: loss object
        """
        ...


LossComponentFactory = Callable[[str, DictConfig], LossComponent]
