from __future__ import annotations
from dataclasses import dataclass
from typing_extensions import Self
from omegaconf import DictConfig

import torch
import torch.nn as nn

from common import registry


@registry.register_loss_component("basic_cls")
@dataclass
class BasicClassificationLoss:
    """
    Basic classification loss for classification purposes
    """

    name: str
    weight: float
    base_loss: nn.Module
    differentiable: bool = True

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
        return self.base_loss(estimation["pred_logits"], target["class"])

    @classmethod
    def from_cfg(cls, name: str, loss_cfg: DictConfig) -> Self:
        """
        Utility method to parse loss parameters from a configuration dictionary

        Args:
            name (str): loss name
            loss_cfg (DictConfig): configuration dictionary

        Returns:
            BasicClassificationLoss: Basic classification loss object
        """
        return cls(
            name,
            loss_cfg.get("weight", 1.0),
            registry.get_loss_module(loss_cfg.get("base_loss", "ce")),
        )


@registry.register_loss_component("percent_correct")
@dataclass
class PercentCorrect:
    """
    Basic metric to count the ratio of the correct number of classifications
    """

    name: str
    weight: float
    base_loss: nn.Module | None = None
    differentiable: bool = False

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
        pred_logits_argmax = torch.argmax(estimation["pred_logits"], dim=1)
        correct = torch.sum(pred_logits_argmax == target["class"])
        return correct / torch.numel(pred_logits_argmax)

    @classmethod
    def from_cfg(cls, name: str, loss_cfg: DictConfig) -> Self:
        """
        Utility method to parse loss parameters from a configuration dictionary

        Args:
            name (str): loss name
            loss_cfg (DictConfig): configuration dictionary

        Returns:
            BasicClassificationLoss: Basic classification loss object
        """
        return cls(
            name,
            loss_cfg.get("weight", 1.0),
            None,
        )
