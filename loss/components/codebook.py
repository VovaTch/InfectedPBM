from dataclasses import dataclass
from typing import Any, Self
from numpy import diff

import torch
import torch.nn as nn

from common import registry


@registry.register_loss_component("align")
@dataclass
class AlignLoss:
    """
    VQ-VAE codebook alignment loss
    """

    name: str
    weight: float
    base_loss: nn.Module
    differentiable: bool = True

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the VQ-VAE codebook alignment loss

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: Loss
        """
        emb = estimation["emb"]
        z_e = target["z_e"]

        return self.base_loss(emb, z_e.detach())

    @classmethod
    def from_cfg(cls, name: str, loss_cfg: dict[str, Any]) -> Self:
        """
        Utility method to parse alignment loss parameters from a configuration dictionary

        Args:
            name (str): loss name
            loss_cfg (DictConfig): configuration dictionary

        Returns:
            AlignLoss: align loss object
        """
        return cls(
            name,
            loss_cfg.get("weight", 1.0),
            registry.get_loss_module(loss_cfg.get("base_loss", "mse")),
        )


@registry.register_loss_component("commit")
@dataclass
class CommitLoss:
    """
    VQ-VAE codebook commitment loss
    """

    name: str
    weight: float
    base_loss: nn.Module
    differentiable: bool = True

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the VQ-VAE codebook commitment loss

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: Loss
        """
        emb = estimation["emb"]
        z_e = target["z_e"]

        return self.base_loss(emb.detach(), z_e)

    @classmethod
    def from_cfg(cls, name: str, loss_cfg: dict[str, Any]) -> Self:
        """
        Utility method to parse commitment loss parameters from a configuration dictionary

        Args:
            name (str): loss name
            loss_cfg (DictConfig): configuration dictionary

        Returns:
            CommitLoss: commit loss object
        """
        return cls(
            name,
            loss_cfg.get("weight", 1.0),
            registry.get_loss_module(loss_cfg.get("base_loss", "mse")),
        )
