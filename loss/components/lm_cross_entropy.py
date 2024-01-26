from dataclasses import dataclass
from typing import Any, Self

import torch
import torch.nn as nn

from common import registry


@registry.register_loss_component("decoder_ce")
@dataclass
class DecoderCrossEntropy:
    """
    Standard cross-entropy-loss for training a decoder transformer for sequence generation.
    """

    name: str
    weight: float
    base_loss: nn.Module

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward method for this configuration

        Args:
            est (Dict[str, torch.Tensor]): Dictionary expecting BS x V x cls in key "logits"
            ref (Dict[str, torch.Tensor]): Dictionary expecting BS x V in key "latent indices"

        Returns:
            torch.Tensor: Loss
        """
        logits = estimation["logits"][:-1]
        target_indices = target["latent indices"][1:]
        return self.base_loss(logits.transpose(1, 2), target_indices.long())

    @classmethod
    def from_cfg(cls, name: str, loss_cfg: dict[str, Any]) -> Self:
        """
        Utility method to parse loss parameters from a configuration dictionary

        Args:
            name (str): loss name
            loss_cfg (DictConfig): configuration dictionary

        Returns:
            AlignLoss: align loss object
        """
        return cls(
            name,
            loss_cfg.get("weight", 1.0),
            registry.get_loss_module(loss_cfg.get("base_loss", "ce")),
        )
