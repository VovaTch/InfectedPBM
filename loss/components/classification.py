from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import LossComponent


@dataclass
class BasicClassificationLoss(LossComponent):
    """
    Basic classification loss for classification purposes
    """

    name: str
    weight: float
    base_loss: nn.Module
    pred_key: str
    ref_key: str
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
        return self.base_loss(estimation[self.pred_key], target[self.ref_key])


@dataclass
class PercentCorrect(LossComponent):
    """
    Basic metric to count the ratio of the correct number of classifications
    """

    name: str
    weight: float
    pred_key: str
    ref_key: str
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
        pred_logits_argmax = torch.argmax(estimation[self.pred_key], dim=1)
        correct = torch.sum(pred_logits_argmax == target[self.ref_key])
        return correct / torch.numel(pred_logits_argmax)


@dataclass
class MaskedClassificationLoss(LossComponent):
    """
    Classification loss with masking, mostly used in Diffusion LLM training
    """

    name: str
    weight: float
    base_loss: nn.Module
    pred_key: str
    ref_key: str
    mask_key: str
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Call method for outputting the loss

        Args:
            pred (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: loss
        """
        mask = ~target[self.mask_key]
        if pred[self.pred_key].dim() == 4:
            logits = pred[self.pred_key][mask].permute(0, 2, 1)
        elif pred[self.pred_key].dim() == 3:
            logits = pred[self.pred_key][mask]
        labels = target[self.ref_key][mask]
        return self.base_loss(logits, labels)


@dataclass
class MaskedPercentCorrect(LossComponent):
    """
    Classification loss with masking, mostly used in Diffusion LLM training
    """

    name: str
    weight: float
    pred_key: str
    ref_key: str
    mask_key: str
    base_loss: nn.Module | None = None
    differentiable: bool = False

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Call method for outputting the loss

        Args:
            pred (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: loss
        """
        mask = ~target[self.mask_key]
        if pred[self.pred_key].dim() == 4:
            logits = pred[self.pred_key][mask].permute(0, 2, 1)
        elif pred[self.pred_key].dim() == 3:
            logits = pred[self.pred_key][mask]
        pred_logits_argmax = torch.argmax(logits, dim=1)
        correct = torch.sum(pred_logits_argmax == target[self.ref_key][mask])
        return correct / torch.numel(pred_logits_argmax)
