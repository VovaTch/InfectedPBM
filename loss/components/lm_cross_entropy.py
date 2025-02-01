from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossComponent


@dataclass
class LLMClassificationLoss(LossComponent):
    """
    Basic classification loss, most commonly cross entropy.

    Args:
        name (str): The name of the loss.
        weight (float): The weight of the loss.
        base_loss (nn.Module): The base loss function.
        ref_key (str): The key for accessing the reference values in the target dictionary.
        pred_key (str): The key for accessing the predicted values in the prediction dictionary.
        differentiable (bool, optional): Whether the loss is differentiable. Defaults to True.
    """

    name: str
    weight: float
    base_loss: nn.Module
    pred_key: str
    ref_key: str
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        logits = pred[self.pred_key].transpose(1, 2).contiguous()[..., :-1]
        target_indices = target[self.ref_key][..., 1:].long()
        return self.base_loss(logits, target_indices)


@dataclass
class TokenEntropy(LossComponent):
    """
    Computes the entropy of the token distribution.

    Attributes:
        name (str): The name of the loss component.
        weight (float): The weight of the loss component.
        logit_key (str): The key for accessing the logits in the prediction dictionary.
        differentiable (bool): Whether the loss component is differentiable.
    """

    name: str
    weight: float
    logit_key: str
    differentiable: bool = False

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        probs = F.softmax(pred[self.logit_key], dim=-1)
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()


@dataclass
class LLMPercentCorrect(LossComponent):
    """
    Computes the percentage of correct predictions.

    Attributes:
        name (str): The name of the loss component.
        weight (float): The weight of the loss component.
        logit_key (str): The key for accessing the logits in the prediction dictionary.
        target_key (str): The key for accessing the target values in the target dictionary.
        differentiable (bool): Whether the loss component is differentiable.
    """

    name: str
    weight: float
    pred_key: str
    ref_key: str
    differentiable: bool = False

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        logits = pred[self.pred_key][:, :-1, :]
        target_indices = target[self.ref_key][..., 1:]
        return torch.mean((torch.argmax(logits, dim=-1) == target_indices).float())
