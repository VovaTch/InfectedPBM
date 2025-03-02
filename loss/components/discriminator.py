from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .base import LossComponent


@dataclass
class DiscriminatorLoss(LossComponent):
    """
    Discriminator loss component for the GAN training.

    Args:
        name (str): The name of the loss component.
        weight (float): The weight of the loss component.
        pred_key_real (str): The key of the real prediction.
        pred_key_fake (str): The key of the fake prediction.
        differentiable (bool): Whether the loss is differentiable. Defaults
            to True.
    """

    name: str
    weight: float
    pred_key_real: str
    pred_key_fake: str
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], _: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return 0.5 * (
            torch.mean(F.softplus(-pred[self.pred_key_real][..., 0]))
            + torch.mean(F.softplus(pred[self.pred_key_fake][..., 0]))
        )


@dataclass
class DiscriminatorHingeLoss(LossComponent):

    name: str
    weight: float
    pred_key_real: str
    pred_key_fake: str
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], _: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        loss_real = torch.mean(F.relu(1 - pred[self.pred_key_real][..., 0]))
        loss_fake = torch.mean(F.relu(1 + pred[self.pred_key_fake][..., 0]))
        return 0.5 * (loss_real + loss_fake)


@dataclass
class GeneratorLoss(LossComponent):

    name: str
    weight: float
    pred_key_disc: str
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], _: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return -torch.mean(pred[self.pred_key_disc][..., 0])


@dataclass
class GeneratorHingeLoss(LossComponent):

    name: str
    weight: float
    pred_key_disc: str
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], _: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.mean(torch.clamp(1 - pred[self.pred_key_disc][..., 0], min=0))
