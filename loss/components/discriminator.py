from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .base import LossComponent


@dataclass
class DiscriminatorLoss(LossComponent):
    name: str
    weight: float
    pred_key_real: str
    pred_key_fake: str
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], _: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        log_sm_real = F.log_softmax(pred[self.pred_key_real], dim=-1)
        log_sm_fake = F.log_softmax(pred[self.pred_key_fake], dim=-1)
        return torch.mean(log_sm_real[..., 1] + log_sm_fake[..., 0])
