from typing import Any

import torch
import torch.nn as nn

from models.modules.base import LossAggregator
from models.modules.music import MusicLightningModule
from utils.containers import LearningParameters


class VqganMusicLightningModule1(MusicLightningModule):

    def __init__(
        self,
        model: nn.Module,
        discriminator: nn.Module,
        learning_params: LearningParameters,
        transforms: nn.Sequential | None = None,
        loss_aggregator: LossAggregator | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model,
            learning_params,
            transforms,
            loss_aggregator,
            optimizer_cfg,
            scheduler_cfg,
        )

        self._discriminator = discriminator

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        vq_outputs = super().forward(input)
        vq_outputs["d_input"] = self._discriminator(input["slice"])
        vq_outputs["d_output"] = self._discriminator(vq_outputs["slice"])
        return vq_outputs
