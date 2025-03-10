from typing import Any

import torch.nn as nn

from utils.containers import LearningParameters
from .base import BaseLightningModule, LossAggregator


class DiffusionLLMLightningModule(BaseLightningModule):
    """
    A module to perform training using the method presented in LLaDa paper.
    https://arxiv.org/pdf/2502.09992
    """

    def __init__(
        self,
        model: nn.Module,
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
