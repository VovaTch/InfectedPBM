from typing import Any

import torch
import torch.nn as nn

from loss.aggregators import LossOutput
from utils.containers import LearningParameters
from .base import BaseLightningModule, LossAggregator


class RippleNetModule(BaseLightningModule):
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

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size, _, slice_length = input["slice"].shape
        arranged_range = (
            torch.arange(slice_length, device=input["slice"].device)
            .reshape(1, -1, 1)
            .repeat(batch_size, 1, 1)
            .to(input["slice"].device)
        )
        reconstructed_slice: torch.Tensor = self.model(arranged_range)
        return {"slice": reconstructed_slice.view(batch_size, 1, -1)}

    def step(self, batch: dict[str, Any], phase: str) -> torch.Tensor | None:
        outputs = self.forward(batch)
        if self.loss_aggregator is None:
            return None
        targets = {"slice": batch["slice"]}
        loss = self.loss_aggregator(outputs, targets)
        loss_total = self.handle_loss(loss, phase)
        return loss_total

    def handle_loss(self, loss: LossOutput, phase: str) -> torch.Tensor:
        """
        Handles the loss calculation and logging (to Tensorboard).

        Args:
            loss (LossOutput): The loss output object containing individual losses.
            phase (str): The phase of the training (e.g., "train", "val").

        Returns:
            torch.Tensor: The total loss.

        """
        for name in loss.individual:
            log_name = f"{phase} {name.replace('_', ' ')}"
            self.log(log_name, loss.individual[name])
        self.log(f"{phase} total loss", loss.total, prog_bar=True)
        return loss.total
