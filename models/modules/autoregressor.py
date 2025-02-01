from typing import Any
import torch

from loss.aggregators import LossOutput

from .base import BaseLightningModule


class AutoRegressorModule(BaseLightningModule):
    """
    Module designed to handle Mamba/Transformer model training.
    """

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the autoregressor module.

        Args:
            input (dict[str, torch.Tensor]): Input dictionary containing the indices.

        Returns:
            dict[str, torch.Tensor]: Output dictionary containing the model predictions.
        """
        indices = input["indices"]
        return self.model(indices)

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

    def step(self, batch: dict[str, Any], phase: str) -> torch.Tensor | None:
        """
        Performs a single step of the autoregressor module.

        Args:
            batch (dict[str, Any]): The input batch.
            phase (str): The phase of the training process.

        Returns:
            torch.Tensor | None: The total loss for the step, or None if the loss aggregator is not defined.
        """
        output = self.forward(batch)
        if self.loss_aggregator is None:
            return None
        targets = {"class": batch["target"]}
        loss = self.loss_aggregator(output, targets)
        loss_total = self.handle_loss(loss, phase)
        return loss_total
