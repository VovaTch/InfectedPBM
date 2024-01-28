from typing import Any, Self

import torch
from omegaconf import DictConfig

from common import registry
from loss.aggregators import LossOutput
from models.base import BaseLightningModule
from utils.containers import LearningParameters


@registry.register_lightning_module("music")
class MusicLightningModule(BaseLightningModule):
    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the music module.

        Args:
            input (dict[str, torch.Tensor]): Input dictionary containing the "slice" tensor.

        Returns:
            dict[str, torch.Tensor]: Output dictionary containing the result of the forward pass.
        """
        slice_inputs = input["slice"]
        return self.model(slice_inputs)

    def handle_loss(self, loss: LossOutput, phase: str) -> torch.Tensor:
        """
        Handles the loss calculation and logging (to Tensorboard).

        Args:
            loss (LossOutput): The loss output object containing individual losses.
            phase (str): The phase of the training (e.g., "train", "val").

        Returns:
            torch.Tensor: The total loss.

        """
        for name in loss.individuals:
            log_name = f"{phase} {name.replace('_', ' ')}"
            self.log(log_name, loss.individuals[name])
        self.log(f"{phase} total loss", loss.total, prog_bar=True)
        return loss.total

    def step(self, batch: dict[str, Any], phase: str) -> torch.Tensor | None:
        """
        Utility method to perform the network step and inference.

        Args:
            batch (dict[str, Any]): Data batch in a form of a dictionary
            phase (str): Phase, used for logging purposes.

        Returns:
            torch.Tensor | None: Either the total loss if there is a loss aggregator, or none if there is no aggregator.
        """
        output = self.forward(batch)
        if self.loss_aggregator is None:
            return
        targets = {"z_e": output["z_e"], "slice": batch["slice"]}
        loss = self.loss_aggregator(output, targets)
        loss_total = self.handle_loss(loss, phase)
        return loss_total

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Create a MusicModule instance from a configuration dictionary.

        Args:
            cfg (DictConfig): The configuration dictionary.

        Returns:
            MusicModule: The created MusicModule instance.
        """
        model = registry.get_model(cfg.model.type).from_cfg(cfg)  # type: ignore
        learning_parameters = LearningParameters.from_cfg(cfg)
        loss_aggregator = (
            registry.get_loss_aggregator(cfg.loss.aggregator.type).from_cfg(cfg)
            if cfg.loss.aggregator.type != "none"
            else None
        )

        optimizer_cfg: dict[str, Any] = cfg.learning.optimizer
        scheduler_cfg: dict[str, Any] = cfg.learning.scheduler

        return cls(
            model,
            learning_parameters,
            None,
            loss_aggregator,
            optimizer_cfg,
            scheduler_cfg,
        )
