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
        self.log(f"{phase} total loss", loss.total)
        return loss.total

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

        optimizer_cfg_raw: dict[str, Any] = cfg.learning.optimizer
        optimizer_cfg = (
            {key: value for (key, value) in optimizer_cfg_raw.items() if key != "type"}
            if optimizer_cfg_raw["type"] != "none"
            else None
        )
        scheduler_cfg_raw: dict[str, Any] = cfg.learning.scheduler
        scheduler_cfg = (
            {key: value for (key, value) in scheduler_cfg_raw.items() if key != "type"}
            if scheduler_cfg_raw["type"] != "none"
            else None
        )

        return cls(
            model,
            learning_parameters,
            None,
            loss_aggregator,
            optimizer_cfg,
            scheduler_cfg,
        )
