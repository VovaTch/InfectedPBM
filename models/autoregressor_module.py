from __future__ import annotations
from typing import Any
from typing_extensions import Self

from omegaconf import DictConfig
import torch

from common import registry
from loss.aggregators import LossOutput
from models.base import BaseLightningModule
from utils.containers import LearningParameters


@registry.register_lightning_module("autoregressor")
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
        for name in loss.individuals:
            log_name = f"{phase} {name.replace('_', ' ')}"
            self.log(log_name, loss.individuals[name])
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

    @classmethod
    def from_cfg(cls, cfg: DictConfig, weights: str | None = None) -> Self:
        """
        Create a MusicModule instance from a configuration dictionary.

        Args:
            cfg (DictConfig): The configuration dictionary.
            weights (str | None): Path to the weights file to load. Defaults to None.

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

        model_params = {
            "model": model,
            "learning_params": learning_parameters,
            "transforms": None,
            "loss_aggregator": loss_aggregator,
            "optimizer_cfg": optimizer_cfg,
            "scheduler_cfg": scheduler_cfg,
        }

        if weights is None:
            return cls(**model_params)
        else:
            return cls.load_from_checkpoint(weights, **model_params)
