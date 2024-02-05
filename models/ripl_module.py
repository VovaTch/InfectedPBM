from typing import Any
from typing_extensions import Self
from matplotlib import pyplot as plt

from omegaconf import DictConfig
import torch
from torch.nn.modules import Module, Sequential

from loss.aggregators import LossAggregator, LossOutput
from models.base import BaseLightningModule
from utils.containers import LearningParameters
from common import registry


@registry.register_lightning_module("ripl")
class RippleNetModule(BaseLightningModule):
    def __init__(
        self,
        model: Module,
        learning_params: LearningParameters,
        transforms: Sequential | None = None,
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
        for name in loss.individuals:
            log_name = f"{phase} {name.replace('_', ' ')}"
            self.log(log_name, loss.individuals[name])
        self.log(f"{phase} total loss", loss.total, prog_bar=True)
        return loss.total

    @classmethod
    def from_cfg(cls, cfg: DictConfig, weights: str | None = None) -> Self:
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
