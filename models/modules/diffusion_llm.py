from typing import Any

import torch
import torch.nn as nn

from loss.aggregators import LossOutput
from utils.containers import LearningParameters
from utils.sample_schedulers.base import SampleScheduler
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
        sample_scheduler: SampleScheduler,
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
        self._sample_scheduler = sample_scheduler

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.model(input["latent"], input["conditional"], input["mask"])
        return {"logits": outputs}

    def step(self, batch: dict[str, Any], phase: str) -> torch.Tensor | None:
        output = self.forward(batch)
        if self.loss_aggregator is None:
            return
        loss = self.loss_aggregator(output, batch)
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
            self.log(
                log_name,
                loss.individual[name],
                sync_dist=True,
                batch_size=self.learning_params.batch_size,
            )
        self.log(
            f"{phase} total loss",
            loss.total,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.learning_params.batch_size,
        )
        return loss.total

    def generate(
        self,
        init_latent: torch.Tensor | None = None,
        init_step: int = 0,
        conditional: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generates a tensor based on the provided initial latent tensor, initial step, and conditional tensor.

        Args:
            init_latent (torch.Tensor, optional): The initial latent tensor to start the generation process.
                Defaults to None.
            init_step (int, optional): The initial step to start the generation process. Defaults to 0.
            conditional (torch.Tensor, optional): A tensor providing conditional information for the generation process.
                Defaults to None.

        Returns:
            torch.Tensor: The generated tensor.
        """
        raise NotImplementedError("TODO: needs to be implemented")
