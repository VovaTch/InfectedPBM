from __future__ import annotations
from abc import abstractmethod
import importlib
import os
from typing import Any, Protocol
import warnings

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn as nn
from loss.aggregators import LossOutput

from utils.learning import LearningParameters


class LossAggregator(Protocol):
    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> LossOutput: ...


class BaseLightningModule(L.LightningModule):
    """
    Base Lightning module class, to be inherited by all models. Contains the basic structure
    of a Lightning module, including the optimizer and scheduler configuration.
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
        """
        Initializes the BaseModel class.

        Args:
        *   model (nn.Module): The neural network model.
        *   learning_params (LearningParameters): The learning parameters for training.
        *   transforms (nn.Sequential | None, optional): The data transforms to be applied. Defaults to None.
        *   loss_aggregator (LossAggregator | None, optional): The loss aggregator for collecting losses.
            Defaults to None.
        *   optimizer_cfg (dict[str, Any] | None, optional): The configuration for the optimizer.
            Defaults to None.
        *   scheduler_cfg (dict[str, Any] | None, optional): The configuration for the scheduler.
            Defaults to None.
        """
        super().__init__()

        self.model = model
        self.learning_params = learning_params
        self.loss_aggregator = loss_aggregator
        self.transforms = transforms

        self.optimizer = self._build_optimizer(optimizer_cfg)
        self.scheduler = self._build_scheduler(scheduler_cfg)

    def _build_optimizer(
        self, optimizer_cfg: dict[str, Any] | None
    ) -> torch.optim.Optimizer:
        """
        Utility method to build the optimizer.

        Args:
            optimizer_cfg (dict[str, Any] | None): Optimizer configuration dictionary.
                The dictionary should contain the following keys:
                - 'type': The type of optimizer to be used (e.g., 'SGD', 'Adam', etc.).
                - Any additional key-value pairs specific to the chosen optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer object.

        Raises:
            AttributeError: If the specified optimizer type is not supported.
        """
        if optimizer_cfg is not None and optimizer_cfg["target"] != "none":
            filtered_optimizer_cfg = {
                key: value for key, value in optimizer_cfg.items() if key != "target"
            }
            optimizer = getattr(
                importlib.import_module(
                    ".".join(optimizer_cfg["target"].split(".")[:-1])
                ),
                optimizer_cfg["target"].split(".")[-1],
            )(self.parameters(), **filtered_optimizer_cfg)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_params.learning_rate,
                weight_decay=self.learning_params.weight_decay,
                amsgrad=True,
            )
        return optimizer

    def _build_scheduler(
        self, scheduler_cfg: dict[str, Any] | None
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        """
        Utility method to build the scheduler.

        Args:
            scheduler_cfg (dict[str, Any] | None): Scheduler configuration dictionary.

        Returns:
            torch.optim.lr_scheduler._LRScheduler | None: The built scheduler object,
            or None if scheduler_cfg is None.
        """
        if scheduler_cfg is not None and scheduler_cfg["target"] != "none":
            filtered_schedulers_cfg = {
                key: value
                for key, value in scheduler_cfg.items()
                if key not in ["target", "module_params"]
            }
            scheduler = getattr(
                importlib.import_module(
                    ".".join(scheduler_cfg["target"].split(".")[:-1])
                ),
                scheduler_cfg["target"].split(".")[-1],
            )(self.optimizer, **filtered_schedulers_cfg)
        else:
            scheduler = None
        return scheduler

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Optimizer configuration Lightning module method. If no scheduler, returns only optimizer.
        If there is a scheduler, returns a settings dictionary and returned to be used during training.

        Returns:
            OptimizerLRScheduler: Method output, used internally.
        """
        if self.scheduler is None:
            return [self.optimizer]

        scheduler_settings = self._configure_scheduler_settings(
            self.learning_params.interval,
            self.learning_params.loss_monitor,
            self.learning_params.frequency,
        )
        return [self.optimizer], [scheduler_settings]  # type: ignore

    def _configure_scheduler_settings(
        self, interval: str, monitor: str, frequency: int
    ) -> dict[str, Any]:
        """
        Utility method to return scheduler configurations to `self.configure_optimizers` method.

        Args:
            interval (str): Intervals to use the scheduler, either 'step' or 'epoch'.
            monitor (str): Loss to monitor and base the scheduler on.
            frequency (int): Frequency to potentially use the scheduler.

        Raises:
            AttributeError: Must include a scheduler

        Returns:
            dict[str, Any]: Scheduler configuration dictionary
        """
        if self.scheduler is None:
            raise AttributeError("Must include a scheduler")
        return {
            "scheduler": self.scheduler,
            "interval": interval,
            "monitor": monitor,
            "frequency": frequency,
        }

    @abstractmethod
    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward method, to be implemented in a subclass

        Args:
            input (dict[str, torch.Tensor]): Input dictionary of tensors

        Returns:
            dict[str, torch.Tensor]: Output dictionary of tensors
        """
        ...

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        """
        Pytorch Lightning standard training step. Uses the loss aggregator to compute the total loss.

        Args:
            batch (dict[str, Any]): Data batch in a form of a dictionary
            batch_idx (int): Data index

        Raises:
            AttributeError: For training, an optimizer is required (usually shouldn't come to this).
            AttributeError: For training, must include a loss aggregator.

        Returns:
            STEP_OUTPUT: total loss output
        """
        if self.optimizer is None:
            raise AttributeError("For training, an optimizer is required.")
        if self.loss_aggregator is None:
            raise AttributeError("For training, must include a loss aggregator.")
        return self.step(batch, "training")  # type: ignore

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> STEP_OUTPUT | None:
        """
        Pytorch lightning validation step. Does not require a loss object this time, but can use it.


        Args:
            batch (dict[str, Any]): Data batch in a form of a dictionary
            batch_idx (int): Data index

        Returns:
            STEP_OUTPUT | None: total loss output if there is an aggregator, none if there isn't.
        """
        return self.step(batch, "validation")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT | None:
        """
        Pytorch lightning test step. Uses the loss aggregator to compute and display all losses during the test
        if there is an aggregator.

        Args:
            batch (dict[str, Any]): Data batch in a form of a dictionary
            batch_idx (int): Data index

        Returns:
            STEP_OUTPUT | None: total loss output if there is an aggregator, none if there isn't.
        """
        output = self.forward(batch)
        if self.loss_aggregator is None:
            return
        loss = self.loss_aggregator(output, batch)
        for ind_loss, value in loss.individual.items():
            self.log(
                f"test_{ind_loss}", value, prog_bar=True, on_step=False, on_epoch=True
            )
        self.log(
            "test_total",
            loss.total,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    @abstractmethod
    def step(self, batch: dict[str, Any], phase: str) -> torch.Tensor | None:
        """
        Utility method to perform the network step and inference.

        Args:
            batch (dict[str, Any]): Data batch in a form of a dictionary
            phase (str): Phase, used for logging purposes.

        Returns:
            torch.Tensor | None: Either the total loss if there is a loss aggregator, or none if there is no aggregator.
        """
        ...


def load_inner_model_state_dict(
    module: BaseLightningModule, checkpoint_path: str
) -> BaseLightningModule:
    """
    Loads the state dictionary of the inner model from a checkpoint file. If the checkpoint file is not found,
    or if an error occurs while loading the checkpoint, the model is returned without loading pre-trained weights.

    Args:
        module (BaseLightningModule): The base lightning module.
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        BaseLightningModule: The base lightning module with the loaded state dictionary.

    """
    if not os.path.isfile(checkpoint_path):
        warnings.warn(
            f"Checkpoint file not found at {checkpoint_path}, skipping weight loading."
        )
        return module

    try:
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["state_dict"]
        module.load_state_dict(state_dict)

    except Exception as e:
        warnings.warn(f"Error loading checkpoint: {e}, loading model without weights.")

    finally:
        return module
