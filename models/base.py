from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol, Self, TYPE_CHECKING

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from omegaconf import DictConfig
import torch
import torch.nn as nn

from utils.containers import LearningParameters

if TYPE_CHECKING:
    from loss.aggregators import LossAggregator, LossOutput


class Tokenizer(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor x into a compressed representation.

        Args:
            x (torch.Tensor): The input tensor to be encoded.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        ...

    @abstractmethod
    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor `x` into indices.

        Args:
            x (torch.Tensor): The input tensor to be encoded.

        Returns:
            torch.Tensor: The encoded tensor with indices.
        """
        ...

    @abstractmethod
    def from_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the input tensor `x` from indices.

        Args:
            x (torch.Tensor): The input tensor to be decoded.

        Returns:
            torch.Tensor: The decoded tensor.
        """
        ...

    @abstractmethod
    def decode(
        self, z_e: torch.Tensor, origin_shape: tuple[int, int, int] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Decodes the given latent tensor `z_e` into the original image representation.

        Args:
            z_e (torch.Tensor): The latent tensor to be decoded.
            origin_shape (tuple[int, int, int] | None, optional): The shape of the original image. Defaults to None.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: A tuple containing the decoded image tensor and additional information.
        """
        ...

    @classmethod
    @abstractmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> Self:
        """
        Create an instance of the class from a configuration dictionary.

        Args:
            cfg (dict[str, Any]): The configuration dictionary.

        Returns:
            Self: An instance of the class.

        """
        ...


class Codebook(Protocol):
    def embed_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Embeds the given indices using the codebook.

        Args:
            indices (torch.Tensor): The indices to be embedded.

        Returns:
            torch.Tensor: The embedded representation of the indices.
        """
        ...

    def apply_codebook(
        self, x_in: torch.Tensor, code_sg: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies a codebook to the input tensor.

        Args:
            x_in (torch.Tensor): The input tensor.
            code_sg (bool, optional): Whether to use codebook for stochastic gradient. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed tensor and the codebook tensor.
        """
        ...

    def update_usage(self, min_enc: torch.Tensor) -> None:
        """
        Update the usage of the model based on the minimum encoding.

        Args:
            min_enc (torch.Tensor): The minimum encoding.

        Returns:
            None
        """
        ...

    def reset_usage(self) -> None:
        """
        Resets the usage of the object.
        """
        ...

    def random_restart(self) -> None:
        """
        Performs a random restart for the optimization algorithm.
        """
        ...


class BaseLightningModule(L.LightningModule):
    """
    Base Pytorch Lightning Module to handle training, validation, testing, logging into Tensorboard, etc.
    The model itself is passed as a Pytorch Module, so this Lightning Module is not limited to a single model.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_params: LearningParameters,
        transforms: nn.Sequential | None = None,
        loss_aggregator: "LossAggregator | None" = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        """
        Constructor method

        Args:
            *   model (nn.Module): Base Pytorch model
            *   learning_params (LearningParameters): Learning parameters object containing all parameters required for
                learning.
            *   transforms (nn.Sequential | None, optional): Image transformation sequence, if None, no
                transforms are performed. Defaults to None.
            *   loss_aggregator (LossAggregator | None, optional): Loss object that is composed of multiple components.
                If None, raises an exception when attempting to train. Defaults to None.
            *   optimizer_builder (OptimizerBuilder | None, optional): Optimizer builder function.
                Programmed in this way because it requires a model, the function is called during initialization.
                If None, then AdamW is used. Defaults to None.
            *   scheduler_builder (SchedulerBuilder | None, optional): Scheduler builder function.
                Programmed in this way because it requires a model, the function is called during initialization.
                If None, no scheduler is used. Defaults to None.
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
        if optimizer_cfg is not None and optimizer_cfg["type"] != "none":
            filtered_optimizer_cfg = {
                key: value for key, value in optimizer_cfg.items() if key != "type"
            }
            optimizer = getattr(torch.optim, optimizer_cfg["type"])(
                self.parameters(), **filtered_optimizer_cfg
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_params.learning_rate,
                weight_decay=self.learning_params.weight_decay,
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
        # Build scheduler
        if scheduler_cfg is not None and scheduler_cfg["type"] != "none":
            filtered_schedulers_cfg = {
                key: value for key, value in scheduler_cfg.items() if key != "type"
            }
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_cfg["type"])(
                self.optimizer, **filtered_schedulers_cfg
            )
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
        # Pytorch-Lightning specific

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
        for ind_loss, value in loss.individuals.items():
            self.log(
                f"test_{ind_loss}", value, prog_bar=True, on_step=False, on_epoch=True
            )
        self.log(f"test_total", loss.total, prog_bar=True, on_step=False, on_epoch=True)

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

    @classmethod
    @abstractmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Create an instance of the class from a configuration dictionary.

        Args:
            cfg (DictConfig): The configuration dictionary.

        Returns:
            Self: An instance of the class.

        """
        ...
