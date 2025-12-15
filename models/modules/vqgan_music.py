from __future__ import annotations
from enum import Enum
import importlib
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.core.optimizer import LightningOptimizer

from loss.aggregators import LossOutput
from loss.components.base import LossComponent
from models.models.base import Tokenizer
from models.models.discriminator.base import Discriminator
from models.modules.base import LossAggregator
from models.modules.music import MusicLightningModule
from utils.containers import LearningParameters
from utils.waveform_tokenization import quantize_waveform_256


class ModelPart(Enum):
    """
    Enum class for model parts.
    """

    GENERATOR = "generator"
    DISCRIMINATOR = "discriminator"


class VqganMusicLightningModule(MusicLightningModule):
    """
    VQGAN version of the Multi-Level VQVAE for music generation. Uses an outside discriminator and generator
    losses to create a GAN training setup.
    """

    def __init__(
        self,
        model: Tokenizer,
        discriminator: Discriminator,
        learning_params: LearningParameters,
        discriminator_loss: LossComponent,
        generator_loss: LossComponent,
        generator_start_step: int = 10000,
        transforms: nn.Sequential | None = None,
        loss_aggregator: LossAggregator | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the VQGAN model with the given parameters.

        Args:
            model (nn.Module): The main model to be used.
            discriminator (Discriminator): The discriminator model for GAN training.
            learning_params (LearningParameters): Parameters related to learning such as learning rate.
            discriminator_loss (LossComponent): Loss component for the discriminator.
            generator_loss (LossComponent): Loss component for the generator.
            generator_start_step (int, optional): The step at which the generator starts training. Defaults to 10000.
            transforms (nn.Sequential, optional): Transformations to be applied to the input data. Defaults to None.
            loss_aggregator (LossAggregator, optional): Aggregator for the loss components. Defaults to None.
            optimizer_cfg (dict[str, Any], optional): Configuration for the optimizer. Defaults to None.
            scheduler_cfg (dict[str, Any], optional): Configuration for the scheduler. Defaults to None.
        """
        super().__init__(
            model,
            learning_params,
            transforms,
            loss_aggregator,
            optimizer_cfg,
            scheduler_cfg,
        )

        self.discriminator = discriminator
        self._generator_start_step = generator_start_step
        self.optimizer_d = self._build_optimizer_gan(
            optimizer_cfg, ModelPart.DISCRIMINATOR
        )
        self.optimizer_g = self._build_optimizer_gan(optimizer_cfg, ModelPart.GENERATOR)
        self.scheduler_d = self._build_scheduler_gan(
            scheduler_cfg, ModelPart.DISCRIMINATOR
        )
        self.scheduler_g = self._build_scheduler_gan(scheduler_cfg, ModelPart.GENERATOR)
        self._current_step = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.automatic_optimization = False  # Necessary for GANs
        self._discriminator_loss = discriminator_loss
        self._generator_loss = generator_loss

        self.optimizer = 0
        del self.scheduler

    @property
    def generator(self) -> Tokenizer:
        """
        Getter for the generator model.

        Returns:
            nn.Module: Generator model of the module
        """
        return self.model  # type: ignore

    def _build_optimizer_gan(
        self, optimizer_cfg: dict[str, Any] | None, model_part: ModelPart
    ) -> torch.optim.Optimizer:
        """
        Build and return an optimizer for the GAN model.

        This method constructs an optimizer for either the generator or discriminator
        part of the GAN model based on the provided configuration.

        Args:
            optimizer_cfg (dict[str, Any] | None): Configuration dictionary for the optimizer.
                If None or if the "target" key is set to "none", a default AdamW optimizer
                will be created.
            model_part (ModelPart): Enum indicating which part of the model the optimizer
                is for. Must be either ModelPart.GENERATOR or ModelPart.DISCRIMINATOR.

        Returns:
            torch.optim.Optimizer: The constructed optimizer for the specified model part.

        Raises:
            ValueError: If an invalid model part is provided.
        """

        if model_part == ModelPart.GENERATOR:
            parameters = self.model.parameters()
        elif model_part == ModelPart.DISCRIMINATOR:
            parameters = self.discriminator.parameters()
        else:
            raise ValueError(f"Invalid model part {model_part}")

        if optimizer_cfg is not None and optimizer_cfg["target"] != "none":
            filtered_optimizer_cfg = {
                key: value for key, value in optimizer_cfg.items() if key != "target"
            }
            optimizer = getattr(
                importlib.import_module(
                    ".".join(optimizer_cfg["target"].split(".")[:-1])
                ),
                optimizer_cfg["target"].split(".")[-1],
            )(parameters, **filtered_optimizer_cfg)
        else:
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.learning_params.learning_rate,
                weight_decay=self.learning_params.weight_decay,
                amsgrad=True,
            )
        return optimizer

    def _build_scheduler_gan(
        self, scheduler_cfg: dict[str, Any] | None, model_part: ModelPart
    ) -> LRScheduler | None:
        """
        Build and return a learning rate scheduler for the GAN model.

        This method constructs a learning rate scheduler for either the generator or discriminator
        part of the GAN model based on the provided configuration.

        Args:
            scheduler_cfg (dict[str, Any] | None): Configuration dictionary for the scheduler.
                If None or if the "target" key is set to "none", no scheduler will be created.
            model_part (ModelPart): Enum indicating which part of the model the scheduler
                is for. Must be either ModelPart.GENERATOR or ModelPart.DISCRIMINATOR.

        Returns:
            LRScheduler | None: The constructed learning rate scheduler for the specified model part.
        """

        if scheduler_cfg is None or scheduler_cfg["target"] == "none":
            return None

        match model_part:
            case ModelPart.GENERATOR:
                optimizer = self.optimizer_g
            case ModelPart.DISCRIMINATOR:
                optimizer = self.optimizer_d
            case _:
                raise ValueError(f"Invalid model part {model_part}")

        filtered_scheduler_cfg = {
            key: value
            for key, value in scheduler_cfg.items()
            if key not in ["target", "module_params"]
        }
        scheduler = getattr(
            importlib.import_module(".".join(scheduler_cfg["target"].split(".")[:-1])),
            scheduler_cfg["target"].split(".")[-1],
        )(optimizer, **filtered_scheduler_cfg)

        return scheduler

    def calculate_adaptive_weight(
        self,
        rec_loss: torch.Tensor,
        generator_loss: torch.Tensor,
        last_layer: nn.Conv1d,
    ):
        """
        Calculate the adaptive weight for the discriminator loss.

        This function computes the adaptive weight based on the gradients of the
        reconstruction loss and the generator loss with respect to the last layer
        of the model. The adaptive weight is used to balance the contributions of
        the reconstruction loss and the generator loss during training.

        Args:
            rec_loss (torch.Tensor): The reconstruction loss.
            generator_loss (torch.Tensor): The generator loss.
            last_layer (torch.Tensor): The last layer of the model.

        Returns:
            torch.Tensor: The adaptive weight for the discriminator loss.
        """
        rec_grads = torch.autograd.grad(rec_loss, last_layer.weight, retain_graph=True)[
            0
        ]
        generator_grads = torch.autograd.grad(
            generator_loss, last_layer.weight, retain_graph=True
        )[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(generator_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def _configure_scheduler_settings(
        self, scheduler: LRScheduler, interval: str, monitor: str, frequency: int
    ) -> dict[str, Any]:
        """
        Utility method to return scheduler configurations to `self.configure_optimizers` method.

        Args:
            interval (str): Intervals to use the scheduler, either 'step' or 'epoch'.
            monitor (str): Loss to monitor and base the scheduler on.
            frequency (int): Frequency to potentially use the scheduler.
            scheduler

        Raises:
            AttributeError: Must include a scheduler

        Returns:
            dict[str, Any]: Scheduler configuration dictionary
        """
        return {
            "scheduler": scheduler,
            "interval": interval,
            "monitor": monitor,
            "frequency": frequency,
        }

    def configure_optimizers(
        self,
    ) -> OptimizerLRScheduler:
        """
        Configures the optimizers and learning rate schedulers for the model.

        Returns:
            OptimizerLRScheduler: A tuple containing a list of optimizers and a list of learning rate schedulers.
                If no scheduler is provided, returns the optimizers with an empty list for schedulers.

        Raises:
            AttributeError: If `self.scheduler` or `self.learning_params` attributes are not defined.

        Note:
            The method assumes that `self.optimizer_d` and `self.optimizer_g` are defined optimizers for the model.
            The method also assumes that `self._configure_scheduler_settings` is a method that configures scheduler
            settings.
        """
        if self.scheduler_d is None or self.scheduler_g is None:
            return [self.optimizer_d, self.optimizer_g], []

        scheduler_settings_g = self._configure_scheduler_settings(
            self.scheduler_g,
            self.learning_params.interval,
            self.learning_params.loss_monitor,
            self.learning_params.frequency,
        )
        scheduler_settings_d = self._configure_scheduler_settings(
            self.scheduler_d,
            self.learning_params.interval,
            self.learning_params.loss_monitor,
            self.learning_params.frequency,
        )
        self.optimizer = self.optimizer_g

        return [self.optimizer_d, self.optimizer_g], [
            scheduler_settings_d,
            scheduler_settings_g,
        ]  # type: ignore

    def step(self, batch: dict[str, Any], phase: str) -> torch.Tensor | None:

        if phase == "training":
            self._current_step.data = self._current_step.data + 1

        optimizer_d, optimizer_g = self.optimizers()  # type: ignore
        scheduler_d, scheduler_g = None, None
        scheduler_list = self.lr_schedulers()
        if scheduler_list is not None and len(scheduler_list) > 0:  # type: ignore
            scheduler_d, scheduler_g = scheduler_list  # type: ignore

        self._discriminator_step(batch, phase, optimizer_d, scheduler_d)
        return self._generator_step(batch, phase, optimizer_g, scheduler_g)

    def _close_generator_phase(
        self,
        phase: str,
        loss: LossOutput,
        d_weight: torch.Tensor | float,
        generator_loss: torch.Tensor | int,
        optimizer_g: LightningOptimizer,
    ) -> None:
        """
        Finalizes the generator phase by logging the generator loss, handling the loss,
        and un-toggling the optimizer.

        Args:
            phase (str): The current phase of the training process.
            loss (LossOutput): The loss output from the model.
            d_weight (torch.Tensor | float): The discriminator weight.
            generator_loss (torch.Tensor | int): The computed generator loss.
            optimizer_g (LightningOptimizer): The optimizer for the generator.
        """
        self.log(
            f"{phase}/generator loss",
            generator_loss / (d_weight + 1e-6),
            sync_dist=True,
            batch_size=self.learning_params.batch_size,
        )
        self.handle_loss(loss, phase)
        self.untoggle_optimizer(optimizer_g)

    def _discriminator_step(
        self,
        batch: dict[str, Any],
        phase: str,
        optimizer_d: LightningOptimizer,
        scheduler_d: LRScheduler | None,
    ) -> None:
        """
        Perform a discriminator step during training or validation.

        Args:
            batch (dict[str, Any]): A dictionary containing the input batch data.
            phase (str): The current phase, either "training" or "validation".
            optimizer_d (LightningOptimizer): The optimizer for the discriminator.
            scheduler_d (LRScheduler | None): The learning rate scheduler for the discriminator, if any.
        """
        self.toggle_optimizer(optimizer_d)
        disc_outputs_real = self.discriminator(batch["slice"])
        with torch.no_grad():
            restructured_outputs_frozen = self.forward(batch)
            fake_slice = restructured_outputs_frozen["slice"].detach()

        disc_outputs_fake = self.discriminator(fake_slice.requires_grad_())
        disc_outputs = {
            "d_input": disc_outputs_real["logits"],
            "d_output": disc_outputs_fake["logits"],
        }
        disc_outputs.update(restructured_outputs_frozen)
        disc_loss = self._discriminator_loss(disc_outputs, {})
        if phase == "training" and self._current_step >= self._generator_start_step:
            self.manual_backward(disc_loss * self._discriminator_loss.weight)
            optimizer_d.step()
            optimizer_d.zero_grad()

        self.log(
            f"{phase}/discriminator loss",
            disc_loss if self._current_step >= self._generator_start_step else 0,
            sync_dist=True,
            batch_size=self.learning_params.batch_size,
        )
        if scheduler_d is not None and phase == "training":
            scheduler_d.step()  # type: ignore
        self.untoggle_optimizer(optimizer_d)

    def _generator_step(
        self,
        batch: dict[str, Any],
        phase: str,
        optimizer_g: LightningOptimizer,
        scheduler_g: LRScheduler | None,
    ) -> torch.Tensor | None:
        """
        Perform a single optimization step for the generator.

        Args:
            batch (dict[str, Any]): A dictionary containing the input batch data.
            phase (str): The current phase of training (e.g., "training", "validation").
            optimizer_g (LightningOptimizer): The optimizer for the generator.
            scheduler_g (LRScheduler | None): The learning rate scheduler for the generator, if any.

        Returns:
            torch.Tensor | None: The total loss tensor if the loss aggregator is defined, otherwise None.
        """
        self.toggle_optimizer(optimizer_g)

        restructured_outputs = self.forward(batch)
        disc_outputs_fake = self.discriminator(restructured_outputs["slice"])
        restructured_outputs["d_output"] = disc_outputs_fake["logits"]
        disc_outputs_real = self.discriminator(batch["slice"])
        restructured_outputs["d_input"] = disc_outputs_real["logits"]
        targets = {
            "z_e": restructured_outputs["z_e"],
            "slice": batch["slice"],
            "class": quantize_waveform_256(batch["slice"]).long(),
        }

        if self.loss_aggregator is None:
            self.untoggle_optimizer(optimizer_g)
            return None

        loss = self.loss_aggregator(restructured_outputs, targets)
        generator_loss = self._generator_loss(restructured_outputs, {})

        if phase != "training":
            self._close_generator_phase(
                phase=phase,
                loss=loss,
                d_weight=1.0,
                generator_loss=generator_loss,
                optimizer_g=optimizer_g,
            )
            return loss.total.clone()

        if self._current_step >= self._generator_start_step:
            d_weight = self.calculate_adaptive_weight(
                loss.total,
                generator_loss,
                self.generator.last_layer,  # type: ignore
            )
            generator_loss = generator_loss * d_weight
            self.log(
                "d_weight",
                d_weight,
                sync_dist=True,
                batch_size=self.learning_params.batch_size,
            )
            self.manual_backward(
                loss.total + generator_loss * self._generator_loss.weight
            )
        else:
            generator_loss = 0
            d_weight = 0
            self.log("d_weight", 0, sync_dist=True)
            self.manual_backward(loss.total)

        optimizer_g.step()
        optimizer_g.zero_grad()

        if scheduler_g is not None and phase == "training":
            scheduler_g.step()  # type: ignore

        self._close_generator_phase(
            phase=phase,
            loss=loss,
            d_weight=d_weight if phase == "training" else 1.0,
            generator_loss=generator_loss,
            optimizer_g=optimizer_g,
        )
        return loss.total.clone()
