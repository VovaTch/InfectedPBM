from typing import Any

import torch

from loss.aggregators import LossOutput
from utils.waveform_tokenization import quantize_waveform_256
from .base import BaseLightningModule


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
        targets = {
            "z_e": output["z_e"],
            "slice": batch["slice"],
            "class": quantize_waveform_256(batch["slice"]).long(),
        }
        loss = self.loss_aggregator(output, targets)
        loss_total = self.handle_loss(loss, phase)
        return loss_total

    def on_train_epoch_end(self) -> None:
        """
        Callback function called at the end of each training epoch.
        Randomly restarts the VQ codebook and resets its usage.
        """
        if hasattr(self.model, "vq_module"):
            num_dead_codes = self.model.vq_module.vq_codebook.random_restart()  # type: ignore # TODO: check if right
            self.model.vq_module.vq_codebook.reset_usage()  # type: ignore # TODO: check if right
            self.log(
                "number of dead codes",
                num_dead_codes,
                sync_dist=True,
                batch_size=self.learning_params.batch_size,
            )
