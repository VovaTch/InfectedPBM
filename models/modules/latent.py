from typing import Any
import torch

from .music import MusicLightningModule
from utils.waveform_tokenization import quantize_waveform_256


class MusicLatentLightningModule(MusicLightningModule):
    """
    A version suited for deeper level latents.
    """

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the music module.

        Args:
            input (dict[str, torch.Tensor]): Input dictionary containing the "slice" tensor.

        Returns:
            dict[str, torch.Tensor]: Output dictionary containing the result of the forward pass.
        """
        slice_inputs = input["latent"]
        return self.model(slice_inputs)

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
            "latent": batch["latent"],
        }
        loss = self.loss_aggregator(output, targets)
        loss_total = self.handle_loss(loss, phase)
        return loss_total
