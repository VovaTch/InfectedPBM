import torch
from models.models.discriminator.base import Discriminator


class EnsembleDiscriminator(Discriminator):
    """
    A discriminator consists of multiple sub-discriminators.
    """

    def __init__(self, discriminators: list[Discriminator]) -> None:
        super().__init__()

        self._discriminators = discriminators

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        discriminator_output = [
            discriminator.forward(x)["logits"] for discriminator in self._discriminators
        ]
        concatenated_output = torch.cat(discriminator_output, dim=1)
        return {"logits": concatenated_output}

    @property
    def last_layer(self) -> torch.nn.Module:
        return self._discriminators[-1].last_layer
