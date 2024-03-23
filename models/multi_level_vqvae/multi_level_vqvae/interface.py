from typing import Protocol

import torch

from models.base import Codebook


class InterfaceVQ1D(Protocol):
    """
    Interface for the VQ1D class.
    """

    def __call__(
        self, z_e: torch.Tensor, extract_losses: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Call method for the VQ1D class.
        """
        ...

    @property
    def vq_codebook(self) -> Codebook:
        """
        Codebook property for the VQ1D class.
        """
        ...
