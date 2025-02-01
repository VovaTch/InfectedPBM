from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AlignLoss:
    """
    VQ-VAE codebook alignment loss
    """

    name: str
    weight: float
    base_loss: nn.Module
    differentiable: bool = True

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the VQ-VAE codebook alignment loss

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: Loss
        """
        emb = estimation["emb"]
        z_e = target["z_e"]

        loss = torch.tensor((0.0)).to(z_e.device)
        for codebook_idx in range(emb.shape[1]):
            # codebook_residual = emb[:, -1, ...] - emb[:, codebook_idx, ...]
            loss += self.base_loss(emb[:, codebook_idx, ...], z_e.detach())

        return loss


@dataclass
class CommitLoss:
    """
    VQ-VAE codebook commitment loss
    """

    name: str
    weight: float
    base_loss: nn.Module
    differentiable: bool = True

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the VQ-VAE codebook commitment loss

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: Loss
        """
        emb = estimation["emb"]
        z_e = target["z_e"]

        loss = torch.tensor((0.0)).to(z_e.device)
        for codebook_idx in range(emb.shape[1]):
            # codebook_residual = emb[:, -1, ...] - emb[:, codebook_idx, ...]
            loss += self.base_loss(emb[:, codebook_idx, ...].detach(), z_e)

        return loss
