import torch
import torch.nn as nn

from ..codebooks.rq import RQCodeBook


class VQ1D(nn.Module):
    """
    VQ1D module that performs vector quantization on 1D input tensors.

    Args:
        token_dim (int): The dimensionality of each token.
        num_tokens (int): The number of tokens in the codebook.

    Attributes:
        vq_codebook (VQCodebook): The codebook used for vector quantization.

    Methods:
        forward(z_e, extract_losses=False): Performs forward pass of the VQ1D module.

    Returns:
        dict: A dictionary containing the output of the forward pass.
    """

    def __init__(
        self, token_dim: int, num_tokens: int = 8192, num_rq_steps: int = 1
    ) -> None:
        """
        Initialize the Block class.

        Args:
            token_dim (int): The dimensionality of each token.
            num_tokens (int, optional): The number of tokens in the codebook. Defaults to 8192.
            num_rq_steps (int, optional): The number of residual steps to use. Defaults to 1.
        """
        super().__init__()
        self.vq_codebook = RQCodeBook(
            token_dim, num_rq_steps=num_rq_steps, num_tokens=num_tokens
        )
        # self.vq_codebook = VQCodebook(token_dim=token_dim, num_tokens=num_tokens)

    def forward(
        self, z_e: torch.Tensor, extract_losses: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the block.

        Args:
            z_e (torch.Tensor): Input tensor.
            extract_losses (bool, optional): Whether to extract losses. Defaults to False.

        Returns:
            dict[str, torch.Tensor]: Output dictionary containing indices and v_q.

        """
        z_q, indices = self.vq_codebook.apply_codebook(z_e, code_sg=True)
        output = {"indices": indices, "v_q": z_q}

        if extract_losses:
            emb, _ = self.vq_codebook.apply_codebook(z_e.detach())
            output.update({"emb": emb})

        return output
