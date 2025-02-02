import torch
import torch.nn as nn

from .vq import VQCodeBook


class RQCodeBook(VQCodeBook):
    def __init__(
        self,
        token_dim: int,
        num_rq_steps: int,
        num_tokens: int = 512,
        usage_threshold: float = 1e-9,
    ) -> None:
        super().__init__(token_dim, num_tokens, usage_threshold)
        self._num_rq_steps = num_rq_steps

    def apply_codebook(
        self, x_in: torch.Tensor, code_sg: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the codebook to the input tensor `x_in` over multiple residual quantization steps.

        Args:
            x_in (torch.Tensor): The input tensor to which the codebook will be applied, size (BS, SeqL, C).
            code_sg (bool, optional): If True, stops the gradient for the codebook. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - z_q_aggregated (torch.Tensor): The aggregated quantized tensor over all residual quantization steps,
                    size (BS, Ncb, SeqL, C).
                - indices (torch.Tensor): The indices of the codebook entries used during quantization,
                    size (BS, SeqL, Ncb).
        """
        x_res = x_in.clone()
        z_q_aggregated = []
        indices = []
        for _ in range(self._num_rq_steps):
            z_q_ind, indices_ind = super().apply_codebook(x_res, code_sg)
            x_res -= z_q_ind.squeeze(1)
            z_q_aggregated.append(
                z_q_ind if len(z_q_aggregated) == 0 else z_q_aggregated[-1] + z_q_ind
            )
            indices.append(indices_ind)

        return torch.cat(z_q_aggregated, dim=1), torch.cat(indices, dim=-1)

    def embed_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Embeds the codebook based on the indices.

        Args:
            indices (torch.Tensor): indices, size (BS, SeqL, Ncb).

        Returns:
            torch.Tensor: The embedded codebook, size (BS, SeqL, C).
        """
        emb = torch.zeros(
            indices.size(0),
            indices.size(1),
            self.code_embedding.size(1),
            device=indices.device,
        )
        for idx in range(self._num_rq_steps):
            emb += self.code_embedding[indices[..., idx]]

        return emb
