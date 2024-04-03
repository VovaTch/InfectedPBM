import torch
import torch.nn as nn

from .vqvae import VQCodebook


class ResidualCodebookCollection(nn.Module):
    def __init__(self, token_dim: int, num_codebooks: int, num_tokens: int) -> None:
        super().__init__()
        self.vq_codebooks = nn.ModuleList(
            [VQCodebook(token_dim, num_tokens=num_tokens) for _ in range(num_codebooks)]
        )
        self.token_dim = token_dim

    def apply_codebook(
        self, x_in: torch.Tensor, code_sg: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the VQ codebook to the input tensor.

        ### Args:
        *   x_in (torch.Tensor): The input tensor to be encoded.
        *   code_sg (bool, optional): Whether to use the straight-through gradient estimator during encoding.
                Defaults to False.

        ### Returns:
        *   tuple[torch.Tensor, torch.Tensor]: A tuple containing the encoded tensor and the indices of the codebook
            vectors used. The indices are at the size of BS x idx_slice x num_codebooks.
        """
        x_res = x_in.clone()
        z_q_aggregated = torch.zeros(
            (x_in.shape[0], 0, x_in.shape[1], x_in.shape[2])
        ).to(x_in.device)
        z_q_ind = torch.zeros((x_in.shape[0], 1, x_in.shape[1], x_in.shape[2])).to(
            x_in.device
        )
        z_q = z_q_ind.clone()
        indices = []
        for codebook in self.vq_codebooks:
            x_res -= z_q_ind.squeeze(1)
            z_q_ind, indices_ind = codebook.apply_codebook(x_res, code_sg)
            z_q += z_q_ind
            z_q_aggregated = torch.cat((z_q_aggregated, z_q), dim=1)
            indices.append(indices_ind)

        indices = torch.cat(indices, dim=-1)
        return (
            z_q_aggregated,
            indices,
        )  # z_q has the dim of BS x num_codebooks x idx_slice_len x emb_size
        # indices are BS x idx_slice_len x num_codebooks

    def embed_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Embeds the codebook using the given indices.

        Args:
            indices (torch.Tensor): The indices to use for embedding.

        Returns:
            torch.Tensor: The embedded codebook.

        """
        emb = torch.zeros((indices.shape[0], indices.shape[1], self.token_dim)).to(
            indices.device
        )
        for idx, codebook in enumerate(self.vq_codebooks):
            emb += codebook.embed_codebook(indices[..., idx])

        return emb.transpose(1, 2)

    def update_usage(self, min_enc: torch.Tensor) -> None:
        """
        Update the usage of the model based on the minimum encoding.

        Args:
            min_enc (torch.Tensor): The minimum encoding.

        Returns:
            None
        """
        for codebook in self.vq_codebooks:
            codebook.update_usage(min_enc)

    def reset_usage(self) -> None:
        """
        Resets the usage of the object.
        """
        for codebook in self.vq_codebooks:
            codebook.reset_usage()

    def random_restart(self) -> float:
        """
        Performs a random restart for the optimization algorithm.

        Returns:
            float: The average number of dead codes
        """
        dead_codes = 0
        for codebook in self.vq_codebooks:
            dead_codes += codebook.random_restart()
        return dead_codes / len(self.vq_codebooks)
