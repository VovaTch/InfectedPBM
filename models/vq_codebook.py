from __future__ import annotations

import torch
import torch.nn as nn


class VQCodebook(nn.Module):
    """
    Parameter holder for the embeddings for the VQVAE. This also references to the function that computes
    gradients past the quantizer.
    """

    def __init__(
        self, token_dim: int, num_tokens: int = 512, usage_threshold: float = 1e-9
    ) -> None:
        """
        Initializes the VQCodebook object.

        Args:
            token_dim (int): The dimensionality of each token.
            num_tokens (int, optional): The number of tokens in the codebook. Defaults to 512.
            usage_threshold (float, optional): The usage threshold for tokens. Defaults to 1e-9.
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.code_embedding = nn.Parameter(torch.rand(num_tokens, token_dim))
        self.usage_threshold = usage_threshold

        # Create a usage instance
        self.register_buffer("usage", torch.ones(self.num_tokens), persistent=False)

    def embed_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Embeds the codebook based on the indices.

        Args:
            indices (torch.Tensor): The indices of the codebook entries.

        Returns:
            torch.Tensor: The embedded codebook.
        """
        return self.code_embedding[indices]

    def apply_codebook(
        self, x_in: torch.Tensor, code_sg: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the codebook to the input tensor.

        Args:
            x_in (torch.Tensor): The input tensor.
            code_sg (bool, optional): Whether to use the detached codebook weights.
                                        Defaults to False.

        Returns:
            torch.Tensor: The quantized tensor.
            torch.Tensor: The indices of the selected codebook entries.
        """
        embedding_weights = self.code_embedding.transpose(0, 1)
        z_q, indices = vq_codebook_select(
            x_in, embedding_weights.detach() if code_sg else embedding_weights
        )  # type: ignore
        self.update_usage(indices)

        return z_q, indices

    # Everything below is the random restart code to try to use the entire codebook and avoid codebook
    # collapse according to OpenAI's Jukebox.
    def update_usage(self, min_enc: torch.Tensor) -> None:
        """
        Update the usage of the codebook based on the minimum encoding.

        Parameters:
        min_enc (numpy.ndarray): The minimum encoding.

        Returns:
        None
        """
        self.usage[min_enc.flatten()] = (  # type: ignore
            self.usage[min_enc.flatten()] + 1  # type: ignore
        )  # if code is used add 1 to usage
        self.usage /= 2  # decay all codes usage # type: ignore

    def reset_usage(self) -> None:
        self.usage.zero_()  # reset usage between epochs # type: ignore

    def random_restart(self) -> float:
        #  randomly restart all dead codes below threshold with random code in codebook
        dead_codes = torch.nonzero(self.usage < self.usage_threshold).squeeze(1)  # type: ignore
        # used_codes = torch.nonzero(self.usage >= self.usage_threshold).squeeze(1)
        # rand_code_idx = torch.randint(used_codes.shape[0], (dead_codes.shape[0],))
        # rand_codes = used_codes[rand_code_idx]
        rand_codes = torch.randperm(self.num_tokens)[0 : len(dead_codes)]
        with torch.no_grad():
            self.code_embedding[dead_codes] = self.code_embedding[rand_codes]
        return dead_codes.shape[0]


class VQCodebookFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x_in: torch.Tensor, embedding_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Autograd function for the index selection. According to the VQ-VAE paper, the gradient for
        x_in should be a mirror, to the x_out.

        Args:
            x_in (torch.Tensor): Input, should be a BS x emb_size x codes (16 x 147)
            embedding_weights (torch.Tensor): Embedding input tensor, should be BS x emb_size x num_codes (4 x 16 x 512)
        """

        ctx.batch_size = x_in.shape[0]

        embedding_batch = embedding_weights.unsqueeze(0).repeat((x_in.shape[0], 1, 1))
        x_in_t = x_in.transpose(1, 2).float()
        embedding_batch_t = embedding_batch.transpose(1, 2).float()
        embedding_batch_flat = embedding_batch_t.flatten(start_dim=0, end_dim=1)

        distances = torch.cdist(x_in_t, embedding_batch_t)  # 4 x 147 x 512
        indices = torch.argmin(distances, dim=2, keepdim=True)  # 4 x 147 x 1
        x_out = torch.index_select(embedding_batch_flat, dim=0, index=indices.flatten())

        x_out = x_out.view((x_in.shape[0], x_in.shape[2], x_in.shape[1]))
        x_out = x_out.transpose(1, 2)

        ctx.save_for_backward(embedding_weights, indices)

        return x_out, indices

    @staticmethod
    def backward(
        ctx, grad_outputs: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        grad_input = None
        grad_emb = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_outputs

        if ctx.needs_input_grad[1]:
            embedding_weights, indices = ctx.saved_variables
            grad_emb = torch.zeros_like(embedding_weights)

            # Feed the gradients into the grad_emb file

            for batch_idx, batch in enumerate(indices.flatten(start_dim=1)):
                running_idx = 0
                for idx in batch:
                    idx_value = idx.item()

                    grad_emb[:, idx_value] += grad_outputs[
                        batch_idx, :, running_idx
                    ] / (indices.flatten().shape[0])
                    running_idx += 1

        return grad_input, grad_emb, None, None


def vq_codebook_select(x_in: torch.Tensor, emb_batch: torch.Tensor):
    """
    Applies the vq codebook function, allowing to pass gradients through it.
    """
    return VQCodebookFunc.apply(x_in, emb_batch)


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

        Args:
            x_in (torch.Tensor): The input tensor to be encoded.
            code_sg (bool, optional): Whether to use the straight-through gradient estimator during encoding.
                Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the encoded tensor and the indices of the codebook
            vectors used.
        """
        x_res = x_in.clone()
        z_q = torch.zeros_like(x_in)
        z_q_ind = z_q.clone()
        indices = []
        for codebook in self.vq_codebooks:
            x_res -= z_q_ind
            z_q_ind, indices_ind = codebook.apply_codebook(x_res, code_sg)
            z_q += z_q_ind
            indices.append(indices_ind)

        indices = torch.cat(indices, dim=-1)
        return z_q, indices

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
