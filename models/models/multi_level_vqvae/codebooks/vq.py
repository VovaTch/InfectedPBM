import torch
import torch.nn as nn

from .funcs import vq_code_book_select
from .base import CodeBook


class VQCodeBook(CodeBook):
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
        embedding_weights = self.code_embedding.transpose(0, 1).contiguous()
        z_q, indices = vq_code_book_select(
            x_in, embedding_weights.detach() if code_sg else embedding_weights
        )  # type: ignore
        self.update_usage(indices)

        return z_q.unsqueeze(1), indices

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
