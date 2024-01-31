import torch
import torch.nn as nn
import torch.nn.functional as F

from common import registry
from ..vq_codebook import VQCodebook


class Res1DBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_res_conv: int,
        dilation_factor: int,
        kernel_size: int,
        activation_type: str = "gelu",
    ) -> None:
        """
        Residual 1D block module.

        Args:
            num_channels (int): Number of input and output channels.
            num_res_conv (int): Number of residual convolutional layers.
            dilation_factor (int): Dilation factor for the convolutional layers.
            kernel_size (int): Kernel size for the convolutional layers.
            activation_type (str, optional): Activation function type. Defaults to "gelu".
        """
        super().__init__()

        self.activation = registry.get_activation_function(activation_type)

        # Create conv, activation, norm blocks
        self.res_block_modules = nn.ModuleList([])
        for idx in range(num_res_conv):
            # Keep output dimension equal to input dim
            dilation = dilation_factor**idx
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

            if idx != num_res_conv - 1:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation,
                        nn.BatchNorm1d(num_channels),
                    )
                )

            else:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation,
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Res1DBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x_init = x.clone()
        for seq_module in self.res_block_modules:
            x = seq_module(x)

        return x + x_init


class Res1DBlockReverse(Res1DBlock):
    def __init__(
        self,
        num_channels: int,
        num_res_conv: int,
        dilation_factor: int,
        kernel_size: int,
        activation_type: str = "gelu",
    ) -> None:
        super().__init__(
            num_channels, num_res_conv, dilation_factor, kernel_size, activation_type
        )

        # Create conv, activation, norm blocks
        self.res_block_modules = nn.ModuleList([])
        for idx in range(num_res_conv):
            # Keep output dimension equal to input dim
            dilation = dilation_factor ** (num_res_conv - idx - 1)
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

            if idx != num_res_conv - 1:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation,
                        nn.BatchNorm1d(num_channels),
                    )
                )

            else:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation,
                    )
                )


class ConvDownsample(nn.Module):
    def __init__(
        self, kernel_size: int, downsample_divide: int, in_dim: int, out_dim: int
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.downsample_divide = downsample_divide
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding_needed = (kernel_size - 2) + (downsample_divide - 2)
        self.padding_needed = (
            0 if self.padding_needed < 0 else self.padding_needed
        )  # Safeguard against negative padding

        # Define the convolutional layer
        self.conv_down = nn.Conv1d(
            in_dim, out_dim, kernel_size=kernel_size, stride=downsample_divide
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[-1]
        x = F.pad(x, (0, self.padding_needed))
        return self.conv_down(x)[..., : sequence_length // self.downsample_divide]


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

    def __init__(self, token_dim: int, num_tokens: int = 8192) -> None:
        """
        Initialize the Block class.

        Args:
            token_dim (int): The dimensionality of each token.
            num_tokens (int, optional): The number of tokens in the codebook. Defaults to 8192.
        """
        super().__init__()
        self.vq_codebook = VQCodebook(token_dim, num_tokens=num_tokens)

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


class ResidualCodebookCollection(nn.Module):
    def __init__(self, token_dim: int, num_codebooks: int, num_tokens: int) -> None:
        super().__init__()
        _codebook_list = [
            VQCodebook(token_dim, num_tokens=num_tokens) for _ in range(num_codebooks)
        ]
        self.vq_codebooks = nn.ModuleList(*_codebook_list)

    def apply_codebook(
        self, x_in: torch.Tensor, code_sg: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_res = x_in.clone()
        z_q = torch.zeros_like(x_in)
        indices = []
        for codebook in self.vq_codebooks:
            x_res -= z_q
            z_q_ind, indices_ind = codebook.apply_codebook(x_res, code_sg)
            z_q += z_q_ind
            indices.append(indices_ind)

        indices = torch.stack(indices, dim=-2)
        return z_q, indices
