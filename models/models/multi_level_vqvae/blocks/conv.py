import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDownsample(nn.Module):
    """
    Convolutional downsample block.
    """

    def __init__(
        self, kernel_size: int, downsample_divide: int, in_dim: int, out_dim: int
    ) -> None:
        """
        Initializes the convolutional block.

        Args:
            kernel_size (int): The size of the convolutional kernel.
            downsample_divide (int): The factor by which to downsample the input.
            in_dim (int): The number of input channels.
            out_dim (int): The number of output channels.
        """
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
