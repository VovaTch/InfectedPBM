import torch
import torch.nn as nn


class SinActivation(nn.Module):
    """
    Applies the sine function element-wise to the input tensor.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)