import torch
import torch.nn as nn

from common import registry


class SinActivation(nn.Module):
    """
    Applies the sine function element-wise to the input tensor.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


registry.activation_functions.update(
    {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "sin": SinActivation(),
    }
)
