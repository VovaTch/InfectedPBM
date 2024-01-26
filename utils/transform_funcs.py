from typing import Callable
import torch

from common import registry


@registry.register_transform_function("none")
def transparent(x: torch.Tensor) -> torch.Tensor:
    """
    A function that returns the input tensor as it is, without any modifications.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The input tensor itself.
    """
    return x


@registry.register_transform_function("tanh")
def tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the hyperbolic tangent function element-wise to the input tensor.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The output tensor with the same shape as the input tensor.
    """
    return torch.tanh(x)


@registry.register_transform_function("sigmoid")
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the sigmoid function element-wise to the input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with sigmoid function applied element-wise.
    """
    return torch.sigmoid(x)
