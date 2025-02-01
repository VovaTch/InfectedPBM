import torch


def transparent(x: torch.Tensor) -> torch.Tensor:
    """
    Transparent transformation function. Returns what is passed in.
    """
    return x


def tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Tanh transformation function.
    """
    return torch.tanh(x)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid transformation function.
    """
    return torch.sigmoid(x)
