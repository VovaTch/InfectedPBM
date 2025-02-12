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


def log_normal(x: torch.Tensor) -> torch.Tensor:
    """
    Applies a log transformation to the input tensor and normalizes it.

    The function first applies a log transformation to the input tensor `x` with a small constant added to avoid
    taking the log of zero. It then computes the mean and standard deviation of the log-transformed tensor and
    normalizes the tensor by subtracting the mean and dividing by the standard deviation.

    Args:
        x (torch.Tensor): The input tensor to be log-transformed and normalized.

    Returns:
        torch.Tensor: The log-transformed and normalized tensor.
    """
    log_transformed = torch.log(x + 1e-6)
    mean = torch.mean(log_transformed)
    std = torch.std(log_transformed)
    return (log_transformed - mean) / std
