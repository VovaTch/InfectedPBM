import torch
import torchaudio.functional as AF


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


def log_normal_fixed(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Applies a log transformation to the input tensor and normalizes it using fixed mean and standard deviation.

    The function first applies a log transformation to the input tensor `x` with a small constant added to avoid
    taking the log of zero. It then normalizes the tensor by subtracting the mean and dividing by the standard deviation.

    Args:
        x (torch.Tensor): The input tensor to be log-transformed and normalized.
        mean (float): The mean value to normalize the log-transformed tensor.
        std (float): The standard deviation value to normalize the log-transformed tensor.

    Returns:
        torch.Tensor: The log-transformed and normalized tensor.
    """
    log_transformed = torch.log(x + 1e-6)
    return (log_transformed - mean) / std


def high_pass_filter(x: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Applies a high-pass filter to the input tensor.

    The function applies a high-pass filter to the input tensor `x` with a cutoff frequency specified by the `cutoff`
    argument. The filter is applied along the last dimension of the tensor.

    Args:
        x (torch.Tensor): The input tensor to which the high-pass filter will be applied.
        cutoff (float): The cutoff frequency of the high-pass filter.

    Returns:
        torch.Tensor: The input tensor with the high-pass filter applied.
    """
    return AF.highpass_biquad(x, sample_rate=44100, cutoff_freq=cutoff, Q=0.707)
