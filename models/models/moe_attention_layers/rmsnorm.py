import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Perform learnable RMS Norm operation on the input tensor.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """
        Initializes the RMSNorm layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value to avoid division by zero. Default is 1e-6.
        """
        super().__init__()
        self._eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to be normalized.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self._eps)

    def reset_parameters(self) -> None:
        """
        Resets the parameters of the layer.

        This method initializes the weights of the layer to ones using the
        PyTorch initialization utility.
        """
        torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS normalization to the input tensor and scales it with a learned weight.

        Args:
            x (torch.Tensor): The input tensor to be normalized.

        Returns:
            torch.Tensor: The normalized and scaled tensor.
        """
        original_type = x.dtype
        x = self._norm(x.float())
        x *= self.weight.float()
        return x.type(original_type)
