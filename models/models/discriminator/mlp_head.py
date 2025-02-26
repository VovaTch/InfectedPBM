import warnings
import torch
import torch.nn as nn

from .base import DiscriminatorHead


class MLP(DiscriminatorHead):
    """
    Simple MLP head implementation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        activation_fn: nn.Module = nn.GELU(),
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the MLP module.

        Args:
            input_dim (int): The dimension of the input features.
            hidden_dim (int): The dimension of the hidden layers.
            num_layers (int): The number of layers in the MLP. Must be at least 1.
            output_dim (int): The dimension of the output features.
            activation_fn (nn.Module, optional): The activation function to use between layers. Defaults to nn.GELU().
            dropout (float, optional): The dropout probability. Defaults to 0.0.

        Raises:
            ValueError: If num_layers is less than 1.
            UserWarning: If num_layers is less than 2, indicating that the MLP is equivalent to a linear layer.
        """
        super().__init__()

        if num_layers < 1:
            raise ValueError("The number of layers must be at least 1")
        if num_layers < 2:
            warnings.warn(
                "A single layer MLP is equivalent to a linear layer, hidden_dim and activation_fn will be ignored"
            )

        # Set parameters
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._output_dim = output_dim
        self._activation_fn = activation_fn

        self._last_layer = nn.Linear(hidden_dim, output_dim)

        # Construct the model
        if num_layers == 1:
            self._mlp = nn.Linear(input_dim, output_dim)
        else:
            self._mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_fn,
                *[
                    nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim),
                        activation_fn,
                    )
                    for _ in range(num_layers - 2)
                ],
                nn.Dropout(dropout),
                self._last_layer,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
        """
        return self._mlp(x)

    @property
    def last_layer(self) -> nn.Module:
        return self._last_layer
