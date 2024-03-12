from __future__ import annotations
import numpy as np
import torch

import torch.nn as nn


class ExpandingMLPDecoder(nn.Module):
    """
    Expanding MLP decoder. This decoder creates a waveform from the latent space.
    """

    def __init__(
        self,
        hidden_dim: int,
        input_len: int,
        output_multiplier: int,
        num_layers: int,
        activation_type: str = "gelu",
        output_channels: int = 1,
    ) -> None:
        """
        Initialize the SimpleDecoder module.

        Args:
            hidden_dim (int): The hidden dimension size.
            input_len (int): The length of the input.
            output_multiplier (int): The multiplier for the output length compared to input_len.
            num_layers (int): The number of layers in the decoder.
            activation_type (str, optional): The type of activation function to use. Defaults to "gelu".
            output_channels (int, optional): The number of output channels. Defaults to 1.

        Raises:
            ValueError: If the number of layers is less than 1.
            ValueError: If the activation type is not supported.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_len = input_len
        self.output_multiplier = output_multiplier
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError(f"Number of layers must be at least 1, got {num_layers}")

        match activation_type:
            case "gelu":
                self.activation = nn.GELU()
            case "relu":
                self.activation = nn.ReLU()
            case "tanh":
                self.activation = nn.Tanh()
            case "sigmoid":
                self.activation = nn.Sigmoid()
            case _:
                raise ValueError(f"Activation type {activation_type} not supported")

        self.output_channels = output_channels

        all_outputs_dims = [hidden_dim * 2**idx for idx in range(num_layers - 1)] + [
            input_len * output_multiplier * output_channels
        ]
        layers = []
        for layer_idx in range(len(all_outputs_dims)):

            # Add linear layer
            if layer_idx == 0:
                layers.append(nn.Linear(input_len, all_outputs_dims[layer_idx]))
            else:
                layers.append(
                    nn.Linear(
                        all_outputs_dims[layer_idx - 1],
                        all_outputs_dims[layer_idx],
                    )
                )

            # Add activation and layer norm
            if layer_idx < len(all_outputs_dims) - 1:
                layers.append(self.activation)
                layers.append(nn.LayerNorm(int(all_outputs_dims[layer_idx])))

        self.sequential = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x_out_flattened = self.sequential(z.flatten(start_dim=1))
        return x_out_flattened.reshape(
            -1, self.output_channels, self.input_len * self.output_multiplier
        )
