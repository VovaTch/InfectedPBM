from dataclasses import dataclass
from typing_extensions import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.multi_level_vqvae.ripplenet import RippleLinear, batch_ripple_linear_func


@dataclass
class RippleDecoderParameters:
    input_dim: int
    hidden_dim: int
    mlp_num_layers: int
    output_dim: int
    ripl_hidden_dim: int
    ripl_num_layers: int
    ripl_coordinate_multipler: int

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        return cls(
            input_dim=cfg.model.input_dim,
            hidden_dim=cfg.model.hidden_dim,
            mlp_num_layers=cfg.model.mlp_num_layers,
            output_dim=cfg.model.output_dim,
            ripl_hidden_dim=cfg.model.ripl_hidden_dim,
            ripl_num_layers=cfg.model.ripl_num_layers,
            ripl_coordinate_multipler=cfg.model.ripl_coordinate_multipler,
        )


class RippleDecoder(nn.Module):
    def __init__(
        self,
        dec_params: RippleDecoderParameters,
    ) -> None:
        super().__init__()
        self.dec_params = dec_params

        ripple_weight_dim = self._compute_mlp_output_dim()
        if self.dec_params.mlp_num_layers < 0:
            raise ValueError(
                f"MLP number of hidden layers must be non negative, got {self.dec_params.mlp_num_layers}"
            )

        layer_list = (
            [
                nn.Linear(self.dec_params.input_dim, self.dec_params.hidden_dim),
                nn.GELU(),
            ]
            + [
                nn.Linear(self.dec_params.hidden_dim, self.dec_params.hidden_dim),
                nn.LayerNorm(self.dec_params.hidden_dim),
                nn.GELU(),
            ]
            * self.dec_params.mlp_num_layers
            + [nn.Linear(self.dec_params.hidden_dim, ripple_weight_dim)]
        )
        self.activation = nn.GELU()
        self.mlp = nn.Sequential(*layer_list)
        self.bypass = RippleLinear(1, 1)
        self.sequence_length = self.dec_params.output_dim
        self.ripl_fully_connected_layer = nn.Linear(1, self.dec_params.ripl_hidden_dim)

    @staticmethod
    def _compute_ripple_weight_dim(in_dim: int, out_dim: int) -> int:
        """
        Computes the dimension of the ripple weights for the decoder.

        Args:
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.

        Returns:
            int: The dimension of the ripple weights.
        """
        num_weights = 2 * in_dim * out_dim
        num_biases = (in_dim + 1) * out_dim
        return num_weights + num_biases

    def _compute_mlp_output_dim(self) -> int:
        """
        Computes the output dimension of the MLP network in the decoder.

        Returns:
            int: The output dimension of the MLP network.
        """
        out_dim = self._compute_ripple_weight_dim(self.dec_params.ripl_hidden_dim, 1)
        middle_dim = (
            self._compute_ripple_weight_dim(
                self.dec_params.ripl_hidden_dim, self.dec_params.ripl_hidden_dim
            )
            * self.dec_params.ripl_num_layers
        )
        return out_dim + middle_dim

    def _split_mlp_output_to_ripple_layers(
        self, mlp_output: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Splits the output of the MLP into ripple layers.

        Args:
            mlp_output (torch.Tensor): The output tensor of the MLP.

        Returns:
            nn.ModuleList: A list of tensors representing the ripple layers.
        """

        out_dim = self._compute_ripple_weight_dim(self.dec_params.ripl_hidden_dim, 1)
        middle_dims = [
            self._compute_ripple_weight_dim(
                self.dec_params.ripl_hidden_dim, self.dec_params.ripl_hidden_dim
            )
        ] * self.dec_params.ripl_num_layers

        # Compute the splitting via a running index
        ripple_weight_layers = []
        running_idx = 0
        for middle_dim in middle_dims:
            ripple_weight_layers.append(
                mlp_output[..., running_idx : running_idx + middle_dim]
            )
            running_idx += middle_dim
        ripple_weight_layers.append(
            mlp_output[..., running_idx : running_idx + out_dim]
        )
        return ripple_weight_layers

    def set_sequence_length(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ripple decoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        mlp_output = self.mlp(x.flatten(start_dim=1))
        ripple_weight_layers = self._split_mlp_output_to_ripple_layers(mlp_output)
        out_weights = ripple_weight_layers[-1]
        middle_weights = ripple_weight_layers[:-1]

        # Forward pass through the ripple-linear layers
        line_coordinates = (
            torch.arange(0, self.sequence_length, step=1)
            .float()
            .to(x.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(x.shape[0], 1, 1)
        ) * self.dec_params.ripl_coordinate_multipler  # Size BS x L x 1

        # Input layer
        lc_x = self.ripl_fully_connected_layer(line_coordinates)
        lc_bypass = lc_x.clone()

        # Middle layers
        for middle_weight in middle_weights:
            lc_x = self._run_ripple_linear(
                lc_x,
                middle_weight,
                self.dec_params.ripl_hidden_dim,
                self.dec_params.ripl_hidden_dim,
            )
            lc_x = self.activation(lc_x)

        # Output layer
        lc_x = self._run_ripple_linear(
            lc_x, out_weights, self.dec_params.ripl_hidden_dim, 1
        )
        lc_x += self.bypass(lc_bypass)
        # lc_x = F.tanh(lc_x)

        return lc_x.transpose(1, 2)

    @staticmethod
    def _run_ripple_linear(
        lc_x: torch.Tensor,
        flattened_weights: torch.Tensor,
        input_dim: int,
        output_dim: int,
    ) -> torch.Tensor:
        """
        Applies the ripple linear transformation to the input tensor.

        Args:
            lc_x (torch.Tensor): The input tensor.
            flattened_weights (torch.Tensor): The flattened weights tensor.
            input_dim (int): The input dimension.
            output_dim (int): The output dimension.

        Returns:
            torch.Tensor: The output tensor after applying the ripple linear transformation.
        """
        ripl_weight_dim = input_dim * output_dim * 2
        ripl_weights = flattened_weights[:, :ripl_weight_dim].view(
            (
                -1,
                output_dim,
                input_dim,
                2,
            )
        )
        ripl_bias = flattened_weights[:, ripl_weight_dim:].view(
            (
                -1,
                output_dim,
                input_dim + 1,
            )
        )
        return batch_ripple_linear_func(lc_x, ripl_weights, ripl_bias)
