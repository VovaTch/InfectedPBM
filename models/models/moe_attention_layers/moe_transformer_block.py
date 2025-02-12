from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import RMSNorm


class MoETransformerDecoderBlock(nn.Module):
    """
    Simple Mixture of Experts Transformer Decoder implementation with gating mechanism. The attention
    is inserted via dependency injection.
    """

    def __init__(
        self,
        hidden_dim: int,
        attention: nn.Module,
        norm_type: Literal["rmsnorm", "layernorm"] = "layernorm",
        num_experts: int = 1,
        top_k_gating: int = 1,
        ff_hidden_dim: int = 2048,
    ) -> None:
        """
        Initializes the base transformer layer.

        Args:
            hidden_dim (int): The dimension of the hidden layer.
            attention (nn.Module): The attention module to be used.
            norm_type (Literal["rmsnorm", "layernorm"], optional): The type of normalization to be used. Defaults to "layernorm".
            num_experts (int, optional): The number of expert layers. Must be at least 1. Defaults to 1.
            top_k_gating (int, optional): The number of top experts to use for gating. Must be at least 1. Defaults to 1.
            ff_hidden_dim (int, optional): The dimension of the feed-forward hidden layer. Defaults to 2048.

        Raises:
            ValueError: If `num_experts` is less than 1.
            ValueError: If `top_k_gating` is less than 1.
        """
        super().__init__()
        if norm_type == "rmsnorm":
            self._norm_1 = RMSNorm(hidden_dim)
            self._norm_2 = RMSNorm(hidden_dim)
        elif norm_type == "layernorm":
            self._norm_1 = nn.LayerNorm(hidden_dim)
            self._norm_2 = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(
                "The normalization type must be either 'rmsnorm' or 'layernorm'"
            )
        self._hidden_dim = hidden_dim
        self._attention = attention

        if num_experts < 1:
            raise ValueError("The number of experts must be at least 1")
        elif num_experts > 1:
            self._gate = nn.Linear(hidden_dim, num_experts)
        else:
            self._gate = None

        if top_k_gating < 1:
            raise ValueError("The top_k_gating must be at least 1")

        self._num_experts = num_experts
        self._ff_hidden_dim = ff_hidden_dim

        self._experts_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_experts)]
        )
        self._top_k_gating = top_k_gating if num_experts > top_k_gating else num_experts

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of the base transformer layer.

        Args:
            x (torch.Tensor): Input tensor size (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: Output tensor after applying layer normalization, attention, and expert layers.
        """
        x = x + self._norm_1(self._attention(x, mask))

        # No gating, when the number of experts is 1
        if self._num_experts == 1:
            x = x + self._norm_2(self._experts_layers[0](x))

        # Apply gating mechanism with top k if the number of experts is greater than 1
        elif self._gate:
            x = x + self._norm_2(self._route_to_experts(x))

        return x

    def _route_to_experts(self, x: torch.Tensor) -> torch.Tensor:
        """
        Routes the input tensor `x` to the appropriate expert layers based on the gating mechanism.

        Args:
            x (torch.Tensor): The input tensor to be routed to the experts.

        Returns:
            torch.Tensor: The output tensor after being processed by the selected expert layers.

        Raises:
            ValueError: If the gating layer is not defined.

        Notes:
            - The gating mechanism determines the probabilities for each expert.
            - The top-k experts are selected based on the gating probabilities.
            - The input is then routed to the selected experts, and their outputs are weighted and combined.
        """
        if (
            not self._gate
        ):  # Shouldn't get there, just in case, also makes the type checker happy
            raise ValueError("The gating layer is not defined")
        gate_scores = self._gate(x)
        gate_probs = F.softmax(gate_scores, dim=-1)
        top_k_values, top_k_indices = torch.topk(gate_probs, self._top_k_gating, dim=-1)

        final_output = torch.zeros_like(x)

        for idx in range(self._top_k_gating):
            expert_idx = top_k_indices[..., idx]
            expert_weight = top_k_values[..., idx].unsqueeze(-1)

            for mask_idx in range(self._num_experts):
                mask = expert_idx == mask_idx
                if not mask.any():
                    continue
                expert_input = torch.zeros_like(x)
                expert_input[mask] = x[mask]
                expert_output = self._experts_layers[mask_idx](expert_input)
                final_output[mask] += expert_weight[mask] * expert_output[mask]
        return final_output
