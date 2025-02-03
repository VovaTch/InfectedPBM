from __future__ import annotations

import torch
from torch.jit._script import script


@script
def ripple_linear_func_batch(
    input: torch.Tensor, out_features: int, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """
    Ripple Linear for meta learning. Each weight and bias are batched.

    Args:
        input (torch.Tensor): Input, size BS x (I)
        out_features (int): num of output features
        weight (torch.Tensor): Weight, size BS x I x O x 2
        bias (torch.Tensor): Bias, size BS x (I + 1) x O

    Returns:
        torch.Tensor: Output, size BS x (O)
    """

    # Register output sizes
    input_size = input.size()
    output_size = list(input_size)
    output_size[-1] = out_features

    # flatten the input
    flattened_input = input.flatten(end_dim=-3)

    # Perform Ripple operation
    operation_result = torch.einsum(
        "bio,bsio->bso",
        weight[:, :, :, 0],
        torch.sin(
            torch.einsum(
                "bsi,bio->bsio",
                flattened_input,
                weight[:, :, :, 1],
            )
            + bias[:, 1:, :].unsqueeze(1)
        ),
    ) + bias[:, 0, :].unsqueeze(1)

    return operation_result.view(output_size)


# @script
def ripple_linear_func_batch_with_channels(
    input: torch.Tensor, out_features: int, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """
    Ripple Linear for meta learning. Each weight and bias are batched and divided by channels

    Args:
        input (torch.Tensor): Input, BS x S x C x I
        out_features (int): num of output features
        weight (torch.Tensor): Weight, size BS x C x I x O x 2
        bias (torch.Tensor): Bias, size BS x C x (I + 1) x O

    Returns:
        torch.Tensor: Output, size BS x S x C x (O)
    """
    # Register output sizes
    input_size = input.size()
    output_size = list(input_size)
    output_size[-1] = out_features

    # Flattened inputs
    flattened_input = input.flatten(end_dim=-4)

    # Perform Ripple operation
    operation_result = torch.einsum(
        "bcio,bscio->bsco",
        weight[:, :, :, :, 0],
        torch.sin(
            torch.einsum(
                "bsci,bcio->bscio",
                flattened_input,
                weight[:, :, :, :, 1],
            )
            + bias[:, :, 1:, :].unsqueeze(1)
        ),
    ) + bias[:, :, 0, :].unsqueeze(1)

    return operation_result.view(output_size)
