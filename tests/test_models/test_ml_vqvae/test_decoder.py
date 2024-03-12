import numpy as np
import torch
import pytest

from models.multi_level_vqvae.decoder import (
    Decoder1D,
    RippleDecoder,
    RippleDecoderParameters,
    ExpandingMLPDecoder,
)


@pytest.fixture
def decoder() -> Decoder1D:
    channel_list = [64, 128, 256]
    dim_change_list = [4, 8]
    return Decoder1D(
        channel_list=channel_list,
        dim_change_list=dim_change_list,
        input_channels=1,
        kernel_size=5,
        dim_add_kernel_add=12,
        num_res_block_conv=3,
        dilation_factor=3,
        activation_type="gelu",
    )


def test_Decoder1D_forward(decoder: Decoder1D) -> None:
    input_tensor = torch.randn(1, 64, 4)
    output_tensor = decoder(input_tensor)
    assert output_tensor.shape == (1, 1, 128)


@pytest.fixture
def ripple_decoder() -> RippleDecoder:
    dec_params = RippleDecoderParameters(
        input_dim=64,
        hidden_dim=128,
        mlp_num_layers=2,
        output_dim=1024,
        ripl_hidden_dim=16,
        ripl_num_layers=1,
        ripl_coordinate_multipler=10,
    )
    return RippleDecoder(dec_params)


def test_RippleDecoder_forward(ripple_decoder: RippleDecoder) -> None:
    input_tensor = torch.randn(6, 4, 16)
    output_tensor = ripple_decoder(input_tensor)
    assert output_tensor.shape == (6, 1, 1024)


def test_expanding_mlp_decoder_forward() -> None:
    hidden_dim = 64
    input_len = 128
    outout_multiplier = 8
    num_layers = 4
    activation_type = "relu"
    output_channels = 1

    decoder = ExpandingMLPDecoder(
        hidden_dim,
        input_len,
        outout_multiplier,
        num_layers,
        activation_type,
        output_channels,
    )

    batch_size = 10
    z = torch.randn(batch_size, input_len // 4, 4)

    output = decoder.forward(z)

    assert output.shape == (batch_size, 1, input_len * outout_multiplier)


def test_expanding_mlp_decoder_output_channels() -> None:
    hidden_dim = 64
    input_len = 128
    outout_multiplier = 8
    num_layers = 4
    activation_type = "relu"
    output_channels = 3

    decoder = ExpandingMLPDecoder(
        hidden_dim,
        input_len,
        outout_multiplier,
        num_layers,
        activation_type,
        output_channels,
    )

    batch_size = 10
    z = torch.randn(batch_size, input_len // 4, 4)

    output = decoder.forward(z)

    assert output.shape == (
        batch_size,
        output_channels,
        input_len * outout_multiplier,
    )
