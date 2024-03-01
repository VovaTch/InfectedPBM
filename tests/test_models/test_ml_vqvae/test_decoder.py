import torch
import pytest

from models.multi_level_vqvae.decoder import (
    Decoder1D,
    RippleDecoder,
    RippleDecoderParameters,
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
