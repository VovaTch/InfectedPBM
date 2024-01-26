import torch
import pytest

from models.multi_level_vqvae.decoder import Decoder1D


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
