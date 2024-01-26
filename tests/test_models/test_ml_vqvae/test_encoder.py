import torch
import pytest

from models.multi_level_vqvae.encoder import Encoder1D


@pytest.fixture
def encoder() -> Encoder1D:
    channel_list = [1, 64, 128]
    dim_change_list = [2, 2]
    return Encoder1D(
        channel_list=channel_list,
        dim_change_list=dim_change_list,
        input_channels=1,
        kernel_size=5,
        num_res_block_conv=3,
        dilation_factor=3,
        dim_change_kernel_size=5,
        activation_type="gelu",
    )


def test_Encoder1D_forward(encoder: Encoder1D) -> None:
    input_tensor = torch.randn(1, 1, 128)
    output_tensor = encoder(input_tensor)
    assert output_tensor.shape == (1, 128, 32)


def test_Encoder1D_forward_padding(encoder: Encoder1D) -> None:
    input_tensor = torch.randn(1, 1, 100)
    output_tensor = encoder(input_tensor)
    assert output_tensor.shape[2] == input_tensor.shape[2] / 4
