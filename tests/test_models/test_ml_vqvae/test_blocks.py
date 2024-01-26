import torch
import pytest

from models.multi_level_vqvae.blocks import Res1DBlock


@pytest.fixture
def res1d_block() -> Res1DBlock:
    return Res1DBlock(
        num_channels=64,
        num_res_conv=3,
        dilation_factor=2,
        kernel_size=3,
        activation_type="gelu",
    )


def test_Res1DBlock_forward(res1d_block: Res1DBlock) -> None:
    input_tensor = torch.randn(1, 64, 100)
    output_tensor = res1d_block(input_tensor)
    assert output_tensor.shape == (1, 64, 100)


def test_Res1DBlock_forward_padding(res1d_block: Res1DBlock) -> None:
    input_tensor = torch.randn(1, 64, 100)
    output_tensor = res1d_block(input_tensor)
    assert output_tensor.shape[2] == input_tensor.shape[2]


def test_Res1DBlock_forward_output_channels(res1d_block: Res1DBlock) -> None:
    input_tensor = torch.randn(1, 64, 100)
    output_tensor = res1d_block(input_tensor)
    assert output_tensor.shape[1] == input_tensor.shape[1]
