import torch
import torch.nn as nn
import pytest
from omegaconf import DictConfig, OmegaConf
import yaml

from models.multi_level_vqvae.blocks import VQ1D
from models.multi_level_vqvae.decoder import Decoder1D, RippleDecoder
from models.multi_level_vqvae.encoder import Encoder1D
from models.multi_level_vqvae.multi_level_vqvae import (
    MultiLvlVQVariationalAutoEncoder,
    RippleVQVariationalAutoEncoder,
)


@pytest.fixture
def multi_lvl_vqvae() -> MultiLvlVQVariationalAutoEncoder:
    # Create a dummy instance of MultiLvlVQVariationalAutoEncoder for testing
    input_channels = 3
    encoder = nn.Conv1d(3, 64, kernel_size=3, padding=1)
    decoder = nn.Conv1d(64, 3, kernel_size=3, padding=1)
    vq_module = VQ1D(64, 128)
    return MultiLvlVQVariationalAutoEncoder(input_channels, encoder, decoder, vq_module)


def test_MultiLvlVQVariationalAutoEncoder_encode(
    multi_lvl_vqvae: MultiLvlVQVariationalAutoEncoder,
) -> None:
    input_tensor = torch.randn(2, 3, 100)
    encoded_tensor = multi_lvl_vqvae.encode(input_tensor)
    assert encoded_tensor.shape == (2, 64, 100)


def test_MultiLvlVQVariationalAutoEncoder_tokenize(
    multi_lvl_vqvae: MultiLvlVQVariationalAutoEncoder,
) -> None:
    input_tensor = torch.randn(2, 3, 100)
    indices_tensor = multi_lvl_vqvae.tokenize(input_tensor)
    assert indices_tensor.shape == (2, 100, 1)
    assert indices_tensor.dtype == torch.int64


def test_MultiLvlVQVariationalAutoEncoder_decode(
    multi_lvl_vqvae: MultiLvlVQVariationalAutoEncoder,
) -> None:
    latent_tensor = torch.randn(2, 64, 100)
    output_tensor, total_output = multi_lvl_vqvae.decode(latent_tensor)
    assert output_tensor.shape == (2, 3, 100)
    assert isinstance(total_output, dict)


def test_MultiLvlVQVariationalAutoEncoder_forward_cpu(
    multi_lvl_vqvae: MultiLvlVQVariationalAutoEncoder,
) -> None:
    input_tensor = torch.randn(2, 3, 100)
    total_output = multi_lvl_vqvae.forward(input_tensor)
    assert isinstance(total_output, dict)
    assert total_output["slice"].size() == input_tensor.size()


def test_MultiLvlVQVariationalAutoEncoder_forward_gpu(
    multi_lvl_vqvae: MultiLvlVQVariationalAutoEncoder,
) -> None:
    multi_lvl_vqvae = multi_lvl_vqvae.to("cuda")
    input_tensor = torch.randn(2, 3, 100).to("cuda")
    total_output = multi_lvl_vqvae.forward(input_tensor)
    assert isinstance(total_output, dict)
    assert total_output["slice"].size() == input_tensor.size()


def test_multi_lvl_vqvae_from_cfg(cfg: DictConfig) -> None:
    """Test the MultiLvlVQVariationalAutoEncoder class."""

    model = MultiLvlVQVariationalAutoEncoder.from_cfg(cfg)
    assert model is not None
    assert isinstance(model, MultiLvlVQVariationalAutoEncoder)
    assert isinstance(model.encoder, Encoder1D)
    assert isinstance(model.decoder, Decoder1D)
    assert isinstance(model.vq_module, VQ1D)


def test_complete_model_forward(real_cfg: DictConfig) -> None:
    model = MultiLvlVQVariationalAutoEncoder.from_cfg(real_cfg)
    model = model.to("cuda")
    input_tensor = torch.randn(128, 1, 1024).to("cuda")
    total_output = model.forward(input_tensor)
    assert isinstance(total_output, dict)
    assert total_output["slice"].size() == input_tensor.size()


@pytest.fixture
def ripple_vqvae() -> RippleVQVariationalAutoEncoder:
    # Create a dummy instance of RippleVQVariationalAutoEncoder for testing
    input_channels = 3
    encoder = nn.Conv1d(3, 64, kernel_size=3, padding=1)
    decoder = nn.Conv1d(64, 3, kernel_size=3, padding=1)
    vq_module = VQ1D(64, 128)
    return RippleVQVariationalAutoEncoder(input_channels, encoder, decoder, vq_module)


def test_from_tokens(multi_lvl_vqvae: MultiLvlVQVariationalAutoEncoder) -> None:
    # Create a sample indices tensor
    indices = torch.randint(0, 128, (10, 1, 1))

    # Test the from_tokens method
    x_out = multi_lvl_vqvae.from_tokens(indices)

    # Assert the shape of the output tensor
    assert x_out.shape == (10, 3, 1)
