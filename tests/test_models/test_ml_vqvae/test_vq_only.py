import torch
import pytest
from omegaconf import OmegaConf
from models.multi_level_vqvae.multi_level_vqvae.vq_only import VQOnlyTokenizer


@pytest.fixture
def tokenizer() -> VQOnlyTokenizer:
    # Create a sample configuration
    config = OmegaConf.create(
        {
            "model": {
                "vocabulary_size": 1024,
                "num_codebooks": 4,
                "slice_length": 256,
                "input_channels": 1,
            },
            "loss": {"aggregator": {"type": "none"}},
        }
    )

    # Create an instance of the tokenizer
    tokenizer = VQOnlyTokenizer.from_cfg(config)
    return tokenizer


def test_from_tokens(tokenizer: VQOnlyTokenizer) -> None:
    # Create a sample indices tensor
    indices = torch.randint(0, 1024, (10, 1, 4))

    # Test the from_tokens method
    x_out = tokenizer.from_tokens(indices)

    # Assert the shape of the output tensor
    assert x_out.shape == (10, 1, 256)


def test_forward(tokenizer: VQOnlyTokenizer) -> None:
    # Create a sample input tensor
    x = torch.randn(10, 1, 256)

    # Test the forward method
    output = tokenizer.forward(x)

    # Assert the presence of certain keys in the output dictionary
    assert "v_q" in output
    assert "slice" in output
    assert "z_e" in output
    assert output["slice"].shape == (10, 1, 256)
