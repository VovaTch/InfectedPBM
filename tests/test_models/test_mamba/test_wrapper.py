import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from models.mamba.wrapper import MambaWrapper
from utils.containers import MambaParams


@pytest.fixture
def mamba_params():
    # Define the MambaParams object with dummy values
    return MambaParams(model_dim=256, ssm_state_dim=128, conv_width=3, expansion=4)


@pytest.fixture
def vocabulary_size():
    # Define the vocabulary size with a dummy value
    return 100


@pytest.fixture
def mamba_wrapper(mamba_params, vocabulary_size):
    # Create an instance of MambaWrapper for testing
    return MambaWrapper(mamba_params, vocabulary_size).to("cuda")


def test_mamba_wrapper_forward(mamba_wrapper):
    # Test the forward method of MambaWrapper
    x = torch.tensor([[1, 2, 3], [4, 5, 6]]).to("cuda")  # Example input tensor
    output = mamba_wrapper.forward(x)
    assert "logits" in output
    assert output["logits"].shape == (2, 3, mamba_wrapper.mamba_params.model_dim)


def test_mamba_wrapper_get_last_logits(mamba_wrapper):
    # Test the get_last_logits method of MambaWrapper
    x = torch.tensor([[1, 2, 3], [4, 5, 6]]).to("cuda")  # Example input tensor
    last_logits = mamba_wrapper.get_last_logits(x)
    assert last_logits.shape == (2, mamba_wrapper.mamba_params.model_dim)


def test_mamba_wrapper_from_cfg():
    # Test the from_cfg class method of MambaWrapper
    cfg = OmegaConf.create(
        {
            "model": {
                "vocabulary_size": 100,
                "model_dim": 256,
                "ssm_state_dim": 16,
                "conv_width": 3,
                "expansion": 2,
            }
        }
    )
    mamba_wrapper = MambaWrapper.from_cfg(cfg).to("cuda")
    assert isinstance(mamba_wrapper, MambaWrapper)
    assert mamba_wrapper.vocabulary_size == 100
