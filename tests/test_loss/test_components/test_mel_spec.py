import torch
import pytest

from loss.components.mel_spec import MelSpecLoss
from common import registry


@pytest.fixture
def mel_spec_loss() -> MelSpecLoss:
    mel_spec_config = {
        "n_fft": 2048,
        "hop_length": 257,
        "n_mels": 128,
        "power": 0.5,
        "f_min": 20,
        "pad": 0,
        "pad_mode": "reflect",
        "norm": "slaney",
        "mel_scale": "htk",
    }
    mel_spec_converter = registry.get_mel_spec_converter("simple").from_cfg(
        mel_spec_config
    )
    transform_func = registry.get_transform_function("tanh")
    base_loss = registry.get_loss_module("mse")
    return MelSpecLoss(
        "mel_spec_loss",
        1.0,
        base_loss,
        mel_spec_converter,
        transform_func=transform_func,
        lin_start=1.0,
        lin_end=1.0,
    )


def test_mel_spec_loss(mel_spec_loss):
    """
    Test function for the mel_spec_loss.

    Args:
        mel_spec_loss: The mel_spec_loss function to be tested.
    """
    torch.manual_seed(1337)
    estimation = {"slice": torch.randn(2, 1, 2048)}
    target = {"slice": torch.randn(2, 1, 2048)}
    result = mel_spec_loss(estimation, target)
    assert torch.allclose(result, torch.tensor(0.007817087695002556))
