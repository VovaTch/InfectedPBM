import torch
import pytest

from loss.components.reconstruction import (
    RecLoss,
    NoisePredLoss,
    DiffReconstructionLoss,
)
from common import registry


@pytest.fixture
def rec_loss() -> RecLoss:
    return RecLoss(
        "rec_loss",
        1.0,
        registry.get_loss_module("mse"),
        transform_func=registry.get_transform_function("none"),
        phase_parameter=1,
    )


@pytest.fixture
def diff_rec_loss() -> DiffReconstructionLoss:
    return DiffReconstructionLoss(
        "diff_rec_loss",
        1.0,
        registry.get_loss_module("mse"),
    )


@pytest.fixture
def noise_pred_loss() -> NoisePredLoss:
    return NoisePredLoss(
        "noise_pred_loss",
        1.0,
        registry.get_loss_module("mse"),
    )


def test_rec_loss(rec_loss: RecLoss) -> None:
    """
    Test the rec_loss function.

    Args:
        rec_loss (RecLoss): The rec_loss function to be tested.
    """
    estimation = {"slice": torch.tensor([[[0.2, 0.8], [0.6, 0.4]]])}
    target = {"slice": torch.tensor([[[0.1, 0.9], [0.3, 0.7]]])}
    result = rec_loss(estimation, target)
    assert torch.isclose(result, torch.tensor(0.04999999701976776))


def test_noise_pred_loss(noise_pred_loss: NoisePredLoss) -> None:
    """
    Test function for the noise prediction loss.

    Args:
        noise_pred_loss (NoisePredLoss): The noise prediction loss function.
    """
    estimation = {"noise_pred": torch.tensor([[[0.2, 0.8], [0.6, 0.4]]])}
    target = {"noise": torch.tensor([[[0.1, 0.9], [0.3, 0.7]]])}
    result = noise_pred_loss(estimation, target)
    assert torch.isclose(result, torch.tensor(0.04999999701976776))


def test_diff_rec_loss(diff_rec_loss: DiffReconstructionLoss) -> None:
    """
    Test function for the diff_rec_loss.

    Args:
        diff_rec_loss (DiffReconstructionLoss): The diff_rec_loss object to be tested.
    """
    estimation = {
        "slice": torch.tensor([[[0.2, 0.8], [0.6, 0.4]]]),
        "noise_pred": torch.tensor([[[0.1, 0.3], [0.5, 0.7]]]),
    }
    target = {
        "slice": torch.tensor([[[0.1, 0.9], [0.3, 0.7]]]),
        "noisy_slice": torch.tensor([[[0.3, 0.7], [0.5, 0.9]]]),
        "noise_scale": torch.tensor([[[0.2, 0.4], [0.6, 0.8]]]),
    }
    result = diff_rec_loss(estimation, target)
    assert torch.isclose(result, torch.tensor(0.04229050874710083))
